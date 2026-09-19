import { classDocXmlToMarkdown } from './enrichment/class-doc-xml';
import {
    Splitter,
    CodeChunk,
    AstCodeSplitter,
    enforceChunkByteLimit,
    MILVUS_CONTENT_MAX_BYTES
} from './splitter';
import {
    Embedding,
    EmbeddingVector,
    OpenAIEmbedding
} from './embedding';
import {
    VectorDatabase,
    VectorDocument,
    VectorSearchResult,
    HybridSearchRequest,
    HybridSearchResult,
    RerankStrategy,
    InsertResult
} from './vectordb';
import { SemanticSearchResult } from './types';
import { envManager } from './utils/env-manager';
import {
    ChunkContextHeaderMode,
    ChunkContextFile,
    INDEX_TEXT_MAX_BYTES,
    resolveChunkContextHeaderMode,
    describeWithHeaderMode,
    headerModeFromDescription,
    buildChunkContextHeader,
    withChunkContextHeader,
    truncateUtf8,
    loadChunkContextFile,
    assertChunkContextCoverage,
    resolveChunkContextFilePath,
} from './utils/chunk-context-header';
import { classifyQuery, weightsForIntent, parseQualifiedName, parseSingleSymbol, isComparisonShape } from './search/query-classifier';
import { routeQuery, ChannelWeights, QueryShape } from './search/query-router';
import { buildSymbolFilter } from './search/symbol-routing';
import { GraphIndex, collectGraphCandidateIds, derivePackageFromPath } from './search/graph-expansion';
import { ProseGraphIndex, collectProseGraphCandidateIds } from './search/prose-graph-expansion';
import { buildComparisonBridgePool } from './search/comparison-bridge';
import { SerenaLspClient } from './search/serena-lsp-client';
import { runSymbolRefsPool, SymbolRefsParsed } from './search/symbol-refs-pool';
import { createSymbolFrequencyGate, loadSymbolFrequencyTable, SymbolFrequencyGate } from './search/symbol-frequency';
import { applyRewriting, RewriteFlags, RewriteResult } from './search/query-rewrite';
import { Reranker } from './reranker';
import { extractCandidateSymbols } from './enrichment';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { FileSynchronizer } from './sync/synchronizer';

/**
 * #19: an insert now reports the rows the vector database refused instead of
 * throwing the whole batch away. Alternative VectorDatabase implementations
 * (and the test doubles) still return void — nothing reported is nothing lost.
 */
function droppedCount(result: InsertResult | void): number {
    return result && Array.isArray(result.dropped) ? result.dropped.length : 0;
}

// rag-graph-layer Phase 2: in-memory accumulator that records, per chunk
// emitted during an indexCodebase run, just enough fields to build the
// cross-domain side-index without re-querying Milvus. `code` chunks contribute
// canonical entries; `doc`/`code_example` chunks contribute reverse-mention
// edges via their mentioned_symbols.
interface GraphAccumulatorChunk {
    chunkId: string;
    relativePath: string;
    contentType: string | undefined;
    symbolName: string | undefined;
    mentionedSymbols: string[] | undefined;
    // rag-graph-comparison-bridge v3-2: additional fields captured per chunk
    // so the side-builder can derive qualified_name + supertype edges without
    // re-querying Milvus.
    parentSymbol?: string;
    extendsName?: string;
    implementsList?: string[];
    // rag-graph-abstract-typedef-edges v3-3: Haxe abstract / typedef relations
    // feed the new `by_abstract_underlying` / `by_typedef_alias` side-index
    // buckets and per-symbol forward attributes used by the comparison bridge.
    abstractUnderlying?: string[];
    typedefAlias?: string;
}

class GraphAccumulator {
    private chunks: GraphAccumulatorChunk[] = [];

    add(entry: GraphAccumulatorChunk): void {
        this.chunks.push(entry);
    }

    snapshot(): readonly GraphAccumulatorChunk[] {
        return this.chunks;
    }
}

// rag-graph-comparison-bridge: helper moved to ./search/graph-expansion.ts as
// `derivePackageFromPath` so the bridge module can import it without a
// circular dep on context.ts. Re-exported here for completeness.
export { derivePackageFromPath } from './search/graph-expansion';

function parseHeadingPath(raw: string | undefined): string[] | undefined {
    if (!raw) return undefined;
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.map(String) : undefined;
    } catch {
        return undefined;
    }
}

// rag-graph-layer Phase 1.3: stored array fields are JSON-encoded VarChar
// columns; decode on the way out. Returns undefined if missing/malformed.
function parseStringArray(raw: string | undefined): string[] | undefined {
    if (!raw) return undefined;
    try {
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return undefined;
        const cleaned = parsed
            .filter((s): s is string => typeof s === 'string' && s.length > 0);
        return cleaned.length > 0 ? cleaned : undefined;
    } catch {
        return undefined;
    }
}

const CODE_DOMAIN_FILTER = `content_type in ["code","docstring"]`;
const DOC_DOMAIN_FILTER = `content_type in ["doc","code_example"]`;
const CODE_DOMAIN_TYPES = new Set(['code', 'docstring']);
const DOC_DOMAIN_TYPES = new Set(['doc', 'code_example']);

// Phase 0+ guarantee-slot threshold: if neither domain has at least this many
// hits in top-N when reranker is off, the floors below kick in to defend
// against RRF dropping a whole domain off the page.
const DEFAULT_GUARANTEE_CODE = 5;
const DEFAULT_GUARANTEE_DOC = 5;

/**
 * sweep-symbol-refs-pool-weight: one row of the pre-rerank candidate dump
 * (CANDIDATE_LOG_DIR). `pools` says which pool put the chunk into the
 * reranker's input and at what rank inside that pool; `survived` says whether
 * it is still there after the reranker's cut and the post-processing. Both are
 * what separates "the references never entered the input" from "they entered
 * and were cut".
 */
interface PreRerankCandidateRow {
    rank: number;
    chunk_id: string | null;
    relativePath: string;
    startLine: number;
    endLine: number;
    symbol_name: string | null;
    parent_symbol: string | null;
    content_type: string | null;
    score: number;
    pools: { pool: string; rank: number }[];
    survived: boolean;
    finalRank: number | null;
}

// Phase 2 reranker pool sizes. INPUT_K is how many merged candidates we
// hand to the reranker; OUTPUT_K is the cut after reranking. Both are
// overridable through env so eval / tuning runs can sweep without rebuilds.
const DEFAULT_RERANKER_INPUT_K = 50;
const DEFAULT_RERANKER_OUTPUT_K = 15;

// Outer weighted-RRF smoothing constant for cross-pool merge (code/doc).
// Smaller k = more weight on top ranks of each pool. Tunable via RRF_K env.
const DEFAULT_RRF_K = 60;

// Inner Milvus per-channel reranker (across dense / sparse_bm25 / sparse_learned).
// 'rrf' is uniform across channels; switch to 'weighted' if any
// CHANNEL_WEIGHT_* env var is set, with weights aligned to the channel order
// emitted by buildRequests (dense, sparse_bm25, sparse_learned).
const DEFAULT_MILVUS_RRF_K = 100;

// rag-graph-layer Phase 3.5: graph-pool weight in the outer weighted RRF.
// Lower than code/doc weights (1.0..1.5) by design — soft-bias rather
// than dominate the merge.
const DEFAULT_GRAPH_POOL_WEIGHT = 0.6;

// rag-graph-layer Phase 3.5: top-K of the *pre-graph* primary RRF that
// becomes seed material for the graph-expansion module. Aligned with
// RERANKER_INPUT_K so we never expand beyond the reranker's working set.
const DEFAULT_GRAPH_SEED_K = 50;

// prose-graph-deterministic §4.3: prose-graph-pool weight + seed-K in the
// outer weighted RRF (mirror the GRAPH_* defaults). Active only when
// PROSE_GRAPH_EXPAND=true.
const DEFAULT_PROSE_GRAPH_POOL_WEIGHT = 0.6;
const DEFAULT_PROSE_GRAPH_SEED_K = 50;

const DEFAULT_SUPPORTED_EXTENSIONS = [
    // Programming languages
    '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    '.dart', '.hx', '.hxml', '.ml', '.mli',
    // Text and markup files
    '.md', '.markdown', '.rst', '.ipynb',
    // '.xml' is admitted only for class-reference documents: processFileList
    // rewrites those to Markdown and SKIPS every other XML, so scene/meta files
    // are read once and dropped rather than indexed as markup.
    '.xml',
    // '.txt',  '.json', '.yaml', '.yml', '.html', '.htm',
    // '.css', '.scss', '.less', '.sql', '.sh', '.bash', '.env'
];

const DEFAULT_IGNORE_PATTERNS = [
    // Common build output and dependency directories
    'node_modules/**',
    'dist/**',
    'build/**',
    'out/**',
    'target/**',
    'coverage/**',
    '.nyc_output/**',

    // IDE and editor files
    '.vscode/**',
    '.idea/**',
    '*.swp',
    '*.swo',

    // Version control
    '.git/**',
    '.svn/**',
    '.hg/**',

    // Cache directories
    '.cache/**',
    '__pycache__/**',
    '.pytest_cache/**',

    // Test directories — kept out of the reference index. This MUST stay in
    // sync with the standalone indexers' EXTRA_IGNORES (infra/index-knowledge.js
    // + infra/update-knowledge.js). Previously only those scripts ignored tests,
    // while the MCP index_codebase / sync-watcher paths did not, so the two
    // fought over the same Milvus collections (scripts purging tests, the MCP
    // watcher re-adding them on the next change). Making it a DEFAULT pattern
    // gives both paths one source of truth.
    'tests/**',
    '**/tests/**',

    // Logs and temporary files
    'logs/**',
    'tmp/**',
    'temp/**',
    '*.log',

    // Environment and config files
    '.env',
    '.env.*',
    '*.local',

    // Minified and bundled files
    '*.min.js',
    '*.min.css',
    '*.min.map',
    '*.bundle.js',
    '*.bundle.css',
    '*.chunk.js',
    '*.vendor.js',
    '*.polyfills.js',
    '*.runtime.js',
    '*.map', // source map files
    'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'out',
    'target', '.vscode', '.idea', '__pycache__', '.pytest_cache',
    'coverage', '.nyc_output', 'logs', 'tmp', 'temp'
];

export interface ContextConfig {
    embedding?: Embedding;
    // prose-embedding-swap: optional second dense embedder for the prose
    // pool (`hybrid_v6_prose_<hash>`) under SPLIT_COLLECTIONS=true. When
    // omitted (or === `embedding`), prose chunks/queries use the same
    // embedder as code → behavior is byte-identical to pre-change v6.
    // The prose embedder MUST share the learned-sparse sidecar (its /sparse
    // call ignores `model`), so its EmbeddingVector still carries the shared
    // learned-sparse lexical_weights for the prose pool's sparse channel.
    proseEmbedding?: Embedding;
    vectorDatabase?: VectorDatabase;
    codeSplitter?: Splitter;
    supportedExtensions?: string[];
    ignorePatterns?: string[];
    customExtensions?: string[]; // New: custom extensions from MCP
    customIgnorePatterns?: string[]; // New: custom ignore patterns from MCP
    collectionNameOverride?: string; // Optional: custom collection name suffix
    reranker?: Reranker; // Phase 2: cross-encoder reranker run after weighted RRF merge
}

export class Context {
    private static readonly MAX_COLLECTION_NAME_LENGTH = 255;

    private embedding: Embedding;
    // prose-embedding-swap: distinct prose-pool dense embedder, or undefined
    // when the prose pool shares the code embedder (the default).
    private proseEmbedding?: Embedding;
    private vectorDatabase: VectorDatabase;
    private codeSplitter: Splitter;
    private supportedExtensions: string[];
    private baseIgnorePatterns: string[];
    private ignorePatterns: string[];
    private collectionNameOverride?: string;
    private warnedOverrideSanitization = new Set<string>();
    private synchronizers = new Map<string, FileSynchronizer>();
    private reranker?: Reranker;
    // Phase 3: per-run bootstrap of the symbol vocabulary. Populated during
    // indexCodebase, written to <codebasePath>/.symbols-vocab.json at the
    // end, then nulled. Search-time vocab loading is independent of this
    // (see loadSymbolVocabulary below).
    private indexedSymbols: Set<string> | null = null;
    private symbolVocabCache = new Map<string, ReadonlySet<string> | null>();
    // sweep-symbol-refs-pool-weight: provenance for the pre-rerank candidate
    // dump. weightedRrfMerge keeps only id -> summed score, so a dumped
    // candidate cannot otherwise say which pool put it into the reranker's
    // input. Collected ONLY while CANDIDATE_LOG_DIR is set (the eval harness;
    // the MCP server never sets it), which also means this instance state
    // assumes one search at a time in that process — the harness runs its
    // queries serially. Null = the branch is dormant, not merely silent.
    private candidateProvenance: Map<string, { pool: string; rank: number }[]> | null = null;
    private pendingCandidateDump: { query: string; rows: PreRerankCandidateRow[] } | null = null;

    // rag-graph-layer Phase 2: in-memory accumulator for the cross-domain
    // graph builder. Populated by processChunkBatch as each chunk gets
    // assigned its Milvus chunk_id; written out as `.symbols-graph.json`
    // at the end of indexCodebase. Null outside an active indexing run.
    private graphAccumulator: GraphAccumulator | null = null;
    // rag-graph-layer Phase 3: search-time cache of `.symbols-graph.json`.
    // Loaded lazily on first semanticSearch per codebasePath; null sentinel
    // stored so we don't re-attempt loading on every query when the file
    // is absent or malformed.
    private graphIndexCache = new Map<string, GraphIndex | null>();
    // Throttle warning rate for stale chunk-id misses (spec scenario
    // "Side-файл stale"): at most one warning per minute per process.
    private staleGraphChunkWarnedAt = 0;
    // rag-graph-layer Phase 3.7: print the wiring banner exactly once per
    // collection so the MCP log is grep-able for the runtime mode.
    private graphStartupBannerLogged = new Set<string>();
    // prose-graph-deterministic §4: lazy cached load of `.prose-graph.json`
    // (one per codebase), its stale-chunk warning throttle, and a one-shot
    // startup banner set. Populated only when PROSE_GRAPH_EXPAND=true → at the
    // default-OFF state none of these are ever touched.
    private proseGraphIndexCache = new Map<string, ProseGraphIndex | null>();
    private staleProseGraphChunkWarnedAt = 0;
    private proseGraphStartupBannerLogged = new Set<string>();
    // rag-symbol-refs-lsp-pool: lazy-init Serena LSP client (one per
    // Context instance, reused across queries). Created on the first
    // pool activation and reused to amortise the SSE handshake.
    private symbolRefsLspClient: SerenaLspClient | null = null;
    // local-rag #85: `close()` is a latch. Once it has run this Context creates
    // no replacement client, so a call that was in flight when it ran cannot
    // resurrect the transport its owner has already stopped holding.
    private closed = false;
    // rag-symbol-refs-multi-hop: print the symbol-refs pool wiring banner
    // exactly once per process so the MCP log is grep-able for hop mode.
    private symbolRefsStartupBannerLogged = false;
    // local-rag #64: the corpus-derived identifier frequency, loaded once per
    // codebase. A null entry is a corpus with no table, which is a real answer
    // (every frequency unknown, no bound, today's behaviour) and is cached as
    // such rather than re-read per query.
    private symbolFrequencyCache = new Map<string, SymbolFrequencyGate>();
    // local-rag #64: what the frequency gate did across this process's pool
    // activations, so a skip is a fact the RUN records rather than only a log
    // line. Read by the arm drivers into the context artifact's
    // `symbol_frequency.pool` block; a spec requirement, not diagnostics.
    // How many skipped subjects the process-lifetime record keeps by name. The
    // count is unbounded and is what a run is read by; the list is a sample, so
    // a long-lived MCP server cannot grow it without limit.
    private static readonly SYMBOL_REFS_SKIP_SAMPLE_CAP = 200;
    // ship-the-reference-expansion-bound: the shipped policy position the pool
    // bounds its reference expansion at. Measured, not chosen: the position
    // #72 nominated and the 7-row probe fired on, priced at $0 in
    // `infra/cross-corpus-runs/symbol-frequency-bound-2026-09-08/` (the skip
    // changed no emitted context on either slice while cutting Σ wall 16.3 %
    // and Serena calls 31 %) and priced for answer quality in
    // `infra/cross-corpus-runs/symbol-frequency-bound-knob-pair-2026-09-08/`
    // (whose negative finding is the ORDERING's, now a separate switch). infra
    // states the same number beside its own resolution
    // (`infra/lib/symbol-frequency.js`); a test reads both and asserts they
    // agree, because the two halves cannot import each other.
    static readonly DEFAULT_SYMBOL_REFS_MAX_SYMBOL_FREQUENCY_QUANTILE = 0.995;
    private symbolRefsFrequencyRecord: {
        activations: number;
        source: string | null;
        bound_documents: number | null;
        policy_quantile: number | null;
        skipped: { symbol: string; document_frequency: number; hop: number }[];
        skipped_total: number;
    } = { activations: 0, source: null, bound_documents: null, policy_quantile: null, skipped: [], skipped_total: 0 };
    // pilot-chunk-context-headers: the context file of a `generated` build,
    // loaded once; the header counters of the current indexing run; and the
    // header mode each searched collection records, for the retrieval stamp.
    private chunkContextFile: ChunkContextFile | null = null;
    private chunkContextReport: { missing_contexts: number; missing_by_file: Record<string, number>; header_truncated_embedding: number; header_truncated_index_text: number } = { missing_contexts: 0, missing_by_file: {}, header_truncated_embedding: 0, header_truncated_index_text: 0 };
    private searchedHeaderModes = new Map<string, string>();

    constructor(config: ContextConfig = {}) {
        // Initialize services
        this.embedding = config.embedding || new OpenAIEmbedding({
            apiKey: envManager.get('OPENAI_API_KEY') || 'your-openai-api-key',
            model: 'text-embedding-3-small',
            ...(envManager.get('OPENAI_BASE_URL') && { baseURL: envManager.get('OPENAI_BASE_URL') })
        });

        // prose-embedding-swap: only retain a distinct prose embedder when it
        // is actually a different instance from the code embedder. Callers
        // that pass the same instance (or none) collapse to the legacy path.
        this.proseEmbedding = (config.proseEmbedding && config.proseEmbedding !== this.embedding)
            ? config.proseEmbedding
            : undefined;
        if (this.proseEmbedding) {
            console.log(`[Context] 🧬 Prose pool uses a distinct dense embedder: ${this.proseEmbedding.getProvider()}`);
        }

        if (!config.vectorDatabase) {
            throw new Error('VectorDatabase is required. Please provide a vectorDatabase instance in the config.');
        }
        this.vectorDatabase = config.vectorDatabase;

        this.codeSplitter = config.codeSplitter || new AstCodeSplitter(2500, 300);

        // Load custom extensions from environment variables
        const envCustomExtensions = this.getCustomExtensionsFromEnv();

        // Combine default extensions with config extensions and env extensions
        const allSupportedExtensions = [
            ...DEFAULT_SUPPORTED_EXTENSIONS,
            ...(config.supportedExtensions || []),
            ...(config.customExtensions || []),
            ...envCustomExtensions
        ];
        // Remove duplicates
        this.supportedExtensions = [...new Set(allSupportedExtensions)];

        // Load custom ignore patterns from environment variables  
        const envCustomIgnorePatterns = this.getCustomIgnorePatternsFromEnv();

        // Start with default ignore patterns and persistent config/env patterns.
        const allIgnorePatterns = [
            ...DEFAULT_IGNORE_PATTERNS,
            ...(config.ignorePatterns || []),
            ...(config.customIgnorePatterns || []),
            ...envCustomIgnorePatterns
        ];
        this.baseIgnorePatterns = this.dedupePatterns(allIgnorePatterns);
        this.ignorePatterns = [...this.baseIgnorePatterns];
        this.collectionNameOverride = config.collectionNameOverride;
        this.reranker = config.reranker;

        console.log(`[Context] 🔧 Initialized with ${this.supportedExtensions.length} supported extensions and ${this.ignorePatterns.length} ignore patterns`);
        if (this.reranker) {
            console.log(`[Context] 🎯 Reranker enabled: provider=${this.reranker.getProvider()} input_k=${this.getRerankerInputK()} output_k=${this.getRerankerOutputK()}`);
        }
        if (envCustomExtensions.length > 0) {
            console.log(`[Context] 📎 Loaded ${envCustomExtensions.length} custom extensions from environment: ${envCustomExtensions.join(', ')}`);
        }
        if (envCustomIgnorePatterns.length > 0) {
            console.log(`[Context] 🚫 Loaded ${envCustomIgnorePatterns.length} custom ignore patterns from environment: ${envCustomIgnorePatterns.join(', ')}`);
        }
    }

    /**
     * local-rag #66: release every client this Context created.
     *
     * The symbol-refs pool creates a `SerenaLspClient` lazily and keeps it for
     * the process lifetime, and its MCP transport holds a standalone SSE `GET` —
     * a request that never settles — so its socket stays ref'd and keeps the
     * host's event loop alive. Without a published disposal every embedder of
     * this engine either leaks that server-side session on shutdown or reaches
     * into a private field of ours, which a rebuild can rename with no signal to
     * them (that is exactly what local-rag's arm drivers had to do in #63).
     *
     * Idempotent, and safe when the pool never activated: a process whose
     * prompts never made a symbol-refs call is the ordinary case. The client's
     * own `close()` terminates the server session before it closes the socket.
     * The reference is dropped BEFORE the close is awaited, so a close that
     * throws still leaves nothing behind.
     *
     * local-rag #85: it is a LATCH, not only a disposal. It used to be the
     * latter, and the discipline that made that safe — "close after your last
     * query, not beside it" — was prose no embedder could enforce. A call in
     * flight when `close()` ran is not abandoned, so it takes its ordinary retry
     * path; that path re-establishes an MCP transport whose standalone SSE `GET`
     * never settles; and nothing owns it, because the field was already nulled.
     * The host process is then held by a socket nobody can name, which is #63
     * arrived at through the method that fixed it. So an activation after
     * `close()` throws, and the in-flight call ends in a recorded error.
     *
     * Deliberately not closed here: the vector database and the embedding.
     * Neither holds a transport this engine established, which is what this
     * disposal is for. The vector database is required in the config and is the
     * caller's; the embedding usually is too, but NOT always — with no
     * `config.embedding` the constructor builds an `OpenAIEmbedding` of its own,
     * which publishes no disposal and keeps no session. "The caller handed it to
     * us" would therefore be false for that one, and the true reason is the one
     * stated here.
     */
    async close(): Promise<void> {
        this.closed = true;
        const lspClient = this.symbolRefsLspClient;
        this.symbolRefsLspClient = null;
        if (!lspClient) return;
        await lspClient.close();
    }

    /**
     * Get embedding instance
     */
    getEmbedding(): Embedding {
        return this.embedding;
    }

    /**
     * prose-embedding-swap: single resolution point for "which dense embedder
     * serves this pool". Code/symbol pools always use `this.embedding`
     * (the code dense model). The prose pool uses `this.proseEmbedding` when
     * one is wired AND split mode is active; otherwise it falls back to
     * `this.embedding` so the shared-embedder default path is byte-identical.
     */
    private hasDistinctProseEmbedding(): boolean {
        return this.proseEmbedding !== undefined;
    }

    private embeddingForPool(pool: 'code' | 'prose'): Embedding {
        if (pool === 'prose' && this.proseEmbedding) return this.proseEmbedding;
        return this.embedding;
    }

    /** pilot-chunk-context-headers: CHUNK_CONTEXT_HEADER, refused when unreadable. */
    private getChunkContextHeaderMode(): ChunkContextHeaderMode {
        return resolveChunkContextHeaderMode(envManager.get('CHUNK_CONTEXT_HEADER'));
    }

    /**
     * The context file a write indexes with. `fresh` (the start of every write: the full-build
     * check and the write guard) loads it from disk and pins it; every header of that write
     * then reads the pinned copy, so a served file rewritten mid-write cannot change what the
     * checked write inserts (serve-generated-chunk-contexts review).
     */
    private getChunkContexts(codebasePath: string, fresh = false): ChunkContextFile {
        if (fresh || !this.chunkContextFile) {
            this.chunkContextFile = loadChunkContextFile(resolveChunkContextFilePath(envManager.get('CHUNK_CONTEXT_FILE'), envManager.get('LOCAL_RAG_KNOWLEDGE_ROOT'), codebasePath));
        }
        return this.chunkContextFile;
    }

    /**
     * serve-generated-chunk-contexts: the chunks the indexer would write for a corpus tree
     * (every indexable file) or for a list of its files — the same discovery, split and
     * chunk ids as `indexCodebase` / `processFileList`. A file that cannot be read or is
     * skipped yields nothing, as in the indexing loop.
     */
    async splitForChunkContexts(
        codebasePath: string,
        options: { files?: string[]; additionalIgnorePatterns?: string[]; additionalSupportedExtensions?: string[]; splitter?: Splitter } = {}
    ): Promise<{ id: string; relativePath: string; startLine: number; endLine: number; content: string }[]> {
        const splitter = options.splitter || this.codeSplitter;
        const files = options.files
            ? options.files.map((f) => (path.isAbsolute(f) ? f : path.join(codebasePath, f)))
            : await this.getCodeFiles(codebasePath, await this.loadIgnorePatterns(codebasePath, options.additionalIgnorePatterns || []), this.getEffectiveSupportedExtensions(options.additionalSupportedExtensions || []));
        const maxBytes = Math.max(1, parseInt(envManager.get('CHUNK_MAX_BYTES') || String(MILVUS_CONTENT_MAX_BYTES), 10));
        const rows: { id: string; relativePath: string; startLine: number; endLine: number; content: string }[] = [];
        for (const filePath of files) {
            let split: { chunks: CodeChunk[] } | null;
            try {
                split = await this.splitFileForIndex(filePath, splitter, maxBytes, false);
            } catch {
                continue; // the indexing loop skips an unreadable file the same way
            }
            if (!split) continue;
            const relativePath = path.relative(codebasePath, filePath);
            for (const chunk of split.chunks) {
                const startLine = chunk.metadata.startLine || 0;
                const endLine = chunk.metadata.endLine || 0;
                rows.push({ id: this.generateId(relativePath, startLine, endLine, chunk.content, chunk.metadata.part), relativePath, startLine, endLine, content: chunk.content });
            }
        }
        return rows;
    }

    /** serve-generated-chunk-contexts: the header counters of the last write. */
    getChunkContextReport(): { missing_contexts: number; missing_by_file: Record<string, number>; header_truncated_embedding: number; header_truncated_index_text: number } {
        return { ...this.chunkContextReport, missing_by_file: { ...this.chunkContextReport.missing_by_file } };
    }

    /**
     * serve-generated-chunk-contexts: every write into collections must pass this before it
     * deletes or inserts a row. It refuses a collection built under another header mode;
     * in `generated` mode it refuses a context file generated over another chunk set than
     * the corpus this write leaves, and — unless CHUNK_CONTEXT_ACCEPT_MISSING=true says the
     * operator accepts metadata headers — a write that would insert a chunk without a
     * context. It resets the header counters for the write.
     */
    async assertWriteReady(
        codebasePath: string,
        targets: string[],
        filesToIndex: string[],
        options: { additionalIgnorePatterns?: string[]; additionalSupportedExtensions?: string[]; splitter?: Splitter } = {}
    ): Promise<{ mode: ChunkContextHeaderMode; inserted: number; missing: number }> {
        const mode = this.getChunkContextHeaderMode();
        for (const target of targets) {
            if (await this.vectorDatabase.hasCollection(target)) await this.assertCollectionHeaderMode(target, mode);
        }
        this.chunkContextReport = { missing_contexts: 0, missing_by_file: {}, header_truncated_embedding: 0, header_truncated_index_text: 0 };
        if (mode !== 'generated') return { mode, inserted: 0, missing: 0 };
        const contexts = this.getChunkContexts(codebasePath, true);
        const whole = await this.splitForChunkContexts(codebasePath, options);
        const { covered, total } = assertChunkContextCoverage(contexts, new Set(whole.map((r) => r.id)), 'CHUNK_CONTEXT_FILE', 0);
        const inserted = filesToIndex.length ? await this.splitForChunkContexts(codebasePath, { ...options, files: filesToIndex }) : [];
        const lacking = inserted.filter((r) => !contexts.contexts.has(r.id));
        console.log(`[Context] 🏷️  CHUNK_CONTEXT_FILE matches the corpus (${covered} of ${total} chunk ids covered); this write inserts ${inserted.length} chunk(s), ${lacking.length} without a context`);
        if (lacking.length && envManager.get('CHUNK_CONTEXT_ACCEPT_MISSING') !== 'true') {
            const files = [...new Set(lacking.map((r) => r.relativePath))];
            throw new Error(`[Context] ${lacking.length} chunk(s) this write inserts have no generated context (${files.slice(0, 5).join(', ')}${files.length > 5 ? `, … ${files.length} files` : ''}); run \`node infra/chunk-context-store.js prepare\` first, or set CHUNK_CONTEXT_ACCEPT_MISSING=true to index them with the metadata header`);
        }
        return { mode, inserted: inserted.length, missing: lacking.length };
    }

    /**
     * Refuse to write into a collection built under another header mode: its
     * rows would mix two indexed texts (and the BM25 input field would be
     * missing on one side).
     */
    private async assertCollectionHeaderMode(collectionName: string, mode: ChunkContextHeaderMode): Promise<void> {
        const recorded = headerModeFromDescription(await this.vectorDatabase.getCollectionDescription(collectionName));
        if (recorded !== mode) {
            throw new Error(`[Context] collection ${collectionName} was built with CHUNK_CONTEXT_HEADER=${recorded}; this run indexes with ${mode}`);
        }
    }

    /**
     * The header mode recorded on each collection a search read, keyed by
     * collection name. Read from the collections, never from the environment,
     * so an artifact states what its index was built with.
     */
    getChunkContextHeaderModes(): Record<string, string> {
        return Object.fromEntries([...this.searchedHeaderModes.entries()].sort(([a], [b]) => a.localeCompare(b)));
    }

    private async noteSearchedHeaderModes(codebasePath: string): Promise<void> {
        const addr = this.getCollectionAddress(codebasePath);
        for (const name of new Set([addr.code, addr.prose])) {
            if (this.searchedHeaderModes.has(name)) continue;
            try {
                this.searchedHeaderModes.set(name, headerModeFromDescription(await this.vectorDatabase.getCollectionDescription(name)));
            } catch (error) {
                console.warn(`[Context] ⚠️ could not read the header mode of ${name}: ${error}`);
            }
        }
    }

    /**
     * prose-embedding-swap: optional indexing-scope filter (env REINDEX_POOLS).
     * When set to `prose` or `code` (split mode only), indexCodebase touches
     * ONLY that pool's collection — the other collection is never created,
     * dropped, embedded, or upserted. This enables a prose-only re-index that
     * provably leaves the code pool untouched. Unset/empty (default) → both
     * pools, byte-identical to pre-change behavior.
     */
    private reindexPoolFilter(): Set<'prose' | 'code'> | null {
        const raw = (envManager.get('REINDEX_POOLS') || '').trim().toLowerCase();
        if (!raw) return null;
        const parts = raw.split(/[,\s]+/).filter(Boolean);
        const allowed = new Set<'prose' | 'code'>();
        for (const p of parts) {
            if (p === 'prose' || p === 'code') allowed.add(p);
        }
        return allowed.size > 0 ? allowed : null;
    }

    /**
     * Get vector database instance
     */
    getVectorDatabase(): VectorDatabase {
        return this.vectorDatabase;
    }

    /**
     * Get code splitter instance
     */
    getCodeSplitter(): Splitter {
        return this.codeSplitter;
    }

    /**
     * Get supported extensions
     */
    getSupportedExtensions(): string[] {
        return [...this.supportedExtensions];
    }

    /**
     * Get supported extensions for the current operation without mutating
     * the Context's persistent extension list.
     */
    getEffectiveSupportedExtensions(additionalExtensions: string[] = []): string[] {
        const normalizedExtensions = this.normalizeExtensions(additionalExtensions);
        return [...new Set([...this.supportedExtensions, ...normalizedExtensions])];
    }

    /**
     * Get ignore patterns
     */
    getIgnorePatterns(): string[] {
        return [...this.ignorePatterns];
    }

    /**
     * Get synchronizers map
     */
    getSynchronizers(): Map<string, FileSynchronizer> {
        return new Map(this.synchronizers);
    }

    /**
     * Set synchronizer for a collection
     */
    setSynchronizer(collectionName: string, synchronizer: FileSynchronizer): void {
        this.synchronizers.set(collectionName, synchronizer);
    }

    /**
     * Public wrapper for loadIgnorePatterns private method
     */
    async getLoadedIgnorePatterns(codebasePath: string): Promise<void> {
        await this.loadIgnorePatterns(codebasePath);
    }

    /**
     * Get the effective ignore patterns for a codebase without relying on
     * codebase-specific patterns already stored on this Context instance.
     */
    async getEffectiveIgnorePatterns(codebasePath: string, additionalIgnorePatterns: string[] = []): Promise<string[]> {
        return this.loadIgnorePatterns(codebasePath, additionalIgnorePatterns);
    }

    /**
     * Public wrapper for prepareCollection private method
     */
    async getPreparedCollection(codebasePath: string): Promise<void> {
        return this.prepareCollection(codebasePath);
    }

    /**
     * Get isHybrid setting from environment variable with default true
     */
    private getIsHybrid(): boolean {
        const isHybridEnv = envManager.get('HYBRID_MODE');
        if (isHybridEnv === undefined || isHybridEnv === null) {
            return true; // Default to true
        }
        return isHybridEnv.toLowerCase() === 'true';
    }

    /**
     * Phase 0+: per-domain multi-query controlled by MULTI_QUERY env. Defaults
     * to true so installs that opt into hybrid mode get domain coverage out
     * of the box. Set MULTI_QUERY=false to fall back to single-pool hybrid.
     */
    private getMultiQuery(): boolean {
        const env = envManager.get('MULTI_QUERY');
        if (env === undefined || env === null) {
            return true;
        }
        return env.toLowerCase() === 'true';
    }

    /**
     * code-collection-split: SPLIT_COLLECTIONS gate. When `true` the index
     * is physically split into two typed Milvus collections:
     *   - `hybrid_v6_prose_<hash>` for content_type IN ('doc', 'code_example')
     *   - `hybrid_v6_code_<hash>`  for content_type IN ('code', 'docstring')
     * Default `false` → legacy single-collection path (byte-identical to
     * pre-change behavior), preserving warm rollback.
     */
    private getSplitCollections(): boolean {
        const env = envManager.get('SPLIT_COLLECTIONS');
        if (env === undefined || env === null) return false;
        return env.toLowerCase() === 'true';
    }

    /**
     * rag-query-static-rewrite: per-feature env flags for the deterministic
     * query rewriters. All default to `false` until the per-feature bake-off
     * gates pass. Flags read lazily per call so .mcp.json edits propagate on
     * the next semanticSearch() invocation without process restart.
     */
    private getRewriteFlags(): RewriteFlags {
        const flag = (name: string): boolean => {
            const v = envManager.get(name);
            return typeof v === 'string' && v.toLowerCase() === 'true';
        };
        return {
            split: flag('QUERY_REWRITE_COMPARISON_SPLIT'),
            case: flag('QUERY_REWRITE_CASE_EXPANSION'),
            abbrev: flag('QUERY_REWRITE_ABBREV_EXPANSION'),
        };
    }

    /**
     * ablate-learned-sparse-channel: whether this query's learned-sparse vector
     * enters the hybrid request. `HYBRID_LEARNED_SPARSE=off` drops the
     * `sparse_learned` channel for every pool (query-side only — indexing never
     * reads it); unset/`on` keeps the channel whenever the vector is present.
     */
    private queriesLearnedSparse(queryEmbedding: EmbeddingVector): boolean {
        const mode = (envManager.get('HYBRID_LEARNED_SPARSE') || 'on').trim().toLowerCase();
        if (mode !== 'on' && mode !== 'off') {
            throw new Error(`HYBRID_LEARNED_SPARSE must be on or off, got ${JSON.stringify(mode)}`);
        }
        return mode === 'on' && !!(queryEmbedding.sparse && queryEmbedding.sparse.indices.length > 0);
    }

    /**
     * Build the per-channel hybrid-search requests for a single subject.
     * Dense and learned-sparse channels use the original `query` (rewriters
     * never touch them — see design D3); the BM25 sparse channel receives
     * `query` plus any whitelist/case extras appended as whitespace-joined
     * tokens.
     */
    private buildHybridRequests(
        queryEmbedding: EmbeddingVector,
        subject: string,
        sparseExtra: string[],
        limit: number,
    ): HybridSearchRequest[] {
        const sparseData = sparseExtra && sparseExtra.length > 0
            ? `${subject} ${sparseExtra.join(' ')}`
            : subject;
        const reqs: HybridSearchRequest[] = [
            { data: queryEmbedding.vector, anns_field: 'vector', param: { nprobe: 10 }, limit },
            { data: sparseData, anns_field: 'sparse_vector', param: { drop_ratio_search: 0.2 }, limit },
        ];
        if (queryEmbedding.sparse && this.queriesLearnedSparse(queryEmbedding)) {
            const sparseDict: Record<string, number> = {};
            const { indices, values } = queryEmbedding.sparse;
            const len = Math.min(indices.length, values.length);
            for (let i = 0; i < len; i++) {
                sparseDict[String(indices[i])] = values[i];
            }
            reqs.push({
                data: sparseDict,
                anns_field: 'sparse_learned',
                param: { drop_ratio_search: 0.2 },
                limit,
            });
        }
        return reqs;
    }

    /**
     * rag-query-static-rewrite (task 3.3): run the full multi-query hybrid
     * pipeline for a single subject string. Used once per call in the
     * non-split path and twice in the comparison-split path. Returns the
     * pre-graph merged result + the underlying pools so the caller can
     * apply graph-expansion / guarantee-slots as it sees fit.
     */
    private async runSubjectHybridPipeline(
        subject: string,
        sparseExtra: string[],
        codebasePath: string,
        collectionName: string,
        filterExpr: string | undefined,
        perPoolK: number,
        mergeLimit: number,
        channelWeights?: ChannelWeights,
    ): Promise<{
        mergedPreGraph: HybridSearchResult[];
        mergePools: { results: HybridSearchResult[]; weight: number; name?: string }[];
        codePool: HybridSearchResult[];
        docPool: HybridSearchResult[];
        symbolPool: HybridSearchResult[];
    }> {
        console.log(`[Context] 🔍 Generating embeddings for subject: "${subject}"`);
        const queryEmbedding: EmbeddingVector = await this.embedding.embed(subject);
        console.log(`[Context] ✅ Generated embedding vector with dimension: ${queryEmbedding.vector.length}`);

        const intent = classifyQuery(subject);
        const weights = weightsForIntent(intent);
        console.log(`[Context] 🔍 Subject intent: codeSignal=${intent.codeSignal} docSignal=${intent.docSignal} → weights code=${weights.code} doc=${weights.doc}`);

        // code-collection-split: address resolution. In split mode each pool
        // queries a different physically-typed collection — content_type
        // filter becomes unnecessary on those pools (the collection IS the
        // filter). In legacy mode addr.prose === addr.code === collectionName
        // and the content_type filter still does its job.
        const addr = this.getCollectionAddress(codebasePath);
        const combinedFilter = (extra: string | undefined): string => {
            if (!extra || extra.trim().length === 0) return '';
            return `(${extra})`;
        };
        const userFilter = filterExpr && filterExpr.trim().length > 0 ? filterExpr : undefined;
        const codeExpr = addr.isSplit
            ? (userFilter ? combinedFilter(userFilter) : '')
            : (userFilter
                ? `${combinedFilter(userFilter)} and ${CODE_DOMAIN_FILTER}`
                : CODE_DOMAIN_FILTER);
        const docExpr = addr.isSplit
            ? (userFilter ? combinedFilter(userFilter) : '')
            : (userFilter
                ? `${combinedFilter(userFilter)} and ${DOC_DOMAIN_FILTER}`
                : DOC_DOMAIN_FILTER);
        const codeCollection = addr.isSplit ? addr.code : collectionName;
        const docCollection = addr.isSplit ? addr.prose : collectionName;

        // prose-embedding-swap: when the prose pool has a distinct dense model
        // (split mode only), embed the query a second time through it. The
        // prose embedder shares the learned-sparse sidecar, so its
        // EmbeddingVector carries the prose dense vector + the shared
        // learned-sparse lexical weights — exactly what the prose pool needs
        // (dense per-pool, sparse shared). Default path: docQueryEmbedding ===
        // queryEmbedding (no extra call, byte-identical).
        let docQueryEmbedding: EmbeddingVector = queryEmbedding;
        if (addr.isSplit && this.hasDistinctProseEmbedding()) {
            docQueryEmbedding = await this.embeddingForPool('prose').embed(subject);
            console.log(`[Context] 🧬 Prose-pool query embedding dim: ${docQueryEmbedding.vector.length}`);
        }
        // Symbol-routing pool stays code-side: its filter targets
        // code/docstring chunks by symbol_name. In split mode that lives
        // entirely in the code-collection.
        const symbolCollection = addr.isSplit ? addr.code : collectionName;

        const innerRerank = this.buildInnerRerankStrategy(
            this.queriesLearnedSparse(queryEmbedding),
            channelWeights,
        );

        const symbolPoolLimit = 10;
        const symbolFilterExpr = await this.buildSymbolPoolFilter(subject, intent, codebasePath);
        const symbolPoolPromise: Promise<HybridSearchResult[]> = symbolFilterExpr
            ? this.vectorDatabase.hybridSearch(
                symbolCollection,
                this.buildHybridRequests(queryEmbedding, subject, sparseExtra, symbolPoolLimit),
                {
                    rerank: innerRerank,
                    limit: symbolPoolLimit,
                    filterExpr: symbolFilterExpr,
                }
            ).catch((err) => {
                console.warn(`[Context] ⚠️ symbol-routing pool failed: ${err}`);
                return [] as HybridSearchResult[];
            })
            : Promise.resolve([] as HybridSearchResult[]);

        // rag-symbol-refs-lsp-pool: 4th retrieval channel via Serena LSP.
        // Gated independently of symbol-routing — this pool finds references
        // and implementations rather than co-located metadata matches.
        // Pool reads chunks by symbol_name + filter `content_type in
        // ['code','docstring']` → code-collection in split mode.
        const symbolRefsPoolPromise = this.runSymbolRefsPoolForSubject(subject, intent, codebasePath, symbolCollection);

        const [codePool, docPool, symbolPool, symbolRefsPool] = await Promise.all([
            this.vectorDatabase.hybridSearch(
                codeCollection,
                this.buildHybridRequests(queryEmbedding, subject, sparseExtra, perPoolK),
                { rerank: innerRerank, limit: perPoolK, filterExpr: codeExpr || undefined },
            ).catch((err) => {
                console.warn(`[Context] ⚠️ code-domain hybrid search failed: ${err}`);
                return [] as HybridSearchResult[];
            }),
            this.vectorDatabase.hybridSearch(
                docCollection,
                this.buildHybridRequests(docQueryEmbedding, subject, sparseExtra, perPoolK),
                { rerank: innerRerank, limit: perPoolK, filterExpr: docExpr || undefined },
            ).catch((err) => {
                console.warn(`[Context] ⚠️ doc-domain hybrid search failed: ${err}`);
                return [] as HybridSearchResult[];
            }),
            symbolPoolPromise,
            symbolRefsPoolPromise,
        ]);

        console.log(`[Context] 🔍 Pool sizes for "${subject}": code=${codePool.length} doc=${docPool.length} symbol=${symbolPool.length} symbolRefs=${symbolRefsPool.length}`);

        const SYMBOL_POOL_WEIGHT = 2.0;
        const mergePools: { results: HybridSearchResult[]; weight: number; name?: string }[] = [
            { results: codePool, weight: weights.code, name: 'code' },
            { results: docPool, weight: weights.doc, name: 'doc' },
        ];
        if (symbolPool.length > 0) {
            mergePools.push({ results: symbolPool, weight: SYMBOL_POOL_WEIGHT, name: 'symbolRouting' });
        }
        // Pool participates in the merge even when weight is 0, so callers
        // can A/B "compute but ignore" via SYMBOL_REFS_POOL_WEIGHT=0.
        if (symbolRefsPool.length > 0 && this.getSymbolRefsPool()) {
            mergePools.push({ results: symbolRefsPool, weight: this.getSymbolRefsPoolWeight(), name: 'symbolRefs' });
        }
        const mergedPreGraph = this.weightedRrfMerge(mergePools, mergeLimit, this.getRrfK());
        return { mergedPreGraph, mergePools, codePool, docPool, symbolPool };
    }

    /**
     * rag-symbol-refs-lsp-pool: build the 4th pool by walking LSP
     * declaration → references + implementations from the Serena daemon.
     * Returns [] when any activation gate fails (env off, no parse, vocab
     * miss, daemon unreachable, engine released). Never throws — pool degrades
     * to a no-op so the other 3 pools continue to serve; the caller's
     * `Promise.all` gives this promise no `.catch()` of its own, so anything
     * that escapes here fails the whole hybrid search.
     */
    private async runSymbolRefsPoolForSubject(
        subject: string,
        intent: { codeSignal: boolean; docSignal: boolean },
        codebasePath: string,
        collectionName: string,
    ): Promise<HybridSearchResult[]> {
        if (!this.getSymbolRefsPool()) return [];
        if (!intent.codeSignal) return [];
        if (!this.symbolRefsStartupBannerLogged) {
            const hops = this.getSymbolRefsMaxHops();
            if (hops >= 2) {
                console.log(`[Context] symbol-refs pool: maxHops=${hops} (hop2 seeds=${this.getSymbolRefsMaxHop1Seeds()}, refs=${this.getSymbolRefsMaxHop2Refs()})`);
            } else {
                console.log(`[Context] symbol-refs pool: single-hop only`);
            }
            this.symbolRefsStartupBannerLogged = true;
        }
        const qualified = parseQualifiedName(subject);
        let parsed: SymbolRefsParsed | null = qualified;
        let vocab: ReadonlySet<string> | null = null;
        try {
            vocab = await this.loadSymbolVocabulary(codebasePath);
        } catch (err) {
            console.warn(`[Context] ⚠️ symbol-refs vocab load failed: ${err}`);
            return [];
        }
        if (qualified) {
            // Same vocab gate as the symbol-routing pool — at minimum the
            // class component must be a known project symbol.
            if (vocab && !vocab.has(qualified.className)) return [];
        } else {
            const single = parseSingleSymbol(subject, vocab);
            if (!single) return [];
            parsed = single;
        }
        if (!parsed) return [];
        try {
            // local-rag #85: INSIDE the try. The creator refuses on a released
            // engine, and this pool's contract — stated in the docblock above and
            // relied on by `Promise.all` in the caller, which gives this promise
            // no `.catch()` of its own the way it does its three siblings — is
            // that it degrades to a no-op rather than failing the query. A latch
            // that failed the whole hybrid search would trade a leaked transport
            // for a lost answer.
            const lspClient = this.getOrCreateSymbolRefsLspClient();
            return await runSymbolRefsPool({
                query: subject,
                parsed,
                lspClient,
                vectorDatabase: this.vectorDatabase,
                collection: collectionName,
                codebasePath,
                maxRefs: this.getSymbolRefsMaxReferences(),
                maxImpls: this.getSymbolRefsMaxImplementations(),
                rrfK: this.getRrfK(),
                maxHops: this.getSymbolRefsMaxHops(),
                maxHop1Seeds: this.getSymbolRefsMaxHop1Seeds(),
                maxHop2Refs: this.getSymbolRefsMaxHop2Refs(),
                symbolFrequency: this.getSymbolFrequencyGate(codebasePath),
                onDiagnostics: (d) => {
                    this.symbolRefsFrequencyRecord.activations += 1;
                    this.symbolRefsFrequencyRecord.source = d.source;
                    this.symbolRefsFrequencyRecord.bound_documents = d.bound_documents;
                    this.symbolRefsFrequencyRecord.policy_quantile = d.policy_quantile;
                    // Bounded: this record lives for the process, and a
                    // long-lived MCP server would otherwise grow it without
                    // limit. The count is what a run is read by; the list is a
                    // sample, and says so when it is one.
                    for (const skip of d.skipped) {
                        this.symbolRefsFrequencyRecord.skipped_total += 1;
                        if (this.symbolRefsFrequencyRecord.skipped.length < Context.SYMBOL_REFS_SKIP_SAMPLE_CAP) this.symbolRefsFrequencyRecord.skipped.push(skip);
                    }
                },
            });
        } catch (err) {
            console.warn(`[Context] ⚠️ symbol-refs pool failed: ${err}`);
            return [];
        }
    }

    /**
     * rag-query-static-rewrite (task 3.4): single-line log describing what
     * the rewriters fired on this query. No log emitted when no flag is on
     * (no-op preserves the pre-change log surface).
     */
    private logQueryRewrite(query: string, rewrite: RewriteResult, flags: RewriteFlags): void {
        const anyFlagOn = flags.split || flags.case || flags.abbrev;
        if (!anyFlagOn) return;
        const parts: string[] = [];
        if (flags.split) {
            if (rewrite.kind === 'split') {
                parts.push(`split={${rewrite.left}}|{${rewrite.right}}`);
            } else if (rewrite.debug.comparisonMatchedTrigger) {
                parts.push(`split=rejected (trigger="${rewrite.debug.comparisonMatchedTrigger}", subject length guard)`);
            }
        }
        if (flags.case && rewrite.debug.caseExpansions.length > 0) {
            const items = rewrite.debug.caseExpansions
                .map((e) => `${e.from}→${e.to.join('|')}`)
                .join(', ');
            parts.push(`case+={${items}}`);
        }
        if (flags.abbrev && rewrite.debug.abbrevExpansions.length > 0) {
            const items = rewrite.debug.abbrevExpansions
                .map((e) => `${e.from}→${e.to.join('|')}`)
                .join(', ');
            parts.push(`abbrev+={${items}}`);
        }
        if (parts.length === 0) return;
        console.log(`[Context] 🔍 Query rewrite: ${parts.join(' | ')}`);
    }

    /**
     * Phase 2: reranker presence flips guarantee-slots off because the
     * reranker arbitrates the top-N better than fixed floors. Driven by
     * whether a Reranker instance was wired into the constructor (the MCP
     * factory only builds one when `RERANKER_PROVIDER` is set).
     */
    private hasReranker(): boolean {
        return !!this.reranker;
    }

    /**
     * Public accessor (mirrors getEmbedding/getVectorDatabase) so consumers
     * can introspect or replace the reranker without subclassing Context.
     */
    getReranker(): Reranker | undefined {
        return this.reranker;
    }

    private getPositiveIntFromEnv(name: string, fallback: number): number {
        const raw = envManager.get(name);
        if (!raw) return fallback;
        const n = Number(raw);
        if (!Number.isInteger(n) || n <= 0) {
            console.warn(`[Context] ⚠️ Ignoring invalid ${name}=${raw}; expected positive integer, falling back to ${fallback}`);
            return fallback;
        }
        return n;
    }

    private getRerankerInputK(): number {
        return this.getPositiveIntFromEnv('RERANKER_INPUT_K', DEFAULT_RERANKER_INPUT_K);
    }

    private getRerankerOutputK(): number {
        return this.getPositiveIntFromEnv('RERANKER_OUTPUT_K', DEFAULT_RERANKER_OUTPUT_K);
    }

    private getRrfK(): number {
        return this.getPositiveIntFromEnv('RRF_K', DEFAULT_RRF_K);
    }

    private getMilvusRrfK(): number {
        return this.getPositiveIntFromEnv('MILVUS_RRF_K', DEFAULT_MILVUS_RRF_K);
    }

    /**
     * rag-graph-layer Phase 3.4: env-gated graph-expansion level.
     *   `0` (default): pipeline runs identically to the pre-graph baseline.
     *   `1`: 1-hop expansion through `.symbols-graph.json`.
     *   `2`: reserved for the deferred 2-hop sub-change (D7).
     */
    private getGraphExpand(): number {
        const raw = envManager.get('GRAPH_EXPAND');
        if (!raw) return 0;
        const n = Number(raw);
        if (!Number.isFinite(n) || n < 0) {
            console.warn(`[Context] ⚠️ Ignoring invalid GRAPH_EXPAND=${raw}; expected non-negative integer, falling back to 0`);
            return 0;
        }
        // Only 0/1 are wired; >=2 maps to "no-op until 2-hop sub-change ships".
        if (n >= 2) return 1;
        return Math.floor(n);
    }

    private getGraphPoolWeight(): number {
        return this.getNonNegativeFloatFromEnv('GRAPH_POOL_WEIGHT', DEFAULT_GRAPH_POOL_WEIGHT);
    }

    private getGraphSeedK(): number {
        return this.getPositiveIntFromEnv('GRAPH_SEED_K', DEFAULT_GRAPH_SEED_K);
    }

    // ---- prose-graph-deterministic §4.3: env getters --------------------

    /**
     * prose-graph-deterministic §4: master flag for prose-graph expansion.
     * default `false` → the prose-graph module is never loaded and the
     * pipeline is identical pre-change. Flip requires the deep-reference-eval
     * gate (see spec «Prose-graph acceptance gate за deep-reference-eval»).
     */
    private getProseGraphExpand(): boolean {
        return (envManager.get('PROSE_GRAPH_EXPAND') || 'false').trim().toLowerCase() === 'true';
    }

    private getProseGraphPoolWeight(): number {
        return this.getNonNegativeFloatFromEnv('PROSE_GRAPH_POOL_WEIGHT', DEFAULT_PROSE_GRAPH_POOL_WEIGHT);
    }

    private getProseGraphSeedK(): number {
        return this.getPositiveIntFromEnv('PROSE_GRAPH_SEED_K', DEFAULT_PROSE_GRAPH_SEED_K);
    }

    // ---- rag-symbol-refs-lsp-pool: env getters --------------------------

    private getSymbolRefsPool(): boolean {
        return (envManager.get('SYMBOL_REFS_POOL') || 'false').toLowerCase() === 'true';
    }

    private getSymbolRefsPoolWeight(): number {
        const raw = this.getNonNegativeFloatFromEnv('SYMBOL_REFS_POOL_WEIGHT', 1.0);
        return Math.min(raw, 3.0);
    }

    private getSymbolRefsLspBaseUrl(): string | undefined {
        const raw = (envManager.get('SYMBOL_REFS_LSP_BASE_URL') || '').trim();
        return raw.length > 0 ? raw : undefined;
    }

    private getSymbolRefsLspTimeoutMs(): number {
        return this.getPositiveIntFromEnv('SYMBOL_REFS_LSP_TIMEOUT_MS', 1500);
    }

    private getSymbolRefsMaxReferences(): number {
        return this.getPositiveIntFromEnv('SYMBOL_REFS_MAX_REFERENCES', 20);
    }

    private getSymbolRefsMaxImplementations(): number {
        return this.getPositiveIntFromEnv('SYMBOL_REFS_MAX_IMPLEMENTATIONS', 10);
    }

    // ---- rag-symbol-refs-multi-hop: env getters -------------------------

    /**
     * local-rag #64, shipped by ship-the-reference-expansion-bound: the policy
     * position in the corpus's own document-frequency distribution above which
     * the pool stops expanding references. A POSITION, not a count: the
     * threshold in documents is a property of the corpus the build ran over
     * and is derived per corpus. Unset or empty means the SHIPPED DEFAULT —
     * the bound is on, at the position priced in
     * `infra/cross-corpus-runs/symbol-frequency-bound-2026-09-08/` (the skip
     * changed no emitted context on either slice while cutting Σ wall 16.3 %
     * and Serena calls 31 %). An invalid value warns and takes the default
     * too: garbage must not fail open to unbounded expansion. Explicitly `0`
     * is the pre-#64 behaviour, by decision. Stamped through the
     * `SYMBOL_REFS_` prefix because it changes what the pool emits.
     */
    private getSymbolRefsMaxSymbolFrequencyQuantile(): number {
        const raw = (envManager.get('SYMBOL_REFS_MAX_SYMBOL_FREQUENCY_QUANTILE') || '').trim();
        if (raw.length === 0) return Context.DEFAULT_SYMBOL_REFS_MAX_SYMBOL_FREQUENCY_QUANTILE;
        const q = Number(raw);
        if (!Number.isFinite(q) || q < 0 || q > 1) {
            console.warn(`[Context] ⚠️ Ignoring invalid SYMBOL_REFS_MAX_SYMBOL_FREQUENCY_QUANTILE=${raw}; expected a quantile in [0, 1], falling back to the shipped default ${Context.DEFAULT_SYMBOL_REFS_MAX_SYMBOL_FREQUENCY_QUANTILE}`);
            return Context.DEFAULT_SYMBOL_REFS_MAX_SYMBOL_FREQUENCY_QUANTILE;
        }
        return q;
    }

    /**
     * local-rag #64: the frequency gate for one codebase, cached. The table is
     * the artifact `infra/lib/symbol-frequency.js` writes beside the symbol
     * vocabulary; a corpus without one yields a gate whose source is `absent`,
     * whose bound is null and whose every frequency is unknown.
     */
    private getSymbolFrequencyGate(codebasePath: string): SymbolFrequencyGate {
        const cached = this.symbolFrequencyCache.get(codebasePath);
        if (cached) return cached;
        const quantile = this.getSymbolRefsMaxSymbolFrequencyQuantile();
        const gate = createSymbolFrequencyGate(loadSymbolFrequencyTable(codebasePath), quantile);
        this.symbolFrequencyCache.set(codebasePath, gate);
        if (gate.source === 'absent') {
            console.log('[Context] 🔢 symbol frequency: no derived table beside the corpus — reference expansion is unbounded (pre-#64 behaviour)');
        } else {
            console.log(`[Context] 🔢 symbol frequency: ${gate.identifiers} identifiers over ${gate.documents} documents${gate.boundDocuments === null ? ', no bound configured' : `, bound ${gate.boundDocuments} documents (quantile ${gate.boundQuantile})`}`);
        }
        return gate;
    }

    /**
     * local-rag #64: what the symbol-refs pool's frequency gate did over this
     * process — how many activations consulted it, which source they read, the
     * bound in documents they applied, and every subject whose reference
     * expansion was skipped. Written into the arm artifacts so the skip is
     * recorded rather than being observable only as an absence of rows.
     */
    getSymbolRefsFrequencyRecord(): {
        activations: number;
        source: string | null;
        bound_documents: number | null;
        policy_quantile: number | null;
        skipped: { symbol: string; document_frequency: number; hop: number }[];
        skipped_total: number;
        skipped_sampled: boolean;
    } {
        return {
            ...this.symbolRefsFrequencyRecord,
            skipped: this.symbolRefsFrequencyRecord.skipped.slice(),
            skipped_sampled: this.symbolRefsFrequencyRecord.skipped_total > this.symbolRefsFrequencyRecord.skipped.length,
        };
    }

    private getSymbolRefsMaxHops(): number {
        const raw = envManager.get('SYMBOL_REFS_MAX_HOPS');
        if (!raw) return 1;
        const n = Number(raw);
        if (!Number.isInteger(n) || n < 1) {
            console.warn(`[Context] ⚠️ Ignoring invalid SYMBOL_REFS_MAX_HOPS=${raw}; expected integer >= 1, falling back to 1`);
            return 1;
        }
        if (n > 2) {
            console.warn(`[Context] ⚠️ SYMBOL_REFS_MAX_HOPS=${n} clamped to 2 (3+ hops not implemented)`);
            return 2;
        }
        return n;
    }

    private getSymbolRefsMaxHop1Seeds(): number {
        const n = this.getPositiveIntFromEnv('SYMBOL_REFS_MAX_HOP1_SEEDS', 3);
        return Math.min(Math.max(n, 1), 10);
    }

    private getSymbolRefsMaxHop2Refs(): number {
        const n = this.getPositiveIntFromEnv('SYMBOL_REFS_MAX_HOP2_REFS', 3);
        return Math.min(Math.max(n, 1), 10);
    }

    // ---- rag-graph-comparison-bridge: env getters -----------------------

    private getComparisonBridgeEnabled(): boolean {
        const raw = (envManager.get('COMPARISON_BRIDGE_ENABLED') || '0').trim();
        return raw === '1' || raw.toLowerCase() === 'true';
    }

    private getComparisonBridgePoolWeight(): number {
        return this.getNonNegativeFloatFromEnv('COMPARISON_BRIDGE_POOL_WEIGHT', 0.6);
    }

    private getComparisonBridgeMaxPartners(): number {
        const n = this.getPositiveIntFromEnv('COMPARISON_BRIDGE_MAX_PARTNERS', 8);
        return Math.min(n, 50);
    }

    private getComparisonBridgeMaxPackageFanout(): number {
        // Default 30 matches the file-level bucket cap, so every bucket
        // passes the fanout filter by default and the cap-30 already keeps
        // the data bounded. Lower values intentionally disable bridging on
        // larger packages (useful for noisy meta-packages).
        const n = this.getPositiveIntFromEnv('COMPARISON_BRIDGE_MAX_PACKAGE_FANOUT', 30);
        return Math.min(n, 100);
    }

    private getComparisonBridgeDebug(): boolean {
        const raw = (envManager.get('DEBUG_COMPARISON_BRIDGE') || '0').trim();
        return raw === '1' || raw.toLowerCase() === 'true';
    }

    // ---- rag-comparison-bridge-reranker-bypass: env getter -------------

    private getComparisonBridgeBypassSlots(): number {
        const raw = (envManager.get('COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS') || '0').trim();
        const n = parseInt(raw, 10);
        if (!Number.isFinite(n) || n < 0) return 0;
        return Math.min(n, 10);
    }

    /**
     * Lazy-init the SerenaLspClient on first activation. Reused across
     * subsequent queries; the client itself caches the service URL for 30s
     * and the MCP connection for the process lifetime. The Serena service is
     * bound to one corpus, so the codebase path is not part of its address.
     */
    private getOrCreateSymbolRefsLspClient(): SerenaLspClient {
        // local-rag #85: a released Context creates nothing. A late retry that
        // reached here would open a transport this Context no longer holds a
        // reference to, so it fails loudly instead — the embedder that closed
        // beside its work learns that it did, rather than losing the process to a
        // socket it cannot find.
        if (this.closed) {
            throw new Error('Context has been closed: its symbol-refs client cannot be re-created. Close the engine after its last query, not beside one.');
        }
        if (!this.symbolRefsLspClient) {
            this.symbolRefsLspClient = new SerenaLspClient({
                baseUrlOverride: this.getSymbolRefsLspBaseUrl(),
                timeoutMs: this.getSymbolRefsLspTimeoutMs(),
            });
        }
        return this.symbolRefsLspClient;
    }

    /**
     * rag-graph-layer Phase 3.4: lazy cached load of the per-codebase
     * `.symbols-graph.json`. Returns null sentinel on missing/parse-error;
     * subsequent calls hit the cache without re-reading the disk.
     */
    private loadGraphIndex(codebasePath: string): GraphIndex | null {
        if (this.graphIndexCache.has(codebasePath)) {
            return this.graphIndexCache.get(codebasePath) ?? null;
        }
        const sidePath = path.join(codebasePath, '.symbols-graph.json');
        if (!fs.existsSync(sidePath)) {
            this.graphIndexCache.set(codebasePath, null);
            return null;
        }
        const idx = GraphIndex.load(sidePath);
        this.graphIndexCache.set(codebasePath, idx);
        if (idx) {
            console.log(`[Context] 🕸️  Loaded graph index (${idx.symbolCount} symbols, version ${idx.version}) from ${sidePath}`);
        }
        return idx;
    }

    /**
     * prose-graph-deterministic §4.1 / §4.4: lazy cached load of the
     * per-codebase `.prose-graph.json`. Returns null sentinel on
     * missing/parse-error/version-mismatch so prose-graph expansion is
     * gracefully disabled regardless of PROSE_GRAPH_EXPAND; subsequent calls
     * hit the cache without re-reading disk. A one-shot banner records the
     * reason so operators can grep the MCP log.
     */
    private loadProseGraphIndex(codebasePath: string): ProseGraphIndex | null {
        if (this.proseGraphIndexCache.has(codebasePath)) {
            return this.proseGraphIndexCache.get(codebasePath) ?? null;
        }
        const sidePath = path.join(codebasePath, '.prose-graph.json');
        const bannerOnce = (msg: string, warn = false): void => {
            if (this.proseGraphStartupBannerLogged.has(sidePath)) return;
            this.proseGraphStartupBannerLogged.add(sidePath);
            if (warn) console.warn(msg); else console.log(msg);
        };
        if (!fs.existsSync(sidePath)) {
            this.proseGraphIndexCache.set(codebasePath, null);
            bannerOnce('[Context] 🪢 prose-graph index not loaded: side-file missing; PROSE_GRAPH_EXPAND ignored', true);
            return null;
        }
        const idx = ProseGraphIndex.load(sidePath);
        this.proseGraphIndexCache.set(codebasePath, idx);
        if (idx) {
            bannerOnce(`[Context] 🪢 prose-graph expansion: enabled, ${idx.nodeCount} nodes / ${idx.edgeCount} edges (version ${idx.version}) from ${sidePath}`);
        } else {
            bannerOnce('[Context] 🪢 prose-graph index not loaded: malformed or version-mismatch; PROSE_GRAPH_EXPAND ignored', true);
        }
        return idx;
    }

    /**
     * rag-graph-layer Phase 3.7: log the runtime graph-expansion status
     * once per (collection, configuration) pair. Distinguishes "explicitly
     * disabled", "side-file missing", "v2 incompatible", and "enabled" so
     * operators can grep the MCP startup log to confirm the mode.
     */
    private logGraphStartupBanner(collectionName: string, codebasePath: string): void {
        const expand = this.getGraphExpand();
        const collectionVersion = (envManager.get('COLLECTION_VERSION') || '').trim().toLowerCase();
        const key = `${collectionName}:${expand}:${collectionVersion}`;
        if (this.graphStartupBannerLogged.has(key)) return;
        this.graphStartupBannerLogged.add(key);
        if (expand === 0) {
            console.log('[Context] 🕸️  graph-expansion: disabled (GRAPH_EXPAND=0)');
            return;
        }
        // code-collection-split: v6 carries the same graph schema (v3-3) as
        // v3 — reindex regenerates side-files unchanged. Accept both.
        if (collectionVersion && collectionVersion !== 'v3' && collectionVersion !== 'v6') {
            console.warn(
                `[Context] ⚠️  GRAPH_EXPAND=${expand} requires COLLECTION_VERSION=v3 or v6; ` +
                `current is "${collectionVersion}" — graph-expansion disabled`,
            );
            return;
        }
        const idx = this.loadGraphIndex(codebasePath);
        if (!idx) {
            console.warn(
                '[Context] ⚠️  graph-expansion: disabled (side-file missing or malformed); ' +
                'GRAPH_EXPAND ignored — reindex with v3 to populate `.symbols-graph.json`',
            );
            return;
        }
        console.log(`[Context] 🕸️  graph-expansion: enabled, ${idx.symbolCount} symbols loaded (hop=${expand})`);
    }

    private getNonNegativeFloatFromEnv(name: string, fallback: number): number {
        const raw = envManager.get(name);
        if (raw === undefined || raw === null || raw === '') return fallback;
        const n = Number(raw);
        if (!Number.isFinite(n) || n < 0) {
            console.warn(`[Context] ⚠️ Ignoring invalid ${name}=${raw}; expected non-negative number, falling back to ${fallback}`);
            return fallback;
        }
        return n;
    }

    /**
     * knowledge-router: QUERY_ROUTER_CHANNEL_WEIGHTS — when true, the inner
     * per-channel ranker weights become a function of the query's
     * lexical_form. Default false → static Phase-4 channel defaults.
     */
    private getQueryRouterChannelWeights(): boolean {
        return (envManager.get('QUERY_ROUTER_CHANNEL_WEIGHTS') || 'false').toLowerCase() === 'true';
    }

    /**
     * knowledge-router: CONCEPT_SPAN_QUOTA — when true, concept-shaped
     * queries get a post-reranker content-type quota. Default false.
     */
    private getConceptSpanQuota(): boolean {
        return (envManager.get('CONCEPT_SPAN_QUOTA') || 'false').toLowerCase() === 'true';
    }

    /**
     * Build the Milvus inner-rerank strategy for the per-pool 3-channel
     * hybrid_search. If any CHANNEL_WEIGHT_* env var is set, switches to
     * Milvus 'weighted' ranker with weights aligned to buildRequests order:
     *   [dense, sparse_bm25, sparse_learned]
     * Otherwise stays on 'rrf' with k = MILVUS_RRF_K (default 100).
     *
     * `hasLearnedSparse` reflects whether the third channel is included in
     * this particular request (some queries have no learned-sparse vector).
     *
     * knowledge-router: when `channelWeights` is supplied (router active via
     * QUERY_ROUTER_CHANNEL_WEIGHTS), it overrides the env-static
     * CHANNEL_WEIGHT_* path with a per-query lexical-form triplet.
     */
    private buildInnerRerankStrategy(
        hasLearnedSparse: boolean,
        channelWeights?: ChannelWeights,
    ): RerankStrategy {
        if (channelWeights) {
            // Weights aligned to buildRequests channel order:
            // [dense, sparse_bm25, sparse_learned].
            const weights = hasLearnedSparse
                ? [channelWeights.dense, channelWeights.sparse_bm25, channelWeights.sparse_learned]
                : [channelWeights.dense, channelWeights.sparse_bm25];
            return { strategy: 'weighted', params: { weights } };
        }
        const dense = envManager.get('CHANNEL_WEIGHT_DENSE');
        const bm25 = envManager.get('CHANNEL_WEIGHT_SPARSE_BM25');
        const learned = envManager.get('CHANNEL_WEIGHT_SPARSE_LEARNED');
        const anySet = [dense, bm25, learned].some((v) => v !== undefined && v !== null && v !== '');
        if (!anySet) {
            return { strategy: 'rrf', params: { k: this.getMilvusRrfK() } };
        }
        const wDense = this.getNonNegativeFloatFromEnv('CHANNEL_WEIGHT_DENSE', 1.0);
        const wBm25 = this.getNonNegativeFloatFromEnv('CHANNEL_WEIGHT_SPARSE_BM25', 0.4);
        const wLearned = this.getNonNegativeFloatFromEnv('CHANNEL_WEIGHT_SPARSE_LEARNED', 0.6);
        const weights = hasLearnedSparse ? [wDense, wBm25, wLearned] : [wDense, wBm25];
        return { strategy: 'weighted', params: { weights } };
    }

    /**
     * Schema-version prefix from COLLECTION_VERSION env. Phase 1 introduces
     * `v1` (BGE-M3 dim-1024 dense + BM25). Phase 4 will switch to `v2` to
     * carry the learned-sparse channel without reorganising existing data.
     * Empty/unset → no version segment, preserving Phase 0/0+ collection names.
     */
    private getCollectionVersionSegment(): string {
        const raw = envManager.get('COLLECTION_VERSION');
        if (!raw) {
            return '';
        }
        const sanitized = raw.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
        return sanitized ? `_${sanitized}` : '';
    }

    /**
     * Generate collection name based on codebase path and hybrid mode.
     *
     * In SPLIT_COLLECTIONS=false mode (legacy) → returns the single
     * `hybrid_v3_code_chunks_<hash>`-style name used by all callers.
     *
     * In SPLIT_COLLECTIONS=true mode (v6+) → returns the **code-side**
     * collection name (`hybrid_v6_code_<hash>`). Callers that need
     * collection-aware routing MUST use `getCollectionAddress` to also
     * obtain the prose-side collection name.
     *
     * The "legacy returns single, split returns code" convention keeps
     * existing internal call-sites (synchronizer key, deleteFileChunks
     * primary, log banners) byte-stable while enabling split-aware paths
     * to use `getCollectionAddress`.
     */
    public getCollectionName(codebasePath: string): string {
        const addr = this.getCollectionAddress(codebasePath);
        return addr.isSplit ? addr.code : addr.legacy;
    }

    /**
     * Every collection a codebase's index lives in (#152). Index STATE — does
     * this codebase have an index, how big is it, is it lost, what does
     * clearing it remove — is a question about all of them: in split mode an
     * all-prose corpus has no code collection at all, and asking
     * `getCollectionName` (the code side by convention) answered "not indexed"
     * for an index holding every row on its prose side.
     */
    public collectionsOf(codebasePath: string): string[] {
        const addr = this.getCollectionAddress(codebasePath);
        return addr.isSplit ? [addr.prose, addr.code] : [addr.legacy];
    }

    /**
     * code-collection-split: returns the full collection-name set for the
     * given codebase under the active SPLIT_COLLECTIONS / COLLECTION_VERSION
     * env state.
     *
     *   - `isSplit=false`: prose === code === legacy (same name everywhere).
     *   - `isSplit=true`:  prose / code are the two v6 collections.
     *
     * `legacy` is always the single-collection name (used by callers that
     * have not been collection-split-ified — symbol-refs-pool, eval queries).
     */
    public getCollectionAddress(codebasePath: string): {
        isSplit: boolean;
        prose: string;
        code: string;
        legacy: string;
    } {
        const isHybrid = this.getIsHybrid();
        const versionSegment = this.getCollectionVersionSegment();
        const normalizedPath = path.resolve(codebasePath);
        const pathHash = crypto.createHash('md5').update(normalizedPath).digest('hex').substring(0, 8);

        const legacyPrefix = isHybrid === true
            ? `hybrid${versionSegment}_code_chunks`
            : `code_chunks${versionSegment}`;

        // Resolve override suffix once; it disambiguates by `<custom>_<pathHash>`
        // appended to the prefix.
        const configOverride = this.getValidOverrideValue(this.collectionNameOverride);
        const envOverride = this.getValidOverrideValue(envManager.get('CODE_CHUNKS_COLLECTION_NAME_OVERRIDE'));
        const overrideSource = configOverride
            ? { value: configOverride, label: 'Context config' }
            : envOverride
                ? { value: envOverride, label: 'CODE_CHUNKS_COLLECTION_NAME_OVERRIDE' }
                : null;

        const buildName = (prefix: string): string => {
            if (overrideSource) {
                const suffix = this.sanitizeCollectionNameSuffix(overrideSource.value, prefix, pathHash, overrideSource.label);
                return `${prefix}_${suffix}`;
            }
            return `${prefix}_${pathHash}`;
        };

        const legacy = buildName(legacyPrefix);

        const isSplit = this.getSplitCollections() && isHybrid === true;
        if (!isSplit) {
            return { isSplit: false, prose: legacy, code: legacy, legacy };
        }

        // v6 split-mode prefixes (no `_code_chunks` segment).
        const prosePrefix = `hybrid${versionSegment}_prose`;
        const codePrefix = `hybrid${versionSegment}_code`;
        return {
            isSplit: true,
            prose: buildName(prosePrefix),
            code: buildName(codePrefix),
            legacy,
        };
    }

    /**
     * code-collection-split: route a single chunk to its target collection
     * by content_type. In legacy (non-split) mode every chunk lands in the
     * same name → returns `addr.legacy`.
     *
     * The mapping mirrors the per-domain pool definitions:
     *   - 'code', 'docstring'      → code-collection
     *   - 'doc',  'code_example'   → prose-collection
     *   - anything else            → code-collection (defensive default;
     *     unknown content_types are extremely rare and we'd rather keep
     *     them addressable as "code-side stuff" than lose them).
     */
    public resolveChunkCollection(
        contentType: string | undefined,
        addr: { isSplit: boolean; prose: string; code: string; legacy: string },
    ): string {
        if (!addr.isSplit) return addr.legacy;
        if (contentType === 'doc' || contentType === 'code_example') return addr.prose;
        return addr.code; // 'code', 'docstring', and any unknown fall here.
    }

    private getValidOverrideValue(value?: string): string | undefined {
        if (!value) {
            return undefined;
        }
        const trimmed = value.trim();
        return trimmed.length > 0 ? trimmed : undefined;
    }

    private sanitizeCollectionNameSuffix(value: string, prefix: string, pathHash: string, source: string): string {
        const hashSuffix = `_${pathHash}`;
        // Leave room for both the prefix and the trailing `_<pathHash>` disambiguator.
        const maxReadableLength = Context.MAX_COLLECTION_NAME_LENGTH - `${prefix}_`.length - hashSuffix.length;
        const normalized = value.trim();
        let sanitized = normalized.replace(/[^A-Za-z0-9_]/g, '_');
        sanitized = sanitized.slice(0, Math.max(0, maxReadableLength));

        if (sanitized.length === 0) {
            sanitized = 'custom';
        }

        const full = `${sanitized}${hashSuffix}`;

        if (sanitized !== normalized) {
            const warningKey = `${source}:${normalized}:${sanitized}`;
            if (!this.warnedOverrideSanitization.has(warningKey)) {
                console.warn(`[Context] ⚠️ Sanitized collection name override from "${normalized}" to "${sanitized}" (${source}); final suffix "${full}"`);
                this.warnedOverrideSanitization.add(warningKey);
            }
        }

        return full;
    }

    /**
     * Index a codebase for semantic search
     * @param codebasePath Codebase root path
     * @param progressCallback Optional progress callback function
     * @param forceReindex Whether to recreate the collection even if it exists
     * @param additionalIgnorePatterns Request-scoped ignore patterns
     * @param additionalSupportedExtensions Request-scoped file extensions
     * @param requestSplitter Request-scoped splitter for this indexing run
     * @returns Indexing statistics
     */
    async indexCodebase(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void,
        forceReindex: boolean = false,
        additionalIgnorePatterns: string[] = [],
        additionalSupportedExtensions: string[] = [],
        requestSplitter?: Splitter
    ): Promise<{ indexedFiles: number; totalChunks: number; droppedChunks: number; status: 'completed' | 'limit_reached'; chunkContextHeader?: { mode: ChunkContextHeaderMode; missing_contexts: number; missing_by_file: Record<string, number>; header_truncated_embedding: number; header_truncated_index_text: number } }> {
        const isHybrid = this.getIsHybrid();
        const searchType = isHybrid === true ? 'hybrid search' : 'semantic search';
        console.log(`[Context] 🚀 Starting to index codebase with ${searchType}: ${codebasePath}`);
        const splitter = requestSplitter || this.codeSplitter;
        // pilot-chunk-context-headers: an unreadable mode is refused before anything runs.
        const headerMode = this.getChunkContextHeaderMode();
        this.chunkContextReport = { missing_contexts: 0, missing_by_file: {}, header_truncated_embedding: 0, header_truncated_index_text: 0 };

        // 1. Compute ignore patterns for this codebase/request without
        // retaining file-based patterns from previous codebases.
        const ignorePatterns = await this.loadIgnorePatterns(codebasePath, additionalIgnorePatterns);

        // Phase 3: start a fresh symbol-vocabulary collector for this run.
        this.indexedSymbols = new Set<string>();

        // rag-graph-layer Phase 2: begin a fresh side-index accumulator.
        this.graphAccumulator = new GraphAccumulator();

        // rag-graph-layer Phase 1.2: if a previous run wrote
        // `.symbols-vocab.json`, hand it to the splitter so doc-side
        // mentioned_symbols can be filtered at split time. On a fresh
        // corpus the vocab is absent → no filter, mentioned_symbols are
        // raw qualified-names; the side-index builder handles filtering
        // through the post-indexing vocab.
        try {
            const priorVocab = await this.loadSymbolVocabulary(codebasePath);
            const vocabProvider = priorVocab ? () => priorVocab : undefined;
            (splitter as any).setMentionedVocabProvider?.(vocabProvider);
        } catch {
            // Vocab is best-effort; never block indexing on its absence.
        }

        if (headerMode === 'generated') {
            await this.checkChunkContextFile(codebasePath, await this.getCodeFiles(codebasePath, ignorePatterns, this.getEffectiveSupportedExtensions(additionalSupportedExtensions)), splitter);
        }

        // 2. Check and prepare vector collection
        progressCallback?.({ phase: 'Preparing collection...', current: 0, total: 100, percentage: 0 });
        console.log(`Debug2: Preparing vector collection for codebase${forceReindex ? ' (FORCE REINDEX)' : ''}`);
        await this.prepareCollection(codebasePath, forceReindex);

        // 3. Recursively traverse codebase to get all supported files
        progressCallback?.({ phase: 'Scanning files...', current: 5, total: 100, percentage: 5 });
        const supportedExtensions = this.getEffectiveSupportedExtensions(additionalSupportedExtensions);
        const codeFiles = await this.getCodeFiles(codebasePath, ignorePatterns, supportedExtensions);
        console.log(`[Context] 📁 Found ${codeFiles.length} code files`);

        if (codeFiles.length === 0) {
            progressCallback?.({ phase: 'No files to index', current: 100, total: 100, percentage: 100 });
            return { indexedFiles: 0, totalChunks: 0, droppedChunks: 0, status: 'completed' };
        }

        // 3. Process each file with streaming chunk processing
        // Reserve 10% for preparation, 90% for actual indexing
        const indexingStartPercentage = 10;
        const indexingEndPercentage = 100;
        const indexingRange = indexingEndPercentage - indexingStartPercentage;

        const result = await this.processFileList(
            codeFiles,
            codebasePath,
            (filePath, fileIndex, totalFiles) => {
                // Calculate progress percentage
                const progressPercentage = indexingStartPercentage + (fileIndex / totalFiles) * indexingRange;

                console.log(`[Context] 📊 Processed ${fileIndex}/${totalFiles} files`);
                progressCallback?.({
                    phase: `Processing files (${fileIndex}/${totalFiles})...`,
                    current: fileIndex,
                    total: totalFiles,
                    percentage: Math.round(progressPercentage)
                });
            },
            splitter
        );

        console.log(`[Context] ✅ Codebase indexing completed! Processed ${result.processedFiles} files in total, generated ${result.totalChunks} code chunks`);
        // #19: never let "completed" stand alone when chunks were lost.
        if (result.droppedChunks > 0) {
            console.error(`[Context] ❌ ${result.droppedChunks} chunk(s) were NOT indexed (rejected by the vector database) — the index is INCOMPLETE`);
        }

        // prose-embedding-swap: a prose-only reindex (REINDEX_POOLS=prose)
        // must NOT overwrite the code-derived side-indexes — the symbol vocab
        // and graph are built from code/docstring chunks which were skipped
        // this run. Preserve the existing files so code-pool features stay on
        // the v6 baseline artifacts.
        const poolFilter = this.reindexPoolFilter();
        const skipCodeSideArtifacts = poolFilter !== null && !poolFilter.has('code');
        if (skipCodeSideArtifacts) {
            console.log('[Context] 🎯 REINDEX_POOLS excludes code → preserving existing .symbols-vocab.json + .symbols-graph.json (not rewritten)');
        } else {
            // Phase 3: persist the collected symbol vocabulary so search-time
            // candidate extraction can filter out false positives.
            await this.persistSymbolVocabulary(codebasePath);

            // rag-graph-layer Phase 2: build & persist `.symbols-graph.json`
            // from accumulator. Best-effort: a failure logs but doesn't abort
            // the whole indexing run (search-side gracefully degrades when the
            // file is missing or malformed).
            await this.persistGraphSideIndex(codebasePath);
        }

        progressCallback?.({
            phase: 'Indexing complete!',
            current: result.processedFiles,
            total: codeFiles.length,
            percentage: 100
        });

        if (headerMode !== 'off') {
            const r = this.chunkContextReport;
            console.log(`[Context] 🏷️  CHUNK_CONTEXT_HEADER=${headerMode}: ${r.missing_contexts} chunk(s) without a context, ${r.header_truncated_embedding} embedding input(s) and ${r.header_truncated_index_text} index_text value(s) truncated because of the header`);
        }

        return {
            ...(headerMode !== 'off' ? { chunkContextHeader: { mode: headerMode, ...this.chunkContextReport } } : {}),
            indexedFiles: result.processedFiles,
            totalChunks: result.totalChunks,
            droppedChunks: result.droppedChunks,
            status: result.status
        };
    }

    async reindexByChange(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void,
        additionalIgnorePatterns: string[] = [],
        additionalSupportedExtensions: string[] = [],
        requestSplitter?: Splitter
    ): Promise<{ added: number, removed: number, modified: number, droppedChunks: number, status: 'completed' | 'limit_reached', processedFiles: number, chunkContextHeader?: { mode: ChunkContextHeaderMode; missing_contexts: number; missing_by_file: Record<string, number>; header_truncated_embedding: number; header_truncated_index_text: number } }> {
        const collectionName = this.getCollectionName(codebasePath);
        // code-collection-split: deletion set spans both v6 collections in
        // split mode. The synchronizer key stays single (`collectionName`,
        // which is the code-collection in split mode and the legacy name
        // otherwise) so file-modification detection is path-stable.
        const addr = this.getCollectionAddress(codebasePath);
        const deletionTargets = addr.isSplit ? [addr.prose, addr.code] : [collectionName];
        const synchronizer = this.synchronizers.get(collectionName);
        const splitter = requestSplitter || this.codeSplitter;
        // pilot-chunk-context-headers: before any chunk is deleted, refuse to
        // update a collection built under another header mode.
        const headerMode = this.getChunkContextHeaderMode();
        for (const target of deletionTargets) {
            if (await this.vectorDatabase.hasCollection(target)) await this.assertCollectionHeaderMode(target, headerMode);
        }
        const headerReport = () => (headerMode !== 'off' ? { chunkContextHeader: { mode: headerMode, ...this.chunkContextReport } } : {});
        await this.assertSnapshotBaseline(codebasePath);

        if (!synchronizer) {
            // Recreate the synchronizer with the same request-scoped options that
            // were used for the original indexing task.
            const ignorePatterns = await this.loadIgnorePatterns(codebasePath, additionalIgnorePatterns);
            const supportedExtensions = this.getEffectiveSupportedExtensions(additionalSupportedExtensions);

            // To be safe, let's initialize if it's not there.
            const newSynchronizer = new FileSynchronizer(codebasePath, ignorePatterns, supportedExtensions);
            await newSynchronizer.initialize();
            this.synchronizers.set(collectionName, newSynchronizer);
        }

        const currentSynchronizer = this.synchronizers.get(collectionName)!;

        progressCallback?.({ phase: 'Checking for file changes...', current: 0, total: 100, percentage: 0 });
        // serve-generated-chunk-contexts: preview the change set without saving the snapshot,
        // check the write, and only then commit — a refused write leaves the changes detectable.
        const preview = await currentSynchronizer.previewChanges();
        const { added, removed, modified } = preview;
        const totalChanges = added.length + removed.length + modified.length;

        if (totalChanges === 0) {
            await FileSynchronizer.writeBaseline(codebasePath, deletionTargets);
            await preview.commit();
            progressCallback?.({ phase: 'No changes detected', current: 100, total: 100, percentage: 100 });
            console.log('[Context] ✅ No file changes detected.');
            return { added: 0, removed: 0, modified: 0, droppedChunks: 0, status: 'completed', processedFiles: 0 };
        }

        await this.assertWriteReady(codebasePath, deletionTargets, [...added, ...modified], { additionalIgnorePatterns, additionalSupportedExtensions, splitter });
        // The record first: a commit without it would lose the change set to a failed write.
        await FileSynchronizer.writeBaseline(codebasePath, deletionTargets);
        await preview.commit();

        console.log(`[Context] 🔄 Found changes: ${added.length} added, ${removed.length} removed, ${modified.length} modified.`);

        let processedChanges = 0;
        const updateProgress = (phase: string) => {
            processedChanges++;
            const percentage = Math.round((processedChanges / (removed.length + modified.length + added.length)) * 100);
            progressCallback?.({ phase, current: processedChanges, total: totalChanges, percentage });
        };

        // Handle removed files
        for (const file of removed) {
            await this.deleteFileChunks(deletionTargets, file);
            updateProgress(`Removed ${file}`);
        }

        // Handle modified and added files. #142: an added file may already have rows —
        // a snapshot that predates the collections does not know it — and an insert
        // does not replace rows with the same id.
        await this.deleteFilesChunks(deletionTargets, [...modified, ...added]);
        for (const file of [...modified, ...added]) updateProgress(`Deleted old chunks for ${file}`);

        // Handle added and modified files
        const filesToIndex = [...added, ...modified].map(f => path.join(codebasePath, f));

        let droppedChunks = 0;
        let scanStatus: 'completed' | 'limit_reached' = 'completed';
        let processedFiles = 0;
        if (filesToIndex.length > 0) {
            const listResult = await this.processFileList(
                filesToIndex,
                codebasePath,
                (filePath, fileIndex, totalFiles) => {
                    updateProgress(`Indexed ${filePath} (${fileIndex}/${totalFiles})`);
                },
                splitter
            );
            droppedChunks = listResult.droppedChunks;
            scanStatus = listResult.status;
            processedFiles = listResult.processedFiles;
        }

        console.log(`[Context] ✅ Re-indexing complete. Added: ${added.length}, Removed: ${removed.length}, Modified: ${modified.length}`);
        if (headerMode !== 'off') {
            const r = this.chunkContextReport;
            console.log(`[Context] 🏷️  Header ${headerMode}: ${r.missing_contexts} chunk(s) without a generated context, ${r.header_truncated_embedding} embedding input(s) and ${r.header_truncated_index_text} index text(s) truncated by the header`);
        }
        // #138: the detected change set is not the processed set. The chunk
        // scan can stop at CHUNK_LIMIT part-way through it with zero rejected
        // batches — files past the cutoff were never embedded, so the #19 tap
        // sees nothing. Print what was actually processed against what was
        // detected, and hand the status to the caller so the run can fail.
        if (scanStatus === 'limit_reached') {
            console.log(`[Context] ⚠️  Processed: ${processedFiles} of ${filesToIndex.length} detected file(s) — the chunk limit stopped the scan`);
            console.error(`[Context] ❌ The index is INCOMPLETE: the chunk limit was reached before every detected file was processed`);
        }
        // #19: "Re-indexing complete! 100%" used to print even when whole
        // batches had been rejected. Say what was lost, and hand the count to
        // the caller so the run can fail.
        if (droppedChunks > 0) {
            console.error(`[Context] ❌ ${droppedChunks} chunk(s) were NOT indexed (rejected by the vector database) — the index is INCOMPLETE`);
        }
        progressCallback?.({ phase: 'Re-indexing complete!', current: totalChanges, total: totalChanges, percentage: 100 });

        return { added: added.length, removed: removed.length, modified: modified.length, droppedChunks, status: scanStatus, processedFiles, ...headerReport() };
    }

    /**
     * stale-snapshot-safe-updates (#143): refuse a change-set write whose merkle snapshot
     * was taken against other collections (e.g. a promoted side build), before anything
     * is deleted. A snapshot without a record predates it and is accepted.
     */
    async assertSnapshotBaseline(codebasePath: string): Promise<void> {
        const addr = this.getCollectionAddress(codebasePath);
        const targets = (addr.isSplit ? [addr.prose, addr.code] : [this.getCollectionName(codebasePath)]).sort();
        const reseed = 'node infra/with-retrieval-env.js node infra/reindex-resume.js --reseed --seed-only';
        let recorded: string[] | null;
        try {
            recorded = await FileSynchronizer.readBaseline(codebasePath);
            if (recorded !== null && !(Array.isArray(recorded) && recorded.every((name) => typeof name === 'string'))) throw new Error('no collection list');
        } catch (error: any) {
            throw new Error(`[Context] The collections record of the merkle snapshot for ${codebasePath} is unreadable (${error.message}); re-seed the snapshot from the served collections: ${reseed}`);
        }
        if (recorded && recorded.join(',') !== targets.join(',')) {
            throw new Error(`[Context] The merkle snapshot for ${codebasePath} was taken against ${recorded.join(', ')}, not ${targets.join(', ')}; re-seed it from the served collections first: ${reseed}`);
        }
    }

    /**
     * Delete chunks for a single file across the active collection set.
     *
     * code-collection-split: in split mode a file's chunks may straddle
     * both collections (an .md file emits doc → prose AND code_example
     * → prose; an .hx file with docstrings emits code + docstring → code
     * — same file might also have an inline doc fragment, etc.). We probe
     * each provided target, so callers that already resolved the full
     * split set pass both; the legacy path passes a single name.
     */
    /**
     * The rows of many files, one query per collection per 500 paths: a single-path query
     * costs ~400 ms on the served v8h2 collections (2026-09-17), a 500-path one ~270 ms.
     */
    async deleteFilesChunks(collectionNames: string[], relativePaths: string[]): Promise<void> {
        const quote = (p: string) => `"${p.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
        for (let i = 0; i < relativePaths.length; i += 500) {
            const batch = relativePaths.slice(i, i + 500);
            for (const target of collectionNames) {
                const rows = await this.vectorDatabase.query(target, `relativePath in [${batch.map(quote).join(', ')}]`, ['id']);
                const ids = rows.map(r => r.id as string).filter(id => id);
                for (let j = 0; j < ids.length; j += 5000) await this.vectorDatabase.delete(target, ids.slice(j, j + 5000));
                if (ids.length) console.log(`[Context] Deleted ${ids.length} chunks of ${batch.length} file(s) (${target})`);
            }
        }
    }

    private async deleteFileChunks(collectionNames: string | string[], relativePath: string): Promise<void> {
        const targets = Array.isArray(collectionNames) ? collectionNames : [collectionNames];
        const escapedPath = relativePath.replace(/\\/g, '\\\\');
        for (const target of targets) {
            const results = await this.vectorDatabase.query(
                target,
                `relativePath == "${escapedPath}"`,
                ['id']
            );

            if (results.length > 0) {
                const ids = results.map(r => r.id as string).filter(id => id);
                if (ids.length > 0) {
                    await this.vectorDatabase.delete(target, ids);
                    console.log(`[Context] Deleted ${ids.length} chunks for file ${relativePath} (${target})`);
                }
            }
        }
    }

    /**
     * Semantic search with unified implementation
     * @param codebasePath Codebase path to search in
     * @param query Search query
     * @param topK Number of results to return
     * @param threshold Similarity threshold
     */
    /**
     * @param queryShape knowledge-router: optional ground-truth query_shape
     *   hint (the eval harness passes the gold-set label). Production callers
     *   omit it; the concept-span quota then stays a no-op since routeQuery
     *   cannot infer `concept` from the deterministic classifiers.
     */
    async semanticSearch(codebasePath: string, query: string, topK: number = 5, threshold: number = 0.5, filterExpr?: string, queryShape?: string): Promise<SemanticSearchResult[]> {
        const isHybrid = this.getIsHybrid();
        const searchType = isHybrid === true ? 'hybrid search' : 'semantic search';
        console.log(`[Context] 🔍 Executing ${searchType}: "${query}" in ${codebasePath}`);

        // sweep-symbol-refs-pool-weight: arm the pre-rerank candidate dump for
        // this search, or leave both fields null so the merge collects nothing.
        this.candidateProvenance = (process.env.CANDIDATE_LOG_DIR || '').trim() ? new Map() : null;
        this.pendingCandidateDump = null;

        // The gate is the whole address (#152): the split pools below are already
        // address-aware and degrade per pool, so a corpus whose code collection
        // was never created is searched on its prose side instead of answering
        // "not indexed" for an index that holds every row.
        const present: string[] = [];
        for (const name of this.collectionsOf(codebasePath)) {
            if (await this.vectorDatabase.hasCollection(name)) present.push(name);
        }
        if (present.length === 0) {
            console.log(`[Context] ⚠️  No collection of ${codebasePath} exists (${this.collectionsOf(codebasePath).join(', ')}). Please index the codebase first.`);
            return [];
        }
        // The single-collection paths below (MULTI_QUERY=false, non-hybrid
        // search) need ONE name: the conventional one while it is there — byte
        // -stable for every existing index — otherwise the side that exists.
        const preferred = this.getCollectionName(codebasePath);
        const collectionName = present.includes(preferred) ? preferred : present[0];
        console.log(`[Context] 🔍 Using collection: ${collectionName}`);
        await this.noteSearchedHeaderModes(codebasePath);

        if (isHybrid === true) {
            try {
                // Check collection stats to see if it has data
                await this.vectorDatabase.query(collectionName, '', ['id'], 1);
                console.log(`[Context] 🔍 Collection '${collectionName}' exists and appears to have data`);
            } catch (error) {
                console.log(`[Context] ⚠️  Collection '${collectionName}' exists but may be empty or not properly indexed:`, error);
            }

            // rag-query-static-rewrite: apply env-gated query rewriters before
            // the retrieval fan-out. `applyRewriting` is a pure function: when
            // all flags are off it returns kind='single' with empty extras
            // (no-op, behavior identical to pre-change baseline).
            const rewriteFlags = this.getRewriteFlags();
            const rewrite = applyRewriting(query, rewriteFlags);
            this.logQueryRewrite(query, rewrite, rewriteFlags);

            // Outer-level intent: used downstream for reranker-bypass decision.
            // The helper recomputes intent per-subject for split fan-out — the
            // outer intent reflects the original (unsplit) query.
            const intent = classifyQuery(query);

            // knowledge-router: aggregate every per-query routing decision in
            // one place so it is visible in diagnostics and the eval dump.
            const channelWeightsEnabled = this.getQueryRouterChannelWeights();
            const conceptQuotaEnabled = this.getConceptSpanQuota();
            const route = routeQuery(query, {
                shapeHint: queryShape as QueryShape | undefined,
                channelWeightsEnabled,
                conceptQuotaEnabled,
            });
            // Only thread channel weights into the pipeline when the flag is
            // on; otherwise `undefined` keeps buildInnerRerankStrategy on the
            // env-static path (byte-identical pre-change behaviour).
            const routedChannelWeights = channelWeightsEnabled ? route.channelWeights : undefined;
            const cw = route.channelWeights;
            console.log(
                `[query-router] lexical_form=${route.lexical_form} query_shape=${route.query_shape} ` +
                `channelWeights=${channelWeightsEnabled ? `${cw.dense}/${cw.sparse_learned}/${cw.sparse_bm25}` : 'static-default'} ` +
                `conceptQuota=${route.conceptQuota}`,
            );

            const multiQuery = this.getMultiQuery();
            const rerankerInputK = this.hasReranker() ? this.getRerankerInputK() : 0;
            const PER_POOL_K = Math.max(topK * 5, 25, rerankerInputK);
            const mergeLimit = this.hasReranker()
                ? Math.max(rerankerInputK, topK * 3, 30)
                : Math.max(topK * 3, 30);

            let mergedResults: SemanticSearchResult[];
            // Pools surfaced from the per-subject helper(s); used by
            // graph-expansion re-merge and guarantee-slots below.
            let codePool: HybridSearchResult[] = [];
            let docPool: HybridSearchResult[] = [];
            let mergePools: { results: HybridSearchResult[]; weight: number; name?: string }[] = [];

            if (multiQuery) {
                console.log(`[Context] 🔍 MULTI_QUERY=true → running parallel code-domain + doc-domain hybrid searches (PER_POOL_K=${PER_POOL_K})`);

                let mergedPreGraph: HybridSearchResult[];

                if (rewrite.kind === 'split' && rewrite.left && rewrite.right) {
                    // Comparison-split fan-out: run the per-subject pipeline
                    // twice, outer-RRF the two pre-graph merges with weights
                    // 0.5/0.5. Graph-expansion (if enabled) runs once on the
                    // outer-merged result with a single-pool re-merge.
                    console.log(`[Context] 🔀 split fan-out: left="${rewrite.left}" right="${rewrite.right}"`);
                    const [leftRun, rightRun] = await Promise.all([
                        this.runSubjectHybridPipeline(
                            rewrite.left,
                            rewrite.sparseExtra,
                            codebasePath,
                            collectionName,
                            filterExpr,
                            PER_POOL_K,
                            mergeLimit,
                            routedChannelWeights,
                        ),
                        this.runSubjectHybridPipeline(
                            rewrite.right,
                            rewrite.sparseExtra,
                            codebasePath,
                            collectionName,
                            filterExpr,
                            PER_POOL_K,
                            mergeLimit,
                            routedChannelWeights,
                        ),
                    ]);
                    mergedPreGraph = this.weightedRrfMerge(
                        [
                            { results: leftRun.mergedPreGraph, weight: 0.5, name: 'splitLeft' },
                            { results: rightRun.mergedPreGraph, weight: 0.5, name: 'splitRight' },
                        ],
                        mergeLimit,
                        this.getRrfK(),
                    );
                    // For downstream graph-expansion re-merge and guarantee-
                    // slots, use the outer-merged pool as the single base
                    // (weight 1.0) and union the per-subject code/doc pools
                    // for guarantee-slots.
                    mergePools = [{ results: mergedPreGraph, weight: 1.0 }];
                    codePool = [...leftRun.codePool, ...rightRun.codePool];
                    docPool = [...leftRun.docPool, ...rightRun.docPool];
                } else {
                    const single = await this.runSubjectHybridPipeline(
                        query,
                        rewrite.sparseExtra,
                        codebasePath,
                        collectionName,
                        filterExpr,
                        PER_POOL_K,
                        mergeLimit,
                        routedChannelWeights,
                    );
                    mergedPreGraph = single.mergedPreGraph;
                    mergePools = single.mergePools;
                    codePool = single.codePool;
                    docPool = single.docPool;
                }

                let semanticMerged: SemanticSearchResult[] = mergedPreGraph.map((r) => this.toSemanticResult(r));

                // rag-graph-layer Phase 3.5: graph-expansion. Build a 3rd
                // pool from the pre-graph top-K's neighbours and re-merge
                // with the original code/doc pools. The re-merge uses the
                // same weighted RRF; graph weight is intentionally lower
                // (default 0.6) so neighbours nudge order rather than
                // dominate it.
                this.logGraphStartupBanner(collectionName, codebasePath);
                const graphExpand = this.getGraphExpand();
                const collectionVersion = (envManager.get('COLLECTION_VERSION') || '').trim().toLowerCase();
                // code-collection-split: v6 retains the v3 graph-schema
                // (v3-3) so all downstream graph paths (expansion +
                // comparison-bridge) accept both versions.
                const v3Compatible = !collectionVersion || collectionVersion === 'v3' || collectionVersion === 'v6';
                if (graphExpand >= 1 && v3Compatible) {
                    const graphIndex = this.loadGraphIndex(codebasePath);
                    if (graphIndex) {
                        const seeds = semanticMerged.slice(0, this.getGraphSeedK());
                        const neighbourIds = collectGraphCandidateIds(seeds, graphIndex);
                        if (neighbourIds.length > 0) {
                            // code-collection-split: graph candidates include
                            // canonical_chunk_ids (code/docstring → code) and
                            // mentioned_by_chunk_ids (mostly code, but
                            // code_example chunks may show up via reverse
                            // mentions). 'either' tries code first then prose;
                            // legacy mode hits the single collection.
                            const fetched = await this.fetchChunksByIds(codebasePath, neighbourIds, 'either');
                            if (fetched.length > 0) {
                                // Convert raw HybridSearchResult to seeds and
                                // re-run the weighted RRF with a 3rd pool.
                                const graphPoolMerge = [
                                    ...mergePools,
                                    { results: fetched, weight: this.getGraphPoolWeight(), name: 'graph' },
                                ];
                                const reMerged = this.weightedRrfMerge(
                                    graphPoolMerge,
                                    mergeLimit,
                                    this.getRrfK(),
                                );
                                semanticMerged = reMerged.map((r) => this.toSemanticResult(r));
                                console.log(`[Context] 🕸️  graph-expansion: ${neighbourIds.length} candidates, ${fetched.length} fetched, weight=${this.getGraphPoolWeight()}`);
                            } else if (Date.now() - this.staleGraphChunkWarnedAt > 60_000) {
                                this.staleGraphChunkWarnedAt = Date.now();
                                console.warn(`[Context] ⚠️ graph index points to ${neighbourIds.length} chunk_id(s) not present in '${collectionName}'; consider reindex`);
                            }
                        }
                    }
                }

                // prose-graph-deterministic §4: optional prose-graph
                // expansion. Orthogonal to GRAPH_EXPAND (that hops code/symbol
                // edges over the code collection; this hops deterministic
                // narrative edges over the prose collection). Seeds come from
                // the prose-pool top-K; their 1-hop neighbours form a separate
                // `prose_graph_pool` merged into the outer weighted RRF with a
                // reduced weight, before the reranker. Default-OFF → the index
                // is never loaded and this block is a full no-op.
                if (this.getProseGraphExpand()) {
                    const proseGraphIndex = this.loadProseGraphIndex(codebasePath);
                    if (proseGraphIndex) {
                        // Prefer prose-pool (doc/code_example) hits as seeds;
                        // fall back to the merged top-K when the doc pool is
                        // empty. Non-prose seeds simply have no graph entries.
                        const proseSeedK = this.getProseGraphSeedK();
                        const proseSeeds = (docPool.length > 0
                            ? docPool.map((r) => this.toSemanticResult(r))
                            : semanticMerged).slice(0, proseSeedK);
                        const cap = this.hasReranker() ? this.getRerankerInputK() : DEFAULT_RERANKER_INPUT_K;
                        const neighbourIds = collectProseGraphCandidateIds(proseSeeds, proseGraphIndex, cap).slice(0, cap);
                        if (neighbourIds.length > 0) {
                            // Collection-aware: prose-graph chunk_ids live in
                            // the prose collection under SPLIT_COLLECTIONS=true.
                            const fetched = (await this.fetchChunksByIds(codebasePath, neighbourIds, 'prose')).slice(0, cap);
                            if (fetched.length > 0) {
                                const proseGraphMerge = [
                                    ...mergePools,
                                    { results: fetched, weight: this.getProseGraphPoolWeight(), name: 'proseGraph' },
                                ];
                                const reMerged = this.weightedRrfMerge(
                                    proseGraphMerge,
                                    mergeLimit,
                                    this.getRrfK(),
                                );
                                const proseGraphIds = new Set(fetched.map((c) => c.document.id));
                                semanticMerged = reMerged.map((r) => {
                                    const s = this.toSemanticResult(r);
                                    if (s.chunk_id && proseGraphIds.has(s.chunk_id)) s.pool = 'proseGraph';
                                    return s;
                                });
                                console.log(`[Context] 🪢 prose-graph expansion: ${neighbourIds.length} candidates, ${fetched.length} fetched, weight=${this.getProseGraphPoolWeight()}`);
                            } else if (Date.now() - this.staleProseGraphChunkWarnedAt > 60_000) {
                                this.staleProseGraphChunkWarnedAt = Date.now();
                                console.warn(`[Context] ⚠️ prose-graph points to ${neighbourIds.length} chunk_id(s) not present in the prose collection; consider rebuild`);
                            }
                        }
                    }
                }

                // rag-graph-comparison-bridge: 5th pool. Activated when the
                // env-flag is on, the query is comparison-shaped, the graph
                // is v3-2, and v3 collections are in use. Failure modes are
                // silent (empty pool); the comparison-bridge module never
                // throws and degrades to a no-op on missing data.
                if (this.getComparisonBridgeEnabled() && v3Compatible && isComparisonShape(query)) {
                    const graphIndex = this.loadGraphIndex(codebasePath);
                    if (graphIndex && graphIndex.supportsComparisonBridge()) {
                        const seeds = semanticMerged.slice(0, this.getGraphSeedK());
                        const bridge = buildComparisonBridgePool(seeds, graphIndex, {
                            maxPartners: this.getComparisonBridgeMaxPartners(),
                            maxPackageFanout: this.getComparisonBridgeMaxPackageFanout(),
                            debug: this.getComparisonBridgeDebug(),
                            queryId: query,
                        });
                        if (bridge.chunkIds.length > 0) {
                            // code-collection-split: comparison-bridge seeds
                            // only from content_type='code' chunks; partners
                            // come from canonical_chunk_ids of symbols (code
                            // or docstring) → always code-collection.
                            const fetched = await this.fetchChunksByIds(codebasePath, bridge.chunkIds, 'code');
                            if (fetched.length > 0) {
                                const bridgePoolMerge = [
                                    ...mergePools,
                                    { results: fetched, weight: this.getComparisonBridgePoolWeight(), name: 'comparisonBridge' },
                                ];
                                const reMerged = this.weightedRrfMerge(
                                    bridgePoolMerge,
                                    mergeLimit,
                                    this.getRrfK(),
                                );
                                // Build a set of bridge chunk_ids so the
                                // marker is only applied to chunks the bridge
                                // actually injected (not original RRF hits
                                // that happen to share a chunk_id, although
                                // the bridge filter guarantees disjointness).
                                const bridgeIds = new Set(fetched.map((c) => c.document.id));
                                semanticMerged = reMerged.map((r) => {
                                    const s = this.toSemanticResult(r);
                                    if (s.chunk_id && bridgeIds.has(s.chunk_id)) {
                                        s.pool = 'comparisonBridge';
                                    }
                                    return s;
                                });
                                console.log(`[Context] 🌉 comparison-bridge: seeds=${bridge.seedsCount} pkg_hits=${bridge.packageHits} sup_hits=${bridge.supertypeHits} fetched=${fetched.length} weight=${this.getComparisonBridgePoolWeight()}`);
                            } else if (Date.now() - this.staleGraphChunkWarnedAt > 60_000) {
                                this.staleGraphChunkWarnedAt = Date.now();
                                console.warn(`[Context] ⚠️ comparison-bridge points to ${bridge.chunkIds.length} chunk_id(s) not present in '${collectionName}'; consider reindex`);
                            }
                        }
                    }
                }

                // Guarantee-slots: defend top-N domain coverage when no
                // reranker is in front of us. Disabled when reranker is
                // active because the reranker handles domain balance.
                if (!this.hasReranker() && (codePool.length > 0 || docPool.length > 0)) {
                    semanticMerged = this.applyGuaranteeSlots(
                        semanticMerged,
                        codePool.map((r) => this.toSemanticResult(r)),
                        docPool.map((r) => this.toSemanticResult(r)),
                        topK,
                        DEFAULT_GUARANTEE_CODE,
                        DEFAULT_GUARANTEE_DOC
                    );
                }

                mergedResults = semanticMerged;
            } else {
                console.log(`[Context] 🔍 MULTI_QUERY=false → single-pool hybrid search`);
                // When reranker is active, pull a wider pool for it to
                // arbitrate over; otherwise topK is enough.
                const singleLimit = this.hasReranker()
                    ? Math.max(this.getRerankerInputK(), topK)
                    : topK;
                console.log(`[Context] 🔍 Generating embeddings for query: "${query}"`);
                const queryEmbedding: EmbeddingVector = await this.embedding.embed(query);
                console.log(`[Context] ✅ Generated embedding vector with dimension: ${queryEmbedding.vector.length}`);
                const innerRerank = this.buildInnerRerankStrategy(
                    this.queriesLearnedSparse(queryEmbedding),
                    routedChannelWeights,
                );
                const searchResults: HybridSearchResult[] = await this.vectorDatabase.hybridSearch(
                    collectionName,
                    this.buildHybridRequests(queryEmbedding, query, rewrite.sparseExtra, singleLimit),
                    {
                        rerank: innerRerank,
                        limit: singleLimit,
                        filterExpr,
                    }
                );
                mergedResults = searchResults.map((r) => this.toSemanticResult(r));
            }

            console.log(`[Context] 🔍 Raw merged results count: ${mergedResults.length}`);
            const dedupedResults = this.deduplicateResults(mergedResults);
            console.log(`[Context] ✅ Found ${mergedResults.length} results, ${dedupedResults.length} after dedup`);

            this.maybeStagePreRerankCandidates(query, dedupedResults);

            // Phase R (rag-code-intent-recall): when the query is a
            // strictly-anchored qualified name AND code-intent only, allow
            // bypass of the cross-encoder reranker. Cross-encoders trained on
            // NL-QA distribute mass over surface-similar wrong-subject chunks
            // for short identifier queries; the merged-RRF order (now
            // boosted by the symbol-routing pool) is more reliable on these.
            const rerankerBypassed = this.shouldBypassReranker(query, intent);

            // rag-comparison-bridge-reranker-bypass: when bypass slots > 0 and
            // the candidate pool contains chunks marker'd `pool:
            // 'comparisonBridge'`, reserve N slots in the final top-K for the
            // highest-RRF-ranked bridge chunks and send only the rest to the
            // reranker. Mirrors the `RERANKER_BYPASS_FOR_QUALIFIED_NAME`
            // precedent at the same call site but partitions by pool marker
            // instead of query shape. When `bypassSlots = 0`, there are no
            // bridge chunks, or the qualified-name bypass already short-
            // circuited the reranker, the path collapses identically to the
            // pre-change behavior.
            const bypassSlots = (this.hasReranker() && !rerankerBypassed)
                ? this.getComparisonBridgeBypassSlots()
                : 0;
            const { reserved: reservedBridge, rerankInput } = this.partitionForBridgeBypass(
                dedupedResults,
                topK,
                bypassSlots,
            );
            if (reservedBridge.length > 0 && this.getComparisonBridgeDebug()) {
                console.log(`[comparison-bridge-bypass] reserved=${reservedBridge.length} rerank_input=${rerankInput.length} final=${topK}`);
            }
            const rerankerSlots = Math.max(topK - reservedBridge.length, 0);
            const rerankedRest = await this.rerankForFinalResults(query, rerankInput, rerankerSlots, rerankerBypassed);
            const finalResults = [...reservedBridge, ...rerankedRest].slice(0, topK);
            if (rerankerBypassed) {
                console.log(`[Context] ⏭️  reranker bypassed for qualified-name code query "${query}"`);
            }

            // knowledge-router: concept-span content-type quota. Runs AFTER
            // the reranker (unlike applyGuaranteeSlots, which is off when a
            // reranker is active) and only for query_shape=concept. No-op
            // when the quota is disabled or the route is not concept-shaped.
            const quotaResults = route.conceptQuota
                ? this.applyConceptSpanQuota(finalResults, dedupedResults, topK)
                : finalResults;

            if (quotaResults.length > 0) {
                console.log(`[Context] 🔍 Top result score: ${quotaResults[0].score}, path: ${quotaResults[0].relativePath}`);
            }
            // Before attachCandidateSymbols: that call awaits the symbol
            // vocabulary and is not wrapped, so a malformed .symbols-vocab.json
            // would take the diagnostic down with a search whose retrieval was
            // fine. The dump needs nothing it produces.
            this.flushPreRerankCandidateDump(quotaResults, {
                topK,
                rerankerBypassed,
                reservedBridgeSlots: reservedBridge.length,
                multiQuery,
            });
            await this.attachCandidateSymbols(quotaResults, codebasePath);
            return quotaResults;
        } else {
            // Regular semantic search
            // 1. Generate query vector
            const queryEmbedding: EmbeddingVector = await this.embedding.embed(query);

            // 2. Search in vector database
            const searchResults: VectorSearchResult[] = await this.vectorDatabase.search(
                collectionName,
                queryEmbedding.vector,
                { topK, threshold, filterExpr }
            );

            // 3. Convert to semantic search result format
            const results: SemanticSearchResult[] = searchResults.map(result => ({
                content: result.document.content,
                relativePath: result.document.relativePath,
                startLine: result.document.startLine,
                endLine: result.document.endLine,
                language: result.document.metadata.language || 'unknown',
                score: result.score,
                content_type: result.document.content_type,
                symbol_name: result.document.symbol_name,
                symbol_kind: result.document.symbol_kind,
                parent_symbol: result.document.parent_symbol,
                heading_path: parseHeadingPath(result.document.heading_path),
                chunk_id: result.document.id,
                imports: parseStringArray(result.document.imports),
                extends: result.document.extends,
                implements: parseStringArray(result.document.implements),
                mentioned_symbols: parseStringArray(result.document.mentioned_symbols),
            }));

            const dedupedResults = this.deduplicateResults(results);
            console.log(`[Context] ✅ Found ${results.length} results, ${dedupedResults.length} after dedup`);
            await this.attachCandidateSymbols(dedupedResults, codebasePath);
            return dedupedResults;
        }
    }

    /**
     * Phase 3: extract candidate symbol names from the final result pool and
     * attach them to the first result. The list belongs to the response (not
     * the chunk), but staying inside SemanticSearchResult keeps the existing
     * return-shape contract intact for older callers.
     */
    private async attachCandidateSymbols(
        results: SemanticSearchResult[],
        codebasePath: string,
    ): Promise<void> {
        if (!results || results.length === 0) return;
        const vocabulary = (await this.loadSymbolVocabulary(codebasePath)) ?? undefined;
        const candidates = extractCandidateSymbols(results, { vocabulary });
        if (candidates.length === 0) return;
        results[0].candidateSymbols = candidates;
        console.log(`[Context] 🧩 Candidate symbols (top ${candidates.length}): ${candidates.slice(0, 5).join(', ')}${candidates.length > 5 ? ', …' : ''}`);
    }

    /**
     * rag-graph-layer Phase 3.5: batch-fetch chunks by their Milvus
     * primary key. Returns a HybridSearchResult[] with score=1 (ranking
     * within the graph pool comes from the weighted RRF over its
     * insertion order; the per-channel score is irrelevant since the
     * pool is already a curated 1-hop expansion). Missing ids are
     * silently dropped per spec scenario "Side-файл stale".
     */
    /**
     * Batch-fetch chunks by Milvus primary key.
     *
     * code-collection-split: in split mode the resolver needs to know
     * which collection holds each chunk_id. Behavior is driven by `mode`:
     *   - 'code'   → query code-collection only (comparison-bridge,
     *                graph canonical_chunk_ids, symbol-refs).
     *   - 'prose'  → query prose-collection only (reserved; no caller
     *                currently uses it).
     *   - 'either' → query code-collection first; whatever id's are not
     *                found there are looked up in the prose-collection
     *                (graph mentioned_by_chunk_ids may include
     *                code_example chunks).
     *
     * In legacy (non-split) mode `addr.legacy` is hit directly and
     * `mode` is ignored — behavior is byte-stable.
     */
    private async fetchChunksByIds(
        codebasePath: string,
        ids: string[],
        mode: 'code' | 'prose' | 'either' = 'either',
    ): Promise<HybridSearchResult[]> {
        if (ids.length === 0) return [];
        // Sanitize ids to a Milvus filter expression. id is VarChar PK;
        // we wrap each id in double-quotes and rely on the chunk_id format
        // (`chunk_<hex16>`) being filter-safe by construction.
        const safeIds = ids
            .filter((id) => typeof id === 'string' && /^[A-Za-z0-9_-]+$/.test(id))
            .map((id) => `"${id}"`);
        if (safeIds.length === 0) return [];
        const filter = `id in [${safeIds.join(',')}]`;
        const outputFields = [
            'id', 'content', 'relativePath', 'startLine', 'endLine', 'fileExtension', 'metadata',
            'content_type', 'symbol_kind', 'symbol_name', 'parent_symbol', 'heading_path',
            'imports', 'extends', 'implements', 'mentioned_symbols',
        ];

        // code-collection-split: pick targets and per-target fetch loop.
        const addr = this.getCollectionAddress(codebasePath);
        const targets: string[] = !addr.isSplit
            ? [addr.legacy]
            : mode === 'code'
                ? [addr.code]
                : mode === 'prose'
                    ? [addr.prose]
                    : [addr.code, addr.prose]; // 'either' — code first, then prose

        const byId = new Map<string, Record<string, any>>();
        for (const target of targets) {
            if (byId.size >= ids.length) break; // every id resolved already
            let rows: Record<string, any>[] = [];
            try {
                rows = await this.vectorDatabase.query(target, filter, outputFields, ids.length);
            } catch (err) {
                console.warn(`[Context] ⚠️ fetchChunksByIds query failed on ${target}: ${err}`);
                continue;
            }
            for (const row of rows) {
                const id = row?.id;
                if (typeof id === 'string' && !byId.has(id)) byId.set(id, row);
            }
        }
        // Preserve the input ordering so the weighted RRF reflects the
        // graph traversal order rather than Milvus's storage order.
        const out: HybridSearchResult[] = [];
        for (const id of ids) {
            const row = byId.get(id);
            if (!row) continue;
            let metadata: Record<string, any> = {};
            try {
                metadata = JSON.parse(row.metadata || '{}');
            } catch {
                /* malformed json — leave empty */
            }
            out.push({
                document: {
                    id,
                    vector: [],
                    content: row.content || '',
                    relativePath: row.relativePath || '',
                    startLine: row.startLine || 0,
                    endLine: row.endLine || 0,
                    fileExtension: row.fileExtension || '',
                    metadata,
                    content_type: row.content_type ?? undefined,
                    symbol_kind: row.symbol_kind ?? undefined,
                    symbol_name: row.symbol_name ?? undefined,
                    parent_symbol: row.parent_symbol ?? undefined,
                    heading_path: row.heading_path ?? undefined,
                    imports: row.imports ?? undefined,
                    extends: row.extends ?? undefined,
                    implements: row.implements ?? undefined,
                    mentioned_symbols: row.mentioned_symbols ?? undefined,
                },
                score: 1,
            });
        }
        return out;
    }

    /**
     * Project a HybridSearchResult onto the SemanticSearchResult shape that
     * external callers consume.
     */
    private toSemanticResult(result: HybridSearchResult): SemanticSearchResult {
        return {
            content: result.document.content,
            relativePath: result.document.relativePath,
            startLine: result.document.startLine,
            endLine: result.document.endLine,
            language: result.document.metadata?.language || 'unknown',
            score: result.score,
            content_type: result.document.content_type,
            symbol_name: result.document.symbol_name,
            symbol_kind: result.document.symbol_kind,
            parent_symbol: result.document.parent_symbol,
            heading_path: parseHeadingPath(result.document.heading_path),
            // rag-graph-layer Phase 1.3: surface graph-edge fields and the
            // chunk_id so search-side graph-expansion can hop without an
            // extra Milvus fetch per neighbour.
            chunk_id: result.document.id,
            imports: parseStringArray(result.document.imports),
            extends: result.document.extends,
            implements: parseStringArray(result.document.implements),
            mentioned_symbols: parseStringArray(result.document.mentioned_symbols),
        };
    }

    /**
     * prose-graph-mcp-tool: public accessor for the lazily-cached per-codebase
     * prose-graph side-index, used by the agent-callable `expand_context` MCP
     * tool. Reuses the same in-process cache as the (archived, default-off)
     * silent-pool path, so repeat calls within one process do not re-read disk
     * (spec «Index loaded once and cached»). Returns null on
     * missing/unparseable/version-mismatch (spec «Side-index missing or
     * unparseable» → graceful-off); the caller renders the unavailable message.
     */
    public getProseGraphIndex(codebasePath: string): ProseGraphIndex | null {
        return this.loadProseGraphIndex(codebasePath);
    }

    /**
     * prose-graph-mcp-tool: batch-fetch prose-collection chunks by chunk_id
     * for `expand_context`, projected onto the external SemanticSearchResult
     * shape. Collection-aware via the existing fetch path: the prose
     * collection (`hybrid_v6_prose_<hash>`) under SPLIT_COLLECTIONS=true, the
     * single legacy collection otherwise. Stale ids absent from Milvus are
     * silently dropped (spec «Stale neighbour chunk_id skipped without
     * error»), so the returned array may be shorter than `ids`; input order
     * is preserved for the surviving ids.
     */
    public async fetchProseChunksByIds(
        codebasePath: string,
        ids: string[],
    ): Promise<SemanticSearchResult[]> {
        const fetched = await this.fetchChunksByIds(codebasePath, ids, 'prose');
        return fetched.map((r) => this.toSemanticResult(r));
    }

    /**
     * Weighted Reciprocal Rank Fusion across pools that already came back
     * from per-domain hybrid_search calls.
     *
     *   score(d) = Σ_pools weight_pool * 1 / (k_rrf + rank_in_pool(d))
     *
     * Documents that appear in multiple pools accumulate score, so a strong
     * cross-domain hit naturally rises. The pool weights come from the
     * intent classifier, so a code-shaped query nudges code-domain hits up
     * without ever excluding doc-domain hits.
     */
    private weightedRrfMerge(
        pools: { results: HybridSearchResult[]; weight: number; name?: string }[],
        k: number,
        kRrf: number = 60
    ): HybridSearchResult[] {
        const scoreById = new Map<string, number>();
        const docById = new Map<string, HybridSearchResult>();

        for (const pool of pools) {
            const { results, weight } = pool;
            for (let rank = 0; rank < results.length; rank++) {
                const r = results[rank];
                const id = r.document.id;
                if (!id) continue;
                const contribution = weight / (kRrf + rank + 1);
                scoreById.set(id, (scoreById.get(id) || 0) + contribution);
                if (this.candidateProvenance && pool.name) {
                    this.noteCandidatePool(id, pool.name, rank + 1);
                }
                if (!docById.has(id)) {
                    docById.set(id, r);
                }
            }
        }

        return Array.from(scoreById.entries())
            .map(([id, score]) => {
                const r = docById.get(id)!;
                return { document: r.document, score } as HybridSearchResult;
            })
            .sort((a, b) => b.score - a.score)
            .slice(0, k);
    }

    /**
     * Phase 2: cross-encoder rerank over the merged candidate pool.
     *
     * Takes the top RERANKER_INPUT_K from `candidates` (already deduped),
     * sends them to the reranker together with the query, sorts by the
     * returned scores, then keeps the top RERANKER_OUTPUT_K (or `topK`,
     * whichever is larger) before slicing to `topK` for the caller.
     *
     * On reranker error we fall back to RRF order — search must not fail
     * because the sidecar hiccupped.
     */
    /**
     * rag-comparison-bridge-reranker-bypass: split the post-dedup candidate
     * list into `reserved` (bridge chunks that bypass the reranker) and
     * `rerankInput` (everything else, including leftover bridge chunks past
     * `bypassSlots` — they still see the reranker so they can compete on
     * cross-encoder score). Pure function; no env reads — callers compute
     * `bypassSlots` upstream and gate qualified-name-bypass / no-reranker
     * cases by passing `0`.
     */
    private partitionForBridgeBypass(
        candidates: SemanticSearchResult[],
        topK: number,
        bypassSlots: number,
    ): { reserved: SemanticSearchResult[]; rerankInput: SemanticSearchResult[] } {
        if (bypassSlots <= 0) {
            return { reserved: [], rerankInput: candidates };
        }
        const bridgeChunks = candidates.filter((r) => r.pool === 'comparisonBridge');
        if (bridgeChunks.length === 0) {
            return { reserved: [], rerankInput: candidates };
        }
        const reservationCount = Math.min(bypassSlots, bridgeChunks.length, topK);
        const reserved = bridgeChunks.slice(0, reservationCount);
        const reservedSet = new Set<SemanticSearchResult>(reserved);
        const rerankInput = candidates.filter((r) => !reservedSet.has(r));
        return { reserved, rerankInput };
    }

    /**
     * The skip-or-rerank decision the final results pass through (#131).
     * Extracted so the vendored suite can pin it with a reranker attached:
     * with a reranker present, an engaged qualified-name bypass skips the
     * cross-encoder entirely — the merged-RRF order passes through as the
     * final order — and a bypass that does not engage hands the sliced input
     * to `applyReranker`. Behaviour-preserving extraction; the search
     * pipeline's call site is the single consumer.
     */
    private async rerankForFinalResults(
        query: string,
        rerankInput: SemanticSearchResult[],
        rerankerSlots: number,
        rerankerBypassed: boolean,
    ): Promise<SemanticSearchResult[]> {
        return this.hasReranker() && !rerankerBypassed
            ? await this.applyReranker(query, rerankInput, rerankerSlots)
            : rerankInput.slice(0, rerankerSlots);
    }

    private async applyReranker(
        query: string,
        candidates: SemanticSearchResult[],
        topK: number,
    ): Promise<SemanticSearchResult[]> {
        if (!this.reranker || candidates.length === 0) {
            return candidates.slice(0, topK);
        }

        const inputK = Math.min(this.getRerankerInputK(), candidates.length);
        const outputK = Math.max(this.getRerankerOutputK(), topK);
        const slice = candidates.slice(0, inputK);
        const docs = slice.map((c) => c.content);

        let rerankResults;
        try {
            const t0 = Date.now();
            rerankResults = await this.reranker.rerank(query, docs);
            console.log(`[Context] 🎯 reranker (${this.reranker.getProvider()}) ranked ${docs.length} candidates in ${Date.now() - t0}ms`);
        } catch (err) {
            console.warn(`[Context] ⚠️ reranker failed, falling back to RRF order: ${err}`);
            return candidates.slice(0, topK);
        }

        rerankResults.sort((a, b) => b.score - a.score);
        const reranked: SemanticSearchResult[] = [];
        for (const r of rerankResults) {
            const src = slice[r.index];
            if (!src) continue;
            reranked.push({ ...src, score: r.score });
            if (reranked.length >= outputK) break;
        }

        return reranked.slice(0, topK);
    }

    /**
     * Phase 0+ guarantee-slots: when the merged top-N is dominated by one
     * domain, swap in the next-best candidates from the under-represented
     * domain until the floor is met. Operates after RRF, before final cut.
     */
    private applyGuaranteeSlots(
        merged: SemanticSearchResult[],
        codePool: SemanticSearchResult[],
        docPool: SemanticSearchResult[],
        topK: number,
        minCode: number,
        minDoc: number
    ): SemanticSearchResult[] {
        // Operate on a wider window so we have room to demote without
        // losing strong general candidates from the merged list.
        const windowSize = Math.max(topK * 2, topK + minCode + minDoc);
        const window = merged.slice(0, windowSize);
        const tail = merged.slice(windowSize);

        const isCode = (r: SemanticSearchResult) => CODE_DOMAIN_TYPES.has(r.content_type || '');
        const isDoc = (r: SemanticSearchResult) => DOC_DOMAIN_TYPES.has(r.content_type || '');

        const idsInWindow = new Set<string>();
        const dedupKey = (r: SemanticSearchResult) => `${r.relativePath}#${r.startLine}-${r.endLine}`;
        for (const r of window) idsInWindow.add(dedupKey(r));

        // Slice the head: we never reorder the absolute top-1 because that
        // would override the strongest signal across both pools.
        const head = window.slice(0, 1);
        const body = window.slice(1);

        const top = head.concat(body).slice(0, topK);
        let codeCount = top.filter(isCode).length;
        let docCount = top.filter(isDoc).length;

        const minCodeEffective = Math.min(minCode, codePool.length);
        const minDocEffective = Math.min(minDoc, docPool.length);

        const promoteFromPool = (
            pool: SemanticSearchResult[],
            need: number,
            counterPredicate: (r: SemanticSearchResult) => boolean,
        ) => {
            if (need <= 0) return;
            for (const candidate of pool) {
                if (need <= 0) break;
                const key = dedupKey(candidate);
                if (idsInWindow.has(key)) continue;
                // Find the lowest-ranked counterPredicate item in the
                // current top to evict.
                let evictIdx = -1;
                for (let i = top.length - 1; i >= 1; i--) {
                    if (counterPredicate(top[i])) { evictIdx = i; break; }
                }
                if (evictIdx === -1) break;
                top[evictIdx] = candidate;
                idsInWindow.add(key);
                need--;
            }
        };

        if (codeCount < minCodeEffective) {
            promoteFromPool(codePool, minCodeEffective - codeCount, isDoc);
        }
        // Recount before doc pass so we don't churn slots we just filled.
        codeCount = top.filter(isCode).length;
        docCount = top.filter(isDoc).length;
        if (docCount < minDocEffective) {
            promoteFromPool(docPool, minDocEffective - docCount, isCode);
        }

        // Stitch back the tail of the wider window so callers that ask for
        // more than topK still get their longer list (the slice(0, topK) at
        // the call-site will trim it down).
        const remainder = window.slice(topK);
        return top.concat(remainder).concat(tail);
    }

    /**
     * knowledge-router: concept-span content-type quota. For
     * query_shape=concept, guarantee the post-reranker top-N spans >=2
     * distinct content_types from {doc, code, code_example}.
     *
     * Unlike applyGuaranteeSlots (disabled when a reranker is active), this
     * runs AFTER the reranker and perturbs its order minimally: it promotes
     * exactly ONE chunk of a missing content_type — the highest-RRF-ranked
     * such candidate — evicting the lowest-ranked chunk of the
     * over-represented type. No-op when the top-N already spans >=2 types or
     * no candidate of a missing type exists anywhere in the pool.
     *
     * `candidatePool` is the pre-rerank merged+deduped pool, in weighted-RRF
     * order — the best available rank for candidates that did not survive
     * into the reranked top-N.
     */
    private applyConceptSpanQuota(
        finalResults: SemanticSearchResult[],
        candidatePool: SemanticSearchResult[],
        topK: number,
    ): SemanticSearchResult[] {
        const SPAN_TYPES = new Set(['doc', 'code', 'code_example']);
        const top = finalResults.slice(0, topK);

        const present = new Set<string>();
        for (const r of top) {
            const ct = r.content_type || '';
            if (SPAN_TYPES.has(ct)) present.add(ct);
        }
        if (present.size >= 2) {
            return finalResults;
        }

        const dedupKey = (r: SemanticSearchResult) => `${r.relativePath}#${r.startLine}-${r.endLine}`;
        const inTop = new Set(top.map(dedupKey));

        // Highest-RRF-ranked candidate of a span-type not yet represented.
        const promote = candidatePool.find((r) => {
            const ct = r.content_type || '';
            return SPAN_TYPES.has(ct) && !present.has(ct) && !inTop.has(dedupKey(r));
        });
        if (!promote) {
            console.log('[concept-quota] no candidate of a missing content_type — top-N unchanged');
            return finalResults;
        }

        // Evict the lowest-ranked chunk of the over-represented span-type;
        // fall back to the last slot when the top-N has no span-type chunk
        // (e.g. all `docstring`) so the promoted type still enters top-N.
        let evictIdx = -1;
        for (let i = top.length - 1; i >= 0; i--) {
            const ct = top[i].content_type || '';
            if (SPAN_TYPES.has(ct) && present.has(ct)) { evictIdx = i; break; }
        }
        if (evictIdx === -1) {
            evictIdx = top.length - 1;
        }
        if (evictIdx < 0) {
            return finalResults;
        }

        const result = [...top];
        const evicted = result[evictIdx];
        result[evictIdx] = promote;
        console.log(
            `[concept-quota] promoted ${promote.content_type} chunk ${dedupKey(promote)} ` +
            `→ slot ${evictIdx + 1}, evicted ${evicted.content_type} ${dedupKey(evicted)}`,
        );
        return result;
    }

    /**
     * Phase R (rag-code-intent-recall): decide whether to skip the
     * cross-encoder reranker for the current query. Gated by env
     * RERANKER_BYPASS_FOR_QUALIFIED_NAME (default false). Same intent gate
     * as the symbol-routing pool — code-intent only, no doc tokens, query
     * must parse as a strictly-anchored qualified name.
     */
    private shouldBypassReranker(query: string, intent: { codeSignal: boolean; docSignal: boolean }): boolean {
        if ((process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME || 'false').toLowerCase() !== 'true') return false;
        // Same intent gate as buildSymbolPoolFilter: codeSignal alone, since
        // the anchored parseQualifiedName already rejects embedded NL like
        // "Lambda.fold reduce list to single value".
        if (!intent.codeSignal) return false;
        return parseQualifiedName(query) !== null;
    }

    /**
     * Phase C (rag-code-intent-recall): build a Milvus filter expression for
     * the symbol-routing pool, or return null when any gate fails (env off,
     * intent mismatch, query is not a qualified name, vocab missing
     * className). Pure read-side — no side effects beyond a debug log.
     */
    private async buildSymbolPoolFilter(
        query: string,
        intent: { codeSignal: boolean; docSignal: boolean },
        codebasePath: string,
    ): Promise<string | null> {
        if ((process.env.SYMBOL_ROUTING || 'true').toLowerCase() === 'false') return null;
        // Gate per spec: codeSignal must be true. Embedded NL is already
        // rejected by parseQualifiedName's anchored regex, so we don't need
        // an additional !docSignal check (which would falsely exclude
        // multi-component qualified names like `haxe.io.Path.join` that the
        // classifier flags as docSignal due to >=3 identifier-shaped words).
        if (!intent.codeSignal) return null;
        const parsed = parseQualifiedName(query);
        if (!parsed) return null;
        let vocab: ReadonlySet<string> | null = null;
        try {
            vocab = await this.loadSymbolVocabulary(codebasePath);
        } catch (err) {
            console.warn(`[Context] ⚠️ symbol-vocab load failed: ${err}`);
            return null;
        }
        if (!vocab) {
            console.warn('[Context] ⚠️ symbol-routing skipped: no .symbols-vocab.json available');
            return null;
        }
        const filterExpr = buildSymbolFilter({ parsed, vocab });
        if (!filterExpr) {
            console.log(`[Context] 🔍 symbol-routing skipped for "${query}" (className "${parsed.className}" not in vocab)`);
            return null;
        }
        console.log(`[Context] 🔎 symbol-routing pool engaged for "${query}" → ${filterExpr.slice(0, 120)}${filterExpr.length > 120 ? '…' : ''}`);
        return filterExpr;
    }

    /**
     * sweep-symbol-refs-pool-weight: record which pool contributed a chunk to
     * the merge, and at what rank inside that pool. Only called while the
     * candidate dump is on. A chunk found by two pools keeps one entry per
     * pool, at its best (lowest) rank, so a subject fanned out twice does not
     * read as two different findings.
     */
    private noteCandidatePool(id: string, pool: string, rank: number): void {
        if (!this.candidateProvenance) return;
        const entries = this.candidateProvenance.get(id);
        if (!entries) {
            this.candidateProvenance.set(id, [{ pool, rank }]);
            return;
        }
        const existing = entries.find((e) => e.pool === pool);
        if (!existing) entries.push({ pool, rank });
        else if (rank < existing.rank) existing.rank = rank;
    }

    /**
     * Key a result the way the provenance map is keyed. `chunk_id` is the
     * Milvus primary key and is present on every hybrid path; the line-range
     * fallback only matters for results that never went through a merge.
     */
    private candidateKey(r: SemanticSearchResult): string {
        return r.chunk_id || `${r.relativePath}:${r.startLine}-${r.endLine}`;
    }

    /**
     * Diagnostic dump of the post-merge / pre-rerank candidate pool.
     * Triggered only when env CANDIDATE_LOG_DIR is set; one JSON file per
     * query, named by env CANDIDATE_LOG_QID (or a sha1 prefix of the query
     * if QID is not provided).
     *
     * Staged here rather than written here: `survived` is only knowable once
     * the reranker has cut the pool and the post-processing has run, so the
     * rows wait for flushPreRerankCandidateDump().
     */
    private maybeStagePreRerankCandidates(query: string, candidates: SemanticSearchResult[]): void {
        if (!this.candidateProvenance) return;
        const rows: PreRerankCandidateRow[] = candidates.slice(0, 50).map((r, i) => ({
            rank: i + 1,
            chunk_id: r.chunk_id ?? null,
            relativePath: r.relativePath,
            startLine: r.startLine,
            endLine: r.endLine,
            symbol_name: r.symbol_name ?? null,
            parent_symbol: r.parent_symbol ?? null,
            content_type: r.content_type ?? null,
            score: typeof r.score === 'number' ? Number(r.score.toFixed(6)) : r.score,
            pools: (this.candidateProvenance!.get(this.candidateKey(r)) || [])
                .map((e) => ({ ...e }))
                .sort((a, b) => a.rank - b.rank),
            survived: false,
            finalRank: null,
        }));
        this.pendingCandidateDump = { query, rows };
    }

    /**
     * Write the staged dump, marking which candidates are still in the results
     * the caller receives. No-op when the dump is off or nothing was staged
     * (an error between the merge and the return leaves no file, the same way
     * an errored query leaves no metrics).
     */
    private flushPreRerankCandidateDump(
        finalResults: SemanticSearchResult[],
        meta: { topK: number; rerankerBypassed: boolean; reservedBridgeSlots: number; multiQuery: boolean },
    ): void {
        const staged = this.pendingCandidateDump;
        this.pendingCandidateDump = null;
        if (!staged) return;
        const dir = (process.env.CANDIDATE_LOG_DIR || '').trim();
        if (!dir) return;
        try {
            fs.mkdirSync(dir, { recursive: true });
            const qidRaw = (process.env.CANDIDATE_LOG_QID || '').trim();
            const safeQid = qidRaw
                ? qidRaw.replace(/[^A-Za-z0-9_.-]/g, '_')
                : crypto.createHash('sha1').update(staged.query).digest('hex').slice(0, 12);
            const finalRankByKey = new Map<string, number>();
            finalResults.forEach((r, i) => {
                const key = this.candidateKey(r);
                if (!finalRankByKey.has(key)) finalRankByKey.set(key, i + 1);
            });
            for (const row of staged.rows) {
                const key = row.chunk_id || `${row.relativePath}:${row.startLine}-${row.endLine}`;
                const finalRank = finalRankByKey.get(key);
                row.finalRank = finalRank ?? null;
                row.survived = finalRank !== undefined;
            }
            const payload = {
                qid: qidRaw || null,
                query: staged.query,
                generatedAt: new Date().toISOString(),
                count: staged.rows.length,
                survived: staged.rows.filter((r) => r.survived).length,
                // `survived` is measured at the list the caller receives —
                // topK — which is the cut every gate metric reads. The
                // reranker's own bounds are stated beside it because they are
                // NOT the same number (topK=10 against RERANKER_OUTPUT_K=15 in
                // the eval harness), and a reader that confuses the two
                // undercounts what lived.
                topK: meta.topK,
                rerankerInputK: this.hasReranker() ? this.getRerankerInputK() : null,
                rerankerOutputK: this.hasReranker() ? this.getRerankerOutputK() : null,
                // Without this a bypassed reranker is indistinguishable from
                // one that ran: on a qualified-name query the "cut" is the RRF
                // order itself, and RERANKER_BYPASS_FOR_QUALIFIED_NAME is
                // served on. Bridge-reserved slots skip the reranker the same
                // way.
                rerankerBypassed: meta.rerankerBypassed,
                reservedBridgeSlots: meta.reservedBridgeSlots,
                // MULTI_QUERY=false never reaches weightedRrfMerge, so every
                // row would read `pools: []` — absence of provenance, not
                // absence of the pool.
                multiQuery: meta.multiQuery,
                // A re-merge that rebuilds from the base pools discards an
                // earlier expansion pool's contribution while its provenance
                // entries remain. Both expansions are served off; when they
                // are off, `pools` is exact.
                graphExpand: this.getGraphExpand(),
                proseGraphExpand: this.getProseGraphExpand(),
                candidates: staged.rows,
            };
            fs.writeFileSync(path.join(dir, `${safeQid}.json`), JSON.stringify(payload, null, 2));
        } catch (err) {
            console.warn(`[Context] ⚠️ candidate dump failed: ${err}`);
        }
    }

    /**
     * Deduplicate search results by file + line range overlap.
     * Keeps higher-scored result when two results from the same file overlap >50%.
     *
     * Phase B (rag-code-intent-recall): when CANONICAL_DEDUP is enabled (default
     * true), follow up with a path-cluster preference pass — within each
     * (symbol_name, basename) cluster of >1 result, demote any path containing a
     * marker from PATH_DEMOTE_MARKERS in favour of canonical paths. This breaks
     * ties from vendored / re-implementation / generated copies of the same
     * symbol, language-agnostically.
     */
    private deduplicateResults(results: SemanticSearchResult[]): SemanticSearchResult[] {
        const kept: SemanticSearchResult[] = [];

        for (const result of results) {
            const overlaps = kept.some((existing) => {
                if (existing.relativePath !== result.relativePath) return false;
                const overlapStart = Math.max(existing.startLine, result.startLine);
                const overlapEnd = Math.min(existing.endLine, result.endLine);
                if (overlapStart > overlapEnd) return false;
                // Line ranges are inclusive (endLine = startLine + N - 1).
                const overlapSize = overlapEnd - overlapStart + 1;
                const resultSize = result.endLine - result.startLine + 1;
                return resultSize > 0 && overlapSize / resultSize > 0.5;
            });
            if (!overlaps) {
                kept.push(result);
            }
        }

        return this.applyCanonicalDedup(kept);
    }

    /**
     * Phase B (rag-code-intent-recall): collapse each (symbol_name, basename)
     * cluster down to its canonical (no demote-marker in path). Among multiple
     * canonicals, keep the shortest path (fewest segments). Among ties, keep
     * the original rerank-input order. No-op when the cluster has zero
     * canonicals (all clones) — leaves the cluster untouched so we never
     * accidentally drop the only available match.
     */
    private applyCanonicalDedup(results: SemanticSearchResult[]): SemanticSearchResult[] {
        if ((process.env.CANONICAL_DEDUP || 'true').toLowerCase() === 'false') {
            return results;
        }
        const markers = this.getPathDemoteMarkers();
        if (markers.length === 0) return results;

        const basenameOf = (p: string): string => {
            const i = p.lastIndexOf('/');
            return i === -1 ? p : p.slice(i + 1);
        };
        const segmentCount = (p: string): number => p.split('/').length;
        const hasDemoteMarker = (p: string): boolean => {
            const segments = p.split('/');
            return segments.some((seg) => markers.includes(seg));
        };

        type Indexed = { result: SemanticSearchResult; idx: number };
        const clusters = new Map<string, Indexed[]>();
        const order: Indexed[] = results.map((r, idx) => ({ result: r, idx }));
        for (const item of order) {
            const sym = item.result.symbol_name;
            if (!sym) continue;
            const base = basenameOf(item.result.relativePath || '');
            if (!base) continue;
            const key = `${sym} ${base}`;
            const bucket = clusters.get(key) || [];
            bucket.push(item);
            clusters.set(key, bucket);
        }

        const dropIdx = new Set<number>();
        for (const bucket of clusters.values()) {
            if (bucket.length <= 1) continue;
            const canonicals = bucket.filter((b) => !hasDemoteMarker(b.result.relativePath || ''));
            if (canonicals.length === 0) continue; // all clones — leave intact
            // Pick the winning canonical: shortest path, then earliest rank.
            canonicals.sort((a, b) => {
                const segDelta = segmentCount(a.result.relativePath) - segmentCount(b.result.relativePath);
                if (segDelta !== 0) return segDelta;
                return a.idx - b.idx;
            });
            const winnerIdx = canonicals[0].idx;
            for (const item of bucket) {
                if (item.idx !== winnerIdx) dropIdx.add(item.idx);
            }
        }

        if (dropIdx.size === 0) return results;
        return order.filter((item) => !dropIdx.has(item.idx)).map((item) => item.result);
    }

    private static DEFAULT_PATH_DEMOTE_MARKERS = [
        '_std', 'vendor', 'node_modules', 'dist', 'build', 'generated',
        '__pycache__', 'target', 'out', '.venv', 'venv', 'site-packages',
    ];

    private getPathDemoteMarkers(): string[] {
        const raw = (process.env.PATH_DEMOTE_MARKERS || '').trim();
        if (!raw) return Context.DEFAULT_PATH_DEMOTE_MARKERS;
        return raw.split(',').map((s) => s.trim()).filter((s) => s.length > 0);
    }

    /**
     * Phase 3: filename for the per-codebase symbol vocabulary cache. We
     * stash it next to the codebase so a `git clean` style refresh wipes
     * the stale vocab automatically.
     */
    private getSymbolVocabPath(codebasePath: string): string {
        return path.join(codebasePath, '.symbols-vocab.json');
    }

    /**
     * Write the symbol vocabulary collected during indexCodebase. Best-effort:
     * a write failure is logged but does not abort the indexing run, since the
     * vocab only gates search-time false-positive filtering.
     */
    private async persistSymbolVocabulary(codebasePath: string): Promise<void> {
        const collected = this.indexedSymbols;
        this.indexedSymbols = null;
        if (!collected || collected.size === 0) return;
        const vocabPath = this.getSymbolVocabPath(codebasePath);
        const sorted = Array.from(collected).sort();
        const payload = JSON.stringify({ symbols: sorted, generatedAt: new Date().toISOString() }, null, 2);
        try {
            await fs.promises.writeFile(vocabPath, payload, 'utf-8');
            this.symbolVocabCache.set(codebasePath, new Set(sorted));
            console.log(`[Context] 📚 Wrote symbol vocabulary (${sorted.length} unique symbols) → ${vocabPath}`);
        } catch (err) {
            console.warn(`[Context] ⚠️ Failed to persist symbol vocabulary: ${err}`);
        }
    }

    /**
     * rag-graph-layer Phase 2 + rag-graph-comparison-bridge + rag-graph-abstract-typedef-edges:
     * build `.symbols-graph.json` from the per-run accumulator and write it
     * next to `.symbols-vocab.json`.
     *
     * Schema (`v3-3`, additive over v3-2):
     * ```
     * {
     *   "version": "v3-3",
     *   "by_symbol": {                       // unchanged top-level shape
     *     "<symbol_name>": {
     *       "canonical_chunk_ids": [...],
     *       "mentioned_by_chunk_ids": [...],
     *       "abstract_underlying": [...],    // v3-3: per-abstract forward types
     *       "typedef_alias": "<type>"        // v3-3: per-typedef alias target
     *     }
     *   },
     *   "by_package": {                      // v3-2: cross-subject siblings
     *     "<pkg>": ["<symbol_name>", ...]    // dedup, alphabetical, cap 30
     *   },
     *   "by_supertype": {                    // v3-2: interface/abstract polymorphism
     *     "<supertype>": ["<symbol_name>", ...]
     *   },
     *   "by_abstract_underlying": {          // v3-3: Haxe abstract underlying/from/to
     *     "<typeName>": ["<symbol_name>", ...]
     *   },
     *   "by_typedef_alias": {                // v3-3: Haxe typedef alias targets
     *     "<typeName>": ["<symbol_name>", ...]
     *   }
     * }
     * ```
     *
     * Package derivation per chunk: parent_symbol (if set) ∥ file-path-based
     * directory (Haxe-aware: strips `_std/` target prefix and `haxe/code/haxe/std/`
     * source root). Empty package → symbol skipped from by_package.
     */
    private async persistGraphSideIndex(codebasePath: string): Promise<void> {
        const accumulator = this.graphAccumulator;
        this.graphAccumulator = null;
        if (!accumulator) return;

        const chunks = accumulator.snapshot();
        if (chunks.length === 0) return;

        // rag-graph-layer Phase 2.1: per-symbol map. Code chunks contribute
        // to canonical_chunk_ids (after demote-marker dedup); doc / code_example
        // chunks contribute to mentioned_by_chunk_ids per their mentioned_symbols.
        const markers = this.getPathDemoteMarkers();
        const hasDemoteMarker = (p: string): boolean => {
            const segments = (p || '').split('/');
            return segments.some((seg) => markers.includes(seg));
        };

        // Vocab gate (spec D8 R4): keep only known-vocab symbols in the
        // mentioned-by edges so docs talking about JS-stdlib `array.push`
        // don't pull random chunks. If vocab is unavailable, no gate.
        const vocab = await this.loadSymbolVocabulary(codebasePath);
        const inVocab = (sym: string): boolean => {
            if (!vocab || vocab.size === 0) return true;
            if (vocab.has(sym)) return true;
            if (sym.includes('.')) {
                for (const seg of sym.split('.')) {
                    if (vocab.has(seg)) return true;
                }
            }
            return false;
        };

        // Group code chunks by symbol_name, then pick canonicals (no demote
        // marker in path). When all candidates are non-canonical (e.g. the
        // symbol only exists in a vendored copy), keep them all so the graph
        // never references a missing chunk.
        const codeBySymbol = new Map<string, GraphAccumulatorChunk[]>();
        const mentionedBySymbol = new Map<string, string[]>();
        const MENTIONED_CAP = 20; // rag-graph-layer Phase 2.2

        for (const c of chunks) {
            if (c.contentType === 'code' && c.symbolName) {
                const bucket = codeBySymbol.get(c.symbolName) || [];
                bucket.push(c);
                codeBySymbol.set(c.symbolName, bucket);
            }
            if ((c.contentType === 'doc' || c.contentType === 'code_example') && c.mentionedSymbols) {
                for (const sym of c.mentionedSymbols) {
                    if (!inVocab(sym)) continue;
                    const bucket = mentionedBySymbol.get(sym) || [];
                    if (bucket.length < MENTIONED_CAP) {
                        bucket.push(c.chunkId);
                        mentionedBySymbol.set(sym, bucket);
                    }
                }
            }
        }

        const bySymbol: Record<string, {
            canonical_chunk_ids: string[];
            mentioned_by_chunk_ids: string[];
            abstract_underlying?: string[];
            typedef_alias?: string;
        }> = {};

        // rag-graph-comparison-bridge: v3-2 inverted indexes. Built from
        // every canonical chunk (post-demote-marker filter), not just the
        // first one — so symbols with multi-target overrides (Haxe stdlib
        // or any monorepo with per-platform copies) appear in every bucket
        // their canonicals span. At runtime the bridge derives the seed's
        // pkg from the seed's own relativePath, so it naturally lands in
        // the bucket containing its same-directory peers.
        const BUCKET_CAP = 30;
        const byPackageSets = new Map<string, Set<string>>();
        const bySupertypeSets = new Map<string, Set<string>>();
        const byAbstractUnderlyingSets = new Map<string, Set<string>>();
        const byTypedefAliasSets = new Map<string, Set<string>>();

        const addToBucket = (
            bucketMap: Map<string, Set<string>>,
            key: string,
            value: string,
        ): void => {
            if (!key || !value) return;
            let bucket = bucketMap.get(key);
            if (!bucket) {
                bucket = new Set<string>();
                bucketMap.set(key, bucket);
            }
            bucket.add(value);
        };

        for (const [sym, bucket] of codeBySymbol.entries()) {
            const canonicals = bucket.filter((b) => !hasDemoteMarker(b.relativePath));
            const chosen = canonicals.length > 0 ? canonicals : bucket;
            // Stable order: shortest path first, then accumulator order.
            chosen.sort((a, b) => {
                const segA = (a.relativePath || '').split('/').length;
                const segB = (b.relativePath || '').split('/').length;
                if (segA !== segB) return segA - segB;
                return 0;
            });
            const ids = chosen.map((c) => c.chunkId);
            const entry = bySymbol[sym] || { canonical_chunk_ids: [], mentioned_by_chunk_ids: [] };
            entry.canonical_chunk_ids = ids;
            // rag-graph-abstract-typedef-edges: collapse per-chunk abstract
            // relations into a per-symbol union — if a symbol has multiple
            // canonical chunks (e.g. multi-target Haxe stdlib), every
            // target's relations participate in the forward attribute.
            const absUnderlying = new Set<string>();
            let typedefAlias: string | undefined;
            for (const c of chosen) {
                if (c.abstractUnderlying) {
                    for (const t of c.abstractUnderlying) {
                        if (t) absUnderlying.add(t);
                    }
                }
                if (c.typedefAlias && !typedefAlias) typedefAlias = c.typedefAlias;
            }
            if (absUnderlying.size > 0) entry.abstract_underlying = Array.from(absUnderlying).sort();
            if (typedefAlias) entry.typedef_alias = typedefAlias;
            bySymbol[sym] = entry;
            // Feed every canonical chunk into the inverted indexes (each
            // contributes the symbol to its own derived package + any
            // supertypes / abstract-underlying / typedef-alias targets
            // declared on that chunk).
            for (const c of chosen) {
                const pkg = c.parentSymbol || derivePackageFromPath(c.relativePath);
                if (pkg) addToBucket(byPackageSets, pkg, sym);
                if (c.extendsName) addToBucket(bySupertypeSets, c.extendsName, sym);
                if (c.implementsList) {
                    for (const s of c.implementsList) {
                        if (s) addToBucket(bySupertypeSets, s, sym);
                    }
                }
                if (c.abstractUnderlying) {
                    for (const t of c.abstractUnderlying) {
                        if (t) addToBucket(byAbstractUnderlyingSets, t, sym);
                    }
                }
                if (c.typedefAlias) {
                    addToBucket(byTypedefAliasSets, c.typedefAlias, sym);
                }
            }
        }

        for (const [sym, ids] of mentionedBySymbol.entries()) {
            const entry = bySymbol[sym] || { canonical_chunk_ids: [], mentioned_by_chunk_ids: [] };
            entry.mentioned_by_chunk_ids = ids;
            bySymbol[sym] = entry;
        }

        const finalizeBucketMap = (m: Map<string, Set<string>>): Record<string, string[]> => {
            const out: Record<string, string[]> = {};
            const keys = Array.from(m.keys()).sort();
            for (const k of keys) {
                const arr = Array.from(m.get(k)!).sort();
                out[k] = arr.length > BUCKET_CAP ? arr.slice(0, BUCKET_CAP) : arr;
            }
            return out;
        };

        const byPackage = finalizeBucketMap(byPackageSets);
        const bySupertype = finalizeBucketMap(bySupertypeSets);
        const byAbstractUnderlying = finalizeBucketMap(byAbstractUnderlyingSets);
        const byTypedefAlias = finalizeBucketMap(byTypedefAliasSets);

        // rag-graph-supertype-extraction-fix + rag-graph-abstract-typedef-edges:
        // one-shot diagnostic so a post-reindex run surfaces extraction
        // quality without manual JSON inspection. Quiet unless
        // DEBUG_COMPARISON_BRIDGE=1.
        if (this.getComparisonBridgeDebug()) {
            const supKeys = Object.keys(bySupertype);
            const supSample = supKeys.slice(0, 5).map((k) => `${k} → ${bySupertype[k].length} impls`);
            console.log(`[graph-side-index] supertype edges: ${supKeys.length} keys, sample: [${supSample.join(', ')}]`);
            const absKeys = Object.keys(byAbstractUnderlying);
            const absSample = absKeys.slice(0, 5).map((k) => `${k} → ${byAbstractUnderlying[k].length}`);
            console.log(`[graph-side-index] abstract_underlying edges: ${absKeys.length} keys, sample: [${absSample.join(', ')}]`);
            const tdKeys = Object.keys(byTypedefAlias);
            const tdSample = tdKeys.slice(0, 5).map((k) => `${k} → ${byTypedefAlias[k].length}`);
            console.log(`[graph-side-index] typedef_alias edges: ${tdKeys.length} keys, sample: [${tdSample.join(', ')}]`);
        }

        const sidePath = path.join(codebasePath, '.symbols-graph.json');
        const payload = {
            version: 'v3-3',
            generatedAt: new Date().toISOString(),
            by_symbol: bySymbol,
            by_package: byPackage,
            by_supertype: bySupertype,
            by_abstract_underlying: byAbstractUnderlying,
            by_typedef_alias: byTypedefAlias,
        };

        try {
            await fs.promises.writeFile(sidePath, JSON.stringify(payload, null, 2), 'utf-8');
            const symbolCount = Object.keys(bySymbol).length;
            const pkgCount = Object.keys(byPackage).length;
            const supCount = Object.keys(bySupertype).length;
            const absCount = Object.keys(byAbstractUnderlying).length;
            const tdCount = Object.keys(byTypedefAlias).length;
            console.log(`[Context] 🕸️  Wrote graph side-index v3-3 (${symbolCount} symbols, ${pkgCount} packages, ${supCount} supertypes, ${absCount} abstract_underlying, ${tdCount} typedef_alias) → ${sidePath}`);
        } catch (err) {
            console.warn(`[Context] ⚠️ Failed to persist graph side-index: ${err}`);
        }
    }

    /**
     * Load (and cache) the symbol vocabulary written by a prior indexCodebase
     * run. Returns null if the file does not exist or fails to parse — callers
     * fall back to unfiltered candidate extraction.
     */
    async loadSymbolVocabulary(codebasePath: string): Promise<ReadonlySet<string> | null> {
        if (this.symbolVocabCache.has(codebasePath)) {
            return this.symbolVocabCache.get(codebasePath) ?? null;
        }
        const vocabPath = this.getSymbolVocabPath(codebasePath);
        try {
            const raw = await fs.promises.readFile(vocabPath, 'utf-8');
            const parsed = JSON.parse(raw);
            const list: unknown = parsed?.symbols;
            if (!Array.isArray(list)) {
                this.symbolVocabCache.set(codebasePath, null);
                return null;
            }
            const set = new Set<string>(list.filter((s): s is string => typeof s === 'string' && s.length > 0));
            this.symbolVocabCache.set(codebasePath, set);
            console.log(`[Context] 📚 Loaded symbol vocabulary (${set.size} symbols) from ${vocabPath}`);
            return set;
        } catch {
            this.symbolVocabCache.set(codebasePath, null);
            return null;
        }
    }

    /**
     * Check if index exists for codebase
     *
     * Asks the WHOLE address (#152): in split mode a codebase's index is the
     * prose and the code collection, and an all-prose corpus has an empty or
     * absent code side. Asking only `getCollectionName` (the code name by
     * convention) answered "not indexed" for a codebase whose prose collection
     * held every row — and the MCP server's recovery, gated on this answer,
     * never ran.
     *
     * @param codebasePath Codebase path to check
     * @returns Whether index exists
     */
    async hasIndex(codebasePath: string): Promise<boolean> {
        for (const name of this.collectionsOf(codebasePath)) {
            if (await this.vectorDatabase.hasCollection(name)) return true;
        }
        return false;
    }

    /**
     * Clear index
     * @param codebasePath Codebase path to clear index for
     * @param progressCallback Optional progress callback function
     */
    async clearIndex(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void
    ): Promise<void> {
        console.log(`[Context] 🧹 Cleaning index data for ${codebasePath}...`);

        progressCallback?.({ phase: 'Checking existing index...', current: 0, total: 100, percentage: 0 });

        // Clearing removes the WHOLE address (#152): dropping only the code side
        // left the prose collection alive, and with an address-wide `hasIndex`
        // the next search recovered the snapshot entry from those surviving rows
        // — a cleared codebase resurrecting itself.
        const names = this.collectionsOf(codebasePath);

        progressCallback?.({ phase: 'Removing index data...', current: 50, total: 100, percentage: 50 });

        for (const name of names) {
            if (await this.vectorDatabase.hasCollection(name)) {
                await this.vectorDatabase.dropCollection(name);
            }
        }

        // Delete snapshot file
        await FileSynchronizer.deleteSnapshot(codebasePath);

        progressCallback?.({ phase: 'Index cleared', current: 100, total: 100, percentage: 100 });
        console.log('[Context] ✅ Index data cleaned');
    }

    /**
     * Update ignore patterns (merges with default patterns and existing patterns)
     * @param ignorePatterns Array of ignore patterns to add to defaults
     */
    updateIgnorePatterns(ignorePatterns: string[]): void {
        // Merge with default patterns and any existing custom patterns, avoiding duplicates
        const mergedPatterns = [...DEFAULT_IGNORE_PATTERNS, ...ignorePatterns];
        this.baseIgnorePatterns = this.dedupePatterns(mergedPatterns);
        this.ignorePatterns = [...this.baseIgnorePatterns];
        console.log(`[Context] 🚫 Updated ignore patterns: ${ignorePatterns.length} new + ${DEFAULT_IGNORE_PATTERNS.length} default = ${this.ignorePatterns.length} total patterns`);
    }

    /**
     * Add custom ignore patterns (from MCP or other sources) without replacing existing ones
     * @param customPatterns Array of custom ignore patterns to add
     */
    addCustomIgnorePatterns(customPatterns: string[]): void {
        if (customPatterns.length === 0) return;

        // Merge persistent base patterns with new custom patterns, avoiding duplicates.
        const mergedPatterns = [...this.baseIgnorePatterns, ...customPatterns];
        this.baseIgnorePatterns = this.dedupePatterns(mergedPatterns);
        this.ignorePatterns = [...this.baseIgnorePatterns];
        console.log(`[Context] 🚫 Added ${customPatterns.length} custom ignore patterns. Total: ${this.ignorePatterns.length} patterns`);
    }

    /**
     * Reset ignore patterns to defaults only
     */
    resetIgnorePatternsToDefaults(): void {
        this.baseIgnorePatterns = [...DEFAULT_IGNORE_PATTERNS];
        this.ignorePatterns = [...this.baseIgnorePatterns];
        console.log(`[Context] 🔄 Reset ignore patterns to defaults: ${this.ignorePatterns.length} patterns`);
    }

    /**
     * Update embedding instance
     * @param embedding New embedding instance
     */
    updateEmbedding(embedding: Embedding): void {
        this.embedding = embedding;
        console.log(`[Context] 🔄 Updated embedding provider: ${embedding.getProvider()}`);
    }

    /**
     * Update vector database instance
     * @param vectorDatabase New vector database instance
     */
    updateVectorDatabase(vectorDatabase: VectorDatabase): void {
        this.vectorDatabase = vectorDatabase;
        console.log(`[Context] 🔄 Updated vector database`);
    }

    /**
     * Update splitter instance
     * @param splitter New splitter instance
     */
    updateSplitter(splitter: Splitter): void {
        this.codeSplitter = splitter;
        console.log(`[Context] 🔄 Updated splitter instance`);
    }

    /**
     * Prepare vector collection
     */
    private async prepareCollection(codebasePath: string, forceReindex: boolean = false): Promise<void> {
        const isHybrid = this.getIsHybrid();
        const collectionType = isHybrid === true ? 'hybrid vector' : 'vector';
        console.log(`[Context] 🔧 Preparing ${collectionType} collection for codebase: ${codebasePath}${forceReindex ? ' (FORCE REINDEX)' : ''}`);
        const addr = this.getCollectionAddress(codebasePath);
        // code-collection-split: in split mode we materialize both prose and
        // code collections with identical schema. In legacy mode the loop
        // runs exactly once over `addr.legacy` — byte-stable behavior.
        let targets: Array<{ name: string; label: 'prose' | 'code' | 'legacy' }> = addr.isSplit
            ? [{ name: addr.prose, label: 'prose' }, { name: addr.code, label: 'code' }]
            : [{ name: addr.legacy, label: 'legacy' }];
        // prose-embedding-swap: restrict to the requested pool(s) so a
        // prose-only reindex never creates/drops the code collection.
        const poolFilter = this.reindexPoolFilter();
        if (addr.isSplit && poolFilter) {
            targets = targets.filter((t) => (t.label === 'prose' || t.label === 'code') ? poolFilter.has(t.label) : true);
            console.log(`[Context] 🎯 REINDEX_POOLS filter active → preparing only: ${targets.map((t) => t.label).join(', ')}`);
        }
        if (addr.isSplit) {
            console.log(`[Context] 🪓 SPLIT_COLLECTIONS=true → preparing collections: ${targets.map((t) => `${t.name} (${t.label})`).join(' + ')}`);
        }

        // prose-embedding-swap: detect dimension per embedder, not once
        // globally — the prose collection MAY have a different dense dim than
        // the code collection. Cache per provider so we don't re-probe.
        const dimCache = new Map<Embedding, number>();
        const detectDimFor = async (emb: Embedding): Promise<number> => {
            const cached = dimCache.get(emb);
            if (cached !== undefined) return cached;
            console.log(`[Context] 🔍 Detecting embedding dimension for ${emb.getProvider()} provider...`);
            const d = await emb.detectDimension();
            console.log(`[Context] 📏 Detected dimension: ${d} for ${emb.getProvider()}`);
            dimCache.set(emb, d);
            return d;
        };

        const headerMode = this.getChunkContextHeaderMode();
        for (const { name: collectionName, label } of targets) {
            const collectionExists = await this.vectorDatabase.hasCollection(collectionName);

            if (collectionExists && !forceReindex) {
                await this.assertCollectionHeaderMode(collectionName, headerMode);
                console.log(`📋 Collection ${collectionName} (${label}) already exists, skipping creation`);
                continue;
            }

            if (collectionExists && forceReindex) {
                console.log(`[Context] 🗑️  Dropping existing collection ${collectionName} (${label}) for force reindex...`);
                await this.vectorDatabase.dropCollection(collectionName);
                console.log(`[Context] ✅ Collection ${collectionName} dropped successfully`);
            }

            // The prose collection uses the prose embedder (when distinct);
            // code/legacy collections use the code embedder.
            const collectionEmbedding = (addr.isSplit && label === 'prose')
                ? this.embeddingForPool('prose')
                : this.embedding;
            const dimension = await detectDimFor(collectionEmbedding);

            // Per-collection dense-model/dim metadata for query-side routing
            // (prose-embedding-swap task 2.4). Only added in split mode so the
            // legacy single-collection description stays byte-stable.
            const denseModelLabel = (collectionEmbedding as any).getModel?.() ?? collectionEmbedding.getProvider();
            // pilot-chunk-context-headers: the header mode is appended only
            // when it is not `off`, so an `off` description is unchanged.
            const description = describeWithHeaderMode(addr.isSplit
                ? `codebasePath:${codebasePath};dense_model:${denseModelLabel};dense_dim:${dimension}`
                : `codebasePath:${codebasePath}`, headerMode);

            if (isHybrid === true) {
                const enableLearnedSparse = this.embedding.hasSparse();
                await this.vectorDatabase.createHybridCollection(
                    collectionName,
                    dimension,
                    description,
                    headerMode === 'off' ? { enableLearnedSparse } : { enableLearnedSparse, indexText: true },
                );
            } else {
                await this.vectorDatabase.createCollection(collectionName, dimension, description);
            }

            console.log(`[Context] ✅ Collection ${collectionName} (${label}) created successfully (dimension: ${dimension})`);
        }
    }

    /**
     * Recursively get all code files in the codebase
     */
    private async getCodeFiles(
        codebasePath: string,
        ignorePatterns: string[] = this.ignorePatterns,
        supportedExtensions: string[] = this.supportedExtensions
    ): Promise<string[]> {
        const files: string[] = [];

        const traverseDirectory = async (currentPath: string) => {
            const entries = await fs.promises.readdir(currentPath, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(currentPath, entry.name);

                // Check if path matches ignore patterns
                if (this.matchesIgnorePattern(fullPath, codebasePath, ignorePatterns)) {
                    continue;
                }

                if (entry.isDirectory()) {
                    await traverseDirectory(fullPath);
                } else if (entry.isFile()) {
                    const ext = path.extname(entry.name);
                    if (supportedExtensions.includes(ext)) {
                        files.push(fullPath);
                    }
                }
            }
        };

        await traverseDirectory(codebasePath);
        return files;
    }

    /**
 * Process a list of files with streaming chunk processing
 * @param filePaths Array of file paths to process
 * @param codebasePath Base path for the codebase
 * @param onFileProcessed Callback called when each file is processed
 * @returns Object with processed file count and total chunk count
 */
    private async processFileList(
        filePaths: string[],
        codebasePath: string,
        onFileProcessed?: (filePath: string, fileIndex: number, totalFiles: number) => void,
        splitter: Splitter = this.codeSplitter
    ): Promise<{ processedFiles: number; totalChunks: number; droppedChunks: number; status: 'completed' | 'limit_reached' }> {
        const isHybrid = this.getIsHybrid();
        const EMBEDDING_BATCH_SIZE = Math.max(1, parseInt(envManager.get('EMBEDDING_BATCH_SIZE') || '100', 10));
        // #19: the Milvus `content` column is VarChar(65535 BYTES) on every
        // collection this engine creates, and no splitter guarantees that — a
        // minified bundle (three.js's 520 KB single-line draco_decoder.js) comes
        // back as one chunk and takes its whole insert batch down with it.
        // Enforced here rather than in each splitter so it holds for every
        // language, every splitter and every collection. Env-overridable only
        // for tests / a differently-declared schema.
        const CHUNK_MAX_BYTES = Math.max(1, parseInt(envManager.get('CHUNK_MAX_BYTES') || String(MILVUS_CONTENT_MAX_BYTES), 10));
        // Env-overridable so a candidate reindex of a corpus larger than the
        // default guard can run without editing code. The default is unchanged:
        // it is a runaway-indexing brake, not a corpus size statement.
        const CHUNK_LIMIT = Math.max(1, parseInt(envManager.get('CHUNK_LIMIT') || '450000', 10));
        console.log(`[Context] 🔧 Using EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE}`);

        let chunkBuffer: Array<{ chunk: CodeChunk; codebasePath: string }> = [];
        let processedFiles = 0;
        let totalChunks = 0;
        let droppedChunks = 0;
        let limitReached = false;

        for (let i = 0; i < filePaths.length; i++) {
            const filePath = filePaths[i];

            // #138: the limit is file-atomic and checked BEFORE a file's chunks
            // are computed, so a file is either fully indexed or not started. A
            // prefix of a file's chunks in the vector database is the worst
            // state the limit could leave: the reseed path derives a file's
            // visibility from its Milvus rows, so a partially-inserted file
            // would read as current and never be re-indexed. The total may
            // overshoot CHUNK_LIMIT by at most the last file's chunk count —
            // the honest alternative to a half-written file.
            if (totalChunks >= CHUNK_LIMIT) {
                console.warn(`[Context] ⚠️  Chunk limit of ${CHUNK_LIMIT} reached. Stopping indexing before ${filePath}.`);
                limitReached = true;
                break;
            }

            try {
                const split = await this.splitFileForIndex(filePath, splitter, CHUNK_MAX_BYTES, true);
                if (split === null) {
                    processedFiles++;
                    onFileProcessed?.(filePath, i + 1, filePaths.length);
                    continue;
                }
                const { content, chunks } = split;

                // Log files with many chunks or large content
                if (chunks.length > 50) {
                    console.warn(`[Context] ⚠️  File ${filePath} generated ${chunks.length} chunks (${Math.round(content.length / 1024)}KB)`);
                } else if (content.length > 100000) {
                    console.log(`📄 Large file ${filePath}: ${Math.round(content.length / 1024)}KB -> ${chunks.length} chunks`);
                }

                // Add chunks to buffer
                for (const chunk of chunks) {
                    chunkBuffer.push({ chunk, codebasePath });
                    totalChunks++;

                    // Process batch when buffer reaches EMBEDDING_BATCH_SIZE
                    if (chunkBuffer.length >= EMBEDDING_BATCH_SIZE) {
                        const batchSize = chunkBuffer.length;
                        try {
                            droppedChunks += await this.processChunkBuffer(chunkBuffer);
                        } catch (error) {
                            // #19: an escaping error means the whole batch is
                            // gone (insertRowsWithRecovery only re-throws when
                            // no row could be written at all). Count it, and
                            // state the size in the "N/M rows rejected" form
                            // infra/lib/index-state.js parses.
                            droppedChunks += batchSize;
                            const searchType = isHybrid === true ? 'hybrid' : 'regular';
                            console.error(`[Context] ❌ Failed to process chunk batch for ${searchType}: ${batchSize}/${batchSize} rows rejected:`, error);
                            if (error instanceof Error) {
                                console.error('[Context] Stack trace:', error.stack);
                            }
                        } finally {
                            chunkBuffer = []; // Always clear buffer, even on failure
                        }
                    }
                }

                processedFiles++;
                onFileProcessed?.(filePath, i + 1, filePaths.length);

            } catch (error) {
                console.warn(`[Context] ⚠️  Skipping file ${filePath}: ${error}`);
            }
        }

        // Process any remaining chunks in the buffer
        if (chunkBuffer.length > 0) {
            const searchType = isHybrid === true ? 'hybrid' : 'regular';
            const batchSize = chunkBuffer.length;
            console.log(`📝 Processing final batch of ${batchSize} chunks for ${searchType}`);
            try {
                droppedChunks += await this.processChunkBuffer(chunkBuffer);
            } catch (error) {
                droppedChunks += batchSize;
                console.error(`[Context] ❌ Failed to process final chunk batch for ${searchType}: ${batchSize}/${batchSize} rows rejected:`, error);
                if (error instanceof Error) {
                    console.error('[Context] Stack trace:', error.stack);
                }
            }
        }

        return {
            processedFiles,
            totalChunks,
            droppedChunks,
            status: limitReached ? 'limit_reached' : 'completed'
        };
    }

    /**
     * The chunks one file is indexed as, or null for a file that is skipped
     * (non-class-reference XML). Shared by the indexing loop and the
     * `generated` build's chunk-set check, so both see the same chunks.
     */
    private async splitFileForIndex(filePath: string, splitter: Splitter, maxBytes: number, logOversized: boolean): Promise<{ content: string; chunks: CodeChunk[] } | null> {
        let content = await fs.promises.readFile(filePath, 'utf-8');
        let language = this.getLanguageFromExtension(path.extname(filePath));

        // Class-reference XML (Godot's `doc/classes/*.xml`) carries the
        // API prose and the GDScript/C# examples that exist nowhere in
        // the C++ sources. Rewrite it to Markdown so it lands in the
        // prose pool with a heading path; any other XML is metadata and
        // is skipped rather than indexed as markup.
        if (language === 'xml') {
            const markdown = classDocXmlToMarkdown(content);
            if (markdown === null) return null;
            content = markdown;
            language = 'markdown';
        }

        const rawChunks = await splitter.split(content, language, filePath);
        const chunks = enforceChunkByteLimit(rawChunks, {
            maxBytes,
            onOversized: logOversized ? (info) => {
                console.warn(
                    `[Context] ✂️  Oversized chunk in ${info.filePath || filePath} at line ${info.startLine}: ` +
                    `${info.bytes}B > ${maxBytes}B → split into ${info.parts} part(s)` +
                    (info.truncatedBytes > 0 ? `, ${info.truncatedBytes}B truncated` : ''),
                );
            } : undefined,
        });
        return { content, chunks };
    }

    /**
     * pilot-chunk-context-headers: a `generated` build refuses, before any
     * collection is touched, a context file generated over another chunk set
     * or covering too few of the chunks this build will index.
     */
    private async checkChunkContextFile(codebasePath: string, codeFiles: string[], splitter: Splitter): Promise<void> {
        const ids = new Set((await this.splitForChunkContexts(codebasePath, { files: codeFiles, splitter })).map((r) => r.id));
        const { covered, total } = assertChunkContextCoverage(this.getChunkContexts(codebasePath, true), ids);
        console.log(`[Context] 🏷️  CHUNK_CONTEXT_FILE covers ${covered} of ${total} chunk ids`);
    }

    /**
     * pilot-chunk-context-headers: the text the dense, learned-sparse and BM25
     * channels read for one chunk under a header mode. `maxChars` is the
     * embedder's truncation point, for counting chunks the header pushed past it.
     */
    private indexTextFor(chunk: CodeChunk, codebasePath: string, mode: ChunkContextHeaderMode, maxChars: number): { embed: string; indexText: string } {
        const meta = chunk.metadata as any;
        const relativePath = path.relative(codebasePath, meta.filePath || '');
        let context: string | undefined;
        if (mode === 'generated') {
            const id = this.generateId(relativePath, meta.startLine || 0, meta.endLine || 0, chunk.content, meta.part);
            context = this.getChunkContexts(codebasePath).contexts.get(id);
            if (!context) {
                this.chunkContextReport.missing_contexts++;
                this.chunkContextReport.missing_by_file[relativePath] = (this.chunkContextReport.missing_by_file[relativePath] || 0) + 1;
            }
        }
        const header = buildChunkContextHeader({
            relativePath,
            heading_path: meta.heading_path,
            symbol_name: meta.symbol_name,
            parent_symbol: meta.parent_symbol,
            symbol_kind: meta.symbol_kind,
        }, context);
        const embed = withChunkContextHeader(header, chunk.content);
        if (embed.length > maxChars && chunk.content.length <= maxChars) this.chunkContextReport.header_truncated_embedding++;
        const indexText = truncateUtf8(embed, INDEX_TEXT_MAX_BYTES);
        if (indexText !== embed && Buffer.byteLength(chunk.content, 'utf8') <= INDEX_TEXT_MAX_BYTES) this.chunkContextReport.header_truncated_index_text++;
        return { embed, indexText };
    }

    /**
 * Process accumulated chunk buffer
 */
    private async processChunkBuffer(chunkBuffer: Array<{ chunk: CodeChunk; codebasePath: string }>): Promise<number> {
        if (chunkBuffer.length === 0) return 0;

        // Extract chunks and ensure they all have the same codebasePath
        const chunks = chunkBuffer.map(item => item.chunk);
        const codebasePath = chunkBuffer[0].codebasePath;

        // Estimate tokens (rough estimation: 1 token ≈ 4 characters)
        const estimatedTokens = chunks.reduce((sum, chunk) => sum + Math.ceil(chunk.content.length / 4), 0);

        const isHybrid = this.getIsHybrid();
        const searchType = isHybrid === true ? 'hybrid' : 'regular';
        console.log(`[Context] 🔄 Processing batch of ${chunks.length} chunks (~${estimatedTokens} tokens) for ${searchType}`);
        // #19: the number of chunks the vector database refused, so the caller
        // can report the loss instead of reporting success.
        return this.processChunkBatch(chunks, codebasePath);
    }

    /**
     * Process a batch of chunks
     */
    private async processChunkBatch(chunks: CodeChunk[], codebasePath: string): Promise<number> {
        const isHybrid = this.getIsHybrid();

        // prose-embedding-swap: REINDEX_POOLS filter — drop chunks whose target
        // pool is excluded so a prose-only reindex never embeds or upserts
        // code-side chunks (the code collection stays exactly as the baseline
        // left it). No-op when the filter is unset.
        const poolFilter = this.reindexPoolFilter();
        if (poolFilter) {
            const addrF = this.getCollectionAddress(codebasePath);
            if (addrF.isSplit) {
                chunks = chunks.filter((c) => {
                    const target = this.resolveChunkCollection((c.metadata as any).content_type, addrF);
                    const label: 'prose' | 'code' = target === addrF.prose ? 'prose' : 'code';
                    return poolFilter.has(label);
                });
                if (chunks.length === 0) return 0;
            }
        }

        // Phase 3: harvest symbol names into the vocabulary collector.
        if (this.indexedSymbols) {
            for (const chunk of chunks) {
                const sym = (chunk.metadata as any).symbol_name;
                const parent = (chunk.metadata as any).parent_symbol;
                if (typeof sym === 'string' && sym.length > 0) this.indexedSymbols.add(sym);
                if (typeof parent === 'string' && parent.length > 0) this.indexedSymbols.add(parent);
            }
        }

        // Generate embedding vectors.
        // prose-embedding-swap: in split mode with a distinct prose embedder,
        // embed prose chunks (doc/code_example) through the prose model and
        // code chunks (code/docstring) through the code model. Both embedders
        // share the learned-sparse sidecar, so every chunk still carries the
        // learned-sparse lexical_weights for the sparse channel. Default path (no distinct
        // prose embedder) runs a single embedBatch — byte-identical to v6.
        const batchAddr = this.getCollectionAddress(codebasePath);
        // pilot-chunk-context-headers: with a header mode the channels embed
        // header + content; `off` embeds exactly the chunk text, as before.
        const headerMode = this.getChunkContextHeaderMode();
        const headerTexts = headerMode === 'off' ? null : chunks.map((chunk) => {
            const ct = (chunk.metadata as any).content_type as string | undefined;
            const pool: 'code' | 'prose' = batchAddr.isSplit && this.resolveChunkCollection(ct, batchAddr) === batchAddr.prose ? 'prose' : 'code';
            const maxTokens = (this.embeddingForPool(pool) as any).maxTokens;
            return this.indexTextFor(chunk, codebasePath, headerMode, typeof maxTokens === 'number' ? maxTokens * 4 : Infinity);
        });
        const chunkContents = headerTexts ? headerTexts.map((t) => t.embed) : chunks.map(chunk => chunk.content);
        let embeddings: EmbeddingVector[];
        if (batchAddr.isSplit && this.hasDistinctProseEmbedding()) {
            const proseIdx: number[] = [];
            const codeIdx: number[] = [];
            for (let i = 0; i < chunks.length; i++) {
                const ct = (chunks[i].metadata as any).content_type as string | undefined;
                if (this.resolveChunkCollection(ct, batchAddr) === batchAddr.prose) proseIdx.push(i);
                else codeIdx.push(i);
            }
            embeddings = new Array<EmbeddingVector>(chunks.length);
            const [proseEmb, codeEmb] = await Promise.all([
                proseIdx.length > 0
                    ? this.embeddingForPool('prose').embedBatch(proseIdx.map(i => chunkContents[i]))
                    : Promise.resolve([] as EmbeddingVector[]),
                codeIdx.length > 0
                    ? this.embedding.embedBatch(codeIdx.map(i => chunkContents[i]))
                    : Promise.resolve([] as EmbeddingVector[]),
            ]);
            proseIdx.forEach((origIdx, k) => { embeddings[origIdx] = proseEmb[k]; });
            codeIdx.forEach((origIdx, k) => { embeddings[origIdx] = codeEmb[k]; });
            console.log(`[Context] 🧬 Split-embed batch: ${proseIdx.length} prose (${this.embeddingForPool('prose').getProvider()}) + ${codeIdx.length} code (${this.embeddingForPool('code').getProvider()})`);
        } else {
            embeddings = await this.embedding.embedBatch(chunkContents);
        }

        // rag-graph-layer Phase 2: accumulator entries are recorded inside
        // the per-chunk map below — each chunk_id gets paired with the
        // raw mentioned_symbols (chunk.metadata, before JSON-encoding) so
        // the side-builder doesn't have to parse the Milvus schema string.
        const recordToGraph = (
            chunkId: string,
            chunk: CodeChunk,
        ): void => {
            if (!this.graphAccumulator) return;
            const meta = chunk.metadata as any;
            this.graphAccumulator.add({
                chunkId,
                relativePath: path.relative(codebasePath, meta.filePath || ''),
                contentType: meta.content_type,
                symbolName: meta.symbol_name,
                mentionedSymbols: Array.isArray(meta.mentioned_symbols)
                    ? meta.mentioned_symbols.slice()
                    : undefined,
                // rag-graph-comparison-bridge v3-2: capture parent/extends/implements
                // so `by_package` (via qualified_name) and `by_supertype` can be
                // built deterministically in persistGraphSideIndex.
                parentSymbol: typeof meta.parent_symbol === 'string' && meta.parent_symbol.length > 0
                    ? meta.parent_symbol
                    : undefined,
                extendsName: typeof meta.extends === 'string' && meta.extends.length > 0
                    ? meta.extends
                    : undefined,
                implementsList: Array.isArray(meta.implements) && meta.implements.length > 0
                    ? meta.implements.slice()
                    : undefined,
                // rag-graph-abstract-typedef-edges v3-3: surface the relations
                // emitted by ast-structural-extractor for Haxe abstracts /
                // typedefs into the graph accumulator. Other languages produce
                // no such fields, so the lookups silently return undefined.
                abstractUnderlying: Array.isArray(meta.abstract_underlying) && meta.abstract_underlying.length > 0
                    ? meta.abstract_underlying.slice()
                    : undefined,
                typedefAlias: typeof meta.typedef_alias === 'string' && meta.typedef_alias.length > 0
                    ? meta.typedef_alias
                    : undefined,
            });
        };

        // code-collection-split: resolve once per batch so we can route each
        // chunk to its target collection by content_type. In legacy mode
        // addr.isSplit=false and the loop collapses to a single name.
        const addr = this.getCollectionAddress(codebasePath);

        if (isHybrid === true) {
            // Create hybrid vector documents
            const documents: VectorDocument[] = chunks.map((chunk, index) => {
                if (!chunk.metadata.filePath) {
                    throw new Error(`Missing filePath in chunk metadata at index ${index}`);
                }

                const relativePath = path.relative(codebasePath, chunk.metadata.filePath);
                const fileExtension = path.extname(chunk.metadata.filePath);
                const {
                    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- destructured only to keep these three out of `...restMetadata`, which is spread into the document below; deleting them adds three columns to every row
                    filePath, startLine, endLine,
                    content_type, symbol_kind, symbol_name, parent_symbol, heading_path,
                    imports, extends: extendsName, implements: implementsList, mentioned_symbols,
                    ...restMetadata
                } = chunk.metadata;

                return {
                    id: this.generateId(relativePath, chunk.metadata.startLine || 0, chunk.metadata.endLine || 0, chunk.content, chunk.metadata.part),
                    content: chunk.content, // Full text content for BM25 and storage
                    vector: embeddings[index].vector, // Dense vector
                    relativePath,
                    startLine: chunk.metadata.startLine || 0,
                    endLine: chunk.metadata.endLine || 0,
                    fileExtension,
                    metadata: {
                        ...restMetadata,
                        codebasePath,
                        language: chunk.metadata.language || 'unknown',
                        chunkIndex: index
                    },
                    content_type,
                    symbol_kind,
                    symbol_name,
                    parent_symbol,
                    heading_path: heading_path ? JSON.stringify(heading_path) : undefined,
                    // rag-graph-layer Phase 1.3: encode array fields as JSON
                    // strings so the v3 Milvus VarChar columns can hold them.
                    imports: imports && imports.length > 0 ? JSON.stringify(imports) : undefined,
                    extends: extendsName || undefined,
                    implements: implementsList && implementsList.length > 0 ? JSON.stringify(implementsList) : undefined,
                    mentioned_symbols: mentioned_symbols && mentioned_symbols.length > 0 ? JSON.stringify(mentioned_symbols) : undefined,
                    // Phase 4: BGE-M3 learned sparse from the sparse sidecar.
                    // Falsy when the embedding provider doesn't expose sparse;
                    // insertHybrid only attaches it when present and non-empty.
                    sparse_learned: embeddings[index].sparse,
                    ...(headerTexts ? { index_text: headerTexts[index].indexText } : {}),
                };
            });

            // rag-graph-layer Phase 2: register every successfully prepared
            // chunk into the graph accumulator. Run before the actual
            // insertHybrid so a network/RPC failure doesn't half-record.
            for (let i = 0; i < documents.length; i++) {
                recordToGraph(documents[i].id, chunks[i]);
            }

            // code-collection-split: partition by content_type and write to
            // each target collection. Legacy path is a single bucket whose
            // key equals addr.legacy (== addr.code == addr.prose).
            if (addr.isSplit) {
                const byCollection = new Map<string, VectorDocument[]>();
                for (let i = 0; i < documents.length; i++) {
                    const ct = (chunks[i].metadata as any).content_type as string | undefined;
                    const target = this.resolveChunkCollection(ct, addr);
                    const bucket = byCollection.get(target) ?? [];
                    bucket.push(documents[i]);
                    byCollection.set(target, bucket);
                }
                let dropped = 0;
                for (const [target, bucket] of byCollection.entries()) {
                    if (bucket.length === 0) continue;
                    dropped += droppedCount(await this.vectorDatabase.insertHybrid(target, bucket));
                }
                return dropped;
            } else {
                return droppedCount(await this.vectorDatabase.insertHybrid(addr.legacy, documents));
            }
        } else {
            // Create regular vector documents
            const documents: VectorDocument[] = chunks.map((chunk, index) => {
                if (!chunk.metadata.filePath) {
                    throw new Error(`Missing filePath in chunk metadata at index ${index}`);
                }

                const relativePath = path.relative(codebasePath, chunk.metadata.filePath);
                const fileExtension = path.extname(chunk.metadata.filePath);
                const {
                    // eslint-disable-next-line @typescript-eslint/no-unused-vars -- destructured only to keep these three out of `...restMetadata`, which is spread into the document below; deleting them adds three columns to every row
                    filePath, startLine, endLine,
                    content_type, symbol_kind, symbol_name, parent_symbol, heading_path,
                    imports, extends: extendsName, implements: implementsList, mentioned_symbols,
                    ...restMetadata
                } = chunk.metadata;

                return {
                    id: this.generateId(relativePath, chunk.metadata.startLine || 0, chunk.metadata.endLine || 0, chunk.content, chunk.metadata.part),
                    vector: embeddings[index].vector,
                    content: chunk.content,
                    relativePath,
                    startLine: chunk.metadata.startLine || 0,
                    endLine: chunk.metadata.endLine || 0,
                    fileExtension,
                    metadata: {
                        ...restMetadata,
                        codebasePath,
                        language: chunk.metadata.language || 'unknown',
                        chunkIndex: index
                    },
                    content_type,
                    symbol_kind,
                    symbol_name,
                    parent_symbol,
                    heading_path: heading_path ? JSON.stringify(heading_path) : undefined,
                    imports: imports && imports.length > 0 ? JSON.stringify(imports) : undefined,
                    extends: extendsName || undefined,
                    implements: implementsList && implementsList.length > 0 ? JSON.stringify(implementsList) : undefined,
                    mentioned_symbols: mentioned_symbols && mentioned_symbols.length > 0 ? JSON.stringify(mentioned_symbols) : undefined,
                };
            });

            for (let i = 0; i < documents.length; i++) {
                recordToGraph(documents[i].id, chunks[i]);
            }

            // code-collection-split: same partitioning as the hybrid path.
            // Non-hybrid mode is rare in production but kept consistent.
            if (addr.isSplit) {
                const byCollection = new Map<string, VectorDocument[]>();
                for (let i = 0; i < documents.length; i++) {
                    const ct = (chunks[i].metadata as any).content_type as string | undefined;
                    const target = this.resolveChunkCollection(ct, addr);
                    const bucket = byCollection.get(target) ?? [];
                    bucket.push(documents[i]);
                    byCollection.set(target, bucket);
                }
                let dropped = 0;
                for (const [target, bucket] of byCollection.entries()) {
                    if (bucket.length === 0) continue;
                    dropped += droppedCount(await this.vectorDatabase.insert(target, bucket));
                }
                return dropped;
            } else {
                return droppedCount(await this.vectorDatabase.insert(addr.legacy, documents));
            }
        }
    }

    /**
     * Get programming language based on file extension
     */
    private getLanguageFromExtension(ext: string): string {
        const languageMap: Record<string, string> = {
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.m': 'objective-c',
            '.mm': 'objective-c',
            '.dart': 'dart',
            '.hx': 'haxe',
            '.hxml': 'hxml',
            '.ml': 'ocaml',
            '.mli': 'ocaml_interface',
            '.ipynb': 'jupyter',
            '.xml': 'xml',
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.rst': 'rst'
        };
        return languageMap[ext] || 'text';
    }

    /**
     * Generate unique ID based on chunk content and location
     * @param relativePath Relative path to the file
     * @param startLine Start line number
     * @param endLine End line number
     * @param content Chunk content
     * @returns Hash-based unique ID
     */
    private generateId(relativePath: string, startLine: number, endLine: number, content: string, part?: string): string {
        // #19: the parts of an oversized chunk that was cut mid-line all carry
        // the same line range, so `part` ("2/8") is what separates them. Absent
        // for every chunk that was not split, which keeps ids byte-identical to
        // what earlier collections were built with.
        const combinedString = `${relativePath}:${startLine}:${endLine}:${part ? `${part}:` : ''}${content}`;
        const hash = crypto.createHash('sha256').update(combinedString, 'utf-8').digest('hex');
        return `chunk_${hash.substring(0, 16)}`;
    }

    /**
     * Read ignore patterns from file (e.g., .gitignore)
     * @param filePath Path to the ignore file
     * @returns Array of ignore patterns
     */
    static async getIgnorePatternsFromFile(filePath: string): Promise<string[]> {
        try {
            const content = await fs.promises.readFile(filePath, 'utf-8');
            return content
                .split('\n')
                .map(line => line.trim())
                .filter(line => line && !line.startsWith('#')); // Filter out empty lines and comments
        } catch (error) {
            console.warn(`[Context] ⚠️  Could not read ignore file ${filePath}: ${error}`);
            return [];
        }
    }

    /**
     * Load ignore patterns from various ignore files in the codebase.
     * Returns the effective patterns for the current codebase/request without
     * allowing file-based patterns from previous codebases to leak forward.
     * @param codebasePath Path to the codebase
     * @param additionalIgnorePatterns Ignore patterns for the current request
     */
    private async loadIgnorePatterns(codebasePath: string, additionalIgnorePatterns: string[] = []): Promise<string[]> {
        try {
            const fileBasedPatterns: string[] = [];

            // Load all .xxxignore files in codebase directory
            const ignoreFiles = await this.findIgnoreFiles(codebasePath);
            for (const ignoreFile of ignoreFiles) {
                const patterns = await this.loadIgnoreFile(ignoreFile, path.basename(ignoreFile));
                fileBasedPatterns.push(...patterns);
            }

            // Load global ~/.context/.contextignore
            const globalIgnorePatterns = await this.loadGlobalIgnoreFile();
            fileBasedPatterns.push(...globalIgnorePatterns);

            const effectiveIgnorePatterns = this.dedupePatterns([
                ...this.baseIgnorePatterns,
                ...additionalIgnorePatterns,
                ...fileBasedPatterns
            ]);
            // Preserve the previous observable getIgnorePatterns() behavior for
            // sequential callers, while all indexing paths use the local return
            // value to avoid shared-state leakage between background tasks.
            this.ignorePatterns = effectiveIgnorePatterns;

            if (fileBasedPatterns.length > 0 || additionalIgnorePatterns.length > 0) {
                console.log(`[Context] 🚫 Loaded total ${fileBasedPatterns.length} ignore patterns from all ignore files and ${additionalIgnorePatterns.length} request ignore patterns`);
            } else {
                console.log('📄 No ignore files found, using base ignore patterns');
            }
            return effectiveIgnorePatterns;
        } catch (error) {
            console.warn(`[Context] ⚠️ Failed to load ignore patterns: ${error}`);
            // Continue with base/request patterns on error - don't reuse
            // previously loaded codebase-specific patterns.
            const fallbackPatterns = this.dedupePatterns([
                ...this.baseIgnorePatterns,
                ...additionalIgnorePatterns
            ]);
            this.ignorePatterns = fallbackPatterns;
            return fallbackPatterns;
        }
    }

    /**
     * Find all .xxxignore files in the codebase directory
     * @param codebasePath Path to the codebase
     * @returns Array of ignore file paths
     */
    private async findIgnoreFiles(codebasePath: string): Promise<string[]> {
        try {
            const entries = await fs.promises.readdir(codebasePath, { withFileTypes: true });
            const ignoreFiles: string[] = [];

            for (const entry of entries) {
                if (entry.isFile() &&
                    entry.name.startsWith('.') &&
                    entry.name.endsWith('ignore')) {
                    ignoreFiles.push(path.join(codebasePath, entry.name));
                }
            }

            if (ignoreFiles.length > 0) {
                console.log(`📄 Found ignore files: ${ignoreFiles.map(f => path.basename(f)).join(', ')}`);
            }

            return ignoreFiles;
        } catch (error) {
            console.warn(`[Context] ⚠️ Failed to scan for ignore files: ${error}`);
            return [];
        }
    }

    /**
     * Load global ignore file from ~/.context/.contextignore
     * @returns Array of ignore patterns
     */
    private async loadGlobalIgnoreFile(): Promise<string[]> {
        try {
            // Stays a `require`, for the same reason as the lazy load in
            // AstCodeSplitter's constructor: as `import * as os` at the top of the
            // file, resolution and namespace capture move from this call to module
            // evaluation. In production that is invisible — `os` is a side-effect-free
            // builtin — but a registry substitution installed after `Context` is
            // imported and before ignore patterns load (jest.doMock('os'),
            // Module._load) is honoured by this form and not by the hoisted one, and
            // would resolve a different ~/.context/.contextignore. #129, cross-model
            // review.
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            const homeDir = require('os').homedir();
            const globalIgnorePath = path.join(homeDir, '.context', '.contextignore');
            return await this.loadIgnoreFile(globalIgnorePath, 'global .contextignore');
        } catch {
            // Global ignore file is optional, don't log warnings
            return [];
        }
    }

    /**
     * Load ignore patterns from a specific ignore file
     * @param filePath Path to the ignore file
     * @param fileName Display name for logging
     * @returns Array of ignore patterns
     */
    private async loadIgnoreFile(filePath: string, fileName: string): Promise<string[]> {
        try {
            await fs.promises.access(filePath);
            console.log(`📄 Found ${fileName} file at: ${filePath}`);

            const ignorePatterns = await Context.getIgnorePatternsFromFile(filePath);

            if (ignorePatterns.length > 0) {
                console.log(`[Context] 🚫 Loaded ${ignorePatterns.length} ignore patterns from ${fileName}`);
                return ignorePatterns;
            } else {
                console.log(`📄 ${fileName} file found but no valid patterns detected`);
                return [];
            }
        } catch {
            if (fileName.includes('global')) {
                console.log(`📄 No ${fileName} file found`);
            }
            return [];
        }
    }

    /**
     * Check if a path matches any ignore pattern
     * @param filePath Path to check
     * @param basePath Base path for relative pattern matching
     * @returns True if path should be ignored
     */
    private matchesIgnorePattern(filePath: string, basePath: string, ignorePatterns: string[] = this.ignorePatterns): boolean {
        const relativePath = path.relative(basePath, filePath);

        // Always ignore dotfiles/dotdirs to stay aligned with
        // FileSynchronizer.shouldIgnore. If these traversals diverge, files
        // indexed here are never hashed by the synchronizer and their stale
        // chunks linger in Milvus forever.
        if (relativePath.split(path.sep).some(part => part.startsWith('.'))) {
            return true;
        }

        if (ignorePatterns.length === 0) {
            return false;
        }

        const normalizedPath = relativePath.replace(/\\/g, '/'); // Normalize path separators

        for (const pattern of ignorePatterns) {
            if (this.isPatternMatch(normalizedPath, pattern)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Simple glob pattern matching
     * @param filePath File path to test
     * @param pattern Glob pattern
     * @returns True if pattern matches
     */
    private isPatternMatch(filePath: string, pattern: string): boolean {
        const cleanPath = filePath.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');
        const normalizedPattern = pattern.replace(/\\/g, '/');
        const cleanPattern = normalizedPattern.replace(/^\/+|\/+$/g, '');
        const isRootAnchored = normalizedPattern.startsWith('/');
        const isDirectoryPattern = normalizedPattern.endsWith('/');

        if (!cleanPath || !cleanPattern) {
            return false;
        }

        // Handle directory patterns (ending with /)
        if (isDirectoryPattern) {
            if (isRootAnchored) {
                return this.simpleGlobMatch(cleanPath, cleanPattern) ||
                    cleanPath.startsWith(`${cleanPattern}/`);
            }

            return this.matchesDirectoryPattern(cleanPath, cleanPattern);
        }

        if (isRootAnchored) {
            return this.simpleGlobMatch(cleanPath, cleanPattern);
        }

        // Handle file patterns
        if (cleanPattern.includes('/')) {
            // Pattern with path separator - match exact path
            return this.simpleGlobMatch(cleanPath, cleanPattern);
        } else {
            // Pattern without path separator - match filename in any directory
            const fileName = path.basename(cleanPath);
            return this.simpleGlobMatch(fileName, cleanPattern);
        }
    }

    private matchesDirectoryPattern(filePath: string, dirPattern: string): boolean {
        const pathParts = filePath.split('/');
        const dirPartCount = dirPattern.split('/').length;

        for (let i = 0; i <= pathParts.length - dirPartCount; i++) {
            const candidate = pathParts.slice(i, i + dirPartCount).join('/');
            if (this.simpleGlobMatch(candidate, dirPattern)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Simple glob matching supporting * wildcard
     * @param text Text to test
     * @param pattern Pattern with * wildcards
     * @returns True if pattern matches
     */
    private simpleGlobMatch(text: string, pattern: string): boolean {
        // Convert glob pattern to regex
        const regexPattern = pattern
            .replace(/[.+^${}()|[\]\\]/g, '\\$&') // Escape regex special chars except *
            .replace(/\*/g, '.*'); // Convert * to .*

        const regex = new RegExp(`^${regexPattern}$`);
        return regex.test(text);
    }

    private dedupePatterns(patterns: string[]): string[] {
        return [...new Set(patterns)];
    }

    /**
     * Get custom extensions from environment variables
     * Supports CUSTOM_EXTENSIONS as comma-separated list
     * @returns Array of custom extensions
     */
    private getCustomExtensionsFromEnv(): string[] {
        const envExtensions = envManager.get('CUSTOM_EXTENSIONS');
        if (!envExtensions) {
            return [];
        }

        try {
            const extensions = envExtensions
                .split(',')
                .map(ext => ext.trim())
                .filter(ext => ext.length > 0)
                .map(ext => ext.startsWith('.') ? ext : `.${ext}`); // Ensure extensions start with dot

            return extensions;
        } catch (error) {
            console.warn(`[Context] ⚠️  Failed to parse CUSTOM_EXTENSIONS: ${error}`);
            return [];
        }
    }

    /**
     * Get custom ignore patterns from environment variables  
     * Supports CUSTOM_IGNORE_PATTERNS as comma-separated list
     * @returns Array of custom ignore patterns
     */
    private getCustomIgnorePatternsFromEnv(): string[] {
        const envIgnorePatterns = envManager.get('CUSTOM_IGNORE_PATTERNS');
        if (!envIgnorePatterns) {
            return [];
        }

        try {
            const patterns = envIgnorePatterns
                .split(',')
                .map(pattern => pattern.trim())
                .filter(pattern => pattern.length > 0);

            return patterns;
        } catch (error) {
            console.warn(`[Context] ⚠️  Failed to parse CUSTOM_IGNORE_PATTERNS: ${error}`);
            return [];
        }
    }

    private normalizeExtensions(extensions: string[]): string[] {
        return extensions
            .map(ext => ext.trim())
            .filter(ext => ext.length > 0)
            .map(ext => ext.startsWith('.') ? ext : `.${ext}`);
    }

    /**
     * Add custom extensions (from MCP or other sources) without replacing existing ones
     * @param customExtensions Array of custom extensions to add
     */
    addCustomExtensions(customExtensions: string[]): void {
        if (customExtensions.length === 0) return;

        const normalizedExtensions = this.normalizeExtensions(customExtensions);

        // Merge current extensions with new custom extensions, avoiding duplicates
        const mergedExtensions = [...this.supportedExtensions, ...normalizedExtensions];
        const uniqueExtensions: string[] = [...new Set(mergedExtensions)];
        this.supportedExtensions = uniqueExtensions;
        console.log(`[Context] 📎 Added ${customExtensions.length} custom extensions. Total: ${this.supportedExtensions.length} extensions`);
    }

    /**
     * Get current splitter information
     */
    getSplitterInfo(): { type: string; hasBuiltinFallback: boolean; supportedLanguages?: string[] } {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            // Typed import: #130 — this call went through `require` (untyped
            // `any`), which is how a call to a method that did not exist hid a
            // runtime `TypeError` from the compiler. The static exists now, and
            // the binding is typed, so the checker holds the seam.
            return {
                type: 'ast',
                hasBuiltinFallback: true,
                supportedLanguages: AstCodeSplitter.getSupportedLanguages()
            };
        } else {
            return {
                type: 'langchain',
                hasBuiltinFallback: false
            };
        }
    }

    /**
     * Check if current splitter supports a specific language
     * @param language Programming language
     */
    isLanguageSupported(language: string): boolean {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            return AstCodeSplitter.isLanguageSupported(language);
        }

        // LangChain splitter supports most languages
        return true;
    }

    /**
     * Get which strategy would be used for a specific language
     * @param language Programming language
     */
    getSplitterStrategyForLanguage(language: string): { strategy: 'ast' | 'langchain'; reason: string } {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const isSupported = AstCodeSplitter.isLanguageSupported(language);

            return {
                strategy: isSupported ? 'ast' : 'langchain',
                reason: isSupported
                    ? 'Language supported by AST parser'
                    : 'Language not supported by AST, will fallback to LangChain'
            };
        } else {
            return {
                strategy: 'langchain',
                reason: 'Using LangChain splitter directly'
            };
        }
    }
}
