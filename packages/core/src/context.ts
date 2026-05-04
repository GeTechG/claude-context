import {
    Splitter,
    CodeChunk,
    AstCodeSplitter
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
    HybridSearchOptions,
    HybridSearchResult,
    RerankStrategy
} from './vectordb';
import { SemanticSearchResult } from './types';
import { envManager } from './utils/env-manager';
import { classifyQuery, weightsForIntent, DomainWeights, parseQualifiedName } from './search/query-classifier';
import { buildSymbolFilter } from './search/symbol-routing';
import { Reranker } from './reranker';
import { extractCandidateSymbols } from './enrichment';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { FileSynchronizer } from './sync/synchronizer';

function parseHeadingPath(raw: string | undefined): string[] | undefined {
    if (!raw) return undefined;
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.map(String) : undefined;
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

const DEFAULT_SUPPORTED_EXTENSIONS = [
    // Programming languages
    '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    '.dart', '.hx', '.hxml',
    // Text and markup files
    '.md', '.markdown', '.rst', '.ipynb',
    // '.txt',  '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
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

    constructor(config: ContextConfig = {}) {
        // Initialize services
        this.embedding = config.embedding || new OpenAIEmbedding({
            apiKey: envManager.get('OPENAI_API_KEY') || 'your-openai-api-key',
            model: 'text-embedding-3-small',
            ...(envManager.get('OPENAI_BASE_URL') && { baseURL: envManager.get('OPENAI_BASE_URL') })
        });

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
     * Get embedding instance
     */
    getEmbedding(): Embedding {
        return this.embedding;
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
     * Build the Milvus inner-rerank strategy for the per-pool 3-channel
     * hybrid_search. If any CHANNEL_WEIGHT_* env var is set, switches to
     * Milvus 'weighted' ranker with weights aligned to buildRequests order:
     *   [dense, sparse_bm25, sparse_learned]
     * Otherwise stays on 'rrf' with k = MILVUS_RRF_K (default 100).
     *
     * `hasLearnedSparse` reflects whether the third channel is included in
     * this particular request (some queries have no learned-sparse vector).
     */
    private buildInnerRerankStrategy(hasLearnedSparse: boolean): RerankStrategy {
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
     * Generate collection name based on codebase path and hybrid mode
     */
    public getCollectionName(codebasePath: string): string {
        const isHybrid = this.getIsHybrid();
        const versionSegment = this.getCollectionVersionSegment();
        const prefix = isHybrid === true ? `hybrid${versionSegment}_code_chunks` : `code_chunks${versionSegment}`;
        const normalizedPath = path.resolve(codebasePath);
        const pathHash = crypto.createHash('md5').update(normalizedPath).digest('hex').substring(0, 8);

        // Overrides always keep the per-codebase `_<pathHash>` suffix so that multiple
        // codebases indexed by the same MCP server can't collapse into one collection.
        const configOverride = this.getValidOverrideValue(this.collectionNameOverride);
        if (configOverride) {
            const suffix = this.sanitizeCollectionNameSuffix(configOverride, prefix, pathHash, 'Context config');
            return `${prefix}_${suffix}`;
        }

        const envOverride = this.getValidOverrideValue(envManager.get('CODE_CHUNKS_COLLECTION_NAME_OVERRIDE'));
        if (envOverride) {
            const suffix = this.sanitizeCollectionNameSuffix(envOverride, prefix, pathHash, 'CODE_CHUNKS_COLLECTION_NAME_OVERRIDE');
            return `${prefix}_${suffix}`;
        }

        return `${prefix}_${pathHash}`;
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
    ): Promise<{ indexedFiles: number; totalChunks: number; status: 'completed' | 'limit_reached' }> {
        const isHybrid = this.getIsHybrid();
        const searchType = isHybrid === true ? 'hybrid search' : 'semantic search';
        console.log(`[Context] 🚀 Starting to index codebase with ${searchType}: ${codebasePath}`);
        const splitter = requestSplitter || this.codeSplitter;

        // 1. Compute ignore patterns for this codebase/request without
        // retaining file-based patterns from previous codebases.
        const ignorePatterns = await this.loadIgnorePatterns(codebasePath, additionalIgnorePatterns);

        // Phase 3: start a fresh symbol-vocabulary collector for this run.
        this.indexedSymbols = new Set<string>();

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
            return { indexedFiles: 0, totalChunks: 0, status: 'completed' };
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

        // Phase 3: persist the collected symbol vocabulary so search-time
        // candidate extraction can filter out false positives.
        await this.persistSymbolVocabulary(codebasePath);

        progressCallback?.({
            phase: 'Indexing complete!',
            current: result.processedFiles,
            total: codeFiles.length,
            percentage: 100
        });

        return {
            indexedFiles: result.processedFiles,
            totalChunks: result.totalChunks,
            status: result.status
        };
    }

    async reindexByChange(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void,
        additionalIgnorePatterns: string[] = [],
        additionalSupportedExtensions: string[] = [],
        requestSplitter?: Splitter
    ): Promise<{ added: number, removed: number, modified: number }> {
        const collectionName = this.getCollectionName(codebasePath);
        const synchronizer = this.synchronizers.get(collectionName);
        const splitter = requestSplitter || this.codeSplitter;

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
        const { added, removed, modified } = await currentSynchronizer.checkForChanges();
        const totalChanges = added.length + removed.length + modified.length;

        if (totalChanges === 0) {
            progressCallback?.({ phase: 'No changes detected', current: 100, total: 100, percentage: 100 });
            console.log('[Context] ✅ No file changes detected.');
            return { added: 0, removed: 0, modified: 0 };
        }

        console.log(`[Context] 🔄 Found changes: ${added.length} added, ${removed.length} removed, ${modified.length} modified.`);

        let processedChanges = 0;
        const updateProgress = (phase: string) => {
            processedChanges++;
            const percentage = Math.round((processedChanges / (removed.length + modified.length + added.length)) * 100);
            progressCallback?.({ phase, current: processedChanges, total: totalChanges, percentage });
        };

        // Handle removed files
        for (const file of removed) {
            await this.deleteFileChunks(collectionName, file);
            updateProgress(`Removed ${file}`);
        }

        // Handle modified files
        for (const file of modified) {
            await this.deleteFileChunks(collectionName, file);
            updateProgress(`Deleted old chunks for ${file}`);
        }

        // Handle added and modified files
        const filesToIndex = [...added, ...modified].map(f => path.join(codebasePath, f));

        if (filesToIndex.length > 0) {
            await this.processFileList(
                filesToIndex,
                codebasePath,
                (filePath, fileIndex, totalFiles) => {
                    updateProgress(`Indexed ${filePath} (${fileIndex}/${totalFiles})`);
                },
                splitter
            );
        }

        console.log(`[Context] ✅ Re-indexing complete. Added: ${added.length}, Removed: ${removed.length}, Modified: ${modified.length}`);
        progressCallback?.({ phase: 'Re-indexing complete!', current: totalChanges, total: totalChanges, percentage: 100 });

        return { added: added.length, removed: removed.length, modified: modified.length };
    }

    private async deleteFileChunks(collectionName: string, relativePath: string): Promise<void> {
        // Escape backslashes for Milvus query expression (Windows path compatibility)
        const escapedPath = relativePath.replace(/\\/g, '\\\\');
        const results = await this.vectorDatabase.query(
            collectionName,
            `relativePath == "${escapedPath}"`,
            ['id']
        );

        if (results.length > 0) {
            const ids = results.map(r => r.id as string).filter(id => id);
            if (ids.length > 0) {
                await this.vectorDatabase.delete(collectionName, ids);
                console.log(`[Context] Deleted ${ids.length} chunks for file ${relativePath}`);
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
    async semanticSearch(codebasePath: string, query: string, topK: number = 5, threshold: number = 0.5, filterExpr?: string): Promise<SemanticSearchResult[]> {
        const isHybrid = this.getIsHybrid();
        const searchType = isHybrid === true ? 'hybrid search' : 'semantic search';
        console.log(`[Context] 🔍 Executing ${searchType}: "${query}" in ${codebasePath}`);

        const collectionName = this.getCollectionName(codebasePath);
        console.log(`[Context] 🔍 Using collection: ${collectionName}`);

        // Check if collection exists and has data
        const hasCollection = await this.vectorDatabase.hasCollection(collectionName);
        if (!hasCollection) {
            console.log(`[Context] ⚠️  Collection '${collectionName}' does not exist. Please index the codebase first.`);
            return [];
        }

        if (isHybrid === true) {
            try {
                // Check collection stats to see if it has data
                const stats = await this.vectorDatabase.query(collectionName, '', ['id'], 1);
                console.log(`[Context] 🔍 Collection '${collectionName}' exists and appears to have data`);
            } catch (error) {
                console.log(`[Context] ⚠️  Collection '${collectionName}' exists but may be empty or not properly indexed:`, error);
            }

            // 1. Generate query vector
            console.log(`[Context] 🔍 Generating embeddings for query: "${query}"`);
            const queryEmbedding: EmbeddingVector = await this.embedding.embed(query);
            console.log(`[Context] ✅ Generated embedding vector with dimension: ${queryEmbedding.vector.length}`);
            console.log(`[Context] 🔍 First 5 embedding values: [${queryEmbedding.vector.slice(0, 5).join(', ')}]`);

            const multiQuery = this.getMultiQuery();
            const intent = classifyQuery(query);
            const weights = weightsForIntent(intent);
            console.log(`[Context] 🔍 Query intent: codeSignal=${intent.codeSignal} docSignal=${intent.docSignal} → weights code=${weights.code} doc=${weights.doc}`);

            // For multi-query we pull a wider candidate pool from each domain
            // so weighted RRF and guarantee-slot reshuffling have something
            // to work with. PER_POOL_K is intentionally generous because
            // RRF on a small pool tends to lock in the leaders. When a
            // reranker is downstream, also make sure each pool can supply
            // ~RERANKER_INPUT_K candidates after merge.
            const rerankerInputK = this.hasReranker() ? this.getRerankerInputK() : 0;
            const PER_POOL_K = Math.max(topK * 5, 25, rerankerInputK);

            const buildRequests = (limit: number): HybridSearchRequest[] => {
                const reqs: HybridSearchRequest[] = [
                    {
                        data: queryEmbedding.vector,
                        anns_field: "vector",
                        param: { "nprobe": 10 },
                        limit
                    },
                    {
                        data: query,
                        anns_field: "sparse_vector",
                        param: { "drop_ratio_search": 0.2 },
                        limit
                    }
                ];
                // Phase 4: add BGE-M3 learned-sparse channel when the embedding
                // provider populated it (m3serve sidecar wired through
                // InfinityEmbedding.sparseURL). Milvus accepts the sparse vector
                // as a dict { "<index>": value } for SPARSE_FLOAT_VECTOR fields
                // without an attached function.
                if (queryEmbedding.sparse && queryEmbedding.sparse.indices.length > 0) {
                    const sparseDict: Record<string, number> = {};
                    const { indices, values } = queryEmbedding.sparse;
                    const len = Math.min(indices.length, values.length);
                    for (let i = 0; i < len; i++) {
                        sparseDict[String(indices[i])] = values[i];
                    }
                    reqs.push({
                        data: sparseDict,
                        anns_field: "sparse_learned",
                        param: { "drop_ratio_search": 0.2 },
                        limit,
                    });
                }
                return reqs;
            };

            let mergedResults: SemanticSearchResult[];

            if (multiQuery) {
                console.log(`[Context] 🔍 MULTI_QUERY=true → running parallel code-domain + doc-domain hybrid searches (PER_POOL_K=${PER_POOL_K})`);

                const combinedFilter = (extra: string | undefined): string => {
                    if (!extra || extra.trim().length === 0) return '';
                    return `(${extra})`;
                };
                const userFilter = filterExpr && filterExpr.trim().length > 0 ? filterExpr : undefined;
                const codeExpr = userFilter
                    ? `${combinedFilter(userFilter)} and ${CODE_DOMAIN_FILTER}`
                    : CODE_DOMAIN_FILTER;
                const docExpr = userFilter
                    ? `${combinedFilter(userFilter)} and ${DOC_DOMAIN_FILTER}`
                    : DOC_DOMAIN_FILTER;

                const innerRerank = this.buildInnerRerankStrategy(
                    !!(queryEmbedding.sparse && queryEmbedding.sparse.indices.length > 0)
                );

                // Phase C (rag-code-intent-recall): symbol-routing pool —
                // when the query is a qualified name AND code-intent AND
                // the className is in the indexed vocab, build a metadata
                // filter that pins candidates to the symbol's chunks.
                const symbolPoolLimit = 10;
                const symbolFilterExpr = await this.buildSymbolPoolFilter(query, intent, codebasePath);
                const symbolPoolPromise: Promise<HybridSearchResult[]> = symbolFilterExpr
                    ? this.vectorDatabase.hybridSearch(
                        collectionName,
                        buildRequests(symbolPoolLimit),
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

                const [codePool, docPool, symbolPool] = await Promise.all([
                    this.vectorDatabase.hybridSearch(
                        collectionName,
                        buildRequests(PER_POOL_K),
                        {
                            rerank: innerRerank,
                            limit: PER_POOL_K,
                            filterExpr: codeExpr,
                        }
                    ).catch((err) => {
                        console.warn(`[Context] ⚠️ code-domain hybrid search failed: ${err}`);
                        return [] as HybridSearchResult[];
                    }),
                    this.vectorDatabase.hybridSearch(
                        collectionName,
                        buildRequests(PER_POOL_K),
                        {
                            rerank: innerRerank,
                            limit: PER_POOL_K,
                            filterExpr: docExpr,
                        }
                    ).catch((err) => {
                        console.warn(`[Context] ⚠️ doc-domain hybrid search failed: ${err}`);
                        return [] as HybridSearchResult[];
                    }),
                    symbolPoolPromise,
                ]);

                console.log(`[Context] 🔍 Pool sizes: code=${codePool.length} doc=${docPool.length} symbol=${symbolPool.length}`);

                // Wider merge when a reranker is downstream so it has
                // ~RERANKER_INPUT_K candidates to arbitrate over.
                const mergeLimit = this.hasReranker()
                    ? Math.max(rerankerInputK, topK * 3, 30)
                    : Math.max(topK * 3, 30);

                const SYMBOL_POOL_WEIGHT = 2.0;
                const mergePools: { results: HybridSearchResult[]; weight: number }[] = [
                    { results: codePool, weight: weights.code },
                    { results: docPool, weight: weights.doc },
                ];
                if (symbolPool.length > 0) {
                    mergePools.push({ results: symbolPool, weight: SYMBOL_POOL_WEIGHT });
                }
                const merged = this.weightedRrfMerge(
                    mergePools,
                    mergeLimit,
                    this.getRrfK()
                );

                let semanticMerged: SemanticSearchResult[] = merged.map((r) => this.toSemanticResult(r));

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
                const innerRerank = this.buildInnerRerankStrategy(
                    !!(queryEmbedding.sparse && queryEmbedding.sparse.indices.length > 0)
                );
                const searchResults: HybridSearchResult[] = await this.vectorDatabase.hybridSearch(
                    collectionName,
                    buildRequests(singleLimit),
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

            this.maybeDumpPreRerankCandidates(query, dedupedResults);

            // Phase R (rag-code-intent-recall): when the query is a
            // strictly-anchored qualified name AND code-intent only, allow
            // bypass of the cross-encoder reranker. Cross-encoders trained on
            // NL-QA distribute mass over surface-similar wrong-subject chunks
            // for short identifier queries; the merged-RRF order (now
            // boosted by the symbol-routing pool) is more reliable on these.
            const rerankerBypassed = this.shouldBypassReranker(query, intent);

            // Phase 2: cross-encoder rerank over the merged-and-deduped pool.
            const finalResults = this.hasReranker() && !rerankerBypassed
                ? await this.applyReranker(query, dedupedResults, topK)
                : dedupedResults.slice(0, topK);
            if (rerankerBypassed) {
                console.log(`[Context] ⏭️  reranker bypassed for qualified-name code query "${query}"`);
            }

            if (finalResults.length > 0) {
                console.log(`[Context] 🔍 Top result score: ${finalResults[0].score}, path: ${finalResults[0].relativePath}`);
            }
            await this.attachCandidateSymbols(finalResults, codebasePath);
            return finalResults;
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
        };
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
        pools: { results: HybridSearchResult[]; weight: number }[],
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
     * Diagnostic dump of the post-merge / pre-rerank candidate pool.
     * Triggered only when env CANDIDATE_LOG_DIR is set; one JSON file per
     * query, named by env CANDIDATE_LOG_QID (or a sha1 prefix of the query
     * if QID is not provided).
     */
    private maybeDumpPreRerankCandidates(query: string, candidates: SemanticSearchResult[]): void {
        const dir = (process.env.CANDIDATE_LOG_DIR || '').trim();
        if (!dir) return;
        try {
            fs.mkdirSync(dir, { recursive: true });
            const qidRaw = (process.env.CANDIDATE_LOG_QID || '').trim();
            const safeQid = qidRaw
                ? qidRaw.replace(/[^A-Za-z0-9_.-]/g, '_')
                : crypto.createHash('sha1').update(query).digest('hex').slice(0, 12);
            const top = candidates.slice(0, 50).map((r, i) => ({
                rank: i + 1,
                relativePath: r.relativePath,
                startLine: r.startLine,
                endLine: r.endLine,
                symbol_name: r.symbol_name ?? null,
                parent_symbol: r.parent_symbol ?? null,
                content_type: r.content_type ?? null,
                score: typeof r.score === 'number' ? Number(r.score.toFixed(6)) : r.score,
            }));
            const payload = {
                qid: qidRaw || null,
                query,
                generatedAt: new Date().toISOString(),
                count: top.length,
                candidates: top,
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
     * @param codebasePath Codebase path to check
     * @returns Whether index exists
     */
    async hasIndex(codebasePath: string): Promise<boolean> {
        const collectionName = this.getCollectionName(codebasePath);
        return await this.vectorDatabase.hasCollection(collectionName);
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

        const collectionName = this.getCollectionName(codebasePath);
        const collectionExists = await this.vectorDatabase.hasCollection(collectionName);

        progressCallback?.({ phase: 'Removing index data...', current: 50, total: 100, percentage: 50 });

        if (collectionExists) {
            await this.vectorDatabase.dropCollection(collectionName);
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
        const collectionName = this.getCollectionName(codebasePath);

        // Check if collection already exists
        const collectionExists = await this.vectorDatabase.hasCollection(collectionName);

        if (collectionExists && !forceReindex) {
            console.log(`📋 Collection ${collectionName} already exists, skipping creation`);
            return;
        }

        if (collectionExists && forceReindex) {
            console.log(`[Context] 🗑️  Dropping existing collection ${collectionName} for force reindex...`);
            await this.vectorDatabase.dropCollection(collectionName);
            console.log(`[Context] ✅ Collection ${collectionName} dropped successfully`);
        }

        console.log(`[Context] 🔍 Detecting embedding dimension for ${this.embedding.getProvider()} provider...`);
        const dimension = await this.embedding.detectDimension();
        console.log(`[Context] 📏 Detected dimension: ${dimension} for ${this.embedding.getProvider()}`);
        const dirName = path.basename(codebasePath);

        if (isHybrid === true) {
            const enableLearnedSparse = this.embedding.hasSparse();
            await this.vectorDatabase.createHybridCollection(
                collectionName,
                dimension,
                `codebasePath:${codebasePath}`,
                { enableLearnedSparse },
            );
        } else {
            await this.vectorDatabase.createCollection(collectionName, dimension, `codebasePath:${codebasePath}`);
        }

        console.log(`[Context] ✅ Collection ${collectionName} created successfully (dimension: ${dimension})`);
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
    ): Promise<{ processedFiles: number; totalChunks: number; status: 'completed' | 'limit_reached' }> {
        const isHybrid = this.getIsHybrid();
        const EMBEDDING_BATCH_SIZE = Math.max(1, parseInt(envManager.get('EMBEDDING_BATCH_SIZE') || '100', 10));
        const CHUNK_LIMIT = 450000;
        console.log(`[Context] 🔧 Using EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE}`);

        let chunkBuffer: Array<{ chunk: CodeChunk; codebasePath: string }> = [];
        let processedFiles = 0;
        let totalChunks = 0;
        let limitReached = false;

        for (let i = 0; i < filePaths.length; i++) {
            const filePath = filePaths[i];

            try {
                const content = await fs.promises.readFile(filePath, 'utf-8');
                const language = this.getLanguageFromExtension(path.extname(filePath));
                const chunks = await splitter.split(content, language, filePath);

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
                        try {
                            await this.processChunkBuffer(chunkBuffer);
                        } catch (error) {
                            const searchType = isHybrid === true ? 'hybrid' : 'regular';
                            console.error(`[Context] ❌ Failed to process chunk batch for ${searchType}:`, error);
                            if (error instanceof Error) {
                                console.error('[Context] Stack trace:', error.stack);
                            }
                        } finally {
                            chunkBuffer = []; // Always clear buffer, even on failure
                        }
                    }

                    // Check if chunk limit is reached
                    if (totalChunks >= CHUNK_LIMIT) {
                        console.warn(`[Context] ⚠️  Chunk limit of ${CHUNK_LIMIT} reached. Stopping indexing.`);
                        limitReached = true;
                        break; // Exit the inner loop (over chunks)
                    }
                }

                processedFiles++;
                onFileProcessed?.(filePath, i + 1, filePaths.length);

                if (limitReached) {
                    break; // Exit the outer loop (over files)
                }

            } catch (error) {
                console.warn(`[Context] ⚠️  Skipping file ${filePath}: ${error}`);
            }
        }

        // Process any remaining chunks in the buffer
        if (chunkBuffer.length > 0) {
            const searchType = isHybrid === true ? 'hybrid' : 'regular';
            console.log(`📝 Processing final batch of ${chunkBuffer.length} chunks for ${searchType}`);
            try {
                await this.processChunkBuffer(chunkBuffer);
            } catch (error) {
                console.error(`[Context] ❌ Failed to process final chunk batch for ${searchType}:`, error);
                if (error instanceof Error) {
                    console.error('[Context] Stack trace:', error.stack);
                }
            }
        }

        return {
            processedFiles,
            totalChunks,
            status: limitReached ? 'limit_reached' : 'completed'
        };
    }

    /**
 * Process accumulated chunk buffer
 */
    private async processChunkBuffer(chunkBuffer: Array<{ chunk: CodeChunk; codebasePath: string }>): Promise<void> {
        if (chunkBuffer.length === 0) return;

        // Extract chunks and ensure they all have the same codebasePath
        const chunks = chunkBuffer.map(item => item.chunk);
        const codebasePath = chunkBuffer[0].codebasePath;

        // Estimate tokens (rough estimation: 1 token ≈ 4 characters)
        const estimatedTokens = chunks.reduce((sum, chunk) => sum + Math.ceil(chunk.content.length / 4), 0);

        const isHybrid = this.getIsHybrid();
        const searchType = isHybrid === true ? 'hybrid' : 'regular';
        console.log(`[Context] 🔄 Processing batch of ${chunks.length} chunks (~${estimatedTokens} tokens) for ${searchType}`);
        await this.processChunkBatch(chunks, codebasePath);
    }

    /**
     * Process a batch of chunks
     */
    private async processChunkBatch(chunks: CodeChunk[], codebasePath: string): Promise<void> {
        const isHybrid = this.getIsHybrid();

        // Phase 3: harvest symbol names into the vocabulary collector.
        if (this.indexedSymbols) {
            for (const chunk of chunks) {
                const sym = (chunk.metadata as any).symbol_name;
                const parent = (chunk.metadata as any).parent_symbol;
                if (typeof sym === 'string' && sym.length > 0) this.indexedSymbols.add(sym);
                if (typeof parent === 'string' && parent.length > 0) this.indexedSymbols.add(parent);
            }
        }

        // Generate embedding vectors
        const chunkContents = chunks.map(chunk => chunk.content);
        const embeddings = await this.embedding.embedBatch(chunkContents);

        if (isHybrid === true) {
            // Create hybrid vector documents
            const documents: VectorDocument[] = chunks.map((chunk, index) => {
                if (!chunk.metadata.filePath) {
                    throw new Error(`Missing filePath in chunk metadata at index ${index}`);
                }

                const relativePath = path.relative(codebasePath, chunk.metadata.filePath);
                const fileExtension = path.extname(chunk.metadata.filePath);
                const {
                    filePath, startLine, endLine,
                    content_type, symbol_kind, symbol_name, parent_symbol, heading_path,
                    ...restMetadata
                } = chunk.metadata;

                return {
                    id: this.generateId(relativePath, chunk.metadata.startLine || 0, chunk.metadata.endLine || 0, chunk.content),
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
                    // Phase 4: BGE-M3 learned sparse from the sparse sidecar.
                    // Falsy when the embedding provider doesn't expose sparse;
                    // insertHybrid only attaches it when present and non-empty.
                    sparse_learned: embeddings[index].sparse,
                };
            });

            // Store to vector database
            await this.vectorDatabase.insertHybrid(this.getCollectionName(codebasePath), documents);
        } else {
            // Create regular vector documents
            const documents: VectorDocument[] = chunks.map((chunk, index) => {
                if (!chunk.metadata.filePath) {
                    throw new Error(`Missing filePath in chunk metadata at index ${index}`);
                }

                const relativePath = path.relative(codebasePath, chunk.metadata.filePath);
                const fileExtension = path.extname(chunk.metadata.filePath);
                const {
                    filePath, startLine, endLine,
                    content_type, symbol_kind, symbol_name, parent_symbol, heading_path,
                    ...restMetadata
                } = chunk.metadata;

                return {
                    id: this.generateId(relativePath, chunk.metadata.startLine || 0, chunk.metadata.endLine || 0, chunk.content),
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
                };
            });

            // Store to vector database
            await this.vectorDatabase.insert(this.getCollectionName(codebasePath), documents);
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
            '.hxml': 'haxe',
            '.ipynb': 'jupyter',
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
    private generateId(relativePath: string, startLine: number, endLine: number, content: string): string {
        const combinedString = `${relativePath}:${startLine}:${endLine}:${content}`;
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
            let fileBasedPatterns: string[] = [];

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
            const homeDir = require('os').homedir();
            const globalIgnorePath = path.join(homeDir, '.context', '.contextignore');
            return await this.loadIgnoreFile(globalIgnorePath, 'global .contextignore');
        } catch (error) {
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
        } catch (error) {
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
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
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
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
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
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
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
