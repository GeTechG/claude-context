// rag-graph-layer Phase 3.1 / 3.2: cross-domain graph expansion module.
// `GraphIndex` loads the `.symbols-graph.json` side-file written by
// indexCodebase (Phase 2). `expandGraphPool` takes the top-K of the
// primary RRF and pulls 1-hop neighbours through the index — forward edges
// from code (parent_symbol, imports, extends, implements), forward edges
// from doc (mentioned_symbols), reverse edges from code
// (mentioned-by). Output is a deduped, capped neighbour pool ready for
// the outer weighted RRF (pool weight 0.6 per design D6).

import * as fs from 'fs';
import { SemanticSearchResult } from '../types';

// rag-graph-comparison-bridge: derive a package label from a repo-relative
// file path. Universal/language-agnostic: package = directory chain of the
// file, joined with `.`. Files at the repo root (no enclosing directory)
// return undefined. This works for any source layout that uses the
// filesystem to group related symbols (Haxe stdlib, Python modules, Java
// packages, npm modules, …). For multi-target stdlibs that store the same
// symbol under several directories (Haxe `eval/_std/...`), the indexer
// picks a single canonical chunk per symbol via shortest-path dedup and
// only that canonical's package is bucketed — so target overrides don't
// fragment the bucket map.
export function derivePackageFromPath(relativePath: string | undefined): string | undefined {
    if (!relativePath) return undefined;
    const p = relativePath.replace(/\\/g, '/');
    const lastSlash = p.lastIndexOf('/');
    if (lastSlash <= 0) return undefined;
    const dir = p.slice(0, lastSlash);
    if (!dir) return undefined;
    return dir.replace(/\//g, '.');
}

export interface GraphSymbolEntry {
    canonical_chunk_ids?: string[];
    mentioned_by_chunk_ids?: string[];
}

interface GraphPayload {
    version?: string;
    by_symbol?: Record<string, GraphSymbolEntry>;
    // rag-graph-comparison-bridge v3-2: optional cross-subject indexes.
    by_package?: Record<string, string[]>;
    by_supertype?: Record<string, string[]>;
}

// rag-graph-comparison-bridge: schema versions for which advanced graph
// pools (graph-expansion + comparison-bridge) are permitted to activate.
// Anything outside this set triggers graceful degradation per spec.
export const RECOGNIZED_GRAPH_VERSIONS: ReadonlySet<string> = new Set(['v3-1', 'v3-2']);

export class GraphIndex {
    private readonly bySymbol: Map<string, GraphSymbolEntry>;
    private readonly byPackage: Map<string, string[]>;
    private readonly bySupertype: Map<string, string[]>;
    public readonly version: string;
    public readonly symbolCount: number;
    public readonly packageCount: number;
    public readonly supertypeCount: number;

    private constructor(
        bySymbol: Map<string, GraphSymbolEntry>,
        byPackage: Map<string, string[]>,
        bySupertype: Map<string, string[]>,
        version: string,
    ) {
        this.bySymbol = bySymbol;
        this.byPackage = byPackage;
        this.bySupertype = bySupertype;
        this.version = version;
        this.symbolCount = bySymbol.size;
        this.packageCount = byPackage.size;
        this.supertypeCount = bySupertype.size;
    }

    static load(filePath: string): GraphIndex | null {
        try {
            const raw = fs.readFileSync(filePath, 'utf-8');
            const parsed = JSON.parse(raw) as GraphPayload;
            if (!parsed || typeof parsed !== 'object') {
                console.warn(`[GraphIndex] ⚠️ ${filePath}: payload is not an object`);
                return null;
            }
            const version = typeof parsed.version === 'string' ? parsed.version : 'unknown';
            if (!RECOGNIZED_GRAPH_VERSIONS.has(version)) {
                console.warn(`[GraphIndex] ⚠️ graph schema version ${version} not recognized; expected one of [${Array.from(RECOGNIZED_GRAPH_VERSIONS).join(', ')}]; advanced pools disabled`);
                return null;
            }
            const map = new Map<string, GraphSymbolEntry>();
            const dict = parsed.by_symbol || {};
            for (const [sym, entry] of Object.entries(dict)) {
                if (!entry || typeof entry !== 'object') continue;
                const canon = Array.isArray(entry.canonical_chunk_ids)
                    ? entry.canonical_chunk_ids.filter((s): s is string => typeof s === 'string')
                    : [];
                const mentioned = Array.isArray(entry.mentioned_by_chunk_ids)
                    ? entry.mentioned_by_chunk_ids.filter((s): s is string => typeof s === 'string')
                    : [];
                if (canon.length === 0 && mentioned.length === 0) continue;
                map.set(sym, { canonical_chunk_ids: canon, mentioned_by_chunk_ids: mentioned });
            }
            const byPackage = new Map<string, string[]>();
            for (const [pkg, names] of Object.entries(parsed.by_package || {})) {
                if (!Array.isArray(names)) continue;
                const clean = names.filter((s): s is string => typeof s === 'string' && s.length > 0);
                if (clean.length > 0) byPackage.set(pkg, clean);
            }
            const bySupertype = new Map<string, string[]>();
            for (const [sup, names] of Object.entries(parsed.by_supertype || {})) {
                if (!Array.isArray(names)) continue;
                const clean = names.filter((s): s is string => typeof s === 'string' && s.length > 0);
                if (clean.length > 0) bySupertype.set(sup, clean);
            }
            return new GraphIndex(map, byPackage, bySupertype, version);
        } catch (err) {
            console.warn(`[GraphIndex] ⚠️ failed to load ${filePath}: ${err}`);
            return null;
        }
    }

    /** Forward lookup: chunk-ids that are canonical definitions of `symbol`. */
    lookupForward(symbol: string): string[] {
        if (!symbol) return [];
        return this.bySymbol.get(symbol)?.canonical_chunk_ids || [];
    }

    /** Reverse lookup: chunk-ids of doc / code_example chunks that mention `symbol`. */
    lookupReverse(symbol: string): string[] {
        if (!symbol) return [];
        return this.bySymbol.get(symbol)?.mentioned_by_chunk_ids || [];
    }

    /** rag-graph-comparison-bridge: symbol_names in the same package bucket. */
    lookupPackage(pkg: string): string[] {
        if (!pkg) return [];
        return this.byPackage.get(pkg) || [];
    }

    /** rag-graph-comparison-bridge: symbol_names that share a supertype. */
    lookupSupertype(sup: string): string[] {
        if (!sup) return [];
        return this.bySupertype.get(sup) || [];
    }

    /** rag-graph-comparison-bridge: per-symbol metadata (package, supertypes). */
    lookupSymbol(symbol: string): GraphSymbolEntry | undefined {
        if (!symbol) return undefined;
        return this.bySymbol.get(symbol);
    }

    /** True only when this index was loaded from a payload that carries v3-2 (or newer) keys. */
    supportsComparisonBridge(): boolean {
        return this.version === 'v3-2';
    }
}

export type GraphChunkFetcher = (chunkId: string) => Promise<SemanticSearchResult | null>;

const DEFAULT_GRAPH_POOL_CAP = 50;

/**
 * Pure-data step: walk the seeds through the graph, collecting unique
 * candidate chunk_ids (forward + reverse 1-hop edges). Production callers
 * use this list to batch-fetch chunks in one Milvus query before handing
 * them to the outer RRF; tests use it directly to assert traversal.
 */
export function collectGraphCandidateIds(
    seeds: SemanticSearchResult[],
    index: GraphIndex,
    cap: number = DEFAULT_GRAPH_POOL_CAP,
): string[] {
    if (!seeds || seeds.length === 0 || !index) return [];

    const seedIds = new Set<string>();
    for (const s of seeds) {
        if (s.chunk_id) seedIds.add(s.chunk_id);
    }

    const out: string[] = [];
    const seen = new Set<string>();
    const enqueue = (id: string | undefined) => {
        if (!id) return;
        if (seedIds.has(id)) return;
        if (seen.has(id)) return;
        seen.add(id);
        out.push(id);
    };

    for (const seed of seeds) {
        if (!seed) continue;
        const ct = seed.content_type;
        if (ct === 'docstring') continue;

        const isCode = ct === 'code';
        const isDocish = ct === 'doc' || ct === 'code_example';

        if (isCode) {
            if (seed.parent_symbol) {
                for (const id of index.lookupForward(seed.parent_symbol)) enqueue(id);
            }
            if (seed.imports) {
                for (const sym of seed.imports) {
                    for (const id of index.lookupForward(sym)) enqueue(id);
                }
            }
            if (seed.extends) {
                for (const id of index.lookupForward(seed.extends)) enqueue(id);
            }
            if (seed.implements) {
                for (const sym of seed.implements) {
                    for (const id of index.lookupForward(sym)) enqueue(id);
                }
            }
            if (seed.symbol_name) {
                for (const id of index.lookupReverse(seed.symbol_name)) enqueue(id);
            }
        } else if (isDocish) {
            if (seed.mentioned_symbols) {
                for (const sym of seed.mentioned_symbols) {
                    for (const id of index.lookupForward(sym)) enqueue(id);
                }
            }
        }
        if (out.length >= cap * 4) break;
    }
    return out.slice(0, cap * 2);
}

/**
 * rag-graph-layer Phase 3.2: 1-hop graph expansion of `seeds`.
 *
 * - For each code-shaped seed: forward edges through parent_symbol,
 *   imports, extends, implements; reverse edges through symbol_name.
 * - For each doc / code_example seed: forward edges through mentioned_symbols.
 * - docstring-shaped seeds are excluded from edge construction (per spec D4).
 * - Dedup by chunk_id, drop ids already present in seeds.
 * - Cap the result at `cap` (default 50, aligned with RERANKER_INPUT_K).
 */
export async function expandGraphPool(
    seeds: SemanticSearchResult[],
    index: GraphIndex,
    fetcher: GraphChunkFetcher,
    cap: number = DEFAULT_GRAPH_POOL_CAP,
): Promise<SemanticSearchResult[]> {
    const candidateIds = collectGraphCandidateIds(seeds, index, cap);
    if (candidateIds.length === 0) return [];

    const limit = Math.min(candidateIds.length, cap * 2);
    const fetched: SemanticSearchResult[] = [];
    for (let i = 0; i < limit && fetched.length < cap; i++) {
        const id = candidateIds[i];
        try {
            const r = await fetcher(id);
            if (r && r.chunk_id) fetched.push(r);
        } catch {
            // Per spec scenario "Side-файл stale": missing chunks are
            // skipped silently. Caller logs at most once a minute via the
            // wiring layer.
        }
    }
    return fetched.slice(0, cap);
}
