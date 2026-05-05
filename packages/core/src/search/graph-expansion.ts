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

interface GraphSymbolEntry {
    canonical_chunk_ids?: string[];
    mentioned_by_chunk_ids?: string[];
}

interface GraphPayload {
    version?: string;
    by_symbol?: Record<string, GraphSymbolEntry>;
}

export class GraphIndex {
    private readonly bySymbol: Map<string, GraphSymbolEntry>;
    public readonly version: string;
    public readonly symbolCount: number;

    private constructor(bySymbol: Map<string, GraphSymbolEntry>, version: string) {
        this.bySymbol = bySymbol;
        this.version = version;
        this.symbolCount = bySymbol.size;
    }

    static load(filePath: string): GraphIndex | null {
        try {
            const raw = fs.readFileSync(filePath, 'utf-8');
            const parsed = JSON.parse(raw) as GraphPayload;
            if (!parsed || typeof parsed !== 'object') {
                console.warn(`[GraphIndex] ⚠️ ${filePath}: payload is not an object`);
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
            const version = typeof parsed.version === 'string' ? parsed.version : 'unknown';
            return new GraphIndex(map, version);
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
