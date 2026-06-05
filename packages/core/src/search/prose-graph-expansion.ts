// prose-graph-deterministic §4: runtime 1-hop expansion over the prose-graph
// side-index. Mirrors `graph-expansion.ts`, but the index is the
// `.prose-graph.json` adjacency (deterministic narrative edges over the prose
// collection) rather than `.symbols-graph.json` symbol edges.
//
// `ProseGraphIndex.load` reads the side-file written by the builder
// (`prose-graph-builder.ts`). `collectProseGraphCandidateIds` walks the seeds'
// 1-hop neighbours through the adjacency; production callers batch-fetch those
// chunk_ids from the prose collection and feed them to the outer weighted RRF
// as `prose_graph_pool`. At `PROSE_GRAPH_EXPAND=false` the module is never
// loaded — pipeline is identical pre-change.

import * as fs from 'fs';
import { SemanticSearchResult } from '../types';
import {
    ProseGraphEdge,
    ProseGraphPayload,
    ProseGraphStats,
    PROSE_GRAPH_VERSION,
} from './prose-graph-builder';

/** Schema versions accepted at load time. Anything else → graceful off. */
export const RECOGNIZED_PROSE_GRAPH_VERSIONS: ReadonlySet<string> = new Set([PROSE_GRAPH_VERSION]);

export class ProseGraphIndex {
    private readonly adjacency: Map<string, ProseGraphEdge[]>;
    public readonly version: string;
    public readonly nodeCount: number;
    public readonly edgeCount: number;
    public readonly stats: ProseGraphStats | null;

    private constructor(
        adjacency: Map<string, ProseGraphEdge[]>,
        version: string,
        stats: ProseGraphStats | null,
    ) {
        this.adjacency = adjacency;
        this.version = version;
        this.nodeCount = adjacency.size;
        this.stats = stats;
        let edges = 0;
        for (const list of adjacency.values()) edges += list.length;
        // Stored on both endpoints → halve for an undirected edge count.
        this.edgeCount = Math.round(edges / 2);
    }

    static load(filePath: string): ProseGraphIndex | null {
        try {
            const raw = fs.readFileSync(filePath, 'utf-8');
            const parsed = JSON.parse(raw) as ProseGraphPayload;
            if (!parsed || typeof parsed !== 'object') {
                console.warn(`[ProseGraphIndex] ⚠️ ${filePath}: payload is not an object`);
                return null;
            }
            const version = typeof parsed.version === 'string' ? parsed.version : 'unknown';
            if (!RECOGNIZED_PROSE_GRAPH_VERSIONS.has(version)) {
                console.warn(`[ProseGraphIndex] ⚠️ prose-graph schema version ${version} not recognized; expected one of [${Array.from(RECOGNIZED_PROSE_GRAPH_VERSIONS).join(', ')}]; prose-graph expansion disabled`);
                return null;
            }
            const adjacency = new Map<string, ProseGraphEdge[]>();
            const dict = parsed.adjacency || {};
            for (const [id, edges] of Object.entries(dict)) {
                if (!id || !Array.isArray(edges)) continue;
                const clean: ProseGraphEdge[] = [];
                for (const e of edges) {
                    if (!e || typeof e !== 'object') continue;
                    const to = (e as ProseGraphEdge).to;
                    const type = (e as ProseGraphEdge).type;
                    if (typeof to !== 'string' || to.length === 0) continue;
                    const weight = Number.isFinite((e as ProseGraphEdge).weight)
                        ? (e as ProseGraphEdge).weight : 1;
                    clean.push({ to, type, weight });
                }
                if (clean.length > 0) adjacency.set(id, clean);
            }
            const stats = parsed.stats && typeof parsed.stats === 'object' ? parsed.stats : null;
            return new ProseGraphIndex(adjacency, version, stats);
        } catch (err) {
            console.warn(`[ProseGraphIndex] ⚠️ failed to load ${filePath}: ${err}`);
            return null;
        }
    }

    /** 1-hop neighbours of `chunkId` across all edge types. */
    neighbours(chunkId: string): ProseGraphEdge[] {
        if (!chunkId) return [];
        return this.adjacency.get(chunkId) || [];
    }

    has(chunkId: string): boolean {
        return this.adjacency.has(chunkId);
    }
}

export type ProseChunkFetcher = (chunkId: string) => Promise<SemanticSearchResult | null>;

const DEFAULT_PROSE_GRAPH_POOL_CAP = 50;

/**
 * Pure-data step: walk the seeds 1-hop through the prose-graph, collecting
 * unique neighbour chunk_ids (any edge type), dropping ids already present in
 * the seeds. Higher-weight neighbours come first (co-mention weight ∝ shared
 * symbols), so the cap keeps the strongest connections. Production callers
 * batch-fetch the returned ids from the prose collection before the outer RRF.
 */
export function collectProseGraphCandidateIds(
    seeds: SemanticSearchResult[],
    index: ProseGraphIndex,
    cap: number = DEFAULT_PROSE_GRAPH_POOL_CAP,
): string[] {
    if (!seeds || seeds.length === 0 || !index) return [];

    const seedIds = new Set<string>();
    for (const s of seeds) {
        if (s.chunk_id) seedIds.add(s.chunk_id);
    }

    // Aggregate neighbour weight across all seeds so a chunk reachable from
    // several seeds (or by several edge types) ranks higher.
    const weightById = new Map<string, number>();
    for (const seed of seeds) {
        if (!seed || !seed.chunk_id) continue;
        for (const edge of index.neighbours(seed.chunk_id)) {
            if (!edge.to || seedIds.has(edge.to)) continue;
            weightById.set(edge.to, (weightById.get(edge.to) || 0) + (edge.weight || 1));
        }
    }
    if (weightById.size === 0) return [];

    const ordered = Array.from(weightById.entries())
        .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
        .map(([id]) => id);
    return ordered.slice(0, cap * 2);
}

/**
 * 1-hop prose-graph expansion of `seeds`. Collects neighbour candidate
 * chunk_ids, fetches them via `fetcher`, dedups, and caps at `cap` (aligned
 * with RERANKER_INPUT_K). Missing chunks are skipped silently (the wiring
 * layer logs a throttled warning).
 */
export async function expandProseGraphPool(
    seeds: SemanticSearchResult[],
    index: ProseGraphIndex,
    fetcher: ProseChunkFetcher,
    cap: number = DEFAULT_PROSE_GRAPH_POOL_CAP,
): Promise<SemanticSearchResult[]> {
    const candidateIds = collectProseGraphCandidateIds(seeds, index, cap);
    if (candidateIds.length === 0) return [];

    const limit = Math.min(candidateIds.length, cap * 2);
    const fetched: SemanticSearchResult[] = [];
    const seen = new Set<string>();
    for (let i = 0; i < limit && fetched.length < cap; i++) {
        const id = candidateIds[i];
        if (seen.has(id)) continue;
        seen.add(id);
        try {
            const r = await fetcher(id);
            if (r && r.chunk_id) fetched.push(r);
        } catch {
            // Missing chunk → skip (spec "Stale chunk_id пропускается без ошибки").
        }
    }
    return fetched.slice(0, cap);
}
