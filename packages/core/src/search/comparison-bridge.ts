// rag-graph-comparison-bridge: 5th retrieval pool that surfaces
// cross-subject partner chunks for comparison-shape queries.
//
// Activation gates (checked by the caller in context.ts):
//   - intent_classifier returns query_shape === 'comparison'
//   - COMPARISON_BRIDGE_ENABLED === '1'
//   - .symbols-graph.json loaded with `version === 'v3-2'`
//
// For each canonical code seed in the primary RRF top-K, the bridge
//   1. derives a package key from the seed itself (parent_symbol if set,
//      otherwise the directory chain of the seed's relativePath);
//   2. consults `by_package[pkg]` for sibling symbol_names — siblings
//      are dropped when the bucket exceeds `maxPackageFanout`;
//   3. consults `by_supertype[<sup>]` for each declared supertype on
//      the seed (`extends` + `implements[]`);
//   4. resolves partner symbol_names back to canonical chunk_ids via
//      `by_symbol[<sym>].canonical_chunk_ids[0]`;
//   5. de-duplicates, drops chunks already in the primary top-K, and
//      caps the pool at `cap` (default 50, aligned with RERANKER_INPUT_K).
//
// The bridge is pure data: it returns chunk_ids only. The caller
// fetches the chunks via the existing Milvus batch path and merges them
// into the outer weighted RRF as a 5th pool with weight
// COMPARISON_BRIDGE_POOL_WEIGHT.

import { SemanticSearchResult } from '../types';
import { GraphIndex, derivePackageFromPath } from './graph-expansion';

export interface ComparisonBridgeOptions {
    maxPartners: number;
    maxPackageFanout: number;
    cap?: number;
    debug?: boolean;
    queryId?: string;
}

export interface ComparisonBridgeResult {
    chunkIds: string[];
    seedsCount: number;
    packageHits: number;
    supertypeHits: number;
}

const DEFAULT_BRIDGE_POOL_CAP = 50;

/**
 * Build the bridge pool (pure data: chunk_id list). Caller fetches chunks.
 *
 * Skipping rules (all silent — return empty pool, never throw):
 *   - if `graph.supportsComparisonBridge()` is false → []
 *   - per-seed: docstring content_type → skip (per spec scenario
 *     "docstring-чанки исключены из seeds")
 *   - per-seed: missing symbol_name or content_type !== 'code' → skip
 *   - per-package: bucket size > maxPackageFanout → skip package lookup
 *     for that seed; supertype lookup still proceeds
 *   - per-partner: canonical_chunk_ids empty (e.g. mentions-only entry) → skip
 *   - chunk_id already present in primaryTopK → skip (no redundant injection)
 */
export function buildComparisonBridgePool(
    primaryTopK: SemanticSearchResult[],
    graph: GraphIndex,
    opts: ComparisonBridgeOptions,
): ComparisonBridgeResult {
    const empty: ComparisonBridgeResult = { chunkIds: [], seedsCount: 0, packageHits: 0, supertypeHits: 0 };
    if (!graph || !graph.supportsComparisonBridge()) return empty;
    if (!primaryTopK || primaryTopK.length === 0) return empty;

    const cap = opts.cap ?? DEFAULT_BRIDGE_POOL_CAP;
    const maxPartners = Math.max(0, opts.maxPartners | 0);
    const maxPackageFanout = Math.max(0, opts.maxPackageFanout | 0);

    const seedIds = new Set<string>();
    for (const s of primaryTopK) {
        if (s.chunk_id) seedIds.add(s.chunk_id);
    }

    const collected: string[] = [];
    const seen = new Set<string>();
    let seedsCount = 0;
    let packageHits = 0;
    let supertypeHits = 0;

    const pushPartner = (partnerSymbol: string, seedSymbol: string): boolean => {
        if (!partnerSymbol || partnerSymbol === seedSymbol) return false;
        const entry = graph.lookupSymbol(partnerSymbol);
        if (!entry || !entry.canonical_chunk_ids || entry.canonical_chunk_ids.length === 0) return false;
        const chunkId = entry.canonical_chunk_ids[0];
        if (!chunkId || seedIds.has(chunkId) || seen.has(chunkId)) return false;
        seen.add(chunkId);
        collected.push(chunkId);
        return true;
    };

    for (const seed of primaryTopK) {
        if (collected.length >= cap) break;
        if (!seed) continue;
        if (seed.content_type !== 'code') continue;
        if (!seed.symbol_name) continue;
        seedsCount++;

        const pkg = seed.parent_symbol || derivePackageFromPath(seed.relativePath);
        if (pkg && maxPartners > 0) {
            const bucket = graph.lookupPackage(pkg);
            if (bucket.length > 0 && bucket.length <= maxPackageFanout) {
                let added = 0;
                for (const partner of bucket) {
                    if (added >= maxPartners) break;
                    if (collected.length >= cap) break;
                    if (pushPartner(partner, seed.symbol_name)) {
                        added++;
                        packageHits++;
                    }
                }
            }
        }

        if (maxPartners > 0) {
            const supertypes: string[] = [];
            if (seed.extends) supertypes.push(seed.extends);
            if (seed.implements) {
                for (const s of seed.implements) {
                    if (s) supertypes.push(s);
                }
            }
            for (const sup of supertypes) {
                const bucket = graph.lookupSupertype(sup);
                if (bucket.length === 0) continue;
                let added = 0;
                for (const partner of bucket) {
                    if (added >= maxPartners) break;
                    if (collected.length >= cap) break;
                    if (pushPartner(partner, seed.symbol_name)) {
                        added++;
                        supertypeHits++;
                    }
                }
            }
        }
    }

    if (opts.debug) {
        const qid = opts.queryId ? ` q=${opts.queryId}` : '';
        console.log(`[comparison-bridge]${qid} seeds=${seedsCount} pkg_hits=${packageHits} sup_hits=${supertypeHits} pool_size=${collected.length}`);
    }

    return {
        chunkIds: collected.slice(0, cap),
        seedsCount,
        packageHits,
        supertypeHits,
    };
}
