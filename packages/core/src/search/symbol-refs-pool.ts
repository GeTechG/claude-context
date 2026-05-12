// rag-symbol-refs-lsp-pool: 4th retrieval pool driven by Serena LSP.
//
// Flow (deviates from design D4's `Promise.all(3 calls)` because Serena's
// `find_symbol` requires a `relative_path` to avoid scanning the whole
// project tree, and a project-wide scan crashes Haxe LSP on test sources):
//
//   1. Resolve the declaration chunk via Milvus `symbol_name == "X"`
//      (canonical preferred over demoted vendored copies). This gives us
//      both the declaration chunk_id AND the `relativePath` we need to
//      pass to the next two LSP calls.
//   2. In parallel, call `find_referencing_symbols(name_path, relPath)`
//      and `find_implementations(name_path, relPath)`.
//   3. Map each LSP location (file + line range) to chunk_ids via Milvus
//      filter `relativePath == "<rel>" AND startLine <= range.end.line
//      AND endLine >= range.start.line`. Hydrated chunks are returned as
//      HybridSearchResult[] tagged with positional ranks so the outer
//      weighted-RRF merger can blend them in alongside code/doc/symbolRouting.
//
// Spec: openspec/changes/rag-symbol-refs-lsp-pool/specs/rag-search/spec.md
// Design: openspec/changes/rag-symbol-refs-lsp-pool/design.md (D4 / D6).
// Implementation note (Haxe LSP): see `infra/eval-summary.md` preflight
// section — `find_symbol` without `relative_path` triggers exhaustive
// project scans that fail on the Haxe test corpus, so we route through
// Milvus's `symbol_name` index instead.

import * as path from 'path';
import { ParsedQName } from './query-classifier';
import { Location, SerenaLspClient } from './serena-lsp-client';
import {
    HybridSearchResult,
    VectorDatabase,
    VectorDocument,
} from '../vectordb/types';

export interface SingleSymbolParsed {
    symbolName: string;
}

export type SymbolRefsParsed = ParsedQName | SingleSymbolParsed;

export interface SymbolRefsPoolOptions {
    /** Original query, kept for diagnostics only. */
    query: string;
    /** Either a structured qualified-name parse or a single-symbol parse. */
    parsed: SymbolRefsParsed;
    /** Connected LSP client (caller manages lifecycle). */
    lspClient: SerenaLspClient;
    /** Milvus connector. */
    vectorDatabase: VectorDatabase;
    /** Target Milvus collection. */
    collection: string;
    /** Indexed codebase root — used to relativize Serena paths if the daemon ever returns absolutes. */
    codebasePath: string;
    /** `SYMBOL_REFS_MAX_REFERENCES`. */
    maxRefs: number;
    /** `SYMBOL_REFS_MAX_IMPLEMENTATIONS`. */
    maxImpls: number;
    /** RRF k-smoothing constant for the per-pool score (mirrors weightedRrfMerge). */
    rrfK?: number;
    /**
     * rag-symbol-refs-multi-hop: maximum LSP expansion depth.
     *  - `1` (default): only declaration + direct refs/impls (hop-1 baseline).
     *  - `2`: after hop-1, run `findReferencingSymbols` on top hop-1 chunks
     *         (refs-of-refs) with fan-out caps `maxHop1Seeds` × `maxHop2Refs`.
     * Values outside `{1, 2}` are clamped by the caller.
     */
    maxHops?: number;
    /**
     * rag-symbol-refs-multi-hop: cap on hop-1 chunks used as seeds for hop-2.
     * Applies only when `maxHops >= 2`. Default 3.
     */
    maxHop1Seeds?: number;
    /**
     * rag-symbol-refs-multi-hop: cap on references per hop-1 seed on hop-2.
     * Applies only when `maxHops >= 2`. Default 3.
     */
    maxHop2Refs?: number;
}

const DEFAULT_RRF_K = 60;
const TOTAL_CHUNK_CAP = 30;
const MAX_DECL_FOR_FANOUT = 1;
const PER_LOCATION_CHUNK_LIMIT = 5;
// rag-symbol-refs-multi-hop: stop-word guard, mirrored from parseSingleSymbol.
// hop-1 chunks whose `symbol_name` is shorter are not promoted to hop-2 seeds.
const HOP2_SEED_MIN_SYMBOL_LEN = 4;

const HYBRID_OUTPUT_FIELDS = [
    'id', 'content', 'relativePath', 'startLine', 'endLine', 'fileExtension', 'metadata',
    'content_type', 'symbol_kind', 'symbol_name', 'parent_symbol', 'heading_path',
    'imports', 'extends', 'implements', 'mentioned_symbols',
];

function escapeMilvusLiteral(s: string): string {
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function namePathFor(parsed: SymbolRefsParsed): string {
    if ('className' in parsed && 'methodName' in parsed) {
        // Serena's name_path syntax accepts `Class/method` for nested lookups.
        return `${parsed.className}/${parsed.methodName}`;
    }
    return parsed.symbolName;
}

function topLevelSymbolName(parsed: SymbolRefsParsed): string {
    if ('className' in parsed) return parsed.className;
    return parsed.symbolName;
}

function toRelative(codebasePath: string, filePath: string): string {
    // Serena returns `relative_path` strings rooted at the project — keep
    // them as-is. Defensive guard for any legacy absolute path the daemon
    // might emit.
    if (path.isAbsolute(filePath)) {
        return path.relative(codebasePath, filePath);
    }
    return filePath;
}

interface OrderedDocument {
    rank: number;
    document: VectorDocument;
}

export async function runSymbolRefsPool(opts: SymbolRefsPoolOptions): Promise<HybridSearchResult[]> {
    const rrfK = opts.rrfK && opts.rrfK > 0 ? opts.rrfK : DEFAULT_RRF_K;
    const maxHops = opts.maxHops && opts.maxHops >= 2 ? 2 : 1;
    const maxHop1Seeds = opts.maxHop1Seeds && opts.maxHop1Seeds > 0 ? opts.maxHop1Seeds : 3;
    const maxHop2Refs = opts.maxHop2Refs && opts.maxHop2Refs > 0 ? opts.maxHop2Refs : 3;

    const lspName = topLevelSymbolName(opts.parsed);
    const callPath = namePathFor(opts.parsed);

    // Step 1: declaration chunk via Milvus (`symbol_name == "X"`). This
    // both seeds the pool with the canonical chunk and gives us the
    // relative_path Serena needs for refs/impls.
    const declChunks = await fetchDeclarationChunks(
        opts.vectorDatabase,
        opts.collection,
        lspName,
        MAX_DECL_FOR_FANOUT,
    );

    let refs: Location[] = [];
    let impls: Location[] = [];
    if (declChunks.length > 0) {
        const declRel = declChunks[0].relativePath;
        const [refsResult, implsResult] = await Promise.allSettled([
            opts.lspClient.findReferencingSymbols(callPath, declRel, opts.maxRefs),
            opts.lspClient.findImplementations(callPath, declRel, opts.maxImpls),
        ]);
        if (refsResult.status === 'fulfilled') refs = refsResult.value;
        if (implsResult.status === 'fulfilled') impls = implsResult.value;
    }

    if (declChunks.length === 0 && refs.length === 0 && impls.length === 0) {
        console.log(`[Context] 🔍 symbol-refs pool: symbol="${lspName}" → decl=0, refs=0, impls=0, hop2=0 → 0 chunks`);
        return [];
    }

    // Rank: decl-chunks first (already deduped + canonical), then ref
    // locations, then impl locations, then hop-2 refs-of-refs. Dedupe by
    // chunk_id across all hops via `seenIds`.
    const orderedChunkIds: string[] = [];
    const seenIds = new Set<string>();
    const declIdCount = declChunks.length;
    for (const d of declChunks) {
        if (!seenIds.has(d.id)) {
            seenIds.add(d.id);
            orderedChunkIds.push(d.id);
        }
    }

    const hop1Locations: Location[] = [...refs, ...impls];
    const hop1NewIds: string[] = [];
    for (const loc of hop1Locations) {
        if (orderedChunkIds.length >= TOTAL_CHUNK_CAP) break;
        const chunkIds = await locationToChunkIds(
            loc,
            opts.vectorDatabase,
            opts.collection,
            opts.codebasePath,
        );
        for (const id of chunkIds) {
            if (!seenIds.has(id)) {
                seenIds.add(id);
                orderedChunkIds.push(id);
                hop1NewIds.push(id);
                if (orderedChunkIds.length >= TOTAL_CHUNK_CAP) break;
            }
        }
    }

    // rag-symbol-refs-multi-hop: hop-2 expansion. Seeds are top hop-1 chunks
    // (in pool-rank order) whose `symbol_name` passes the stop-word guard.
    // Hop-2 fail/timeout degrades gracefully — pool still returns hop-0 + hop-1.
    let hop2CountAdded = 0;
    if (maxHops >= 2 && hop1NewIds.length > 0 && orderedChunkIds.length < TOTAL_CHUNK_CAP) {
        // Hydrate hop-1 docs (id + symbol_name + relativePath) to pick eligible seeds.
        // We hydrate the full ordered list once at the end anyway; here we issue a
        // cheap projection query just for the hop-1 ids we just added.
        const seedDocs = await fetchSeedMetadata(opts.vectorDatabase, opts.collection, hop1NewIds);
        const eligibleSeeds = seedDocs
            .filter((d) => typeof d.symbolName === 'string' && d.symbolName.length >= HOP2_SEED_MIN_SYMBOL_LEN
                && typeof d.relativePath === 'string' && d.relativePath.length > 0)
            .slice(0, maxHop1Seeds);

        if (eligibleSeeds.length > 0) {
            const hop2Results = await Promise.allSettled(
                eligibleSeeds.map((seed) =>
                    opts.lspClient.findReferencingSymbols(seed.symbolName, seed.relativePath, maxHop2Refs),
                ),
            );
            const hop2Locations: Location[] = [];
            for (let i = 0; i < hop2Results.length; i++) {
                const r = hop2Results[i];
                if (r.status === 'fulfilled') {
                    hop2Locations.push(...r.value);
                } else {
                    console.warn(`[Context] ⚠️ symbol-refs hop-2 failed for ${eligibleSeeds[i].symbolName}: ${r.reason}`);
                }
            }

            for (const loc of hop2Locations) {
                if (orderedChunkIds.length >= TOTAL_CHUNK_CAP) break;
                const chunkIds = await locationToChunkIds(
                    loc,
                    opts.vectorDatabase,
                    opts.collection,
                    opts.codebasePath,
                );
                for (const id of chunkIds) {
                    if (!seenIds.has(id)) {
                        seenIds.add(id);
                        orderedChunkIds.push(id);
                        hop2CountAdded++;
                        if (orderedChunkIds.length >= TOTAL_CHUNK_CAP) break;
                    }
                }
            }
        }
    }

    if (orderedChunkIds.length === 0) {
        console.log(`[Context] 🔍 symbol-refs pool: symbol="${lspName}" → decl=${declIdCount}, refs=${refs.length}, impls=${impls.length}, hop2=0 → 0 chunks (no Milvus matches)`);
        return [];
    }

    const documents = await fetchOrderedDocuments(opts.vectorDatabase, opts.collection, orderedChunkIds);
    const results: HybridSearchResult[] = documents.map((d) => ({
        document: d.document,
        score: 1 / (rrfK + d.rank),
    }));

    const hop2LogPart = maxHops >= 2 ? `, hop2=${hop2CountAdded}` : '';
    console.log(`[Context] 🔍 symbol-refs pool: symbol="${lspName}" → decl=${declIdCount}, refs=${refs.length}, impls=${impls.length}${hop2LogPart} → ${results.length} chunks`);
    return results;
}

interface SeedMetadata {
    id: string;
    symbolName: string;
    relativePath: string;
}

/**
 * rag-symbol-refs-multi-hop: project-only Milvus query to pull
 * (id, symbol_name, relativePath) for hop-1 chunks so the caller can
 * pick eligible hop-2 seeds without paying the full hydration cost yet.
 * Falls back to empty on Milvus errors so hop-2 simply no-ops.
 */
async function fetchSeedMetadata(
    vectorDatabase: VectorDatabase,
    collection: string,
    ids: string[],
): Promise<SeedMetadata[]> {
    const safe = ids.filter((id) => /^[A-Za-z0-9_-]+$/.test(id));
    if (safe.length === 0) return [];
    const filter = `id in [${safe.map((id) => `"${id}"`).join(',')}]`;
    let rows: Record<string, any>[];
    try {
        rows = await vectorDatabase.query(collection, filter, ['id', 'symbol_name', 'relativePath'], safe.length);
    } catch (err) {
        console.warn(`[Context] ⚠️ symbol-refs hop-2 seed-metadata query failed: ${err}`);
        return [];
    }
    // Preserve caller-side ordering so hop-2 seeds the highest-ranked hop-1 chunks first.
    const byId = new Map<string, SeedMetadata>();
    for (const row of rows) {
        const id = row?.id;
        if (typeof id !== 'string') continue;
        byId.set(id, {
            id,
            symbolName: typeof row.symbol_name === 'string' ? row.symbol_name : '',
            relativePath: typeof row.relativePath === 'string' ? row.relativePath : '',
        });
    }
    const ordered: SeedMetadata[] = [];
    for (const id of ids) {
        const m = byId.get(id);
        if (m) ordered.push(m);
    }
    return ordered;
}

interface DeclarationChunk {
    id: string;
    relativePath: string;
}

async function fetchDeclarationChunks(
    vectorDatabase: VectorDatabase,
    collection: string,
    symbolName: string,
    maxResults: number,
): Promise<DeclarationChunk[]> {
    const safeName = escapeMilvusLiteral(symbolName);
    // Restrict to code/docstring chunks — the LSP works on source files,
    // not docs. Limit to a small number; we only need the canonical
    // file path to seed refs/impls.
    const filter = `symbol_name == "${safeName}" and content_type in ["code","docstring"]`;
    let rows: Record<string, any>[];
    try {
        rows = await vectorDatabase.query(collection, filter, ['id', 'relativePath'], maxResults * 4);
    } catch (err) {
        console.warn(`[Context] ⚠️ symbol-refs declaration lookup failed: ${err}`);
        return [];
    }
    const out: DeclarationChunk[] = [];
    const seenPaths = new Set<string>();
    for (const row of rows) {
        const id = row?.id;
        const rel = row?.relativePath;
        if (typeof id !== 'string' || typeof rel !== 'string' || rel.length === 0) continue;
        if (seenPaths.has(rel)) continue;
        seenPaths.add(rel);
        out.push({ id, relativePath: rel });
        if (out.length >= maxResults) break;
    }
    return out;
}

async function locationToChunkIds(
    loc: Location,
    vectorDatabase: VectorDatabase,
    collection: string,
    codebasePath: string,
): Promise<string[]> {
    const rel = toRelative(codebasePath, loc.filePath);
    if (!rel) return [];
    const startLine = Math.max(0, loc.range.start.line);
    const endLine = Math.max(startLine, loc.range.end.line);
    const filter = `relativePath == "${escapeMilvusLiteral(rel)}" AND startLine <= ${endLine} AND endLine >= ${startLine}`;
    let rows: Record<string, any>[];
    try {
        rows = await vectorDatabase.query(collection, filter, ['id'], PER_LOCATION_CHUNK_LIMIT);
    } catch (err) {
        console.warn(`[Context] ⚠️ symbol-refs Milvus query failed for ${rel}: ${err}`);
        return [];
    }
    const ids: string[] = [];
    for (const row of rows) {
        const id = row?.id;
        if (typeof id === 'string' && id.length > 0) ids.push(id);
    }
    return ids;
}

async function fetchOrderedDocuments(
    vectorDatabase: VectorDatabase,
    collection: string,
    orderedIds: string[],
): Promise<OrderedDocument[]> {
    const safe = orderedIds.filter((id) => /^[A-Za-z0-9_-]+$/.test(id));
    if (safe.length === 0) return [];
    const filter = `id in [${safe.map((id) => `"${id}"`).join(',')}]`;
    let rows: Record<string, any>[];
    try {
        rows = await vectorDatabase.query(collection, filter, HYBRID_OUTPUT_FIELDS, safe.length);
    } catch (err) {
        console.warn(`[Context] ⚠️ symbol-refs hydration query failed: ${err}`);
        return [];
    }
    const byId = new Map<string, Record<string, any>>();
    for (const row of rows) {
        const id = row?.id;
        if (typeof id === 'string') byId.set(id, row);
    }
    const out: OrderedDocument[] = [];
    for (let i = 0; i < orderedIds.length; i++) {
        const row = byId.get(orderedIds[i]);
        if (!row) continue;
        let metadata: Record<string, any> = {};
        try {
            metadata = JSON.parse(row.metadata || '{}');
        } catch {
            /* malformed json — leave empty */
        }
        out.push({
            rank: i + 1,
            document: {
                id: orderedIds[i],
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
        });
    }
    return out;
}
