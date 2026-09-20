// find-references-as-a-tool D2: the index-side reference lookups, in one place.
//
// Lifted verbatim out of `symbol-refs-pool.ts` so the retrieval pool and the
// agent-callable `find_references` tool ask the index the SAME question. The
// part that must not be copied twice is the quoting rule: `mentioned_symbols`
// and `implements` are JSON-encoded `string[]`, so a name is matched WITH its
// quotes — `"Bytes"` never matches `"BytesBuffer"`, which a bare substring
// would. That was a measured bug fix, not a style choice, and a second copy is
// how the next one gets fixed in one place only.

import { VectorDatabase } from '../vectordb/types';

/** The relations the index itself records between a symbol and a code row. */
export type ReferenceRelation = 'mentions' | 'extends' | 'implements';

export const REFERENCE_RELATIONS: readonly ReferenceRelation[] = ['mentions', 'extends', 'implements'];

export function escapeMilvusLiteral(s: string): string {
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

/**
 * The scalar filter for one relation. `extends` holds a single name and is
 * compared exactly; the two JSON-array fields are matched through the quoted
 * name.
 */
export function referenceFilter(relation: ReferenceRelation, symbolName: string): string {
    const exact = escapeMilvusLiteral(symbolName);
    const needle = escapeMilvusLiteral(`"${symbolName}"`);
    switch (relation) {
        case 'mentions': return `mentioned_symbols like "%${needle}%"`;
        case 'extends': return `extends == "${exact}"`;
        case 'implements': return `implements like "%${needle}%"`;
    }
}

/**
 * The declaration lookup: code/docstring chunks whose `symbol_name` is this
 * name. The LSP works on source files, not docs, and the tool reports a
 * declaration for the same reason — it is where the symbol is defined.
 */
export function declarationFilter(symbolName: string): string {
    return `symbol_name == "${escapeMilvusLiteral(symbolName)}" and content_type in ["code","docstring"]`;
}

export interface DeclarationChunk {
    id: string;
    relativePath: string;
}

export async function fetchDeclarationChunks(
    vectorDatabase: VectorDatabase,
    collection: string,
    symbolName: string,
    maxResults: number,
): Promise<DeclarationChunk[]> {
    // Limit to a small number; the pool only needs the canonical file path to
    // seed refs/impls.
    let rows: Record<string, any>[];
    try {
        rows = await vectorDatabase.query(collection, declarationFilter(symbolName), ['id', 'relativePath'], maxResults * 4);
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

// reference-edges-from-the-index: the chunks that REFERENCE a symbol, read from
// the index instead of from a language server.
//
// Why the index is the primary source here: measured 2026-09-19 over 98 type
// declarations of the Haxe standard library, Serena returned three or more
// external referencing files for 5 of them and implementations for none, and it
// returned no cross-file references at all for C++ or JS/TS. A pool whose only
// source answers that rarely runs empty whatever its weight.
export async function fetchReferencingChunkIds(
    vectorDatabase: VectorDatabase,
    collection: string,
    symbolName: string,
    limit: number,
): Promise<string[]> {
    if (!symbolName) return [];
    try {
        const rows = await vectorDatabase.query(collection, referenceFilter('mentions', symbolName), ['id'], limit);
        return rows.map((r) => r?.id).filter((id): id is string => typeof id === 'string' && id.length > 0);
    } catch (err) {
        console.warn(`[Context] ⚠️ symbol-refs index references query failed for ${symbolName}: ${err}`);
        return [];
    }
}

// The chunks that DECLARE a type derived from this one.
export async function fetchImplementingChunkIds(
    vectorDatabase: VectorDatabase,
    collection: string,
    symbolName: string,
    limit: number,
): Promise<string[]> {
    if (!symbolName) return [];
    try {
        const rows = await vectorDatabase.query(
            collection,
            `${referenceFilter('extends', symbolName)} or ${referenceFilter('implements', symbolName)}`,
            ['id'],
            limit,
        );
        return rows.map((r) => r?.id).filter((id): id is string => typeof id === 'string' && id.length > 0);
    } catch (err) {
        console.warn(`[Context] ⚠️ symbol-refs index implementations query failed for ${symbolName}: ${err}`);
        return [];
    }
}
