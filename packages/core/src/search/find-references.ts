// find-references-as-a-tool: the agent-callable reference lookup.
//
// It answers from the index's own scalar fields and nothing else: no embedding,
// no RRF, no reranker, no score. The reason it exists is measured — four
// ranking levers failed to move the structural question (archive
// `2026-09-20-reserve-slots-for-reference-chunks`), because a cross-encoder
// scores "does this passage answer this question" and a file that USES a symbol
// does not look like an answer to a question about that symbol. The edges are
// in the index; what was missing was a way to ask for them directly.
//
// D3: the frequency bound ANSWERS. A symbol the corpus uses everywhere is not a
// useful reference question, and picking which 30 of a thousand rows to show
// would be a ranking decision — the exact thing this tool exists to avoid.

import { VectorDatabase } from '../vectordb/types';
import { identifierOf, SymbolFrequencyGate } from './symbol-frequency';
import { REFERENCE_RELATIONS, ReferenceRelation, referenceFilter } from './symbol-index-refs';

/** Rows are grouped per file, and inside a file the chunks of ONE symbol that
 *  touch or overlap are merged into a single span.
 *
 *  Both halves are measured, not stylistic. Phase 0 found `impact` returning
 *  168 raw chunks over 10-14 distinct files, and the first end-to-end call of
 *  the built tool printed `core_bind.cpp` as 35 consecutive `CoreBind` chunk
 *  lines — a chunked-up class, one edge. The agent reads ROWS, so the size bar
 *  is a row bar, and a row is a span. Merging is same-symbol only: collapsing
 *  across symbols would have swallowed `add_resource_format_saver` (L188-190),
 *  which is the row the question was actually about. */
export interface ReferenceSpan {
    /** The first chunk of the span — what `expand_context` is given. */
    chunkId: string;
    startLine: number;
    endLine: number;
    symbolName: string;
    symbolKind: string;
    contentType: string;
    /** How many chunks this span merged. 1 when nothing was merged. */
    chunks: number;
}

export interface ReferenceFile {
    relativePath: string;
    spans: ReferenceSpan[];
    /** Chunks behind the spans, so a collapsed file still says how big it is. */
    totalChunks: number;
}

export interface ReferenceGroup {
    relation: ReferenceRelation;
    files: ReferenceFile[];
    /** Files found before `limit` was applied. */
    totalFiles: number;
    /** The scalar query hit its own fetch ceiling, so `totalFiles` is a floor. */
    atFetchCeiling: boolean;
    /**
     * The query threw. An empty group and a FAILED group are the same rows and
     * opposite claims — "the index holds no such edge" against "we do not
     * know" — so the failure is carried to the answer instead of being logged
     * and dropped.
     */
    failed: string | null;
}

export interface FindReferencesAnswer {
    symbol: string;
    groups: ReferenceGroup[];
    /**
     * Non-null when the frequency bound refused the question. No rows are
     * returned in that case — the bound is the answer.
     */
    bounded: {
        /** The identifier the count belongs to: `frequencyOf` prices the LEAF
         *  of a name path, so `Foo.bar` is refused on `bar`'s frequency. */
        identifier: string;
        documentFrequency: number;
        boundDocuments: number;
        boundQuantile: number | null;
    } | null;
    /** The file limit actually applied, and whether the caller asked for more
     *  than the tool allows — a bound the answer has to be able to state. */
    limit: number;
    limitClampedFrom: number | null;
}

export interface FindReferencesOptions {
    vectorDatabase: VectorDatabase;
    collection: string;
    symbol: string;
    /** Which relations to ask for. Default: all three. */
    relations?: ReferenceRelation[];
    /** An extra scalar filter ANDed onto every relation (the category scope). */
    scopeFilter?: string;
    /** Max files returned per relation. */
    limit?: number;
    frequency?: SymbolFrequencyGate | null;
}

export const DEFAULT_FIND_REFERENCES_LIMIT = 30;
export const MAX_FIND_REFERENCES_LIMIT = 100;

// The frequency bound already keeps the big symbols out, so this ceiling is a
// guard against an unbounded scan and not a silent top-N: when it is hit the
// answer says so.
const FETCH_CEILING = 512;

// The relation's own field comes back with the row so the match can be
// confirmed exactly, in code. Milvus `like` is a WILDCARD match: `%` spans any
// run of characters and `_` any single one, and a backslash does NOT escape
// either (probed on the served collection, 2026-09-21: `"ValueExceptio_"` and
// `"ValueExceptio%"` both return `ValueException`'s 33 rows, and the escaped
// form returns 0). So the filter is a PREFILTER and the exact test is below —
// otherwise `Nod%` would come back as exact edges of `Node`, `NodePath` and
// `Node2D`, and every `_` in a real name would quietly widen its own match.
const OUTPUT_FIELDS = ['id', 'relativePath', 'startLine', 'endLine', 'content_type', 'symbol_kind', 'symbol_name', 'mentioned_symbols', 'implements', 'extends'];

const FIELD_OF: Record<ReferenceRelation, string> = {
    mentions: 'mentioned_symbols',
    extends: 'extends',
    implements: 'implements',
};

/** Does this row really carry the name, or did a wildcard let it through? */
export function rowNamesSymbol(relation: ReferenceRelation, row: Record<string, any>, symbol: string): boolean {
    const value = row?.[FIELD_OF[relation]];
    if (typeof value !== 'string' || value.length === 0) return false;
    // `extends` is a single name, compared exactly by the filter already.
    if (relation === 'extends') return value === symbol;
    try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed)) return parsed.some((n) => n === symbol);
    } catch {
        /* not JSON — fall through to the quoted-substring test */
    }
    return value.includes(`"${symbol}"`);
}

function clampLimit(limit: number | undefined): { limit: number; clampedFrom: number | null } {
    if (!Number.isFinite(limit as number)) return { limit: DEFAULT_FIND_REFERENCES_LIMIT, clampedFrom: null };
    const n = Math.floor(limit as number);
    if (n < 1) return { limit: DEFAULT_FIND_REFERENCES_LIMIT, clampedFrom: n };
    if (n > MAX_FIND_REFERENCES_LIMIT) return { limit: MAX_FIND_REFERENCES_LIMIT, clampedFrom: n };
    return { limit: n, clampedFrom: null };
}

/** Same symbol, touching or overlapping line ranges → one span. The input is
 *  already sorted by line, so one pass does it. */
function mergeSpans(sorted: ReferenceSpan[]): ReferenceSpan[] {
    const out: ReferenceSpan[] = [];
    for (const span of sorted) {
        const last = out[out.length - 1];
        if (last && last.symbolName === span.symbolName && span.startLine <= last.endLine + 1) {
            last.endLine = Math.max(last.endLine, span.endLine);
            last.chunks += span.chunks;
            continue;
        }
        out.push({ ...span });
    }
    return out;
}

async function groupFor(
    opts: FindReferencesOptions,
    relation: ReferenceRelation,
    limit: number,
): Promise<ReferenceGroup> {
    const base = referenceFilter(relation, opts.symbol);
    const filter = opts.scopeFilter && opts.scopeFilter.trim().length > 0
        ? `(${base}) and (${opts.scopeFilter})`
        : base;
    let rows: Record<string, any>[];
    try {
        rows = await opts.vectorDatabase.query(opts.collection, filter, OUTPUT_FIELDS, FETCH_CEILING);
    } catch (err) {
        console.warn(`[Context] ⚠️ find_references ${relation} query failed for ${opts.symbol}: ${err}`);
        return { relation, files: [], totalFiles: 0, atFetchCeiling: false, failed: String((err as any)?.message || err) };
    }

    const byPath = new Map<string, ReferenceSpan[]>();
    for (const row of rows) {
        const id = row?.id;
        const rel = row?.relativePath;
        if (typeof id !== 'string' || id.length === 0) continue;
        if (typeof rel !== 'string' || rel.length === 0) continue;
        if (!rowNamesSymbol(relation, row, opts.symbol)) continue;
        const chunks = byPath.get(rel) || [];
        chunks.push({
            chunkId: id,
            startLine: Number(row.startLine) || 0,
            endLine: Number(row.endLine) || 0,
            symbolName: typeof row.symbol_name === 'string' ? row.symbol_name : '',
            symbolKind: typeof row.symbol_kind === 'string' ? row.symbol_kind : '',
            contentType: typeof row.content_type === 'string' ? row.content_type : '',
            chunks: 1,
        });
        byPath.set(rel, chunks);
    }

    // Milvus returns scalar-query rows in no defined order, and the same query
    // twice can return them differently. The order here is the path, then the
    // line — lexicographic and locale-free, so two runs agree.
    const files: ReferenceFile[] = [...byPath.entries()]
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([relativePath, chunks]) => ({
            relativePath,
            spans: mergeSpans(chunks.sort((a, b) => a.startLine - b.startLine || (a.chunkId < b.chunkId ? -1 : 1))),
            totalChunks: chunks.length,
        }));

    return {
        relation,
        files: files.slice(0, limit),
        totalFiles: files.length,
        atFetchCeiling: rows.length >= FETCH_CEILING,
        failed: null,
    };
}

export async function findReferences(opts: FindReferencesOptions): Promise<FindReferencesAnswer> {
    const symbol = (opts.symbol || '').trim();
    const { limit, clampedFrom } = clampLimit(opts.limit);
    if (symbol.length === 0) return { symbol, groups: [], bounded: null, limit, limitClampedFrom: clampedFrom };

    const frequency = opts.frequency || null;
    if (frequency && frequency.atOrAboveBound(symbol)) {
        return {
            symbol,
            groups: [],
            bounded: {
                identifier: identifierOf(symbol) || symbol,
                documentFrequency: frequency.frequencyOf(symbol) as number,
                boundDocuments: frequency.boundDocuments as number,
                boundQuantile: frequency.boundQuantile,
            },
            limit,
            limitClampedFrom: clampedFrom,
        };
    }

    const relations = (opts.relations && opts.relations.length > 0 ? opts.relations : REFERENCE_RELATIONS)
        .filter((r) => REFERENCE_RELATIONS.includes(r));
    const groups = await Promise.all(relations.map((r) => groupFor({ ...opts, symbol }, r, limit)));
    return { symbol, groups, bounded: null, limit, limitClampedFrom: clampedFrom };
}
