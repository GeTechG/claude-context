/**
 * prefer-the-named-engine-family: named-family prose demotion.
 *
 * When a query names an engine (resolved by the HOST — this module knows no
 * corpus), prose from outside every named family is stable-moved below the rest
 * of the final list. Code never moves, in-family prose never moves, nothing is
 * dropped or added. The reranker scores passages and never who they are about;
 * this is the one place the final order learns which corpus the query named.
 */

/** What the host resolved for one query. */
export interface NamedFamilyMatch {
    /** Source ids the query names. */
    named: string[];
    /** Every source of every named family (a named source plus its specialisations). */
    members: string[];
    /** The members' registered content paths, relative to the knowledge root. */
    pathPrefixes: string[];
}

/** Host-supplied: the families a query names, or null when it names none. */
export type NamedFamilyResolver = (query: string, codebasePath?: string) => NamedFamilyMatch | null;

/** What the step did on the last search (diagnostics and the eval harness). */
export interface NamedFamilyDemotion {
    named: string[];
    members: string[];
    /** How many prose rows from outside the families were moved below the rest. */
    demoted: number;
    /** Whether the final order actually changed (a demoted tail already last changes nothing). */
    moved: boolean;
    /** The demoted rows' paths, in their final order. */
    demotedPaths: string[];
}

function normalisePrefix(prefix: string): string {
    return String(prefix || '').replace(/\\/g, '/').replace(/^\.?\/+/, '').replace(/\/+$/, '');
}

/** Whether `relativePath` lies under one of the prefixes (whole path segments). */
export function isUnderAnyPrefix(relativePath: string, prefixes: string[]): boolean {
    const p = normalisePrefix(relativePath);
    return prefixes.some((raw) => {
        const prefix = normalisePrefix(raw);
        return prefix.length > 0 && (p === prefix || p.startsWith(prefix + '/'));
    });
}

/**
 * Stable partition: rows that are prose (`content_type === 'doc'`) and outside
 * every prefix go last, in their own order; every other row keeps its order.
 * Returns the SAME array when there is nothing to move.
 */
export function demoteProseOutsideNamedFamilies<T extends { relativePath: string; content_type?: string }>(
    rows: T[],
    pathPrefixes: string[],
): { rows: T[]; demoted: number; moved: boolean; demotedPaths: string[] } {
    const prefixes = (pathPrefixes || []).map(normalisePrefix).filter((p) => p.length > 0);
    if (!rows || rows.length === 0 || prefixes.length === 0) return { rows, demoted: 0, moved: false, demotedPaths: [] };
    const keep: T[] = [];
    const out: T[] = [];
    for (const row of rows) {
        if (row.content_type === 'doc' && !isUnderAnyPrefix(row.relativePath, prefixes)) out.push(row);
        else keep.push(row);
    }
    if (out.length === 0) return { rows, demoted: 0, moved: false, demotedPaths: [] };
    const next = [...keep, ...out];
    const moved = next.some((row, i) => row !== rows[i]);
    return { rows: moved ? next : rows, demoted: out.length, moved, demotedPaths: out.map((row) => row.relativePath) };
}
