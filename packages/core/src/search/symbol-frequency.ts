// local-rag #64: the corpus-derived identifier frequency, read beside the
// symbol vocabulary.
//
// `<codebasePath>/.local-rag/symbol-frequency.json` is written by the harness
// (`infra/lib/symbol-frequency.js`, refreshed by `update-knowledge` beside the
// clangd compilation database). It carries the DOCUMENT frequency of every
// identifier — the number of the corpus's own files it occurs in — over the
// files of the languages this deployment's language servers resolve, plus the
// distribution of those frequencies as an exact histogram.
//
// The pool consults it before expanding references: with a warm index, a
// `find_referencing_symbols` on a name the corpus uses everywhere enumerates
// the whole corpus (17.4s and 487 048 characters on the live corpus, refused by
// Serena itself as too long), and `SYMBOL_REFS_MAX_REFERENCES` is 5, so the
// five references such a call would keep are an arbitrary five out of thousands.
//
// Two rules this file exists to keep:
//   - a name the table does not carry is UNKNOWN, never a frequency of zero.
//     Read as zero, a partial source ranks the most common identifier in the
//     corpus first, which is the failure the gate exists to prevent.
//   - the bound is a position in the corpus's own distribution, never a number
//     of documents stated in code, so one policy yields different thresholds on
//     corpora of different sizes.

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

export const SYMBOL_FREQUENCY_SCHEMA = 'local-rag-symbol-frequency-v1';
export const SYMBOL_FREQUENCY_RELPATH = path.join('.local-rag', 'symbol-frequency.json');
// The provenance the harness writes beside the table: when it was derived, how
// long it took, and the sha256 of the table's own bytes. The pool cannot walk
// the corpus to re-check the signature, but it CAN refuse a table the report
// beside it does not describe — a half-written file, a hand-edited one, or one
// left behind by another corpus.
export const SYMBOL_FREQUENCY_REPORT_RELPATH = path.join('.local-rag', 'symbol-frequency-report.json');

export interface SymbolFrequencyDistribution {
    identifiers: number;
    /** frequency value -> how many identifiers have it. Exact and small. */
    histogram: Record<string, number>;
    quantiles?: Record<string, number>;
}

export interface SymbolFrequencyTable {
    schema: string;
    signature?: string;
    generated_at?: string;
    documents: number;
    distribution: SymbolFrequencyDistribution;
    frequencies: Record<string, number>;
}

export interface SymbolFrequencyGate {
    /** `derived` when a table was read, `absent` when there is none. */
    source: 'derived' | 'absent';
    documents: number | null;
    identifiers: number | null;
    signature: string | null;
    /** The threshold in documents this build applies, or null for no bound. */
    boundDocuments: number | null;
    /** The policy position in the distribution the bound came from. */
    boundQuantile: number | null;
    /** A number, or null for UNKNOWN. Never zero for a name nothing carries. */
    frequencyOf(name: string): number | null;
    /** False when there is no bound, and false when the frequency is unknown. */
    atOrAboveBound(name: string): boolean;
}

/** The leaf of a name path (`Class/method`, `A::b`, `pkg.Thing`). */
export function identifierOf(name: string): string | null {
    const text = String(name ?? '').trim();
    if (!text) return null;
    const parts = text.split(/::|\/|\./).filter((part) => part.length > 0);
    const leaf = parts.length > 0 ? parts[parts.length - 1] : '';
    if (!leaf) return null;
    return /^[A-Za-z_][A-Za-z0-9_]*$/.test(leaf) ? leaf : null;
}

/**
 * The document-frequency value at a position in the distribution, walked over
 * the histogram rather than over an expanded list of 358 191 numbers.
 */
export function quantileFromHistogram(distribution: SymbolFrequencyDistribution | null, quantile: number): number | null {
    if (!distribution || !distribution.histogram) return null;
    if (!Number.isFinite(quantile) || quantile < 0 || quantile > 1) return null;
    const values = Object.keys(distribution.histogram).map(Number).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    if (values.length === 0) return null;
    let total = 0;
    for (const value of values) total += distribution.histogram[String(value)];
    if (total === 0) return null;
    const target = Math.ceil(quantile * (total - 1));
    let seen = 0;
    for (const value of values) {
        seen += distribution.histogram[String(value)];
        if (seen > target) return value;
    }
    return values[values.length - 1];
}

/**
 * The bound in DOCUMENTS a policy quantile yields on this corpus. A quantile of
 * 0 (unset, empty, out of range) is the policy "no bound" — the inert default —
 * and never "skip everything", which is what a 0 read as a position would mean.
 */
export function boundForQuantile(distribution: SymbolFrequencyDistribution | null, quantile: number | null): number | null {
    const q = Number(quantile);
    if (!Number.isFinite(q) || q <= 0 || q > 1) return null;
    return quantileFromHistogram(distribution, q);
}

/**
 * Read the table written beside the corpus, and only accept it when the report
 * beside it describes exactly these bytes. Null when there is none, when it is
 * malformed, or when the two disagree — in every one of those cases the pool
 * expands references for every activated subject, which is the behaviour that
 * predates the bound.
 */
export function loadSymbolFrequencyTable(codebasePath: string): SymbolFrequencyTable | null {
    let raw: string;
    try {
        raw = fs.readFileSync(path.join(codebasePath, SYMBOL_FREQUENCY_RELPATH), 'utf-8');
    } catch {
        return null;
    }
    let parsed: any;
    try {
        parsed = JSON.parse(raw);
    } catch {
        return null;
    }
    if (!parsed || parsed.schema !== SYMBOL_FREQUENCY_SCHEMA) return null;
    if (!parsed.frequencies || !parsed.distribution || !parsed.distribution.histogram) return null;
    // The table has to be the one its own report describes. A table with no
    // report, or one whose bytes have moved under it, is not applied: the gate
    // would otherwise bound references by a table nothing in the run vouches for
    // and no artifact describes.
    try {
        const report = JSON.parse(fs.readFileSync(path.join(codebasePath, SYMBOL_FREQUENCY_REPORT_RELPATH), 'utf-8'));
        const digest = crypto.createHash('sha256').update(raw).digest('hex');
        if (!report || report.sha256 !== digest || report.signature !== parsed.signature) {
            console.warn('[Context] 🔢 symbol frequency: the table and the report beside it disagree — not applying a bound');
            return null;
        }
    } catch {
        console.warn('[Context] 🔢 symbol frequency: no report beside the table — not applying a bound');
        return null;
    }
    return parsed as SymbolFrequencyTable;
}

/**
 * The gate both the pool and the run record read. With `table === null` the
 * source is `absent`, every frequency is unknown, there is no bound, and every
 * activated subject expands references exactly as it did before this gate.
 */
export function createSymbolFrequencyGate(table: SymbolFrequencyTable | null, quantile: number | null): SymbolFrequencyGate {
    const distribution = table ? table.distribution : null;
    const bound = boundForQuantile(distribution, quantile);
    const frequencyOf = (name: string): number | null => {
        if (!table) return null;
        const identifier = identifierOf(name);
        if (!identifier) return null;
        return Object.prototype.hasOwnProperty.call(table.frequencies, identifier) ? table.frequencies[identifier] : null;
    };
    return {
        source: table ? 'derived' : 'absent',
        documents: table ? table.documents : null,
        identifiers: distribution ? distribution.identifiers : null,
        signature: table && table.signature ? table.signature : null,
        boundDocuments: bound,
        boundQuantile: bound === null ? null : Number(quantile),
        frequencyOf,
        atOrAboveBound(name: string): boolean {
            if (bound === null) return false;
            const value = frequencyOf(name);
            return value !== null && value >= bound;
        },
    };
}
