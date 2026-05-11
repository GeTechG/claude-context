// rag-query-static-rewrite: deterministic, LLM-free query rewriters for the
// retrieval-side hybrid pipeline. Three independent transformations gated by
// env flags (`QUERY_REWRITE_COMPARISON_SPLIT`, `QUERY_REWRITE_CASE_EXPANSION`,
// `QUERY_REWRITE_ABBREV_EXPANSION`):
//
//   - splitComparison(query)      — detects EN/RU comparison triggers and
//                                   returns {left, right} subjects for
//                                   independent multi-query fan-out.
//   - expandCaseVariants(token)   — emits sibling case-forms (camelCase →
//                                   [snake_case, kebab-case], etc.) for
//                                   BM25 sparse-channel concatenation.
//   - expandAbbreviations(query)  — whitelist-driven token expansion
//                                   (`cfg → config`, …) for the same channel.
//
// applyRewriting(query, flags) composes the three into a single RewriteResult.
//
// Design ref: openspec/changes/rag-query-static-rewrite/design.md.

export interface ComparisonSplit {
    left: string;
    right: string;
}

export type RewriteKind = 'single' | 'split';

export interface RewriteResult {
    kind: RewriteKind;
    /** Populated only when kind === 'split'. */
    left?: string;
    right?: string;
    /** Tokens appended to the BM25 sparse-channel input. Empty when no expansion fired. */
    sparseExtra: string[];
    /** Diagnostic payload for the [Context] log line. */
    debug: {
        comparisonMatchedTrigger?: string;
        comparisonRejectedReason?: 'subject_empty' | 'subject_too_long' | 'guard_fail';
        caseExpansions: Array<{ from: string; to: string[] }>;
        abbrevExpansions: Array<{ from: string; to: string[] }>;
    };
}

export interface RewriteFlags {
    split: boolean;
    case: boolean;
    abbrev: boolean;
}

// --- Comparison triggers --------------------------------------------------

interface ComparisonPattern {
    re: RegExp;
    /** 'binary' = split at trigger token; 'phrase' = left/right captured by regex groups. */
    kind: 'binary' | 'phrase';
    /** For 'phrase' kind: how to extract left/right from regex match. */
    extract?: (m: RegExpExecArray) => { left: string; right: string } | null;
}

const PATTERNS: ComparisonPattern[] = [
    // EN binary triggers — split point is the trigger token itself.
    { re: /\bvs\b/i, kind: 'binary' },
    { re: /\bversus\b/i, kind: 'binary' },
    // EN phrase triggers — regex captures left/right directly.
    {
        re: /\bcompared\s+to\b/i,
        kind: 'phrase',
        extract: (m) => {
            const full = m.input;
            const pre = full.slice(0, m.index).trim();
            const post = full.slice(m.index + m[0].length).trim();
            if (!pre || !post) return null;
            return { left: pre, right: post };
        },
    },
    {
        re: /\bdifference\s+between\s+(.+?)\s+and\s+(.+)/i,
        kind: 'phrase',
        extract: (m) => (m[1] && m[2] ? { left: m[1].trim(), right: m[2].trim() } : null),
    },
    // RU binary triggers — same split-at-token behavior as `vs`.
    // JS `\b` is ASCII-only; for Cyrillic words we anchor on whitespace or
    // start/end of string. Using lookarounds keeps the split-at-token
    // semantics consistent with `vs`/`versus`.
    { re: /(?:^|\s)против(?:\s|$)/iu, kind: 'binary' },
    // RU phrase triggers.
    {
        re: /разница\s+между\s+(.+?)\s+и\s+(.+)/iu,
        kind: 'phrase',
        extract: (m) => (m[1] && m[2] ? { left: m[1].trim(), right: m[2].trim() } : null),
    },
    {
        re: /чем\s+(.+?)\s+отличается\s+от\s+(.+)/iu,
        kind: 'phrase',
        extract: (m) => (m[1] && m[2] ? { left: m[1].trim(), right: m[2].trim() } : null),
    },
    {
        re: /отличие\s+(.+?)\s+от\s+(.+)/iu,
        kind: 'phrase',
        extract: (m) => (m[1] && m[2] ? { left: m[1].trim(), right: m[2].trim() } : null),
    },
];

// `(?:compare|comparing) … (?:and|with|to)` requires both halves; treat as
// phrase but pull pre-trigger as right context discard (the comparison verb
// can sit before either subject). We anchor on the verb + connector pair.
const COMPARE_VERB = /\b(?:compare|comparing)\b\s+(.+?)\s+(?:and|with|to)\s+(.+)/i;
PATTERNS.push({
    re: COMPARE_VERB,
    kind: 'phrase',
    extract: (m) => (m[1] && m[2] ? { left: m[1].trim(), right: m[2].trim() } : null),
});

function countTokens(s: string): number {
    const trimmed = s.trim();
    if (trimmed.length === 0) return 0;
    return trimmed.split(/\s+/).length;
}

/**
 * Detect comparison-shape queries and split them into two independent
 * subjects. Returns null when no trigger matches or when the length-guard
 * rejects the candidate split.
 *
 * Length-guard: each subject must be ≥ 1 token and ≤ 6 tokens. This
 * protects against false-positive splits on NL prose like
 * "compared to last year" (left empty) or "different from prior years
 * where ..." (right too long).
 */
export function splitComparison(query: string): ComparisonSplit | null {
    if (!query) return null;
    const trimmed = query.trim();
    if (trimmed.length === 0) return null;

    for (const pat of PATTERNS) {
        // Use exec to capture index for binary split-at-trigger paths.
        const re = new RegExp(pat.re.source, pat.re.flags);
        const m = re.exec(trimmed);
        if (!m) continue;

        let left: string;
        let right: string;

        if (pat.kind === 'binary') {
            left = trimmed.slice(0, m.index).trim();
            right = trimmed.slice(m.index + m[0].length).trim();
        } else {
            const extracted = pat.extract!(m);
            if (!extracted) continue;
            left = extracted.left;
            right = extracted.right;
        }

        const lTok = countTokens(left);
        const rTok = countTokens(right);
        if (lTok < 1 || rTok < 1) continue;
        if (lTok > 6 || rTok > 6) continue;

        return { left, right };
    }
    return null;
}

/** For diagnostics — returns the matched trigger token / phrase, or null. */
function detectComparisonTrigger(query: string): { trigger: string; matched: boolean } | null {
    if (!query) return null;
    const trimmed = query.trim();
    for (const pat of PATTERNS) {
        const re = new RegExp(pat.re.source, pat.re.flags);
        const m = re.exec(trimmed);
        if (m) return { trigger: m[0], matched: true };
    }
    return null;
}

// --- Case-style expansion -------------------------------------------------

const CAMEL_RE = /^[a-z]+(?:[A-Z][a-zA-Z0-9]*)+$/;
const PASCAL_RE = /^[A-Z][a-z]+(?:[A-Z][a-zA-Z0-9]*)+$/;
const SNAKE_RE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;
const KEBAB_RE = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$/;

function splitCamel(token: string): string[] {
    // "parseConfig" → ["parse", "Config"]; "URLEncode" → ["URL", "Encode"]
    // (acronym block is preserved as one segment).
    const parts: string[] = [];
    let buf = '';
    for (let i = 0; i < token.length; i++) {
        const c = token[i];
        const isUpper = c >= 'A' && c <= 'Z';
        if (isUpper && buf.length > 0) {
            // Boundary heuristic: new segment when previous char was lower OR
            // next char is lower (end of acronym block).
            const prev = token[i - 1];
            const next = token[i + 1];
            const prevLower = prev && prev >= 'a' && prev <= 'z';
            const nextLower = next && next >= 'a' && next <= 'z';
            if (prevLower || nextLower) {
                parts.push(buf);
                buf = '';
            }
        }
        buf += c;
    }
    if (buf.length > 0) parts.push(buf);
    return parts.filter((p) => p.length > 0);
}

/**
 * Emit sibling case-form variants for a single token. Returns the input
 * token's complement set: `parseConfig` → `[parse_config, parse-config]`;
 * `parse_config` → `[parseConfig, parse-config]`; `parse-config` →
 * `[parseConfig, parse_config]`. Returns [] for tokens with no case shape
 * or single-form tokens (all-lowercase, all-uppercase).
 */
export function expandCaseVariants(token: string): string[] {
    if (!token) return [];
    const t = token.trim();
    if (t.length === 0) return [];

    // Detect form and split into segments.
    let segments: string[] | null = null;
    let inputForm: 'camel' | 'pascal' | 'snake' | 'kebab' | null = null;
    if (CAMEL_RE.test(t)) {
        inputForm = 'camel';
        segments = splitCamel(t);
    } else if (PASCAL_RE.test(t)) {
        inputForm = 'pascal';
        segments = splitCamel(t);
    } else if (SNAKE_RE.test(t)) {
        inputForm = 'snake';
        segments = t.split('_');
    } else if (KEBAB_RE.test(t)) {
        inputForm = 'kebab';
        segments = t.split('-');
    }
    if (!segments || segments.length < 2) return [];

    const lower = segments.map((s) => s.toLowerCase());
    const variants: string[] = [];

    const snake = lower.join('_');
    const kebab = lower.join('-');
    const camel = lower[0] + lower.slice(1).map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join('');
    const pascal = lower.map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join('');

    if (inputForm !== 'snake' && snake !== t) variants.push(snake);
    if (inputForm !== 'kebab' && kebab !== t) variants.push(kebab);
    if (inputForm !== 'camel' && camel !== t) variants.push(camel);
    if (inputForm !== 'pascal' && pascal !== t) variants.push(pascal);

    // Dedup
    return Array.from(new Set(variants)).filter((v) => v !== t);
}

// --- Abbreviation expansion ----------------------------------------------

/**
 * Static abbreviation whitelist for sparse-channel expansion. Each entry maps
 * a short form to one or more canonical long forms. Membership rationale:
 *
 *   cfg     → config, configuration   — pervasive in configuration code paths
 *   auth    → authentication          — auth flows in docs vs. code
 *   db      → database                — short form in NL queries, long form indexed
 *   lib     → library                 — package/library refs in markdown
 *   repo    → repository              — Git/source-tree terminology
 *   init    → initialize, initializer — covers verb + noun forms in symbol names
 *   impl    → implementation          — common in docstrings and identifiers
 *
 * Short or polysemous abbreviations (`net`, `id`, `op`, `sys`) are
 * intentionally EXCLUDED — false-positive risk on the N=46 tuning gold-set
 * outweighs marginal lift (whitelist tuned in change `rag-query-static-rewrite`
 * task 2.2). Audited eval queries: only 1 of 58 contains a whitelist token
 * (q55 `init` in "build expression init"), so expected impact on this corpus
 * is small — whitelist remains for forward gold-set coverage.
 */
export const ABBREV_WHITELIST: ReadonlyMap<string, readonly string[]> = new Map([
    ['cfg', ['config', 'configuration']],
    ['auth', ['authentication']],
    ['db', ['database']],
    ['lib', ['library']],
    ['repo', ['repository']],
    ['init', ['initialize', 'initializer']],
    ['impl', ['implementation']],
]);

const WORD_RE = /[A-Za-z][A-Za-z0-9_]*/g;

/**
 * Walk through tokens in `query` and emit expanded forms for any whitelist
 * abbreviation. Skips expansions when the expanded form is already present
 * in the query (idempotency: `auth authentication` does not produce
 * duplicate `authentication`).
 */
export function expandAbbreviations(query: string): string[] {
    if (!query) return [];
    const lowerQuery = query.toLowerCase();
    const out: string[] = [];
    const seen = new Set<string>();
    let m: RegExpExecArray | null;
    const re = new RegExp(WORD_RE.source, 'g');
    while ((m = re.exec(query)) !== null) {
        const tok = m[0].toLowerCase();
        const expansions = ABBREV_WHITELIST.get(tok);
        if (!expansions) continue;
        for (const exp of expansions) {
            if (seen.has(exp)) continue;
            // Idempotency: do not duplicate a long form already present.
            const wordRe = new RegExp(`\\b${exp}\\b`, 'i');
            if (wordRe.test(lowerQuery)) continue;
            seen.add(exp);
            out.push(exp);
        }
    }
    return out;
}

// --- Composition ----------------------------------------------------------

/**
 * Build the sparse-channel extras (case + abbrev) by walking tokens in the
 * query. Returns a flat list of expansion strings ready for concatenation,
 * plus the per-feature debug payload.
 */
function collectExpansions(
    query: string,
    flags: { case: boolean; abbrev: boolean },
): {
    extras: string[];
    caseExpansions: Array<{ from: string; to: string[] }>;
    abbrevExpansions: Array<{ from: string; to: string[] }>;
} {
    const caseExpansions: Array<{ from: string; to: string[] }> = [];
    const abbrevExpansions: Array<{ from: string; to: string[] }> = [];
    const extras: string[] = [];
    const seen = new Set<string>();

    if (!query) return { extras, caseExpansions, abbrevExpansions };

    if (flags.case) {
        const re = new RegExp(WORD_RE.source, 'g');
        let m: RegExpExecArray | null;
        while ((m = re.exec(query)) !== null) {
            const tok = m[0];
            const variants = expandCaseVariants(tok);
            if (variants.length === 0) continue;
            caseExpansions.push({ from: tok, to: variants });
            for (const v of variants) {
                if (seen.has(v)) continue;
                seen.add(v);
                extras.push(v);
            }
        }
    }

    if (flags.abbrev) {
        const lowerQuery = query.toLowerCase();
        const re = new RegExp(WORD_RE.source, 'g');
        let m: RegExpExecArray | null;
        const perFromSeen = new Set<string>();
        while ((m = re.exec(query)) !== null) {
            const raw = m[0];
            const tok = raw.toLowerCase();
            const expansions = ABBREV_WHITELIST.get(tok);
            if (!expansions || perFromSeen.has(tok)) continue;
            perFromSeen.add(tok);
            const toAdd: string[] = [];
            for (const exp of expansions) {
                const wordRe = new RegExp(`\\b${exp}\\b`, 'i');
                if (wordRe.test(lowerQuery)) continue;
                if (seen.has(exp)) continue;
                seen.add(exp);
                toAdd.push(exp);
                extras.push(exp);
            }
            if (toAdd.length > 0) abbrevExpansions.push({ from: raw, to: toAdd });
        }
    }

    return { extras, caseExpansions, abbrevExpansions };
}

/**
 * Apply the three rewriters according to flags. Returns a uniform
 * RewriteResult that callers can act on without re-running detection:
 *
 *   kind === 'split'  → run two independent multi-query pipelines for
 *                       `left` and `right`, then outer-RRF 0.5/0.5.
 *   kind === 'single' → standard single multi-query path; if
 *                       sparseExtra.length > 0, append to the BM25
 *                       sparse-channel `data` input.
 */
export function applyRewriting(query: string, flags: RewriteFlags): RewriteResult {
    const result: RewriteResult = {
        kind: 'single',
        sparseExtra: [],
        debug: {
            caseExpansions: [],
            abbrevExpansions: [],
        },
    };

    if (flags.split) {
        const trigger = detectComparisonTrigger(query);
        if (trigger) {
            result.debug.comparisonMatchedTrigger = trigger.trigger;
            const split = splitComparison(query);
            if (split) {
                result.kind = 'split';
                result.left = split.left;
                result.right = split.right;
                // When split fires, case/abbrev still apply but per-subject
                // — collect extras for each subject and stash them on the
                // result. For wiring simplicity we keep a single sparseExtra
                // representing the union; the wiring layer can also call
                // collectExpansions per-subject if it prefers per-side
                // injection. The simpler union approach is what we ship.
                const expL = collectExpansions(split.left, { case: flags.case, abbrev: flags.abbrev });
                const expR = collectExpansions(split.right, { case: flags.case, abbrev: flags.abbrev });
                const dedup = new Set<string>();
                for (const e of [...expL.extras, ...expR.extras]) {
                    if (!dedup.has(e)) {
                        dedup.add(e);
                        result.sparseExtra.push(e);
                    }
                }
                result.debug.caseExpansions = [...expL.caseExpansions, ...expR.caseExpansions];
                result.debug.abbrevExpansions = [...expL.abbrevExpansions, ...expR.abbrevExpansions];
                return result;
            }
            // Trigger matched but split rejected by guards — fall through to
            // single-path, but tag the debug payload so the wiring layer can
            // log "comparison detected but split rejected".
            result.debug.comparisonRejectedReason = 'guard_fail';
        }
    }

    const exp = collectExpansions(query, { case: flags.case, abbrev: flags.abbrev });
    result.sparseExtra = exp.extras;
    result.debug.caseExpansions = exp.caseExpansions;
    result.debug.abbrevExpansions = exp.abbrevExpansions;
    return result;
}
