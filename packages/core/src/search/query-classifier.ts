import { envManager } from '../utils/env-manager';

// Phase 0+: regex-based query intent classifier.
//
// Returns a coarse signal pair {codeSignal, docSignal} used by per-domain
// multi-query to weight the code-domain pool vs. the doc-domain pool in
// weighted RRF. Heuristics are deliberately cheap and language-agnostic so
// they add zero latency to the search path.
//
// Heuristics:
//   codeSignal — camelCase, snake_case, qualified names (Foo.bar), Haxe
//   metadata (`@:build`), C#-style attributes, decorators, function-call
//   parentheses, type sigils, code punctuation.
//   docSignal — natural-language tokens of length >= 3 with no code-shape.
//
// The classifier returns booleans rather than scores; callers map intent
// to weights (see weightedRrfMerge in context.ts).

export interface QueryIntent {
    codeSignal: boolean;
    docSignal: boolean;
}

// rag-graph-comparison-bridge: cheap heuristic for comparison-shape queries.
// Activates whenever the query contains one of the natural-language
// comparison connectors. Designed to match the gold-set's `query_shape ==
// "comparison"` queries (which all use " vs " or "compared to") while
// staying ignorant of language/domain — no special tokens, no LLM.
const COMPARISON_PATTERNS: RegExp[] = [
    /\bvs\.?\b/i,
    /\bversus\b/i,
    /\bcompared to\b/i,
    /\bdifference between\b/i,
    /\bcompare\s+\w+\s+(?:and|to)\b/i,
];

export function isComparisonShape(query: string | undefined | null): boolean {
    if (!query) return false;
    return COMPARISON_PATTERNS.some((p) => p.test(query));
}

const CAMEL_CASE = /\b[a-z]+[A-Z][a-zA-Z0-9]*\b/;
const PASCAL_CASE = /\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b/;
const SNAKE_CASE = /\b[a-z][a-z0-9]*_[a-z0-9_]+\b/;
const QUALIFIED_NAME = /\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b/;
const HAXE_METADATA = /@:[A-Za-z_][A-Za-z0-9_]*/;
const PY_DECORATOR = /(?:^|\s)@[A-Za-z_][A-Za-z0-9_]*/;
const CALL_PARENS = /[A-Za-z_][A-Za-z0-9_]*\s*\(/;
const CODE_PUNCT = /(?:::|->|=>|<<|>>|\+\+|--|&&|\|\||::<)/;
const TYPE_SIGIL = /(?:^|\s)[A-Z][a-zA-Z0-9_]*<[^>]+>/;
const ANGLE_TYPE_PARAM = /[A-Za-z_][A-Za-z0-9_]*<[A-Z][A-Za-z0-9_,\s]*>/;

const CODE_REGEXES: RegExp[] = [
    CAMEL_CASE,
    PASCAL_CASE,
    SNAKE_CASE,
    QUALIFIED_NAME,
    HAXE_METADATA,
    PY_DECORATOR,
    CALL_PARENS,
    CODE_PUNCT,
    TYPE_SIGIL,
    ANGLE_TYPE_PARAM,
];

// Words shorter than this are filtered out of the natural-language token
// count so single-letter typos and acronyms in code-only queries do not
// trip the doc signal.
const MIN_WORD_LEN = 3;
const MIN_WORDS_FOR_DOC = 3;

const NL_WORD = /\b[A-Za-z]{3,}\b/g;

// Count natural-language words of length >= MIN_WORD_LEN. PascalCase /
// camelCase tokens still match \b[A-Za-z]{3,}\b so we subtract anything
// that looks code-shaped. Shared by docSignal (classifyQuery) and the
// lexical-form axis (classifyLexicalForm) so the two stay calibrated to
// the same notion of "natural-language word".
function countPlainWords(trimmed: string): number {
    const nlWords = trimmed.match(NL_WORD) || [];
    const plainWords = nlWords.filter((w) => {
        if (CAMEL_CASE.test(w)) return false;
        if (PASCAL_CASE.test(w)) return false;
        if (SNAKE_CASE.test(w)) return false;
        // PascalCase single tokens like "Reflect" without inner caps look
        // like a real symbol if they're capitalized AND the query is short
        // (single-symbol lookup).
        if (/^[A-Z]/.test(w) && nlWords.length <= 2) return false;
        return true;
    });
    return plainWords.length;
}

export function classifyQuery(query: string): QueryIntent {
    if (!query || query.trim().length === 0) {
        return { codeSignal: false, docSignal: false };
    }
    const trimmed = query.trim();

    let codeSignal = false;
    for (const rx of CODE_REGEXES) {
        if (rx.test(trimmed)) {
            codeSignal = true;
            break;
        }
    }

    const docSignal = countPlainWords(trimmed) >= MIN_WORDS_FOR_DOC;

    return { codeSignal, docSignal };
}

// knowledge-router: lexical-form axis — orthogonal to {codeSignal, docSignal}.
//
// {codeSignal, docSignal} answers "which DOMAIN to hit" (code-pool vs
// doc-pool). `lexical_form` answers "which CHANNEL carries the query"
// (sparse/BM25 vs dense). The two axes are independent: a descriptive query
// about code wants the code-pool but the dense channel; an identifier query
// about code wants the code-pool and the sparse channel. Folding them into
// one signal would lose exactly the identifier-vs-descriptive combination on
// which the bipolar dense effect is observed (diagnostic 2026-05-23).
//
//   identifier  — dominated by a code token (qualified name, camelCase,
//                 snake_case, PascalCase) with fewer than MIN_WORDS_FOR_DOC
//                 natural-language words.
//   descriptive — MIN_WORDS_FOR_DOC+ natural-language words and no
//                 dominating code token.
//   mixed       — both signals present (a descriptive query with an embedded
//                 qualified name), or neither clearly present — the neutral
//                 bucket that keeps the Phase-4 channel defaults.
export type LexicalForm = 'identifier' | 'descriptive' | 'mixed';

// Identifier-shaped tokens — the four "code token" regexes reused from the
// codeSignal heuristics. CALL_PARENS / CODE_PUNCT / decorators / sigils are
// deliberately excluded: those mark code *shape* but are not the literal
// symbol token the BM25 channel keys off.
const IDENTIFIER_TOKEN_REGEXES: RegExp[] = [
    CAMEL_CASE,
    PASCAL_CASE,
    SNAKE_CASE,
    QUALIFIED_NAME,
];

// Global variant of QUALIFIED_NAME for stripping. A dotted qualified name
// (`haxe.Json.parse`) splits into components that each match NL_WORD —
// `haxe`, `parse` look like lowercase prose words. Counting them would tip
// a pure-symbol query into the descriptive/mixed buckets, so they are
// removed before the natural-language word count (calibrated on the
// gold-set, task 2.2).
const QUALIFIED_NAME_GLOBAL = new RegExp(QUALIFIED_NAME.source, 'g');

export function classifyLexicalForm(query: string): LexicalForm {
    if (!query || query.trim().length === 0) {
        return 'mixed';
    }
    const trimmed = query.trim();

    const hasCodeToken = IDENTIFIER_TOKEN_REGEXES.some((rx) => rx.test(trimmed));
    // Strip qualified-name tokens before counting NL words so their dotted
    // components do not inflate the count (see QUALIFIED_NAME_GLOBAL above).
    const nlWordCount = countPlainWords(trimmed.replace(QUALIFIED_NAME_GLOBAL, ' '));

    if (hasCodeToken && nlWordCount < MIN_WORDS_FOR_DOC) {
        return 'identifier';
    }
    if (!hasCodeToken && nlWordCount >= MIN_WORDS_FOR_DOC) {
        return 'descriptive';
    }
    // Both signals (qualified name embedded in NL prose) or neither clearly
    // present (short pure-NL query) — neutral bucket, Phase-4 channel
    // defaults let weighted RRF arbitrate.
    return 'mixed';
}

// Weight presets for weighted RRF merge:
//   code-only query  → boost code-domain pool
//   doc-only query   → boost doc-domain pool
//   mixed / unclear  → equal weight (let RRF arbitrate)
export interface DomainWeights {
    code: number;
    doc: number;
}

// code-collection-split: cross-pool RRF weights are env-overridable so the
// split-regime BM25 balance can be re-tuned without a rebuild. Defaults are
// the original single-collection values, so an unset env reproduces the
// pre-split behaviour bit-for-bit. Branch keys:
//   code-intent  → WEIGHT_CODE_INTENT_{CODE,DOC}   (default 1.5 / 1.0)
//   doc-intent   → WEIGHT_DOC_INTENT_{CODE,DOC}    (default 1.0 / 1.3)
//   mixed/ambig  → WEIGHT_MIXED_{CODE,DOC}         (default 1.0 / 1.0)
function envWeight(name: string, fallback: number): number {
    const raw = envManager.get(name);
    if (raw === undefined || raw === null || raw === '') return fallback;
    const n = Number(raw);
    return Number.isFinite(n) && n >= 0 ? n : fallback;
}

export function weightsForIntent(intent: QueryIntent): DomainWeights {
    if (intent.codeSignal && !intent.docSignal) {
        return { code: envWeight('WEIGHT_CODE_INTENT_CODE', 1.5), doc: envWeight('WEIGHT_CODE_INTENT_DOC', 1.0) };
    }
    if (!intent.codeSignal && intent.docSignal) {
        return { code: envWeight('WEIGHT_DOC_INTENT_CODE', 1.0), doc: envWeight('WEIGHT_DOC_INTENT_DOC', 1.3) };
    }
    return { code: envWeight('WEIGHT_MIXED_CODE', 1.0), doc: envWeight('WEIGHT_MIXED_DOC', 1.0) };
}

// Phase C (rag-code-intent-recall): qualified-name parser for symbol routing.
//
// Recognises three universal forms:
//   Foo.Bar.baz       (dot-separated; JS, Python, Haxe, …)
//   Foo::Bar::baz     (double-colon; Rust, C++)
//   Foo/Bar/baz       (slash; path-style)
//
// Returns the trailing component as `methodName` and the immediately preceding
// component as `className`. Returns null for natural-language phrases, single
// identifiers without a separator, and malformed inputs (`a.`, `.b`, `a..b`).
//
// Anchored: the entire trimmed query must be a qualified name. This protects
// downstream consumers (symbol-routing 3rd pool, reranker bypass) from
// mixed-intent queries that contain a qualified name embedded in NL prose
// (e.g. "Lambda.fold reduce list to single value" — these belong to the
// general reranked pool, not symbol-routing).

export interface ParsedQName {
    className: string;
    methodName: string;
    fullyQualified: string;
}

const COMPONENT = /[A-Za-z_][A-Za-z0-9_]*/;
// Whole-string qualified names with a single separator class.
const QNAME_DOT = new RegExp(`^${COMPONENT.source}(?:\\.${COMPONENT.source})+$`);
const QNAME_COLON = new RegExp(`^${COMPONENT.source}(?:::${COMPONENT.source})+$`);
const QNAME_SLASH = new RegExp(`^${COMPONENT.source}(?:/${COMPONENT.source})+$`);

// rag-symbol-refs-lsp-pool D5: single-identifier symbol parser used as a
// fallback gate when `parseQualifiedName` returns null but the query still
// contains a recognisable symbol token (e.g. "BytesBuffer", "Hmac").
//
// Behaviour:
//   1. Extract candidate tokens via `extractCodeTokens` (Unicode word regex).
//   2. Keep only tokens present in the per-codebase symbol vocabulary.
//   3. Drop matches shorter than 4 characters (stop-word guard — `data`
//      and `result` are vocab-listed but not distinctive).
//   4. Pick the most distinctive remaining token: longest first, with
//      PascalCase as tiebreak (these are most often the class subjects
//      the LSP pool can fan out from).
//
// Caller convention: only invoke this when `parseQualifiedName` already
// returned null — otherwise the structured form is preferred.

const MIN_SINGLE_SYMBOL_LEN = 4;
const CODE_TOKEN = /[\p{L}_][\p{L}\p{N}_]*/gu;

export interface SingleSymbol {
    symbolName: string;
}

export function extractCodeTokens(query: string): string[] {
    if (!query) return [];
    const matches = query.match(CODE_TOKEN);
    return matches ? Array.from(matches) : [];
}

export function parseSingleSymbol(
    query: string,
    vocab: ReadonlySet<string> | null | undefined,
): SingleSymbol | null {
    if (!query || !vocab || vocab.size === 0) return null;
    const tokens = extractCodeTokens(query);
    if (tokens.length === 0) return null;
    const matched = tokens.filter((t) => t.length >= MIN_SINGLE_SYMBOL_LEN && vocab.has(t));
    if (matched.length === 0) return null;
    matched.sort((a, b) => {
        if (a.length !== b.length) return b.length - a.length;
        const aPascal = /^[A-Z]/.test(a) ? 0 : 1;
        const bPascal = /^[A-Z]/.test(b) ? 0 : 1;
        return aPascal - bPascal;
    });
    return { symbolName: matched[0] };
}

export function parseQualifiedName(query: string): ParsedQName | null {
    if (!query) return null;
    const trimmed = query.trim();
    if (trimmed.length === 0) return null;

    let separator: string | null = null;
    if (QNAME_DOT.test(trimmed)) separator = '.';
    else if (QNAME_COLON.test(trimmed)) separator = '::';
    else if (QNAME_SLASH.test(trimmed)) separator = '/';
    if (separator === null) return null;

    const components = trimmed.split(separator);
    if (components.length < 2) return null;
    // Reject empty components (`a..b` would split to ['a','','b'] for dot).
    if (components.some((c) => c.length === 0)) return null;

    const methodName = components[components.length - 1];
    const className = components[components.length - 2];
    return { className, methodName, fullyQualified: trimmed };
}
