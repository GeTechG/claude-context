// Phase 3: extract candidate symbol names from a search result pool so the
// agent can hand them to Serena (`find_symbol`) for full-definition lookup.
//
// Two sources per chunk:
//   code / docstring chunks → metadata (`symbol_name`, `parent_symbol`)
//                             plus regex over the body for nested defs.
//   doc / code_example chunks → regex over the body for qualified names
//                               (`Foo.bar.baz`) and markdown code spans
//                               (`` `func()` ``).
//
// Candidates are deduped, ranked by (frequency, has-qualifier, length),
// optionally filtered against a project-level vocabulary, and the top-N
// are returned. Default N = 10.

import { SemanticSearchResult } from '../types';

export interface SymbolExtractorOptions {
    topN?: number;
    vocabulary?: ReadonlySet<string>;
}

const DEFAULT_TOP_N = 10;

const NESTED_DEF = /\b(?:class|function|def|fn|interface|enum|typedef|abstract)\s+([A-Za-z_][A-Za-z0-9_]*)/g;
// rag-graph-layer Phase 1.2: shared with markdown-splitter at split time
// (mentioned_symbols metadata) and with this module at search time
// (candidateSymbols enrichment).
export const QUALIFIED_NAME = /\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b/g;
export const MD_CODE_SPAN = /`([^`\n]{1,120})`/g;
const BARE_IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;
const QUALIFIED_IDENT = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$/;

export const SYMBOL_STOPWORDS: ReadonlySet<string> = new Set([
    'true', 'false', 'null', 'none', 'undefined', 'self', 'this', 'super',
    'class', 'function', 'def', 'fn', 'interface', 'enum', 'typedef',
    'abstract', 'return', 'import', 'from', 'as', 'if', 'else', 'for',
    'while', 'try', 'catch', 'throw', 'new', 'delete', 'const', 'let',
    'var', 'public', 'private', 'protected', 'static', 'async', 'await',
]);
const STOPWORDS = SYMBOL_STOPWORDS;

function isCodeDomain(r: SemanticSearchResult): boolean {
    const t = r.content_type;
    return t === 'code' || t === 'docstring';
}

function isDocDomain(r: SemanticSearchResult): boolean {
    const t = r.content_type;
    return t === 'doc' || t === 'code_example';
}

function looksLikeSymbol(token: string): boolean {
    if (!token) return false;
    if (STOPWORDS.has(token.toLowerCase())) return false;
    if (token.length < 2) return false;
    return BARE_IDENT.test(token) || QUALIFIED_IDENT.test(token);
}

/**
 * rag-graph-layer Phase 1.2: extract qualified-names and markdown code-span
 * tokens from a doc / code_example body. Used at split time to populate
 * `chunk.metadata.mentioned_symbols`. Optionally vocab-filtered with the
 * same logic as `extractCandidateSymbols`.
 */
export function extractMentionedSymbolsFromText(
    text: string,
    vocabulary?: ReadonlySet<string>,
): string[] {
    if (!text) return [];
    const seen = new Set<string>();
    const out: string[] = [];
    const push = (token: string) => {
        if (!looksLikeSymbol(token)) return;
        if (seen.has(token)) return;
        seen.add(token);
        out.push(token);
    };

    let m: RegExpExecArray | null;
    QUALIFIED_NAME.lastIndex = 0;
    while ((m = QUALIFIED_NAME.exec(text)) !== null) {
        push(m[0]);
    }
    MD_CODE_SPAN.lastIndex = 0;
    while ((m = MD_CODE_SPAN.exec(text)) !== null) {
        const inner = m[1].trim();
        if (!inner) continue;
        const stripped = inner.replace(/\s*\([^)]*\)\s*$/, '');
        push(stripped);
    }

    if (!vocabulary || vocabulary.size === 0) return out;
    return out.filter((name) => {
        if (vocabulary.has(name)) return true;
        if (!name.includes('.')) return false;
        for (const seg of name.split('.')) {
            if (vocabulary.has(seg)) return true;
        }
        return false;
    });
}

function harvestFromCode(r: SemanticSearchResult, sink: Map<string, number>): void {
    if (r.symbol_name && looksLikeSymbol(r.symbol_name)) {
        bump(sink, r.symbol_name);
    }
    if (r.parent_symbol && looksLikeSymbol(r.parent_symbol)) {
        bump(sink, r.parent_symbol);
        if (r.symbol_name && looksLikeSymbol(r.symbol_name)) {
            const qualified = `${r.parent_symbol}.${r.symbol_name}`;
            if (looksLikeSymbol(qualified)) bump(sink, qualified);
        }
    }
    if (!r.content) return;
    let m: RegExpExecArray | null;
    NESTED_DEF.lastIndex = 0;
    while ((m = NESTED_DEF.exec(r.content)) !== null) {
        const name = m[1];
        if (looksLikeSymbol(name)) bump(sink, name);
    }
}

function harvestFromDoc(r: SemanticSearchResult, sink: Map<string, number>): void {
    if (!r.content) return;
    let m: RegExpExecArray | null;

    QUALIFIED_NAME.lastIndex = 0;
    while ((m = QUALIFIED_NAME.exec(r.content)) !== null) {
        const name = m[0];
        if (looksLikeSymbol(name)) bump(sink, name);
    }

    MD_CODE_SPAN.lastIndex = 0;
    while ((m = MD_CODE_SPAN.exec(r.content)) !== null) {
        const inner = m[1].trim();
        if (!inner) continue;
        // Strip trailing call parens so `parse_args()` becomes `parse_args`.
        const stripped = inner.replace(/\s*\([^)]*\)\s*$/, '');
        if (looksLikeSymbol(stripped)) bump(sink, stripped);
    }
}

function bump(sink: Map<string, number>, key: string): void {
    sink.set(key, (sink.get(key) || 0) + 1);
}

export function extractCandidateSymbols(
    results: SemanticSearchResult[],
    options: SymbolExtractorOptions = {},
): string[] {
    const topN = options.topN ?? DEFAULT_TOP_N;
    if (!results || results.length === 0 || topN <= 0) return [];

    const counts = new Map<string, number>();
    for (const r of results) {
        if (isCodeDomain(r)) {
            harvestFromCode(r, counts);
        } else if (isDocDomain(r)) {
            harvestFromDoc(r, counts);
        } else {
            // Unknown content_type (legacy chunks): try both, cheap enough.
            harvestFromCode(r, counts);
            harvestFromDoc(r, counts);
        }
    }

    let entries = Array.from(counts.entries());
    if (options.vocabulary && options.vocabulary.size > 0) {
        const vocab = options.vocabulary;
        entries = entries.filter(([name]) => {
            if (vocab.has(name)) return true;
            // For qualified names, accept if any segment is in vocab —
            // covers cases where the doc references `argparse.ArgumentParser`
            // but the indexed symbol is just `ArgumentParser`.
            if (name.includes('.')) {
                for (const seg of name.split('.')) {
                    if (vocab.has(seg)) return true;
                }
            }
            return false;
        });
    }

    entries.sort((a, b) => {
        // (a) frequency desc
        if (b[1] !== a[1]) return b[1] - a[1];
        // (b) qualified names rank higher (more specific)
        const aQual = a[0].includes('.') ? 1 : 0;
        const bQual = b[0].includes('.') ? 1 : 0;
        if (bQual !== aQual) return bQual - aQual;
        // (c) longer names rank higher (more specific tokens)
        if (b[0].length !== a[0].length) return b[0].length - a[0].length;
        // Stable: alphabetical
        return a[0].localeCompare(b[0]);
    });

    return entries.slice(0, topN).map(([name]) => name);
}
