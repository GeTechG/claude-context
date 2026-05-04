// Phase C (rag-code-intent-recall): symbol-aware Milvus filter routing.
//
// When a query parses as a qualified name (Class.method, Foo::bar, Foo/Bar)
// and both components exist in the per-codebase symbol vocabulary, build a
// Milvus filter expression that pins the candidate pool to chunks whose
// metadata matches the symbol exactly. The third pool runs in parallel with
// the existing code/doc pools and is merged through outer weighted RRF with
// a higher pool weight so the deterministic match dominates.
//
// Falls back to a basename LIKE filter when `parent_symbol` is empty in the
// indexed metadata (the AST splitter for that language did not populate it),
// using the language-extension map to constrain the file extension.
//
// Design ref: openspec/changes/rag-code-intent-recall/design.md (D7).

import { ParsedQName } from './query-classifier';

export interface BuildSymbolFilterOptions {
    parsed: ParsedQName;
    vocab?: ReadonlySet<string> | null;
    /** Map className → expected file extension (e.g. 'Std' won't be in here, falls back to '.hx'). */
    languageExtensions?: string[];
}

/**
 * Default file-extension guesses for the basename fallback. Order matters
 * only for documentation — the LIKE filter accepts any of them via OR.
 */
export const DEFAULT_LANGUAGE_EXTENSIONS = [
    '.hx', '.ts', '.tsx', '.js', '.jsx', '.py', '.java',
    '.cs', '.go', '.rs', '.rb', '.swift', '.kt', '.scala',
    '.cpp', '.c', '.h', '.hpp', '.dart', '.php',
];

function escapeMilvusString(s: string): string {
    // Milvus filter expressions use double-quoted strings with backslash escapes.
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

/**
 * Build a Milvus boolean filter expression that targets a qualified-name match.
 * Returns null when the vocabulary disqualifies the query (className unknown)
 * — callers fall back to the existing 2-pool flow.
 */
export function buildSymbolFilter(opts: BuildSymbolFilterOptions): string | null {
    const { parsed } = opts;
    const vocab = opts.vocab ?? null;
    const exts = opts.languageExtensions && opts.languageExtensions.length > 0
        ? opts.languageExtensions
        : DEFAULT_LANGUAGE_EXTENSIONS;

    // Vocab gate: only run the third pool when at least the className is
    // known to the index. Without it we have no defensible signal.
    if (vocab && !vocab.has(parsed.className)) {
        return null;
    }
    // If method is also known, this is the strongest signal — anchor on both.
    const methodKnown = vocab ? vocab.has(parsed.methodName) : true;

    const method = escapeMilvusString(parsed.methodName);
    const klass = escapeMilvusString(parsed.className);

    if (methodKnown) {
        // Primary form: parent_symbol + symbol_name. AST splitters that fill
        // both fields will return the canonical chunk(s).
        const exact = `(symbol_name == "${method}" and parent_symbol == "${klass}")`;
        // Fallback form (OR'd): parent_symbol may be unset for some languages,
        // so accept matches by basename of the file. Language extension OR'd.
        const likeClauses = exts
            .map((ext) => `relativePath like "%${klass}${escapeMilvusString(ext)}"`)
            .join(' or ');
        const fallback = `(symbol_name == "${method}" and (${likeClauses}))`;
        return `${exact} or ${fallback}`;
    }

    // Method unknown: degrade to "any chunk in a file matching the className
    // basename". This still narrows the pool dramatically.
    const likeClauses = exts
        .map((ext) => `relativePath like "%${klass}${escapeMilvusString(ext)}"`)
        .join(' or ');
    return `(${likeClauses})`;
}
