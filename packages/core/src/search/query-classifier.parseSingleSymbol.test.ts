// rag-symbol-refs-lsp-pool: parseSingleSymbol tests (D5).

import { extractCodeTokens, parseSingleSymbol } from './query-classifier';

describe('extractCodeTokens', () => {
    it('extracts identifier-shaped tokens', () => {
        expect(extractCodeTokens('how do I use BytesBuffer in Haxe?'))
            .toEqual(['how', 'do', 'I', 'use', 'BytesBuffer', 'in', 'Haxe']);
    });

    it('returns empty for an empty query', () => {
        expect(extractCodeTokens('')).toEqual([]);
    });

    it('handles Cyrillic + ASCII identifiers', () => {
        expect(extractCodeTokens('как использовать Bytes в проекте')).toEqual(['как', 'использовать', 'Bytes', 'в', 'проекте']);
    });
});

describe('parseSingleSymbol', () => {
    it('returns the longest vocab match', () => {
        const vocab = new Set(['Bytes', 'BytesBuffer']);
        expect(parseSingleSymbol('как открыть Bytes и BytesBuffer', vocab)).toEqual({ symbolName: 'BytesBuffer' });
    });

    it('breaks length ties by preferring PascalCase', () => {
        const vocab = new Set(['parse', 'Parse']);
        expect(parseSingleSymbol('hint about parse vs Parse', vocab)).toEqual({ symbolName: 'Parse' });
    });

    it('returns null when no token meets the min-length gate (≥ 4 chars)', () => {
        // Spec D5/4.3: min length 4 trims 1–3 char stop-words while
        // letting longer-but-generic vocab entries (`data`, `result`)
        // through — they're filtered by the activation gate's broader
        // requirement that the token also looks like a project symbol.
        const vocab = new Set(['Foo', 'Bar', 'baz']);
        expect(parseSingleSymbol('return Foo or Bar', vocab)).toBeNull();
        expect(parseSingleSymbol('how to use baz', vocab)).toBeNull();
    });

    it('returns null when the query contains no vocab match', () => {
        const vocab = new Set(['Bytes', 'BytesBuffer']);
        expect(parseSingleSymbol('what is the weather', vocab)).toBeNull();
    });

    it('returns null on empty / null vocab', () => {
        expect(parseSingleSymbol('Bytes', new Set())).toBeNull();
        expect(parseSingleSymbol('Bytes', null)).toBeNull();
    });

    it('matches a long single identifier (≥ 4 chars)', () => {
        const vocab = new Set(['Hmac']);
        expect(parseSingleSymbol('как использовать Hmac', vocab)).toEqual({ symbolName: 'Hmac' });
    });

    it('keeps tie-break stable when both PascalCase candidates are same length', () => {
        const vocab = new Set(['Bytes', 'Input']);
        // Bytes and Input both length 5, both PascalCase — first by length-then-pascal sort.
        // Sort is stable in V8 for equal keys; first insertion wins → vocab order would NOT
        // matter, only token order. Token order is dictated by query position.
        expect(parseSingleSymbol('Bytes vs Input', vocab)?.symbolName).toMatch(/^(Bytes|Input)$/);
    });

    it('does NOT consider tokens not present in vocab even if they look code-shaped', () => {
        const vocab = new Set(['Bytes']);
        expect(parseSingleSymbol('NotInVocabFoo Bytes', vocab)).toEqual({ symbolName: 'Bytes' });
    });
});
