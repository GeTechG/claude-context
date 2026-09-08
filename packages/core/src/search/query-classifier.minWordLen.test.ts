// MIN_WORD_LEN is read by the code that claims to read it (local-rag #129).
//
// The constant used to be dead: the comment above it said the natural-language
// word filter used it, while NL_WORD hard-coded `{3,}`. Nothing would have
// noticed if the two had disagreed — editing the constant changed nothing, and
// editing the regex left the constant lying. #129 builds NL_WORD from the
// constant, which is behaviour-preserving only because at 3 the constructed
// source is character-for-character the literal it replaced.
//
// So the boundary is pinned here, from outside: a word one character shorter
// than MIN_WORD_LEN does not count towards the doc signal, and a word exactly
// MIN_WORD_LEN long does. Move the constant and these fail — which is the whole
// point of a constant the code actually reads.

import { classifyQuery } from './query-classifier';

describe('the natural-language word filter is bounded by MIN_WORD_LEN (#129)', () => {
    it('counts words of exactly the minimum length', () => {
        // three plain words, each exactly 3 characters — MIN_WORDS_FOR_DOC of them
        expect(classifyQuery('the cat sat').docSignal).toBe(true);
    });

    it('does not count words one character shorter', () => {
        // three plain words, each 2 characters: below the floor, so none counts
        expect(classifyQuery('to be or').docSignal).toBe(false);
    });

    it('still needs enough of them, so the floor is a length and not a count', () => {
        expect(classifyQuery('the cat').docSignal).toBe(false);
    });
});
