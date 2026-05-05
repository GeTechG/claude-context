// rag-graph-layer Phase 1.5: tests for `mentioned_symbols` extraction in
// markdown / code_example chunks, plus optional vocab gating.

import { MarkdownSplitter } from './markdown-splitter';
import { extractMentionedSymbolsFromText } from '../enrichment/symbol-extractor';

describe('extractMentionedSymbolsFromText', () => {
    it('extracts qualified names from prose', () => {
        const out = extractMentionedSymbolsFromText('See Foo.bar for details.');
        expect(out).toEqual(expect.arrayContaining(['Foo.bar']));
    });

    it('extracts code-span tokens, stripping call parens', () => {
        const out = extractMentionedSymbolsFromText('Use `Bytes.alloc()` to allocate.');
        expect(out).toEqual(expect.arrayContaining(['Bytes.alloc']));
    });

    it('returns empty array when no code-tokens present', () => {
        const out = extractMentionedSymbolsFromText('This is a plain prose sentence about nothing.');
        expect(out).toEqual([]);
    });

    it('vocab filter keeps known symbols and drops unknown ones', () => {
        const vocab = new Set(['Bytes', 'alloc', 'Foo']);
        const text = 'Call `Bytes.alloc()` not `randomString` or `Foo.bar`.';
        const out = extractMentionedSymbolsFromText(text, vocab);
        // Bytes.alloc → segment Bytes is in vocab → kept.
        // Foo.bar → segment Foo is in vocab → kept.
        // randomString → not in vocab → dropped.
        expect(out).toEqual(expect.arrayContaining(['Bytes.alloc', 'Foo.bar']));
        expect(out).not.toContain('randomString');
    });

    it('deduplicates repeated mentions', () => {
        const out = extractMentionedSymbolsFromText('`Bytes.alloc()` and again `Bytes.alloc()`.');
        const matches = out.filter((s) => s === 'Bytes.alloc');
        expect(matches.length).toBe(1);
    });

    it('drops stopwords and single-character tokens', () => {
        const out = extractMentionedSymbolsFromText('Use `if` or `a` as needed.');
        // `if` is a stopword and `a` is too short — neither should be kept.
        expect(out).toEqual([]);
    });
});

describe('MarkdownSplitter — mentioned_symbols on chunks', () => {
    it('attaches mentioned_symbols to doc chunks', async () => {
        const md = `# Title

See \`Foo.bar()\` and \`Bytes.alloc()\` for the canonical examples.
`;
        const splitter = new MarkdownSplitter();
        const chunks = await splitter.split(md, 'markdown', '/tmp/x.md');
        const doc = chunks.find((c) => c.metadata.content_type === 'doc');
        expect(doc).toBeDefined();
        expect(doc!.metadata.mentioned_symbols).toEqual(
            expect.arrayContaining(['Foo.bar', 'Bytes.alloc']),
        );
    });

    it('attaches mentioned_symbols to code_example chunks', async () => {
        const md = `# Title

\`\`\`haxe
class Demo {
  static function go() {
    Bytes.alloc(8);
  }
}
\`\`\`
`;
        const splitter = new MarkdownSplitter();
        const chunks = await splitter.split(md, 'markdown', '/tmp/y.md');
        const code = chunks.find((c) => c.metadata.content_type === 'code_example');
        expect(code).toBeDefined();
        // Inside fenced blocks, Bytes.alloc appears as a qualified name and
        // should be picked up.
        expect(code!.metadata.mentioned_symbols).toEqual(
            expect.arrayContaining(['Bytes.alloc']),
        );
    });

    it('vocab provider filters chunk-level mentioned_symbols', async () => {
        const md = `# Title

We compare \`Foo.bar()\` with \`unknownSymbol\` here.
`;
        const splitter = new MarkdownSplitter();
        splitter.setMentionedVocabProvider(() => new Set(['Foo']));
        const chunks = await splitter.split(md, 'markdown', '/tmp/z.md');
        const doc = chunks.find((c) => c.metadata.content_type === 'doc');
        expect(doc!.metadata.mentioned_symbols).toEqual(['Foo.bar']);
        // Without the vocab the unknown symbol would be picked up; the gate
        // should drop it.
        expect(doc!.metadata.mentioned_symbols).not.toContain('unknownSymbol');
    });

    it('does not filter when vocab provider returns null', async () => {
        const md = `See \`Foo.bar\` and \`anotherCallable\` here.`;
        const splitter = new MarkdownSplitter();
        splitter.setMentionedVocabProvider(() => null);
        const chunks = await splitter.split(md, 'markdown', '/tmp/q.md');
        const doc = chunks.find((c) => c.metadata.content_type === 'doc');
        expect(doc!.metadata.mentioned_symbols).toEqual(
            expect.arrayContaining(['Foo.bar', 'anotherCallable']),
        );
    });
});
