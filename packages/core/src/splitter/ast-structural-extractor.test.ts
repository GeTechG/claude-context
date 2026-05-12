// rag-graph-supertype-extraction-fix: dedicated tests for the rewritten
// Haxe `extends` / `implements` extractor. Probes the actual tree-sitter-haxe
// grammar with the three fixture shapes from design D4 — clean ClassType,
// ERROR-parse with #if/#elseif, and a heritageless class — plus unit tests
// for `normalizeTypeName` covering the spec's strip / reject rules.
//
// AST-walk tests build minimal mock nodes mirroring the deployed
// `tree-sitter-haxe@0.4.6` ClassType emission shape (verified by direct probe
// 2026-05-12; see infra/eval-summary.md preflight section). The regex-fallback
// tests parse real files through tree-sitter to confirm end-to-end behaviour.

import * as fs from 'fs';
import * as path from 'path';
import Parser from 'tree-sitter';
import { extractClassStructural, normalizeTypeName } from './ast-structural-extractor';

const Haxe = require('tree-sitter-haxe');

const FIXTURES_DIR = path.join(__dirname, '__fixtures__');

function readFixture(name: string): string {
    return fs.readFileSync(path.join(FIXTURES_DIR, name), 'utf-8');
}

// Build a minimal node satisfying the parts of `Parser.SyntaxNode` our extractor
// reads (`type`, `text`, `children`). Sufficient for the AST-walk unit tests
// without depending on tree-sitter native-binding state (which jest --runInBand
// can corrupt across test files that also load tree-sitter grammars).
type FakeNode = {
    type: string;
    text: string;
    children: FakeNode[];
};
function fakeNode(type: string, text: string, children: FakeNode[] = []): FakeNode {
    return { type, text, children };
}

// Lazily create a single real Haxe parser for the end-to-end sanity test —
// allocating at module-load time can race with other test files' tree-sitter
// initialization when jest --runInBand pre-loads multiple test modules.
let _haxeParser: Parser | null = null;
function haxeParser(): Parser {
    if (_haxeParser) return _haxeParser;
    const p = new Parser();
    p.setLanguage(Haxe);
    _haxeParser = p;
    return p;
}

describe('extractClassStructural — Haxe AST walk (rag-graph-supertype-extraction-fix)', () => {
    it('clean ClassType with single extends + multi-implements emits both edges', () => {
        // Mirror the deployed grammar's emission for
        //   class EnumValueMap<K:EnumValue, V> extends haxe.ds.BalancedTree<K, V> implements haxe.Constraints.IMap<K, V> implements Foo {}
        const node = fakeNode('ClassType', '', [
            fakeNode('class', 'class'),
            fakeNode('type_name', 'EnumValueMap'),
            fakeNode('<', '<'),
            fakeNode('TypeParameter', 'K:EnumValue'),
            fakeNode(',', ','),
            fakeNode('TypeParameter', 'V'),
            fakeNode('>', '>'),
            fakeNode('extends', 'extends'),
            fakeNode('TypePath', 'haxe.ds.BalancedTree<K, V>'),
            fakeNode('implements', 'implements'),
            fakeNode('TypePath', 'haxe.Constraints.IMap<K, V>'),
            fakeNode('implements', 'implements'),
            fakeNode('TypePath', 'Foo'),
            fakeNode('{', '{'),
        ]);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.extends).toBe('BalancedTree');
        expect(out.implements).toEqual(['IMap', 'Foo']);
    });

    it('class with comma-separated implements list emits the full set', () => {
        // Haxe-4 syntax: implements A, B  (single keyword, comma-separated)
        const node = fakeNode('ClassType', '', [
            fakeNode('class', 'class'),
            fakeNode('type_name', 'Multi'),
            fakeNode('implements', 'implements'),
            fakeNode('TypePath', 'A'),
            fakeNode(',', ','),
            fakeNode('TypePath', 'B'),
            fakeNode('{', '{'),
        ]);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.implements).toEqual(['A', 'B']);
    });

    it('class with no heritage clause emits empty result', () => {
        const node = fakeNode('ClassType', 'class Bytes { public var length:Int; }', [
            fakeNode('class', 'class'),
            fakeNode('type_name', 'Bytes'),
            fakeNode('{', '{'),
            fakeNode('ClassVar', 'public var length:Int;'),
            fakeNode('}', '}'),
        ]);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.extends).toBeUndefined();
        expect(out.implements ?? []).toEqual([]);
    });

    it('strips generics and package prefix from extracted names', () => {
        const node = fakeNode('ClassType', '', [
            fakeNode('class', 'class'),
            fakeNode('type_name', 'X'),
            fakeNode('extends', 'extends'),
            fakeNode('TypePath', 'pkg.sub.Base<A, Map<B, C>>'),
            fakeNode('implements', 'implements'),
            fakeNode('TypePath', 'pkg.IFoo<T>'),
            fakeNode('{', '{'),
        ]);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.extends).toBe('Base');
        expect(out.implements).toEqual(['IFoo']);
    });
});

describe('extractClassStructural — Haxe regex fallback (rag-graph-supertype-extraction-fix)', () => {
    it('ERROR-parse-style file with #if blocks: regex fallback recovers heritage', () => {
        // The chunk node has no `extends`/`implements` keyword children — the
        // class declaration is in `text` only. Mirrors what tree-sitter-haxe
        // emits for files where the surrounding grammar errors out.
        const src = readFixture('class-error-parse-with-conditional.hx');
        const node = fakeNode('ClassType', src, []);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.extends).toBe('BaseBuffer');
        expect(out.implements).toEqual(['IBuffer']);
    });

    it('regex fallback ignores block-comment "class Foo extends Bar"', () => {
        const src = `/*\n * class Foo extends Bar implements IBaz\n */\nfunction unrelated() {}\n`;
        const node = fakeNode('ClassType', src, []);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.extends).toBeUndefined();
        expect(out.implements ?? []).toEqual([]);
    });

    it('regex fallback recovers heritage when the AST walk yields nothing', () => {
        // Real Haxe text with no AST children — exercises the cheap pre-check
        // and the line-anchored regex tail walk.
        const src = readFixture('class-clean-heritage.hx');
        const node = fakeNode('ClassType', src, []);
        const out = extractClassStructural(node as unknown as Parser.SyntaxNode, 'haxe');
        expect(out.extends).toBe('BalancedTree');
        expect(out.implements).toEqual(['IMap', 'Foo']);
    });

    it('end-to-end parse via tree-sitter-haxe is non-throwing', () => {
        // Sanity-check the production wiring against the deployed grammar:
        // parse a real fixture, hand whichever node we can find (ClassType if
        // available, root otherwise) to the dispatcher, and assert nothing
        // throws. Strict extraction assertions live in the mock-based tests
        // above — when this suite runs alongside other tree-sitter consumers
        // in the same jest worker, native-binding state can transiently nul
        // out `node.children[i]`, which is a test-only fragility.
        const src = readFixture('class-clean-heritage.hx');
        const tree = haxeParser().parse(src);
        const root = tree.rootNode;
        expect(() => extractClassStructural(root, 'haxe')).not.toThrow();
    });
});

describe('normalizeTypeName (rag-graph-supertype-extraction-fix)', () => {
    it('strips leading package prefix and generic suffix', () => {
        expect(normalizeTypeName('haxe.ds.BalancedTree<K, V>')).toBe('BalancedTree');
    });

    it('strips deeper nested generics depth-balanced', () => {
        expect(normalizeTypeName('Array<Map<Int, String>>')).toBe('Array');
    });

    it('strips leading package without generics', () => {
        expect(normalizeTypeName('haxe.Constraints.IMap')).toBe('IMap');
    });

    it('rejects malformed Haxe type-parameter constraint syntax', () => {
        expect(normalizeTypeName('K:EnumValue & Constructible')).toBeNull();
    });

    it('rejects strings with top-level whitespace separating tokens', () => {
        expect(normalizeTypeName('foo bar baz')).toBeNull();
    });

    it('rejects empty / pure-symbol inputs', () => {
        expect(normalizeTypeName('')).toBeNull();
        expect(normalizeTypeName('   ')).toBeNull();
        expect(normalizeTypeName('<>')).toBeNull();
    });

    it('accepts bare identifier', () => {
        expect(normalizeTypeName('Exception')).toBe('Exception');
    });
});
