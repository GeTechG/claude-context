// rag-graph-abstract-typedef-edges: dedicated tests for the Haxe
// `abstract` / `typedef` relation extractors. Mirrors the structure of
// ast-structural-extractor.test.ts: mock-node AST-walk tests for the happy
// paths, childless-node tests for the regex fallback (that is what a chunk node
// looks like when the walk yields nothing), and a real-grammar block at the
// bottom that pins the mocks to what `tree-sitter-haxe@0.4.6` actually emits.
//
// This file used to say the grammar's emission was "broken by `@:meta`
// prefixes". It is not (#123): meta entries come out as MetaDataEntry siblings
// and the AbstractType beside them is intact — see the real-parse test for
// abstract-bare-meta-form.hx.

import * as fs from 'fs';
import * as path from 'path';
import Parser from 'tree-sitter';
import {
    extractAbstractHaxe,
    extractTypedefHaxe,
    extractTypeRelations,
} from './ast-structural-extractor';

const Haxe = require('tree-sitter-haxe');

const FIXTURES_DIR = path.join(__dirname, '__fixtures__');

function readFixture(name: string): string {
    return fs.readFileSync(path.join(FIXTURES_DIR, name), 'utf-8');
}

type FakeNode = {
    type: string;
    text: string;
    children: FakeNode[];
};
function fakeNode(type: string, text: string, children: FakeNode[] = []): FakeNode {
    return { type, text, children };
}

let _haxeParser: Parser | null = null;
function haxeParser(): Parser {
    if (_haxeParser) return _haxeParser;
    const p = new Parser();
    p.setLanguage(Haxe);
    _haxeParser = p;
    return p;
}

/** First node of `type` in a real parse, depth-first. */
function findFirstByType(node: Parser.SyntaxNode, type: string): Parser.SyntaxNode | null {
    if (node.type === type) return node;
    // No falsy-child guard here on purpose: children are never null, and
    // skipping falsy ones would hide exactly the regression #75 was about.
    for (const child of node.children) {
        const found = findFirstByType(child, type);
        if (found) return found;
    }
    return null;
}

describe('extractAbstractHaxe — AST walk (parens-form)', () => {
    it('emits underlying + from + to type names for `abstract Bytes(BytesData) from Array<UInt8> to BytesData`', () => {
        // Mirrors the deployed tree-sitter-haxe AbstractType emission.
        const node = fakeNode('AbstractType', '', [
            fakeNode('abstract', 'abstract'),
            fakeNode('type_name', 'Bytes'),
            fakeNode('(', '('),
            fakeNode('ComplexType', 'BytesData', [
                fakeNode('TypePath', 'BytesData', [fakeNode('type_name', 'BytesData')]),
            ]),
            fakeNode(')', ')'),
            fakeNode('from', 'from'),
            fakeNode('TypePath', 'Array<UInt8>', [fakeNode('type_name', 'Array')]),
            fakeNode('to', 'to'),
            fakeNode('TypePath', 'BytesData', [fakeNode('type_name', 'BytesData')]),
            fakeNode('{', '{'),
        ]);
        const out = extractAbstractHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.abstract_underlying).toBeDefined();
        const set = new Set(out.abstract_underlying);
        expect(set.has('BytesData')).toBe(true);
        expect(set.has('Array')).toBe(true);
        // Duplicate BytesData (underlying + to) collapses to one entry.
        expect(out.abstract_underlying!.length).toBe(2);
    });

    it('emits underlying for parens-form abstract with no from/to', () => {
        const node = fakeNode('AbstractType', '', [
            fakeNode('abstract', 'abstract'),
            fakeNode('type_name', 'MyAbs'),
            fakeNode('(', '('),
            fakeNode('ComplexType', 'Int', [
                fakeNode('TypePath', 'Int', [fakeNode('type_name', 'Int')]),
            ]),
            fakeNode(')', ')'),
            fakeNode('{', '{'),
        ]);
        const out = extractAbstractHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.abstract_underlying).toEqual(['Int']);
    });

    it('emits generic underlying with package prefix normalized to bare name', () => {
        // `abstract Foo<T>(haxe.ds.IntMap<T>) from haxe.ds.IntMap<T> {}`
        const node = fakeNode('AbstractType', '', [
            fakeNode('abstract', 'abstract'),
            fakeNode('type_name', 'Foo'),
            fakeNode('<', '<'),
            fakeNode('TypeParameter', 'T'),
            fakeNode('>', '>'),
            fakeNode('(', '('),
            fakeNode('ComplexType', 'haxe.ds.IntMap<T>', [
                fakeNode('TypePath', 'haxe.ds.IntMap<T>', [
                    fakeNode('type_name', 'IntMap'),
                ]),
            ]),
            fakeNode(')', ')'),
            fakeNode('from', 'from'),
            fakeNode('TypePath', 'haxe.ds.IntMap<T>', [fakeNode('type_name', 'IntMap')]),
            fakeNode('{', '{'),
        ]);
        const out = extractAbstractHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.abstract_underlying).toEqual(['IntMap']);
    });
});

describe('extractAbstractHaxe — regex fallback (bare-form / @:meta)', () => {
    it('recovers from/to for `@:coreType @:notNull abstract Single to Float from Float {}`', () => {
        const src = readFixture('abstract-bare-meta-form.hx');
        const node = fakeNode('AbstractType', src, []);
        const out = extractAbstractHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.abstract_underlying).toBeDefined();
        expect(out.abstract_underlying).toEqual(['Float']);
    });

    it('recovers underlying + from + to for parens-form with meta when AST gives nothing', () => {
        const src = `@:forward
@:notNull
abstract Wrapper<T>(Inner<T>) from Inner<T> to Outer<T> {
    public var size:Int;
}`;
        const node = fakeNode('AbstractType', src, []);
        const out = extractAbstractHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.abstract_underlying).toBeDefined();
        const set = new Set(out.abstract_underlying);
        expect(set.has('Inner')).toBe(true);
        expect(set.has('Outer')).toBe(true);
    });

    it('returns empty for `@:coreType abstract Void {}` (no relations at all)', () => {
        const node = fakeNode('AbstractType', `@:coreType abstract Void {}`, []);
        const out = extractAbstractHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.abstract_underlying).toBeUndefined();
    });
});

describe('extractTypedefHaxe — AST walk', () => {
    it('emits alias for `typedef Null<T> = T`', () => {
        const node = fakeNode('DefType', '', [
            fakeNode('typedef', 'typedef'),
            fakeNode('type_name', 'Null'),
            fakeNode('<', '<'),
            fakeNode('TypeParameter', 'T'),
            fakeNode('>', '>'),
            fakeNode('=', '='),
            fakeNode('ComplexType', 'T', [
                fakeNode('TypePath', 'T', [fakeNode('type_name', 'T')]),
            ]),
        ]);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBe('T');
    });

    it('emits bare-name alias for `typedef Map<K, V> = haxe.ds.IntMap<V>`', () => {
        const node = fakeNode('DefType', '', [
            fakeNode('typedef', 'typedef'),
            fakeNode('type_name', 'Map'),
            fakeNode('=', '='),
            fakeNode('ComplexType', 'haxe.ds.IntMap<V>', [
                fakeNode('TypePath', 'haxe.ds.IntMap<V>', [
                    fakeNode('type_name', 'IntMap'),
                ]),
            ]),
        ]);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBe('IntMap');
    });

    it('skips structural / anonymous-struct aliases (`typedef Iterator<T> = { ... }`)', () => {
        const node = fakeNode('DefType', '', [
            fakeNode('typedef', 'typedef'),
            fakeNode('type_name', 'Iterator'),
            fakeNode('=', '='),
            fakeNode('ComplexType', '{...}', [
                fakeNode('TAnonymous', '{}'),
            ]),
        ]);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBeUndefined();
    });

    it('skips self-referential typedef silently', () => {
        const node = fakeNode('DefType', '', [
            fakeNode('typedef', 'typedef'),
            fakeNode('type_name', 'Foo'),
            fakeNode('=', '='),
            fakeNode('ComplexType', 'Foo<T>', [
                fakeNode('TypePath', 'Foo<T>', [fakeNode('type_name', 'Foo')]),
            ]),
        ]);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBeUndefined();
    });
});

describe('extractTypedefHaxe — regex fallback', () => {
    it('recovers alias when AST yields nothing', () => {
        const src = readFixture('typedef-name-alias.hx');
        const node = fakeNode('DefType', src, []);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBe('Iterator');
    });

    it('skips self-reference via regex too', () => {
        const src = readFixture('typedef-self-reference.hx');
        const node = fakeNode('DefType', src, []);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBeUndefined();
    });

    it('skips structural alias via regex', () => {
        const src = `typedef Anon = { foo:Int, bar:String };`;
        const node = fakeNode('DefType', src, []);
        const out = extractTypedefHaxe(node as unknown as Parser.SyntaxNode);
        expect(out.typedef_alias).toBeUndefined();
    });
});

describe('extractTypeRelations dispatch', () => {
    it('returns empty for non-Haxe languages', () => {
        const node = fakeNode('class_declaration', 'class X {}', []);
        expect(extractTypeRelations(node as unknown as Parser.SyntaxNode, 'typescript', 'class')).toEqual({});
        expect(extractTypeRelations(node as unknown as Parser.SyntaxNode, 'java', 'class')).toEqual({});
    });

    it('routes abstract symbol_kind to extractAbstractHaxe', () => {
        const node = fakeNode('AbstractType', `abstract A(B) {}`, []);
        const out = extractTypeRelations(node as unknown as Parser.SyntaxNode, 'haxe', 'abstract');
        expect(out.abstract_underlying).toEqual(['B']);
    });

    it('routes typedef symbol_kind to extractTypedefHaxe', () => {
        const node = fakeNode('DefType', `typedef A = B;`, []);
        const out = extractTypeRelations(node as unknown as Parser.SyntaxNode, 'haxe', 'typedef');
        expect(out.typedef_alias).toBe('B');
    });

    it('returns empty for unrelated symbol_kind', () => {
        const node = fakeNode('ClassType', 'class X {}', []);
        expect(extractTypeRelations(node as unknown as Parser.SyntaxNode, 'haxe', 'class')).toEqual({});
    });
});

// Real-grammar assertions. These could not be written while #75 was live: a
// second execution of `tree-sitter/index.js` in one jest process left
// `tree.rootNode` undefined for every parse, so only one test file per worker
// could parse at all. `tree-sitter-registry-guard.ts` (jest `setupFiles`)
// reinstalls the first execution's prototype patches per test file.
describe('extractAbstractHaxe / extractTypedefHaxe — real tree-sitter-haxe parse (#123)', () => {
    it('parens-form abstract: underlying + from + to, deduplicated', () => {
        // abstract Bytes(BytesData) from Array<UInt8> to BytesData
        // — `BytesData` is both the underlying type and the `to` target, and
        // collapses to a single entry.
        const tree = haxeParser().parse(readFixture('abstract-clean-parens-form.hx'));
        expect(tree.rootNode.hasError).toBe(false);
        const node = findFirstByType(tree.rootNode, 'AbstractType');
        expect(node).not.toBeNull();
        expect(extractAbstractHaxe(node!).abstract_underlying).toEqual(['BytesData', 'Array']);
    });

    it('the real grammar wraps from/to targets in ComplexType > TypePath', () => {
        // The mock suite above hangs a bare TypePath off `from` / `to`. The
        // deployed grammar puts a ComplexType in between; assert the real shape
        // so the mocks cannot quietly stop describing the grammar.
        const tree = haxeParser().parse(readFixture('abstract-clean-parens-form.hx'));
        const node = findFirstByType(tree.rootNode, 'AbstractType')!;
        const kinds = node.children.map(c => c.type);
        expect(kinds).toContain('from');
        expect(kinds).toContain('to');
        const fromTarget = node.children[kinds.indexOf('from') + 1];
        expect(fromTarget.type).toBe('ComplexType');
        expect(fromTarget.children[0].type).toBe('TypePath');
        expect(fromTarget.text).toBe('Array<UInt8>');
    });

    it('bare `@:meta` abstract parses cleanly — the meta prefixes are siblings', () => {
        // `@:coreType @:notNull @:runtimeValue abstract Single to Float from Float {}`.
        // The old comment in this file claimed the grammar's emission was broken
        // by `@:meta` prefixes. It is not: they are MetaDataEntry siblings of an
        // intact AbstractType, and the AST walk extracts from it.
        const tree = haxeParser().parse(readFixture('abstract-bare-meta-form.hx'));
        expect(tree.rootNode.hasError).toBe(false);
        expect(tree.rootNode.children.map(c => c.type)).toEqual([
            'MetaDataEntry', 'MetaDataEntry', 'MetaDataEntry', 'AbstractType',
        ]);
        const node = findFirstByType(tree.rootNode, 'AbstractType')!;
        expect(extractAbstractHaxe(node).abstract_underlying).toEqual(['Float']);
    });

    it('typedef alias: the bare name of the aliased type', () => {
        // typedef KeyValueIterator<K, V> = Iterator<{key:K, value:V}>;
        const tree = haxeParser().parse(readFixture('typedef-name-alias.hx'));
        expect(tree.rootNode.hasError).toBe(false);
        const node = findFirstByType(tree.rootNode, 'DefType');
        expect(node).not.toBeNull();
        expect(extractTypedefHaxe(node!).typedef_alias).toBe('Iterator');
    });

    it('a real self-referential typedef yields no alias', () => {
        // typedef Foo<T> = Foo<T>; — a self edge would make the alias index
        // point a symbol at itself, so it must be dropped on the AST path too.
        const tree = haxeParser().parse(readFixture('typedef-self-reference.hx'));
        expect(tree.rootNode.hasError).toBe(false);
        const node = findFirstByType(tree.rootNode, 'DefType')!;
        expect(extractTypedefHaxe(node).typedef_alias).toBeUndefined();
    });
});
