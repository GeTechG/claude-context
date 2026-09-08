// rag-graph-supertype-extraction-fix: dedicated tests for the rewritten
// Haxe `extends` / `implements` extractor. Probes the actual tree-sitter-haxe
// grammar with the three fixture shapes from design D4 — clean ClassType,
// ERROR-parse with #if/#elseif, and a heritageless class — plus unit tests
// for `normalizeTypeName` covering the spec's strip / reject rules.
//
// Three layers, deliberately: mock nodes exercise the AST walk over shapes that
// are cheap to write down; childless nodes carrying fixture text exercise the
// regex fallback (that is what a chunk node looks like when the walk yields
// nothing); and the end-to-end block at the bottom parses the fixtures with the
// real `tree-sitter-haxe@0.4.6` grammar and asserts the values, so the mocks
// cannot drift away from what the grammar actually emits.

import * as fs from 'fs';
import * as path from 'path';
import Parser from 'tree-sitter';
import { extractClassStructural, normalizeTypeName } from './ast-structural-extractor';

// tree-sitter grammar packages declare their own structural `Language` type,
// which is not assignable to the `Language` that `tree-sitter`'s `setLanguage`
// takes, and this file hands the grammar STRAIGHT to `setLanguage`, so an
// `import` here only compiles behind an `as any`. `require` is typed `any` by
// @types/node, which is why this file builds. (`ast-splitter.imports.test.ts`
// passes its grammars through a helper typed `any`, where the two types are
// never compared — so it imports them and this file cannot.) See #129.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const Haxe = require('tree-sitter-haxe');

const FIXTURES_DIR = path.join(__dirname, '__fixtures__');

function readFixture(name: string): string {
    return fs.readFileSync(path.join(FIXTURES_DIR, name), 'utf-8');
}

// Build a minimal node satisfying the parts of `Parser.SyntaxNode` our extractor
// reads (`type`, `text`, `children`) — enough to state an AST-walk case without
// writing out a whole parse. The real-grammar block at the bottom of this file
// is what keeps these shapes honest.
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
    for (const child of node.children) {
        const found = findFirstByType(child, type);
        if (found) return found;
    }
    return null;
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

});

// The end-to-end block. Until #75 these assertions could not be written: a
// second execution of `tree-sitter/index.js` in one jest process permanently
// captured `undefined` for `Tree.prototype.rootNode`, so exactly one test file
// per worker got a usable parse and file order decided which. The setup file
// `tree-sitter-registry-guard.ts` reinstalls the first execution's prototype
// patches before every test file, which makes a real parse deterministic again.
describe('extractClassStructural — real tree-sitter-haxe parse (#123)', () => {
    it('the deployed grammar emits bare `extends` / `implements` keyword tokens', () => {
        // This is the shape the whole AST walk is built on, so assert it
        // directly rather than only through the extractor's output: a grammar
        // bump that starts wrapping the heritage in ExtendsClause /
        // ImplementsClause nodes has to fail here.
        const tree = haxeParser().parse(readFixture('class-clean-heritage.hx'));
        expect(tree.rootNode).toBeDefined();
        expect(tree.rootNode.hasError).toBe(false);
        const classType = findFirstByType(tree.rootNode, 'ClassType');
        expect(classType).not.toBeNull();
        const kinds = classType!.children.map(c => c.type);
        expect(kinds).toContain('extends');
        expect(kinds.filter(k => k === 'implements')).toHaveLength(2);
        // `haxe.Constraints.IMap<K, V>` comes out as a TypePath whose own
        // children are package_name / type_name / identifier, NOT a single
        // type_name — which is why normalization runs off TypePath.text.
        const firstTypePath = classType!.children.find(c => c.type === 'TypePath');
        expect(firstTypePath!.text).toBe('haxe.ds.BalancedTree<K, V>');
    });

    it('heritage is extracted from the real ClassType node', () => {
        const tree = haxeParser().parse(readFixture('class-clean-heritage.hx'));
        const classType = findFirstByType(tree.rootNode, 'ClassType')!;
        const out = extractClassStructural(classType, 'haxe');
        expect(out.extends).toBe('BalancedTree');
        expect(out.implements).toEqual(['IMap', 'Foo']);
    });

    it('the real ClassType carries the heritage as named fields as well', () => {
        // The extractor reads these fields BEFORE walking siblings, so record
        // what the grammar puts in them. Which route produced a given answer is
        // pinned separately, in the routing suite below — on a real node all
        // three routes agree, so no real-node test can tell them apart.
        const tree = haxeParser().parse(readFixture('class-clean-heritage.hx'));
        const classType = findFirstByType(tree.rootNode, 'ClassType')!;
        expect(classType.childrenForFieldName('extends').map(c => c.text))
            .toEqual(['haxe.ds.BalancedTree<K, V>']);
        expect(classType.childrenForFieldName('implements').map(c => c.text))
            .toEqual(['haxe.Constraints.IMap<K, V>', 'Foo']);
    });

    it('from the module root the heritage comes from the regex, not the walk', () => {
        // Worth stating exactly, because it is easy to assume otherwise: the
        // module root has no heritage fields, its own children are `package` and
        // `ClassType`, and the sibling walk is NOT recursive — so neither AST
        // route sees anything here, and `extractClassHaxe` falls through to the
        // source-text regex. The values are right; the path is the fallback.
        const tree = haxeParser().parse(readFixture('class-clean-heritage.hx'));
        const root = tree.rootNode;
        expect(root.childrenForFieldName('extends')).toEqual([]);
        expect(root.childrenForFieldName('implements')).toEqual([]);
        expect(root.children.map(c => c.type)).toEqual(['package', 'ClassType']);

        const out = extractClassStructural(root, 'haxe');
        expect(out.extends).toBe('BalancedTree');
        expect(out.implements).toEqual(['IMap', 'Foo']);

        // Same node with its source text removed: if either AST route had
        // produced that answer this would still return it. It does not.
        const withoutText = {
            type: root.type,
            text: '',
            children: root.children,
            childrenForFieldName: (field: string) => root.childrenForFieldName(field),
        };
        expect(extractClassStructural(withoutText as unknown as Parser.SyntaxNode, 'haxe')).toEqual({});
    });

    it('a real heritageless class yields nothing — not a spurious edge', () => {
        const tree = haxeParser().parse(readFixture('class-no-heritage.hx'));
        expect(tree.rootNode.hasError).toBe(false);
        const out = extractClassStructural(findFirstByType(tree.rootNode, 'ClassType')!, 'haxe');
        expect(out.extends).toBeUndefined();
        expect(out.implements ?? []).toEqual([]);
    });

    it('a `#if` / `#elseif` class parses cleanly and the AST walk handles it', () => {
        // The fixture is named for an ERROR parse and this file used to say the
        // grammar could not cope with conditional compilation. It can:
        // `tree-sitter-haxe@0.4.6` parses this with hasError=false and folds the
        // `#if` body into a `conditional` child of an otherwise ordinary
        // ClassType, so the heritage is on the AST path and the regex fallback
        // is never reached for it. The fallback is exercised by the childless
        // node in the suite above, which is what actually triggers it.
        const tree = haxeParser().parse(readFixture('class-error-parse-with-conditional.hx'));
        expect(tree.rootNode.hasError).toBe(false);
        const classType = findFirstByType(tree.rootNode, 'ClassType')!;
        expect(classType.children.map(c => c.type)).toContain('conditional');
        const out = extractClassStructural(classType, 'haxe');
        expect(out.extends).toBe('BaseBuffer');
        expect(out.implements).toEqual(['IBuffer']);
    });
});

// Haxe heritage extraction has THREE routes, tried in this order:
//
//   1. the named fields — `childrenForFieldName('extends' | 'implements')`
//   2. the sibling walk — bare `extends` / `implements` keyword tokens among
//      `node.children`, each followed by its TypePath
//   3. the source-text regex, when neither AST route yields anything
//
// On any real node all three agree, which is exactly why none of the tests above
// can tell them apart: forcing route 1 off a real ClassType still returns the
// right answer through route 2, so route 1 — the one production actually takes —
// would go unprotected. The nodes below make the three routes disagree on
// purpose, so removing any one of them reddens exactly one test.
type RouteNode = {
    type: string;
    text: string;
    children: FakeNode[];
    childrenForFieldName?: (field: string) => { text: string }[];
};

const SIBLING_HERITAGE: FakeNode[] = [
    fakeNode('class', 'class'),
    fakeNode('type_name', 'Discriminator'),
    fakeNode('extends', 'extends'),
    fakeNode('TypePath', 'SiblingBase'),
    fakeNode('implements', 'implements'),
    fakeNode('TypePath', 'ISibling'),
    fakeNode('{', '{'),
];
const NO_SIBLING_HERITAGE: FakeNode[] = [
    fakeNode('class', 'class'),
    fakeNode('type_name', 'Discriminator'),
    fakeNode('{', '{'),
];
// Route 3's input. Deliberately a third pair of names.
const REGEX_SOURCE = 'class Discriminator extends TextBase implements IText {\n}\n';

function routeNode(opts: {
    fields?: { extends?: string[]; implements?: string[] };
    children: FakeNode[];
    text: string;
}): Parser.SyntaxNode {
    const node: RouteNode = { type: 'ClassType', text: opts.text, children: opts.children };
    if (opts.fields) {
        const { extends: ext = [], implements: impl = [] } = opts.fields;
        node.childrenForFieldName = (field: string) =>
            (field === 'extends' ? ext : field === 'implements' ? impl : []).map(text => ({ text }));
    }
    return node as unknown as Parser.SyntaxNode;
}

describe('extractClassStructural — the three heritage routes, pinned separately (#123)', () => {
    it('route 1: the named fields win over both the siblings and the text', () => {
        const out = extractClassStructural(routeNode({
            fields: { extends: ['pkg.FieldBase<T>'], implements: ['pkg.IFieldOne', 'IFieldTwo'] },
            children: SIBLING_HERITAGE,
            text: REGEX_SOURCE,
        }), 'haxe');
        expect(out.extends).toBe('FieldBase');
        expect(out.implements).toEqual(['IFieldOne', 'IFieldTwo']);
    });

    it('route 2: with the fields empty, the sibling walk wins over the text', () => {
        const out = extractClassStructural(routeNode({
            fields: { extends: [], implements: [] },
            children: SIBLING_HERITAGE,
            text: REGEX_SOURCE,
        }), 'haxe');
        expect(out.extends).toBe('SiblingBase');
        expect(out.implements).toEqual(['ISibling']);
    });

    it('route 2 is also taken when the node has no field accessor at all', () => {
        // Chunk nodes reaching the extractor are not always full SyntaxNodes.
        const out = extractClassStructural(routeNode({
            children: SIBLING_HERITAGE,
            text: REGEX_SOURCE,
        }), 'haxe');
        expect(out.extends).toBe('SiblingBase');
        expect(out.implements).toEqual(['ISibling']);
    });

    it('route 3: with both AST routes empty, the source-text regex answers', () => {
        const out = extractClassStructural(routeNode({
            fields: { extends: [], implements: [] },
            children: NO_SIBLING_HERITAGE,
            text: REGEX_SOURCE,
        }), 'haxe');
        expect(out.extends).toBe('TextBase');
        expect(out.implements).toEqual(['IText']);
    });

    it('no route yields anything when the node declares no heritage anywhere', () => {
        const out = extractClassStructural(routeNode({
            fields: { extends: [], implements: [] },
            children: NO_SIBLING_HERITAGE,
            text: 'class Discriminator {\n}\n',
        }), 'haxe');
        expect(out.extends).toBeUndefined();
        expect(out.implements ?? []).toEqual([]);
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
