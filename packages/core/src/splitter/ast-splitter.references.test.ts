// reference-edges-from-the-index: a code chunk records the vocabulary-known
// names it references. Until this change `mentioned_symbols` was written only
// by the markdown splitter and was empty on every one of the 365,022 code rows
// in the served index, so "what depends on X" had nothing to read.
//
// Tests the extractor against parsed trees, like its sibling
// `ast-splitter.imports.test.ts` — grammars are imported statically and handed
// to `parseAs`, because the splitter's own dynamic grammar load needs
// `--experimental-vm-modules` that this suite does not run with.

import Parser from 'tree-sitter';
import { collectReferencedSymbols } from './ast-splitter';

import Haxe from 'tree-sitter-haxe';
import Cpp from 'tree-sitter-cpp';

function parseAs(grammar: any, code: string): Parser.SyntaxNode {
    const parser = new Parser();
    parser.setLanguage(grammar);
    return parser.parse(code).rootNode;
}

const vocab = new Set([
    'Bytes', 'alloc', 'ofString', 'Helper', 'compute', 'Widget', 'render', 'runLater', 'go',
    'godotengine.org', 'AUTHORS.md', 'class_db.h', 'ClassDB', 'bind_method',
]);

describe('collectReferencedSymbols', () => {
    it('records a bare call', () => {
        const root = parseAs(Haxe, `class Thing {
    public function go() { compute(1, 2); }
}`);
        expect(collectReferencedSymbols(root, vocab, ['Thing'])).toContain('compute');
    });

    it('records a qualified call and a type named in a signature', () => {
        const root = parseAs(Haxe, `class Thing {
    public function make(w:Widget) { return Bytes.alloc(8); }
}`);
        const got = collectReferencedSymbols(root, vocab, ['Thing']);
        expect(got).toEqual(expect.arrayContaining(['Bytes', 'alloc', 'Widget']));
    });

    it('sees a bare C++ call, which the text extractor never did', () => {
        const root = parseAs(Cpp, `void Node::setup() { bind_method("x"); }`);
        expect(collectReferencedSymbols(root, vocab, ['setup'])).toContain('bind_method');
    });

    it('excludes the chunk own symbol and its enclosing scope', () => {
        const root = parseAs(Haxe, `class Helper {
    public function compute() { return runLater(); }
}`);
        const got = collectReferencedSymbols(root, vocab, ['compute', 'Helper']);
        expect(got).not.toContain('compute');
        expect(got).not.toContain('Helper');
        expect(got).toContain('runLater');
    });

    it('drops a name too short for the pool to ever be asked about', () => {
        // `go` is in the vocabulary and is genuinely called here, but the pool
        // activates only on names of 4 characters or more, so storing it would
        // spend the field's clamp on something no query can reach.
        const root = parseAs(Haxe, `class Thing {
    public function make() { go(); compute(); }
}`);
        const got = collectReferencedSymbols(root, vocab, ['Thing', 'make']);
        expect(got).not.toContain('go');
        expect(got).toContain('compute');
    });

    it('keeps comment URLs, document names and include targets out', () => {
        // These are exactly what the text extractor returned over real files on
        // 2026-09-19: godotengine.org, AUTHORS.md, class_db.h.
        const root = parseAs(Cpp, `/* see https://godotengine.org and AUTHORS.md */
#include "class_db.h"
void f() {}`);
        const got = collectReferencedSymbols(root, vocab, []);
        expect(got).not.toContain('godotengine.org');
        expect(got).not.toContain('AUTHORS.md');
        expect(got).not.toContain('class_db.h');
    });

    it('records nothing without a vocabulary', () => {
        const root = parseAs(Haxe, `class Thing { public function go() { compute(); } }`);
        expect(collectReferencedSymbols(root, undefined, [])).toEqual([]);
        expect(collectReferencedSymbols(root, new Set(), [])).toEqual([]);
    });

    it('excludes a name the vocabulary does not know', () => {
        const root = parseAs(Haxe, `class Thing { public function go() { somethingNobodyDeclared(); } }`);
        expect(collectReferencedSymbols(root, vocab, ['Thing'])).not.toContain('somethingNobodyDeclared');
    });

    it('reports each name once', () => {
        const root = parseAs(Haxe, `class Thing {
    public function go() { compute(); compute(); compute(); }
}`);
        const got = collectReferencedSymbols(root, vocab, ['Thing']);
        expect(got.filter((n) => n === 'compute')).toHaveLength(1);
    });
});
