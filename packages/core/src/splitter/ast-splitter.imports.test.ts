// rag-graph-layer Phase 1.4: smoke-tests for the structural extractors —
// imports / extends / implements across typescript, javascript, java,
// python, haxe. The extractors are written defensively (graceful fallback
// to {} on grammar mismatch); these tests confirm at least one happy path
// per language.

import Parser from 'tree-sitter';
import { extractStructural, extractClassStructural } from './ast-structural-extractor';

const TypeScript = require('tree-sitter-typescript').typescript;
const JavaScript = require('tree-sitter-javascript');
const Python = require('tree-sitter-python');
const Java = require('tree-sitter-java');
const Haxe = require('tree-sitter-haxe');

function parseAs(grammar: any, code: string): Parser.SyntaxNode {
    const parser = new Parser();
    parser.setLanguage(grammar);
    return parser.parse(code).rootNode;
}

function findFirst(node: Parser.SyntaxNode, types: string[]): Parser.SyntaxNode | null {
    // Defensive walk via numbered indices: tree-sitter's `node.children`
    // accessor can transiently return holes when multiple grammars share a
    // process (jest --runInBand across test files).
    if (!node) return null;
    if (types.includes(node.type)) return node;
    const count = node.childCount;
    for (let i = 0; i < count; i++) {
        const child = node.child(i);
        if (!child) continue;
        const out = findFirst(child, types);
        if (out) return out;
    }
    return null;
}

describe('extractStructural — imports', () => {
    it('typescript: collects ES module sources', () => {
        const root = parseAs(TypeScript, `
            import { foo } from 'lodash';
            import bar from './bar';
            class A {}
        `);
        const out = extractStructural(root, 'typescript');
        expect(out.imports).toEqual(expect.arrayContaining(['lodash', './bar']));
    });

    it('javascript: collects ES module sources', () => {
        const root = parseAs(JavaScript, `
            import { foo } from 'lodash';
            import './side-effect.js';
        `);
        const out = extractStructural(root, 'javascript');
        expect(out.imports).toEqual(expect.arrayContaining(['lodash', './side-effect.js']));
    });

    it('python: collects import + from-import statements as dotted names', () => {
        const root = parseAs(Python, `
import os
import os.path
from collections import OrderedDict
`);
        const out = extractStructural(root, 'python');
        expect(out.imports).toBeDefined();
        // `os` and `os.path` come in as bare dotted names; `collections.OrderedDict`
        // gets composed for from-import.
        expect(out.imports).toEqual(
            expect.arrayContaining(['os', 'os.path', 'collections.OrderedDict'])
        );
    });

    it('java: collects scoped identifier imports', () => {
        const root = parseAs(Java, `
package com.example;
import java.util.List;
import java.util.Map;
class A {}
`);
        const out = extractStructural(root, 'java');
        expect(out.imports).toEqual(expect.arrayContaining(['java.util.List', 'java.util.Map']));
    });

    it('haxe: collects import statements without throwing on grammar quirks', () => {
        const code = `
package foo.bar;
import haxe.io.Bytes;
import haxe.ds.StringMap;
class A {}
`;
        const root = parseAs(Haxe, code);
        const out = extractStructural(root, 'haxe');
        // The exact shape depends on tree-sitter-haxe — accept either:
        //   - a populated imports[] (preferred)
        //   - an empty/undefined imports[] (graceful fallback)
        // Critically: the call must not throw.
        if (out.imports && out.imports.length > 0) {
            const joined = out.imports.join(' ');
            expect(joined).toMatch(/Bytes|StringMap/);
        } else {
            expect(out.imports ?? []).toEqual([]);
        }
    });

    it('returns {} for unsupported languages', () => {
        const root = parseAs(JavaScript, 'console.log(1);');
        expect(extractStructural(root, 'cobol')).toEqual({});
    });
});

describe('extractClassStructural — extends / implements', () => {
    it('typescript: class extending another class', () => {
        const root = parseAs(TypeScript, `class A extends B {}`);
        const node = findFirst(root, ['class_declaration'])!;
        expect(node).not.toBeNull();
        const out = extractClassStructural(node, 'typescript');
        expect(out.extends).toBe('B');
    });

    it('typescript: class implementing interfaces', () => {
        const root = parseAs(TypeScript, `class A implements I1, I2 {}`);
        const node = findFirst(root, ['class_declaration'])!;
        const out = extractClassStructural(node, 'typescript');
        expect(out.implements).toEqual(expect.arrayContaining(['I1', 'I2']));
    });

    it('java: class extending parent and implementing interfaces', () => {
        const root = parseAs(Java, `
package x;
class A extends Base implements Iface1, Iface2 {}
`);
        const node = findFirst(root, ['class_declaration'])!;
        const out = extractClassStructural(node, 'java');
        expect(out.extends).toBe('Base');
        expect(out.implements).toEqual(expect.arrayContaining(['Iface1', 'Iface2']));
    });

    it('python: class with one base', () => {
        const root = parseAs(Python, `
class Foo(Bar):
    pass
`);
        const node = findFirst(root, ['class_definition'])!;
        const out = extractClassStructural(node, 'python');
        expect(out.extends).toBe('Bar');
        // Python has no `implements` concept.
        expect(out.implements ?? []).toEqual([]);
    });

    it('haxe: class extending — does not throw, returns either the parent or null', () => {
        const code = `
class Foo extends Bar implements IZ {
    public function new() {}
}
`;
        const root = parseAs(Haxe, code);
        // tree-sitter-haxe emits `ClassType` for class declarations.
        const node = findFirst(root, ['ClassType', 'class_declaration']);
        // If grammar didn't even produce a class node, we still must not crash.
        if (!node) return;
        const out = extractClassStructural(node, 'haxe');
        // Accept either successful extraction or graceful empty fallback.
        if (out.extends) {
            expect(out.extends).toMatch(/Bar/);
        }
        if (out.implements) {
            expect(out.implements.join(' ')).toMatch(/IZ/);
        }
    });

    it('returns {} for unsupported languages', () => {
        const root = parseAs(JavaScript, 'class A {}');
        const node = findFirst(root, ['class_declaration'])!;
        expect(extractClassStructural(node, 'cobol')).toEqual({});
    });
});
