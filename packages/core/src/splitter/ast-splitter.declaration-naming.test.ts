// name-declarations-and-refresh-the-vocabulary: a declaration wrapped by the grammar is
// still a declaration, and its name has to reach `symbol_name`.
//
// Why this is not cosmetic: an unnamed declaration never enters the symbol vocabulary, the
// vocabulary is what `collectReferencedSymbols` filters through, so the symbol's edges are
// missing from EVERY chunk in the index and the symbol-references pool cannot find its
// declaration either. Measured 2026-09-20 on the served index: 100% of OCaml's 24,340 rows,
// 25.6% of TypeScript's 35,783 and 21.8% of JavaScript's 41,286 carried no name.
//
// Grammars are imported statically and handed to `parseAs`, like
// `ast-splitter.references.test.ts` beside it — the splitter's own grammar load is dynamic
// and needs `--experimental-vm-modules`, which this suite does not run with.

import Parser from 'tree-sitter';
import { AstCodeSplitter, collectReferencedSymbols, declarationPayload, symbolNameFor } from './ast-splitter';
import { astSupportedLanguages, getSplittableTypes, isAstSupported } from './grammar-registry';

// "Does this payload emit its own chunk?" is a per-grammar question, so the walk takes the
// language's own splittable list — the same one the splitter's `emit` reads.
const TS_SPLITTABLE = new Set(getSplittableTypes('typescript')!);
const OCAML_SPLITTABLE = new Set(getSplittableTypes('ocaml')!);

import Cpp from 'tree-sitter-cpp';
import Haxe from 'tree-sitter-haxe';
import TypeScriptGrammars from 'tree-sitter-typescript';
import OCamlGrammars from 'tree-sitter-ocaml';

// Both packages ship several grammars under one entry point.
const TypeScript = (TypeScriptGrammars as any).typescript;
const OCaml = (OCamlGrammars as any).ocaml;

function parseAs(grammar: any, code: string): Parser.SyntaxNode {
    const parser = new Parser();
    parser.setLanguage(grammar);
    return parser.parse(code).rootNode;
}
const firstNamed = (root: Parser.SyntaxNode) => root.namedChildren[0];

describe('symbolNameFor — declarations the grammar wraps', () => {
    it('names an exported abstract class, which no chunk carried before', () => {
        // `export abstract class PoseNode` is the shape that put `symbol_name: null` on the
        // cocos rows: `export_statement` is the splittable node, and it has no name of its
        // own while its methods were named all along.
        const root = parseAs(TypeScript, 'export abstract class PoseNode extends Base { evaluate() {} }');
        expect(symbolNameFor(firstNamed(root))).toBe('PoseNode');
    });

    it('names an exported abstract class and an exported enum — the declarations that emit nothing of their own', () => {
        // `abstract_class_declaration` and `enum_declaration` are not splittable types, so
        // the wrapper's chunk is the ONLY chunk these declarations get. Both corpus
        // subjects this change was measured on — `PoseNode`, `SettingPass` — are
        // `export abstract class`.
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export abstract class SettingPass extends BasePass {}')), 0, TS_SPLITTABLE)).toBe('SettingPass');
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export enum Mode { A }')), 0, TS_SPLITTABLE)).toBe('Mode');
    });

    it('leaves the wrapper of a declaration that emits its own chunk unnamed', () => {
        // `export class X {}` already produced a `class_declaration` chunk named X with its
        // `extends` extracted. Naming the wrapper too would put a second row with the same
        // name beside it, and stamp it `function` — `export_statement` is registered as
        // `fn()` — so the walk stops when the payload is itself a registered declaration.
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export class Plain extends B {}')), 0, TS_SPLITTABLE)).toBeUndefined();
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export interface Iface {}')), 0, TS_SPLITTABLE)).toBeUndefined();
    });

    it('never names a re-export after the symbol it re-exports', () => {
        // `class Foo {…}\n\nexport { Foo };` is the three.js idiom — 1,312 of them in this
        // corpus. Walking into `export_clause > export_specifier` named each one-line
        // re-export `Foo`, giving the real declaration a decoy to compete with in every
        // symbol lookup. `export_clause` is not a registered declaration type, so the walk
        // stops at it; note that it stopped for `export { a, b }` by accident already, and
        // the single-specifier form is the one that mattered.
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export { Foo };')))).toBeUndefined();
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export { internalName as publicName };')))).toBeUndefined();
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export { a, b };')))).toBeUndefined();
    });

    it('never walks into a body and names a declaration after its own member', () => {
        // `export default class { foo() {} }` walked `class > class_body >
        // method_definition` and came back with `foo`: a class chunk named after its only
        // method, and two chunks then claiming `foo`. A wrong name is worse than none — it
        // enters the vocabulary and gates every other chunk's references.
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export default class { foo() {} }')))).toBeUndefined();
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export default { alpha: 1 };')))).toBeUndefined();
    });

    it('does not name a binding the walk cannot justify', () => {
        // `lexical_declaration` is not a registered declaration type, so `export const f`
        // stays as unnamed as it is today. The conservative side of the same rule.
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export const f = () => 1;')))).toBeUndefined();
    });

    it('names an OCaml value binding and a type binding', () => {
        // `value_definition > let_binding > value_name`, `type_definition > type_binding >
        // type_constructor`. Every one of the corpus's 24,340 OCaml rows was unnamed.
        expect(symbolNameFor(firstNamed(parseAs(OCaml, 'let add_class_flag c f = c.cl_flags <- f')), 0, OCAML_SPLITTABLE)).toBe('add_class_flag');
        expect(symbolNameFor(firstNamed(parseAs(OCaml, 'type tclass = { mutable cl_flags : int }')), 0, OCAML_SPLITTABLE)).toBe('tclass');
    });

    it('leaves a declaration it cannot resolve unnamed rather than guessing', () => {
        // Two declarators under one `export const`: there is no single name to take, so the
        // walk stops. An anonymous default export has no name at all.
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export const a = 1, b = 2;')))).toBeUndefined();
        expect(symbolNameFor(firstNamed(parseAs(TypeScript, 'export default function () {}')))).toBeUndefined();
    });

    it('does not move a name the direct rules already resolve', () => {
        // The descent runs only after every existing rule has failed, so C++ declarator
        // chains and Haxe classes keep resolving exactly as before.
        expect(symbolNameFor(firstNamed(parseAs(Cpp, 'void CoreConstants::get_enum_values(int a) { }')))).toBe('CoreConstants::get_enum_values');
        expect(symbolNameFor(firstNamed(parseAs(Haxe, 'class Thing { public function go() {} }')))).toBe('Thing');
    });

    it('the descent is bounded', () => {
        // A guard, not a rule: a pathological wrapper chain costs a bounded walk.
        const deep = parseAs(TypeScript, 'export class Outer {}');
        expect(symbolNameFor(firstNamed(deep), 99)).toBeUndefined();
    });
});

describe('declarationPayload', () => {
    it('is the declaration field when the wrapper has one, else the only named child', () => {
        expect(declarationPayload(firstNamed(parseAs(TypeScript, 'export abstract class X {}')))!.type).toBe('abstract_class_declaration');
        expect(declarationPayload(firstNamed(parseAs(OCaml, 'let f x = x')))!.type).toBe('let_binding');
    });

    it('is nothing when the payload emits its own chunk', () => {
        expect(declarationPayload(firstNamed(parseAs(TypeScript, 'export class X {}')), TS_SPLITTABLE)).toBeUndefined();
    });

    it('is nothing when the only named child is the node\'s own body', () => {
        const anonymousClass = declarationPayload(firstNamed(parseAs(TypeScript, 'export default class { foo() {} }')))!;
        expect(anonymousClass.type).toBe('class');
        expect(declarationPayload(anonymousClass)).toBeUndefined();
    });

    it('is nothing when the node is not a wrapper', () => {
        // Several named children and no `declaration` field: taking one of them would be a
        // guess, and a wrong `symbol_name` is worse than none — it lands in the vocabulary
        // and then gates every other chunk's references.
        const twoDeclarators = declarationPayload(firstNamed(parseAs(TypeScript, 'export const a = 1, b = 2;')))!;
        expect(twoDeclarators.type).toBe('lexical_declaration');
        expect(declarationPayload(twoDeclarators)).toBeUndefined();
    });
});

describe('collectReferencedSymbols — OCaml application callees', () => {
    const vocab = new Set(['add_class_flag', 'set_flag', 'int_of_class_flag', 'CoreType']);

    it('records the function an OCaml expression calls', () => {
        // tree-sitter-ocaml names the callee `value_name` under `value_path`. Without it the
        // corpus's OCaml chunks recorded their CONSTRUCTORS (`constructor_name` was already
        // collected) and never the functions they call: measured 4,134 uncollected
        // `value_name` nodes in `typer.ml` alone.
        const root = parseAs(OCaml, 'let f c flag = add_class_flag c flag');
        expect(collectReferencedSymbols(root, vocab, ['f'])).toContain('add_class_flag');
    });

    it('never records the chunk\'s own name', () => {
        const root = parseAs(OCaml, 'let set_flag flags flag = flags land flag');
        expect(collectReferencedSymbols(root, vocab, ['set_flag'])).not.toContain('set_flag');
    });
});

describe('the supported-language list is the registry', () => {
    it('answers for every language the registry has a grammar for', () => {
        // The hand-written list had drifted: `ocaml` has been in the registry since the Haxe
        // compiler's sources were indexed, and `getSplitterStrategyForLanguage('ocaml')`
        // still reported `langchain` for 24,340 rows the AST splitter had parsed.
        expect(AstCodeSplitter.isLanguageSupported('ocaml')).toBe(true);
        expect(AstCodeSplitter.isLanguageSupported('ml')).toBe(true);
        expect(AstCodeSplitter.getSupportedLanguages().sort()).toEqual(astSupportedLanguages().sort());
        for (const lang of AstCodeSplitter.getSupportedLanguages()) expect(isAstSupported(lang)).toBe(true);
    });

    it('does not claim a language the registry has no grammar for', () => {
        expect(AstCodeSplitter.isLanguageSupported('ruby')).toBe(false);
        expect(AstCodeSplitter.isLanguageSupported('kotlin')).toBe(false);
    });
});
