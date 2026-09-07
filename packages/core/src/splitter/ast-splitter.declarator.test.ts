// C/C++ symbol naming: tree-sitter-cpp gives `function_definition` no `name`
// field — the identifier sits under a `declarator` chain. Without the walk,
// every C++ function was indexed with symbol_name undefined and ast-splitter's
// loose identifier fallback stored the RETURN TYPE instead. Measured against
// clangd on Godot/cocos sources: 0.02 name recall before, 1.00 after.
//
// These cases parse real C++ with the deployed `tree-sitter-cpp` (#123). They
// used to be hand-built stub nodes, written that way while #75 was live —
// `tree-sitter/index.js` is not idempotent, and a second execution of it in one
// jest process left `tree.rootNode` undefined for every parse, so only one test
// file per worker could parse at all. `tree-sitter-registry-guard.ts` (wired as
// jest `setupFiles`) reinstalls the first execution's prototype patches before
// every test file, so a real parse is deterministic again. A stub asserts the
// author's belief about a third-party grammar; only a real parse notices when
// the next `tree-sitter-cpp` bump changes the chain.

import Parser from 'tree-sitter';
import { declaratorName } from './ast-splitter';

const Cpp = require('tree-sitter-cpp');

// One translation unit holding every shape, so the fixtures stay readable as
// C++ rather than as a list of node types.
const SOURCE = `
StringName _global_enums(int p_index) { return StringName(); }
int CoreConstants::get_global_constant_count() { return 0; }
char *make_buffer(int n) { return nullptr; }
class C { public: bool is_global_constant(const String &n); };
`;

let _parser: Parser | null = null;
function cppParser(): Parser {
    if (_parser) return _parser;
    const p = new Parser();
    p.setLanguage(Cpp);
    _parser = p;
    return p;
}

function parsed(): Parser.SyntaxNode {
    const tree = cppParser().parse(SOURCE);
    expect(tree.rootNode).toBeDefined();
    expect(tree.rootNode.hasError).toBe(false);
    return tree.rootNode;
}

/** Every node of one of `types`, depth-first, in source order. */
function collect(node: Parser.SyntaxNode, types: string[], acc: Parser.SyntaxNode[] = []): Parser.SyntaxNode[] {
    if (types.includes(node.type)) acc.push(node);
    for (const child of node.children) collect(child, types, acc);
    return acc;
}

/** The declaration whose text starts with `prefix`. */
function declaration(prefix: string): Parser.SyntaxNode {
    const found = collect(parsed(), ['function_definition', 'field_declaration', 'class_specifier'])
        .find(n => n.text.startsWith(prefix));
    if (!found) throw new Error(`no declaration starting with ${JSON.stringify(prefix)}`);
    return found;
}

describe('declaratorName — C/C++ declarator chain, real tree-sitter-cpp parse', () => {
    it('reaches the identifier through function_declarator', () => {
        const node = declaration('StringName _global_enums');
        // function_definition → function_declarator → identifier
        expect(node.childForFieldName('declarator')!.type).toBe('function_declarator');
        expect(declaratorName(node)).toBe('_global_enums');
    });

    it('keeps the qualifier of an out-of-line definition', () => {
        const node = declaration('int CoreConstants::');
        // function_definition → function_declarator → qualified_identifier,
        // whose own text carries the whole `A::b`.
        expect(declaratorName(node)).toBe('CoreConstants::get_global_constant_count');
    });

    it('walks nested pointer declarators', () => {
        const node = declaration('char *make_buffer');
        // function_definition → pointer_declarator → function_declarator → identifier
        expect(node.childForFieldName('declarator')!.type).toBe('pointer_declarator');
        expect(declaratorName(node)).toBe('make_buffer');
    });

    it('names a member declaration via field_identifier', () => {
        const node = declaration('bool is_global_constant');
        expect(node.type).toBe('field_declaration');
        expect(declaratorName(node)).toBe('is_global_constant');
    });

    it('returns undefined when the node has no declarator', () => {
        const node = declaration('class C {');
        expect(node.type).toBe('class_specifier');
        expect(node.childForFieldName('declarator')).toBeNull();
        expect(declaratorName(node)).toBeUndefined();
    });

    it('gives up instead of looping on a cyclic declarator chain', () => {
        // Stays a hand-built node deliberately: a real parse cannot produce a
        // cycle, and the depth bound in declaratorName exists precisely for a
        // node that did not come from one.
        const cyclic: any = { type: 'pointer_declarator', text: '' };
        cyclic.childForFieldName = (f: string) => (f === 'declarator' ? cyclic : null);
        expect(declaratorName(cyclic)).toBeUndefined();
    });
});
