// C/C++ symbol naming: tree-sitter-cpp gives `function_definition` no `name`
// field — the identifier sits under a `declarator` chain. Without the walk,
// every C++ function was indexed with symbol_name undefined and ast-splitter's
// loose identifier fallback stored the RETURN TYPE instead. Measured against
// clangd on Godot/cocos sources: 0.02 name recall before, 1.00 after.
//
// Stub nodes rather than a real parse: node-tree-sitter hands out malformed
// trees when a second suite parses with another grammar in the same
// `jest --runInBand` process (see the `findFirst` note in
// ast-splitter.imports.test.ts), and the shapes below are exactly what
// tree-sitter-cpp produces for these sources.

import { declaratorName } from './ast-splitter';

type Stub = { type: string; text?: string; declarator?: Stub };

/** Minimal SyntaxNode surface: declaratorName only uses type/text/declarator. */
function node(stub: Stub): any {
    return {
        type: stub.type,
        text: stub.text ?? '',
        childForFieldName: (field: string) =>
            field === 'declarator' && stub.declarator ? node(stub.declarator) : null,
    };
}

describe('declaratorName — C/C++ declarator chain', () => {
    it('reaches the identifier through function_declarator', () => {
        // StringName _global_enums(int p_index) { … }
        expect(declaratorName(node({
            type: 'function_definition',
            declarator: { type: 'function_declarator', declarator: { type: 'identifier', text: '_global_enums' } },
        }))).toBe('_global_enums');
    });

    it('keeps the qualifier of an out-of-line definition', () => {
        // int CoreConstants::get_global_constant_count() { … }
        expect(declaratorName(node({
            type: 'function_definition',
            declarator: {
                type: 'function_declarator',
                declarator: { type: 'qualified_identifier', text: 'CoreConstants::get_global_constant_count' },
            },
        }))).toBe('CoreConstants::get_global_constant_count');
    });

    it('walks nested pointer declarators', () => {
        // char *make_buffer(int n) { … }
        expect(declaratorName(node({
            type: 'function_definition',
            declarator: {
                type: 'pointer_declarator',
                declarator: { type: 'function_declarator', declarator: { type: 'identifier', text: 'make_buffer' } },
            },
        }))).toBe('make_buffer');
    });

    it('names a member declaration via field_identifier', () => {
        // class C { public: bool is_global_constant(const String &n); };
        expect(declaratorName(node({
            type: 'field_declaration',
            declarator: {
                type: 'function_declarator',
                declarator: { type: 'field_identifier', text: 'is_global_constant' },
            },
        }))).toBe('is_global_constant');
    });

    it('returns undefined when the node has no declarator', () => {
        expect(declaratorName(node({ type: 'class_specifier', text: 'class C { int x; };' }))).toBeUndefined();
    });

    it('gives up instead of looping on a cyclic declarator chain', () => {
        const cyclic: any = { type: 'pointer_declarator', text: '' };
        cyclic.childForFieldName = (f: string) => (f === 'declarator' ? cyclic : null);
        expect(declaratorName(cyclic)).toBeUndefined();
    });
});
