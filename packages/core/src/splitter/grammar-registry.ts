/**
 * Dynamic, data-driven grammar registry for the AST splitter.
 *
 * Adding a language is a one-entry change here (plus its npm dependency and the
 * extension → language mapping in context.ts `getLanguageFromExtension`). The
 * splitter no longer hardcodes `require(...)` calls or per-language maps; it
 * derives everything from `GRAMMARS` below.
 *
 * Grammars are loaded lazily via dynamic `import()` so the registry works with
 * BOTH CommonJS and pure-ESM grammar packages. Newer tree-sitter grammars
 * (e.g. `tree-sitter-c-sharp` >= 0.23) ship as ESM with top-level `await`, which
 * `require()` cannot load ("require() cannot be used on an ESM graph with
 * top-level await"). The `Function` wrapper preserves a real runtime `import()`
 * that tsc's `module: commonjs` target would otherwise down-level to `require()`.
 */

// A real dynamic import that survives tsc's commonjs down-levelling.
const dynamicImport: (specifier: string) => Promise<any> =
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    new Function('specifier', 'return import(specifier)') as any;

export interface NodeKindSpec {
    /** Normalized symbol_kind tag. Omitted for structural-only splittables (e.g. hxml `section`). */
    kind?: string;
    /** Node introduces a symbol scope inherited by descendants as parent_symbol. */
    parentScope?: boolean;
}

export interface GrammarSpec {
    /** Language ids as produced by Context.getLanguageFromExtension (lowercased). */
    langs: string[];
    /** npm package providing the tree-sitter grammar. */
    module: string;
    /** Sub-export within the package (e.g. 'typescript', 'ocaml'); default export when omitted. */
    export?: string;
    /** Splittable node types → symbol metadata. */
    nodes: Record<string, NodeKindSpec>;
}

const fn = (): NodeKindSpec => ({ kind: 'function' });
const method = (): NodeKindSpec => ({ kind: 'method' });
const cls = (): NodeKindSpec => ({ kind: 'class', parentScope: true });
const iface = (): NodeKindSpec => ({ kind: 'interface', parentScope: true });
const enm = (): NodeKindSpec => ({ kind: 'enum', parentScope: true });
const typedef = (): NodeKindSpec => ({ kind: 'typedef' });

export const GRAMMARS: GrammarSpec[] = [
    {
        langs: ['javascript', 'js'],
        module: 'tree-sitter-javascript',
        nodes: {
            function_declaration: fn(), arrow_function: fn(), export_statement: fn(),
            method_definition: method(), class_declaration: cls(),
        },
    },
    {
        langs: ['typescript', 'ts'],
        module: 'tree-sitter-typescript', export: 'typescript',
        nodes: {
            function_declaration: fn(), arrow_function: fn(), export_statement: fn(),
            method_definition: method(), class_declaration: cls(),
            interface_declaration: iface(), type_alias_declaration: typedef(),
        },
    },
    {
        langs: ['python', 'py'],
        module: 'tree-sitter-python',
        nodes: {
            function_definition: fn(), async_function_definition: fn(), decorated_definition: fn(),
            class_definition: cls(),
        },
    },
    {
        langs: ['java'],
        module: 'tree-sitter-java',
        nodes: {
            method_declaration: method(), constructor_declaration: method(),
            class_declaration: cls(), interface_declaration: iface(),
        },
    },
    {
        langs: ['cpp', 'c++', 'c'],
        module: 'tree-sitter-cpp',
        nodes: {
            function_definition: fn(), declaration: fn(),
            class_specifier: cls(), namespace_definition: cls(),
        },
    },
    {
        langs: ['go'],
        module: 'tree-sitter-go',
        nodes: {
            function_declaration: fn(), method_declaration: method(),
            type_declaration: typedef(), var_declaration: typedef(), const_declaration: typedef(),
        },
    },
    {
        langs: ['rust', 'rs'],
        module: 'tree-sitter-rust',
        nodes: {
            function_item: fn(), impl_item: cls(), struct_item: cls(),
            enum_item: enm(), trait_item: iface(), mod_item: cls(),
        },
    },
    {
        langs: ['csharp', 'cs'],
        module: 'tree-sitter-c-sharp',
        nodes: {
            method_declaration: method(), class_declaration: cls(), interface_declaration: iface(),
            struct_declaration: cls(), enum_declaration: enm(),
        },
    },
    {
        langs: ['scala'],
        module: 'tree-sitter-scala',
        nodes: {
            method_declaration: method(), constructor_declaration: method(),
            class_declaration: cls(), interface_declaration: iface(),
        },
    },
    {
        langs: ['haxe', 'hx'],
        module: 'tree-sitter-haxe',
        nodes: {
            ClassType: cls(), EnumType: enm(), AbstractType: { kind: 'abstract', parentScope: true },
            DefType: { kind: 'typedef', parentScope: true }, ClassMethod: method(),
        },
    },
    {
        langs: ['hxml'],
        module: 'tree-sitter-hxml',
        nodes: { section: {} },
    },
    {
        langs: ['ocaml', 'ml'],
        module: 'tree-sitter-ocaml', export: 'ocaml',
        nodes: {
            value_definition: fn(), external: fn(), method_definition: method(),
            type_definition: typedef(), exception_definition: typedef(),
            module_definition: cls(), class_definition: cls(),
            module_type_definition: iface(), class_type_definition: iface(),
        },
    },
    {
        langs: ['ocaml_interface', 'mli'],
        module: 'tree-sitter-ocaml', export: 'ocaml_interface',
        nodes: {
            value_specification: fn(), external: fn(),
            type_definition: typedef(), exception_definition: typedef(),
            module_definition: cls(), class_definition: cls(),
            module_type_definition: iface(), class_type_definition: iface(),
        },
    },
];

// ---- Derived lookups (built once at module load) ----

const specByLang = new Map<string, GrammarSpec>();
const splittableByLang = new Map<string, string[]>();

/** node type → normalized symbol_kind tag, merged across all grammars. */
export const NODE_TYPE_TO_SYMBOL_KIND: Record<string, string> = {};
/** node types that introduce a parent symbol scope, merged across all grammars. */
export const PARENT_SCOPE_NODE_TYPES = new Set<string>();

for (const spec of GRAMMARS) {
    const splittable = Object.keys(spec.nodes);
    for (const lang of spec.langs) {
        specByLang.set(lang, spec);
        splittableByLang.set(lang, splittable);
    }
    for (const [nodeType, meta] of Object.entries(spec.nodes)) {
        if (meta.kind) NODE_TYPE_TO_SYMBOL_KIND[nodeType] = meta.kind;
        if (meta.parentScope) PARENT_SCOPE_NODE_TYPES.add(nodeType);
    }
}

/** Splittable node types for a language id, or null if unsupported by the AST splitter. */
export function getSplittableTypes(language: string): string[] | null {
    return splittableByLang.get(language.toLowerCase()) ?? null;
}

/** True if the AST splitter has a grammar registered for this language id. */
export function isAstSupported(language: string): boolean {
    return specByLang.has(language.toLowerCase());
}

// Loaded tree-sitter Language objects, cached per language id.
const languageCache = new Map<string, any>();

/**
 * Lazily load (and cache) the tree-sitter Language for a language id.
 * Returns null if the language is unsupported or the grammar fails to load.
 */
export async function loadLanguage(language: string): Promise<any | null> {
    const lang = language.toLowerCase();
    if (languageCache.has(lang)) return languageCache.get(lang);

    const spec = specByLang.get(lang);
    if (!spec) return null;

    const mod = await dynamicImport(spec.module);
    // import() of a CJS package puts module.exports on `.default`; an ESM package
    // exposes its `export default` on `.default` too. Sub-exports (e.g.
    // `.typescript`, `.ocaml`) live on the resolved base object.
    const base = mod?.default ?? mod;
    const grammar = spec.export ? (base?.[spec.export] ?? mod?.[spec.export]) : base;
    if (!grammar) {
        throw new Error(`grammar ${spec.module}${spec.export ? '.' + spec.export : ''} resolved to empty`);
    }
    languageCache.set(lang, grammar);
    return grammar;
}
