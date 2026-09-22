import Parser from 'tree-sitter';
import { Splitter, CodeChunk } from './index';
import { MarkdownSplitter, MentionedVocabProvider } from './markdown-splitter';
import { extractStructural, extractClassStructural, extractTypeRelations } from './ast-structural-extractor';
import { envManager } from '../utils/env-manager';

// Language grammars, splittable node types, symbol-kind mapping and parent-scope
// set all come from the data-driven registry. Adding a language = one entry there.
import {
    astSupportedLanguages,
    getSplittableTypes,
    isAstSupported,
    loadLanguage,
    NODE_TYPE_TO_SYMBOL_KIND,
    PARAMETER_LIST_NODE_TYPES,
    PARENT_SCOPE_NODE_TYPES,
} from './grammar-registry';

// Symbol kinds that declare a type (grammar-registry kinds); inside a function body they stay chunks.
const TYPE_SYMBOL_KINDS = new Set(['class', 'enum', 'interface', 'typedef', 'abstract']);

// Terminal identifier node types at the end of a C/C++ declarator chain.
const DECLARATOR_NAME_TYPES = new Set([
    'identifier', 'field_identifier', 'qualified_identifier',
    'destructor_name', 'operator_name',
]);

// reference-edges-from-the-index: the node types whose text is a name a chunk
// can reference. Deliberately the same shallow list across languages — the point
// is "this chunk names X", not a typed call graph, and the symbol vocabulary is
// what keeps the list from becoming noise.
const REFERENCE_NAME_TYPES = new Set([
    'identifier', 'type_identifier', 'field_identifier', 'qualified_identifier',
    'scoped_identifier', 'property_identifier', 'namespace_identifier',
    'constructor_name', 'class_name',
    // tree-sitter-haxe names a type reference `type_name`, not `type_identifier`
    // (`ComplexType > TypePath > type_name`), so without it a type in a
    // signature — the thing a signature change breaks — was invisible.
    'type_name',
    // name-declarations-and-refresh-the-vocabulary: tree-sitter-ocaml names the
    // callee of an application `value_name` (`application_expression >
    // value_path > value_name`), so every OCaml function call was invisible while
    // `constructor_name` and `class_name` above were collected. Measured
    // 2026-09-20 on `typer.ml`: 4,134 `value_name` nodes uncollected, and the
    // chunk calling `add_class_flag` stored `None`, `WithType`, `TFunction` — its
    // constructors — and not the function it calls.
    'value_name',
]);

// Depth guard, not correctness: a pathological subtree should cost a bounded
// walk, and a chunk is already size-capped by the splitter.
const REFERENCE_SCAN_MAX_NODES = 20000;

// The same floor the symbol-references pool activates on
// (`MIN_SINGLE_SYMBOL_LEN` in query-classifier, `HOP2_SEED_MIN_SYMBOL_LEN` in
// symbol-refs-pool): a name shorter than this can never be the symbol a query
// resolves to, so storing it buys nothing and spends the field's 4096-character
// clamp. Over real files it is what drops `b`, `i`, `pos`, `get` and `new` —
// locals and parameters that happen to be in a 128,562-name vocabulary — from
// the references of `haxe.io.Bytes`.
const REFERENCE_MIN_NAME_LEN = 4;

/**
 * The vocabulary-known names a chunk's own subtree mentions, minus the chunk's
 * own symbol and its enclosing scope, so a chunk never references itself.
 *
 * Read from the AST rather than the chunk text on measured grounds (2026-09-19):
 * `extractMentionedSymbolsFromText` matches qualified dotted names and markdown
 * code spans, so over real source it returns `this.length`, `b.endian`,
 * `#include` targets and the URL in a licence header, and never sees the bare
 * call that is the common case in C++ and Haxe.
 */
export function collectReferencedSymbols(
    node: Parser.SyntaxNode,
    vocabulary: ReadonlySet<string> | null | undefined,
    exclude: ReadonlyArray<string | undefined>,
): string[] {
    if (!vocabulary || vocabulary.size === 0) return [];
    const skip = new Set(exclude.filter((n): n is string => Boolean(n)));
    const seen = new Set<string>();
    const out: string[] = [];
    let budget = REFERENCE_SCAN_MAX_NODES;
    const walk = (cur: Parser.SyntaxNode) => {
        if (budget-- <= 0) return;
        if (REFERENCE_NAME_TYPES.has(cur.type)) {
            const name = cur.text;
            if (name && name.length >= REFERENCE_MIN_NAME_LEN
                && !skip.has(name) && !seen.has(name) && vocabulary.has(name)) {
                seen.add(name);
                out.push(name);
            }
        }
        for (const child of cur.children) walk(child);
    };
    walk(node);
    return out;
}

/**
 * Resolve the identifier hiding under a C/C++ `declarator` chain, e.g.
 * `function_definition` → `function_declarator` → `qualified_identifier`.
 *
 * tree-sitter-cpp gives `function_definition` NO `name` field, so without this
 * every C/C++ function was stored with symbol_name = undefined — and the loose
 * identifier scan in extractSymbolName picked up the RETURN TYPE instead
 * (`StringName _global_enums(...)` was indexed as the symbol "StringName").
 * Pointer/reference/array/init declarators nest, hence the walk.
 */
/**
 * keep-doc-comments-with-their-declarations: `CODE_CHUNK_LEADING_COMMENTS=on` extends a code
 * chunk over the comments written for its declaration. `off` (the default) is the splitter
 * as it was, byte for byte.
 */
export function leadingCommentsEnabled(): boolean {
    return (envManager.get('CODE_CHUNK_LEADING_COMMENTS') || '').trim().toLowerCase() === 'on';
}

/**
 * The first of the comments that directly document `node`, or null. A comment is the node's
 * preceding sibling (or precedes such a comment), is a comment by its GRAMMAR node type —
 * every grammar here names them `*comment*`, so there is no language table — begins its own
 * line (a trailing `x(); // note` belongs to the statement it ends), and has no blank line
 * between it and what follows. A license header above a blank line therefore stays out.
 *
 * ponytail: stops at annotations/metadata between a comment and its declaration
 * (`@:keep`, `@Override`); skipping those is the upgrade if the corpus measurement says
 * they hide a large share of doc comments.
 */
export function firstLeadingComment(node: Parser.SyntaxNode, code: string): Parser.SyntaxNode | null {
    let first: Parser.SyntaxNode | null = null;
    let below: Parser.SyntaxNode = node;
    for (let prev = node.previousSibling; prev && prev.type.includes('comment'); prev = prev.previousSibling) {
        if (below.startPosition.row - prev.endPosition.row > 1) break;
        const lineStart = code.lastIndexOf('\n', prev.startIndex - 1) + 1;
        if (code.slice(lineStart, prev.startIndex).trim() !== '') break;
        first = prev;
        below = prev;
    }
    return first;
}

export function declaratorName(node: Parser.SyntaxNode): string | undefined {
    let cur: Parser.SyntaxNode | null = (node as any).childForFieldName?.('declarator') ?? null;
    for (let depth = 0; cur && depth < 8; depth++) {
        if (DECLARATOR_NAME_TYPES.has(cur.type)) return cur.text || undefined;
        cur = (cur as any).childForFieldName?.('declarator') ?? null;
    }
    return undefined;
}

// name-declarations-and-refresh-the-vocabulary: how far `extractSymbolName` may
// step through wrapper nodes. Three covers `export_statement > lexical_declaration
// > variable_declarator`, the deepest wrapper chain in this corpus; the bound is a
// guard, not a rule.
const WRAPPER_DESCENT_LIMIT = 3;

/**
 * The declaration a wrapper node wraps, or undefined when the node is not a
 * wrapper. A wrapper names nothing itself and hands its whole payload to one
 * child: `export …` exposes it as the `declaration` field, OCaml's
 * `value_definition` / `type_definition` as their only named child.
 *
 * Deliberately structural — no grammar and no language is named — so a grammar
 * added to the registry later is covered without an entry here.
 *
 * TWO GUARDS, both there because a WRONG name is worse than none: it enters the
 * symbol vocabulary and then gates every other chunk's references.
 *
 *  - the single-named-child step is taken only from a node the grammar REGISTRY
 *    knows as a declaration. Without it the walk followed `export { a };` into
 *    `export_clause > export_specifier` and named that one-line re-export after
 *    the class it re-exports — 1,312 of them in this corpus, each a decoy row
 *    competing with the real declaration. `export_statement` is registered so
 *    the first step is allowed; `export_clause` is not, so the walk stops there.
 *  - never step into the node's own `body`. Without it `export default class {
 *    foo() {} }` walked `class > class_body > method_definition` and named the
 *    class after its only method.
 *
 * A third rule, for duplicates rather than for wrong names: the walk stops when the
 * payload is ITSELF a registered declaration type, because that node emits its own chunk
 * and already carries the name, the kind and the `extends`/`implements` the wrapper cannot
 * work out. `export class X {}` therefore leaves the wrapper chunk unnamed exactly as
 * today, and `export abstract class X {}` — whose `abstract_class_declaration` emits
 * nothing, which is why those rows were nameless — is named.
 *
 * What this deliberately does NOT name: `export const thing = 3`, because
 * `lexical_declaration` is not a registered declaration type. That is the
 * conservative side of the same rule — those rows stay exactly as unnamed as
 * they are today rather than risk a name the walk cannot justify.
 */
export function declarationPayload(node: Parser.SyntaxNode, splittable?: ReadonlySet<string>): Parser.SyntaxNode | undefined {
    const body = (node as any).childForFieldName?.('body');
    const notBody = (child: Parser.SyntaxNode | null | undefined): Parser.SyntaxNode | undefined =>
        (child && (!body || child.id !== body.id)) ? child : undefined;
    // "Emits its own chunk" is a property of THIS grammar: `enum_declaration` is splittable
    // in Java and not in TypeScript, and the kind map is global across grammars. With the
    // language's own list the rule is exact; without one (a caller testing the walk in
    // isolation) it falls back to the global map, which refuses more and invents nothing.
    const emitsItsOwnChunk = (child: Parser.SyntaxNode | undefined): Parser.SyntaxNode | undefined => {
        if (!child) return undefined;
        const emits = splittable ? splittable.has(child.type) : (child.type in NODE_TYPE_TO_SYMBOL_KIND);
        return emits ? undefined : child;
    };
    const declared = notBody((node as any).childForFieldName?.('declaration'));
    if (declared) return emitsItsOwnChunk(declared);
    if (!(node.type in NODE_TYPE_TO_SYMBOL_KIND)) return undefined;
    const named = node.namedChildren;
    return named.length === 1 ? emitsItsOwnChunk(notBody(named[0])) : undefined;
}

/**
 * Find the identifier child of a tree-sitter node and return its text.
 *
 * A free function, like `declaratorName` and `collectReferencedSymbols` beside it: the
 * splitter's own grammar load is dynamic and this suite cannot run it, so naming is tested
 * against parsed trees instead.
 * Tree-sitter conventions vary across grammars; we try a few common shapes.
 *
 * `depth` bounds the walk into wrapper nodes (see the end of the method); a
 * caller never passes it.
 */
export function symbolNameFor(node: Parser.SyntaxNode, depth = 0, splittable?: ReadonlySet<string>): string | undefined {
    // Try field names that grammars commonly use for the symbol identifier.
    const fieldCandidates = ['name', 'identifier'];
    for (const field of fieldCandidates) {
        const fieldNode = (node as any).childForFieldName?.(field);
        if (fieldNode && fieldNode.text) {
            return fieldNode.text;
        }
    }

    // C/C++ hide the identifier under a `declarator` chain, and the loose
    // scan below would return the return type — resolve declarators first.
    const declared = declaratorName(node);
    if (declared) return declared;

    // Fall back to the first identifier-shaped direct child.
    for (const child of node.children) {
        if (
            child.type === 'identifier' ||
            child.type === 'type_identifier' ||
            child.type === 'property_identifier' ||
            child.type === 'field_identifier' ||
            child.type === 'name' ||
            child.type === 'IDENTIFIER' ||
            // name-declarations-and-refresh-the-vocabulary: OCaml binds names
            // under `value_name` (`let f x = …`) and `type_constructor`
            // (`type t = …`), which no other grammar here uses, so adding them
            // to this scan cannot rename a declaration another grammar already
            // resolves.
            child.type === 'value_name' ||
            child.type === 'type_constructor'
        ) {
            if (child.text) return child.text;
        }
    }

    // decorated_definition (Python) wraps a function/class — recurse into its inner def.
    if (node.type === 'decorated_definition') {
        for (const child of node.children) {
            if (
                child.type === 'function_definition' ||
                child.type === 'class_definition' ||
                child.type === 'async_function_definition'
            ) {
                return symbolNameFor(child, depth + 1, splittable);
            }
        }
    }

    // name-declarations-and-refresh-the-vocabulary: a grammar may wrap the
    // declaration in a node that carries no name of its own — TS/JS
    // `export_statement > class_declaration`, OCaml `value_definition >
    // let_binding` and `type_definition > type_binding`. The chunk then went
    // in with no `symbol_name`, and an unnamed declaration is not a cosmetic
    // loss: its name never enters the symbol vocabulary, so it is dropped from
    // every other chunk's `mentioned_symbols` too, and the whole index loses
    // that symbol's edges. Measured 2026-09-20 on the served index: 100% of
    // OCaml's 24,340 rows, 25.6% of TypeScript's, 21.8% of JavaScript's.
    //
    // One rule for every grammar: step into the node's `declaration` field if
    // it has one, else into its only named child, and ask again. Reached ONLY
    // when the rules above found nothing, so no name that resolves today moves.
    if (depth < WRAPPER_DESCENT_LIMIT) {
        const payload = declarationPayload(node, splittable);
        if (payload) return symbolNameFor(payload, depth + 1, splittable);
    }

    return undefined;
}

export class AstCodeSplitter implements Splitter {
    private chunkSize: number = 2500;
    private chunkOverlap: number = 300;
    private parser: Parser;
    private langchainFallback: any; // LangChainCodeSplitter for fallback
    private markdownSplitter: MarkdownSplitter;
    private mentionedVocabProvider?: MentionedVocabProvider;

    constructor(chunkSize?: number, chunkOverlap?: number) {
        if (chunkSize) this.chunkSize = chunkSize;
        if (chunkOverlap) this.chunkOverlap = chunkOverlap;
        this.parser = new Parser();

        // Initialize fallback splitter.
        // Stays a `require`: as an import it would emit at module load, moving the
        // evaluation of `langchain/text_splitter` from "an AstCodeSplitter is
        // constructed" to "ast-splitter is loaded" — a different load order and a
        // different failure mode if langchain cannot load. #129.
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { LangChainCodeSplitter } = require('./langchain-splitter');
        this.langchainFallback = new LangChainCodeSplitter(chunkSize, chunkOverlap);
        this.markdownSplitter = new MarkdownSplitter(chunkSize, chunkOverlap);
    }

    async split(code: string, language: string, filePath?: string): Promise<CodeChunk[]> {
        // Markdown / reStructuredText go through a structure-aware splitter that
        // preserves heading_path and breaks fenced code blocks out as code_example.
        const lower = language.toLowerCase();
        if (lower === 'markdown' || lower === 'md' || lower === 'rst' || lower === 'restructuredtext') {
            return this.markdownSplitter.split(code, language, filePath);
        }

        // Check if language is supported by AST splitter
        const langConfig = await this.getLanguageConfig(language);
        if (!langConfig) {
            console.log(`📝 Language ${language} not supported by AST, using LangChain splitter for: ${filePath || 'unknown'}`);
            return await this.langchainFallback.split(code, language, filePath);
        }

        try {
            console.log(`🌳 Using AST splitter for ${language} file: ${filePath || 'unknown'}`);

            this.parser.setLanguage(langConfig.parser);
            // Stream input via callback to bypass tree-sitter@0.21 32KB string limit
            const CHUNK_SIZE = 8 * 1024;
            const parserInput = (index: number) =>
                index >= code.length ? null : code.slice(index, index + CHUNK_SIZE);
            const tree = this.parser.parse(parserInput);

            if (!tree.rootNode) {
                console.warn(`[ASTSplitter] ⚠️  Failed to parse AST for ${language}, falling back to LangChain: ${filePath || 'unknown'}`);
                return await this.langchainFallback.split(code, language, filePath);
            }

            // Extract chunks based on AST nodes
            const chunks = this.extractChunks(tree.rootNode, code, langConfig.nodeTypes, language, filePath);

            // If chunks are too large, split them further
            const refinedChunks = await this.refineChunks(chunks, code);

            return refinedChunks;
        } catch (error) {
            console.warn(`[ASTSplitter] ⚠️  AST splitter failed for ${language}, falling back to LangChain: ${error}`);
            return await this.langchainFallback.split(code, language, filePath);
        }
    }

    setChunkSize(chunkSize: number): void {
        this.chunkSize = chunkSize;
        this.langchainFallback.setChunkSize(chunkSize);
    }

    setChunkOverlap(chunkOverlap: number): void {
        this.chunkOverlap = chunkOverlap;
        this.langchainFallback.setChunkOverlap(chunkOverlap);
    }

    /**
     * rag-graph-layer Phase 1.2: forward the mentioned-symbols vocabulary
     * provider to the embedded MarkdownSplitter so doc / code_example
     * chunks get vocab-filtered `mentioned_symbols[]` at split time.
     */
    setMentionedVocabProvider(provider: MentionedVocabProvider | undefined): void {
        // reference-edges-from-the-index: code chunks need it too. Until this
        // change the provider only reached the markdown splitter, so
        // `mentioned_symbols` was populated for doc / code_example chunks and
        // empty on every one of the 365,022 code rows.
        this.mentionedVocabProvider = provider;
        this.markdownSplitter.setMentionedVocabProvider(provider);
    }

    private async getLanguageConfig(language: string): Promise<{ parser: any; nodeTypes: string[] } | null> {
        const nodeTypes = getSplittableTypes(language);
        if (!nodeTypes) return null;
        const parser = await loadLanguage(language);
        if (!parser) return null;
        return { parser, nodeTypes };
    }

    private extractChunks(
        node: Parser.SyntaxNode,
        code: string,
        splittableTypes: string[],
        language: string,
        filePath?: string
    ): CodeChunk[] {
        const chunks: CodeChunk[] = [];
        const codeLines = code.split('\n');

        // rag-graph-layer Phase 1: file-level imports computed once and
        // attached to every code chunk emitted from this file. Per-symbol
        // extends/implements are computed inline below per node.
        const fileStructural = extractStructural(node, language);
        // reference-edges-from-the-index: resolved once per file, not per chunk.
        const referenceVocab = this.mentionedVocabProvider?.();
        const withLeadingComments = leadingCommentsEnabled();

        // no-chunks-inside-function-bodies: inside a function body only a named type declaration
        // or a named function with a parameter list becomes a chunk — not `int i = 0;`,
        // `let n = … in` or a callback. Types stay because a class the parser misreads as a
        // function (`class CC_DLL Foo {…}`) holds its public enums and nested classes in that body.
        // The same list `isSplittable` reads, as a set, so the naming walk can tell a payload
        // that will emit its own chunk in THIS grammar from one that will not.
        const splittableSet = new Set(splittableTypes);
        const traverse = (currentNode: Parser.SyntaxNode, parentScope?: string, insideBody = false) => {
            const isSplittable = splittableTypes.includes(currentNode.type);
            let scopeForChildren = parentScope;
            const kind = NODE_TYPE_TO_SYMBOL_KIND[currentNode.type];
            const emit = isSplittable && (!insideBody
                || (Boolean(this.extractSymbolName(currentNode, splittableSet)) && (TYPE_SYMBOL_KINDS.has(kind) || hasParameterList(currentNode))));
            // A body is the node's own `body` field. Wrappers (`export …`), declarations holding a
            // type (C++ `field_declaration` → `struct`) and OCaml `let` bindings have none, so OCaml
            // chunks as before: a local `let result = if … in` answered an exact-symbol query (u213).
            const bodyForChildren = insideBody || (isSplittable && (kind === 'function' || kind === 'method') && currentNode.childForFieldName('body') !== null);

            if (emit) {
                // keep-doc-comments-with-their-declarations: only the START moves; everything
                // read off the node below (symbol, kind, structure, references) is the declaration's.
                const lead = withLeadingComments ? firstLeadingComment(currentNode, code) : null;
                const startNode = lead ?? currentNode;
                const startLine = startNode.startPosition.row + 1;
                const endLine = currentNode.endPosition.row + 1;
                const nodeText = code.slice(startNode.startIndex, currentNode.endIndex);

                if (nodeText.trim().length > 0) {
                    const rawSymbolName = this.extractSymbolName(currentNode, splittableSet);
                    // C++ out-of-line definitions carry their class in the name
                    // (`CoreConstants::get_enum_values`). Store the bare name so
                    // lookups work as in every other language, and keep the
                    // qualifier as parent_symbol.
                    const qualifierEnd = rawSymbolName ? rawSymbolName.lastIndexOf('::') : -1;
                    const symbolName = qualifierEnd >= 0
                        ? rawSymbolName!.slice(qualifierEnd + 2)
                        : rawSymbolName;
                    const symbolScope = qualifierEnd > 0
                        ? rawSymbolName!.slice(0, qualifierEnd)
                        : undefined;
                    const symbolKind = NODE_TYPE_TO_SYMBOL_KIND[currentNode.type];
                    const classStructural = (symbolKind === 'class' || symbolKind === 'abstract')
                        ? extractClassStructural(currentNode, language)
                        : {};
                    // rag-graph-abstract-typedef-edges: Haxe abstract/typedef
                    // relations feed the v3-3 side-index buckets.
                    const typeRelations = (symbolKind === 'abstract' || symbolKind === 'typedef')
                        ? extractTypeRelations(currentNode, language, symbolKind)
                        : {};
                    // The names this chunk references. Broader than "calls": a
                    // type in a signature and a field declaration count too,
                    // because "what breaks if this changes" reaches them.
                    const referenced = collectReferencedSymbols(
                        currentNode,
                        referenceVocab,
                        [symbolName, rawSymbolName, symbolScope ?? parentScope],
                    );

                    chunks.push({
                        content: nodeText,
                        metadata: {
                            startLine,
                            endLine,
                            language,
                            filePath,
                            content_type: 'code',
                            symbol_kind: symbolKind,
                            symbol_name: symbolName,
                            parent_symbol: symbolScope ?? parentScope,
                            ...(fileStructural.imports && fileStructural.imports.length > 0
                                ? { imports: fileStructural.imports }
                                : {}),
                            ...(classStructural.extends ? { extends: classStructural.extends } : {}),
                            ...(classStructural.implements && classStructural.implements.length > 0
                                ? { implements: classStructural.implements }
                                : {}),
                            ...(typeRelations.abstract_underlying && typeRelations.abstract_underlying.length > 0
                                ? { abstract_underlying: typeRelations.abstract_underlying }
                                : {}),
                            ...(typeRelations.typedef_alias ? { typedef_alias: typeRelations.typedef_alias } : {}),
                            ...(referenced.length > 0 ? { mentioned_symbols: referenced } : {}),
                        }
                    });

                    // If this node introduces a parent scope (class/interface/struct/etc.),
                    // its descendants inherit it as parent_symbol.
                    if (PARENT_SCOPE_NODE_TYPES.has(currentNode.type) && symbolName) {
                        scopeForChildren = symbolName;
                    }
                }
            }

            for (const child of currentNode.children) {
                traverse(child, scopeForChildren, bodyForChildren);
            }
        };

        traverse(node);

        // If no meaningful chunks found, create a single chunk with the entire code
        if (chunks.length === 0) {
            chunks.push({
                content: code,
                metadata: {
                    startLine: 1,
                    endLine: codeLines.length,
                    language,
                    filePath,
                    content_type: 'code',
                }
            });
        }

        return chunks;
    }

    private extractSymbolName(node: Parser.SyntaxNode, splittable?: ReadonlySet<string>): string | undefined {
        return symbolNameFor(node, 0, splittable);
    }

    private async refineChunks(chunks: CodeChunk[], _originalCode: string): Promise<CodeChunk[]> {
        const refined: CodeChunk[] = [];
        for (const chunk of chunks) {
            if (chunk.content.length <= this.chunkSize) refined.push(chunk);
            else refined.push(...this.splitLargeChunk(chunk));
        }
        return collapseIdenticalChunks(refined);
    }

    /**
     * Cut a node longer than `chunkSize` into pieces on line boundaries.
     *
     * fix-code-chunk-line-ranges: a piece is a span of the node's lines, so its
     * range always locates its text — the range is computed after the trim
     * (D3), and the overlap with the previous piece is whole lines of that
     * piece's span within `chunkOverlap` characters (D2). Overlap never crosses
     * into another node: before this, every chunk got the last 300 characters
     * of whatever chunk was emitted before it — often the enclosing class —
     * and `startLine` was moved back by a line count that did not match.
     */
    private splitLargeChunk(chunk: CodeChunk): CodeChunk[] {
        const lines = chunk.content.split('\n');

        // Piece boundaries as before the fix: add lines until the next one
        // would push the piece past chunkSize. `to` is exclusive.
        const spans: Array<{ from: number; to: number }> = [];
        let from = 0;
        let length = 0;
        for (let i = 0; i < lines.length; i++) {
            const lineLength = lines[i].length + (i === lines.length - 1 ? 0 : 1);
            if (length + lineLength > this.chunkSize && length > 0) {
                spans.push({ from, to: i });
                from = i;
                length = 0;
            }
            length += lineLength;
        }
        spans.push({ from, to: lines.length });

        const subChunks: CodeChunk[] = [];
        for (let k = 0; k < spans.length; k++) {
            const { to } = spans[k];
            let spanFrom = spans[k].from;
            if (k > 0 && this.chunkOverlap > 0) {
                // Whole trailing lines of the previous piece, within the budget,
                // leaving at least one of its lines out of the overlap.
                const previous = spans[k - 1];
                let budget = this.chunkOverlap;
                let overlapFrom = previous.to;
                while (overlapFrom - 1 > previous.from) {
                    const cost = lines[overlapFrom - 1].length + 1;
                    if (cost > budget) break;
                    budget -= cost;
                    overlapFrom--;
                }
                spanFrom = overlapFrom;
            }

            const raw = lines.slice(spanFrom, to).join('\n');
            const content = raw.trim();
            if (content.length === 0) continue;
            const trimmedHead = raw.slice(0, raw.length - raw.trimStart().length);
            const startLine = chunk.metadata.startLine + spanFrom + countNewlines(trimmedHead);
            subChunks.push({
                content,
                // Sub-chunks inherit content_type / symbol info from the parent
                // AST node so every piece keeps its semantic tag.
                metadata: {
                    ...chunk.metadata,
                    startLine,
                    endLine: startLine + countNewlines(content),
                },
            });
        }

        return subChunks;
    }

    /**
     * Check if AST splitting is supported for the given language
     */
    static isLanguageSupported(language: string): boolean {
        return isAstSupported(language);
    }

    /**
     * The languages this splitter actually supports. #130 made this static the
     * one list two readers share; it was still a SECOND list beside the grammar
     * registry, and the two drifted: the registry has had `ocaml` since the Haxe
     * compiler's own sources were indexed, the hand-written list never got it, so
     * `getSplitterStrategyForLanguage('ocaml')` reported `langchain` for 24,340
     * rows the AST splitter had in fact parsed — and a caller that believed the
     * report skipped them (2026-09-20, the vocabulary pass of
     * name-declarations-and-refresh-the-vocabulary did exactly that).
     *
     * The registry is the source; `split()` has always read it and nothing else.
     * Adding a language stays one entry there.
     */
    static getSupportedLanguages(): string[] {
        return astSupportedLanguages();
    }
}

/** A parameter list among the node's descendants within four levels, not looking into its body. */
function hasParameterList(node: Parser.SyntaxNode, depth = 0): boolean {
    if (depth > 4) return false;
    const body = node.childForFieldName('body');
    for (const child of node.children) {
        if (PARAMETER_LIST_NODE_TYPES.has(child.type)) return true;
        if (body && child.id === body.id) continue;
        if (hasParameterList(child, depth + 1)) return true;
    }
    return false;
}

function countNewlines(text: string): number {
    let count = 0;
    for (let i = text.indexOf('\n'); i !== -1; i = text.indexOf('\n', i + 1)) count++;
    return count;
}

/**
 * fix-code-chunk-line-ranges (D4): chunks with the same range and text hash to
 * the same chunk id. They come from nested nodes — a class and a method both cut
 * into pieces, or a JS/TS `export_statement` and the declaration it wraps, which
 * span the same lines. Nodes are emitted before their descendants, so the later
 * copy is the inner node, with the specific symbol: it takes the earlier slot.
 */
function collapseIdenticalChunks(chunks: CodeChunk[]): CodeChunk[] {
    const slot = new Map<string, number>();
    const out: CodeChunk[] = [];
    for (const chunk of chunks) {
        const key = `${chunk.metadata.startLine}:${chunk.metadata.endLine}:${chunk.content}`;
        const at = slot.get(key);
        if (at === undefined) {
            slot.set(key, out.length);
            out.push(chunk);
        } else {
            out[at] = chunk;
        }
    }
    return out;
}
