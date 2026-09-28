import Parser from 'tree-sitter';
import { Splitter, CodeChunk } from './index';
import { MarkdownSplitter, MentionedVocabProvider } from './markdown-splitter';
import { extractStructural, extractClassStructural, extractTypeRelations } from './ast-structural-extractor';
import { envManager } from '../utils/env-manager';

// Language grammars, splittable node types, symbol-kind mapping and parent-scope
// set all come from the data-driven registry. Adding a language = one entry there.
import {
    astSupportedLanguages,
    ATTACHED_NODE_TYPES,
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
    return leadingCommentsMode() !== 'off';
}

/**
 * cover-code-outside-declarations-and-attach-doc-blocks: `doc` attaches only doc comments
 * (see `isDocComment`); `on` every leading comment, as measured for v8q0; anything else `off`.
 */
export function leadingCommentsMode(): 'off' | 'on' | 'doc' {
    const value = (envManager.get('CODE_CHUNK_LEADING_COMMENTS') || '').trim().toLowerCase();
    return value === 'on' || value === 'doc' ? value : 'off';
}

/** cover-code-outside-declarations-and-attach-doc-blocks: pack the code outside every declaration. */
export function coverGapsEnabled(): boolean {
    return (envManager.get('CODE_CHUNK_COVER_GAPS') || '').trim().toLowerCase() === 'on';
}

/** split-oversized-declarations-on-syntax: how a declaration longer than the chunk size is cut. */
export function splitOversizedMode(): 'lines' | 'syntax' {
    return (envManager.get('CODE_CHUNK_SPLIT_OVERSIZED') || '').trim().toLowerCase() === 'syntax' ? 'syntax' : 'lines';
}

/**
 * index-class-skeletons-and-dedup-clones-across-paths: `skeleton` indexes an oversized type
 * declaration with member chunks as its skeleton instead of body pieces; `pieces` (the default)
 * is the splitter as it was, byte for byte.
 */
export function classBodyMode(): 'pieces' | 'skeleton' {
    return (envManager.get('CODE_CHUNK_CLASS_BODY') || '').trim().toLowerCase() === 'skeleton' ? 'skeleton' : 'pieces';
}

/** What stands for an elided member body in a skeleton; the only text of a skeleton not in the file. */
export const SKELETON_ELISION = '…';

/**
 * A doc comment by its own syntax: `/**` (not `/**` + `/` or a `/***` banner), `/*!`, `///`
 * (not a `////` rule), `//!`, OCaml `(**` (not a `(***` banner), GDScript `##` (not `###`). The doc-comment forms of the corpus's languages — a
 * property of the languages, not a list of corpus files.
 */
/**
 * A gap group whose text outside comments and whitespace is shorter than this is not emitted:
 * stray braces, `namespace x {` openers, a lone `#endif`.
 * ponytail: a fixed knob; the pre-build count reports the lines it drops.
 */
const GAP_MIN_CHARS = 40;

/** The line (from 1) holding `offset`, over `lineStartOffsets`. */
function lineOfOffset(lineStarts: number[], offset: number): number {
    let lo = 0, hi = lineStarts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (lineStarts[mid] <= offset) lo = mid; else hi = mid - 1; }
    return lo + 1;
}

/** Where a chunk's text starts in `code`: on its start line, or -1. */
function locateChunk(chunk: CodeChunk, code: string, lineStarts: number[]): number {
    const lineStart = lineStarts[(chunk.metadata.startLine ?? 1) - 1];
    if (lineStart === undefined) return -1;
    const from = code.indexOf(chunk.content, lineStart);
    return from < 0 || from > (lineStarts[chunk.metadata.startLine ?? 1] ?? code.length) ? -1 : from;
}

/** The offset each line of `code` starts at; line n (from 1) starts at index n - 1. */
function lineStartOffsets(code: string): number[] {
    const starts = [0];
    for (let i = code.indexOf('\n'); i >= 0; i = code.indexOf('\n', i + 1)) starts.push(i + 1);
    return starts;
}

/** A comment or an import by the grammar's own node type — no language table. */
function isCommentOrImport(node: Parser.SyntaxNode): boolean {
    return /comment|import|include|using|use_declaration|open_module/.test(node.type);
}

export function isDocComment(text: string): boolean {
    return /^(\/\*\*(?![\/*])|\/\*!|\/\/\/(?!\/)|\/\/!|\(\*\*(?![)*])|##(?!#))/.test(text.trimStart());
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
export function firstLeadingComment(node: Parser.SyntaxNode, code: string, docOnly = false): Parser.SyntaxNode | null {
    let first: Parser.SyntaxNode | null = null;
    let below: Parser.SyntaxNode = node;
    for (let prev = node.previousSibling; prev && prev.type.includes('comment'); prev = prev.previousSibling) {
        if (below.startPosition.row - prev.endPosition.row > 1) break;
        // `doc`: the run stops at the first comment that is not a doc comment.
        if (docOnly && !isDocComment(prev.text)) break;
        const lineStart = code.lastIndexOf('\n', prev.startIndex - 1) + 1;
        if (code.slice(lineStart, prev.startIndex).trim() !== '') break;
        first = prev;
        below = prev;
    }
    return first;
}

/**
 * attach-haxe-docs-across-metadata: the first of the sibling nodes directly above `node` that the
 * grammar registry marks as attached to a declaration (Haxe `@:keep`), or `node` when there is
 * none. The run stops at a blank line, as the comment run does.
 */
export function attachedHead(node: Parser.SyntaxNode): Parser.SyntaxNode {
    let head = node;
    for (let prev = node.previousSibling; prev && ATTACHED_NODE_TYPES.has(prev.type); prev = prev.previousSibling) {
        if (head.startPosition.row - prev.endPosition.row > 1) break;
        head = prev;
    }
    return head;
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
    // split-oversized-declarations-on-syntax: the declaration node behind each chunk the walk
    // emitted, so an oversized one can be cut on its own syntax. Keyed by the chunk object, so
    // nothing leaks into the chunk itself.
    private declarationNodes = new WeakMap<CodeChunk, Parser.SyntaxNode>();
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
        const commentsMode = leadingCommentsMode();
        const splitSyntax = splitOversizedMode() === 'syntax';
        const keepNodes = splitSyntax || classBodyMode() === 'skeleton';
        const withLeadingComments = commentsMode !== 'off';
        // cover-code-outside-declarations-and-attach-doc-blocks: what each declaration chunk
        // covers, as [start, end) offsets, so the gap pass can tell what lies outside them.
        const covered: Array<[number, number]> = [];

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
                // attach-haxe-docs-across-metadata: the head (metadata above it) is the declaration's too.
                const head = withLeadingComments ? attachedHead(currentNode) : currentNode;
                const lead = withLeadingComments ? firstLeadingComment(head, code, commentsMode === 'doc') : null;
                const startNode = lead ?? head;
                const startLine = startNode.startPosition.row + 1;
                const endLine = currentNode.endPosition.row + 1;
                const nodeText = code.slice(startNode.startIndex, currentNode.endIndex);

                if (nodeText.trim().length > 0) {
                    covered.push([startNode.startIndex, currentNode.endIndex]);
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

                    const declarationChunk: CodeChunk = {
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
                    };
                    chunks.push(declarationChunk);
                    if (keepNodes) this.declarationNodes.set(declarationChunk, currentNode);

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

        if (chunks.length > 0 && coverGapsEnabled()) {
            chunks.push(...this.packGaps(node, code, covered, chunks, language, filePath, fileStructural.imports, referenceVocab, splittableSet, commentsMode));
            // Document order, stable: a declaration keeps its place before what follows it.
            chunks.sort((a, b) => (a.metadata.startLine ?? 0) - (b.metadata.startLine ?? 0));
        }

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

    /**
     * cover-code-outside-declarations-and-attach-doc-blocks, decision 2 — cAST's split-then-merge
     * over what the declaration walk left out. A node disjoint from every covered range is a gap
     * node; one that overlaps a covered range is descended into; one inside a covered range is a
     * wall. Gap nodes are packed greedily in document order while the group's text stays within
     * `chunkSize`; a wall closes the group, so no chunk spans a declaration chunk; a gap node
     * larger than `chunkSize` is packed through its children. Groups of only comments or
     * imports, or with under GAP_MIN_CHARS characters outside comments and whitespace, are not
     * emitted. Boundaries are node boundaries only, never a line window.
     */
    private packGaps(
        root: Parser.SyntaxNode,
        code: string,
        covered: Array<[number, number]>,
        declarations: CodeChunk[],
        language: string,
        filePath: string | undefined,
        imports: string[] | undefined,
        referenceVocab: ReadonlySet<string> | null | undefined,
        splittableSet: ReadonlySet<string>,
        mode: 'off' | 'on' | 'doc',
    ): CodeChunk[] {
        const inside = (s: number, e: number) => covered.some(([a, b]) => a <= s && e <= b);
        const overlaps = (s: number, e: number) => covered.some(([a, b]) => a < e && s < b);
        const groups: Parser.SyntaxNode[][] = [];
        let open: Parser.SyntaxNode[] = [];
        const close = () => { if (open.length) groups.push(open); open = []; };
        const visit = (n: Parser.SyntaxNode) => {
            if (n.endIndex <= n.startIndex) return; // zero-width (a MISSING node)
            if (inside(n.startIndex, n.endIndex)) { close(); return; }
            if (overlaps(n.startIndex, n.endIndex)) {
                if (n.childCount === 0) { close(); return; }
                for (const child of n.children) visit(child);
                return;
            }
            // A declaration the grammar table does not emit (a TypeScript `enum`) is its own
            // chunk, with its doc block, whole — cut by size later like any declaration.
            if (n.type in NODE_TYPE_TO_SYMBOL_KIND && this.extractSymbolName(n, splittableSet)) {
                // attach-haxe-docs-across-metadata: metadata above the declaration goes with it.
                const head = mode === 'off' ? n : attachedHead(n);
                const lead = mode === 'off' ? null : firstLeadingComment(head, code, mode === 'doc');
                const from = (lead ?? head).startIndex;
                const docs = open.filter((m) => m.startIndex >= from);
                open = open.filter((m) => !docs.includes(m));
                close();
                groups.push([...docs, n]);
                return;
            }
            if (n.endIndex - n.startIndex > this.chunkSize) {
                close();
                // A single token longer than a chunk (a base64 or wasm blob in a string literal)
                // is data, not code: it is not indexed.
                if (n.childCount === 0) return;
                for (const child of n.children) visit(child);
                close();
                return;
            }
            if (open.length && n.endIndex - open[0].startIndex > this.chunkSize) close();
            open.push(n);
        };
        visit(root);
        close();

        const chunks: CodeChunk[] = [];
        const fragments: Parser.SyntaxNode[][] = [];
        for (const group of groups) {
            if (group.every((n) => isCommentOrImport(n))) continue;
            const first = group[0];
            const last = group[group.length - 1];
            // Named when its code is one declaration node (any other nodes are its doc comments).
            const codeNodes = group.filter((n) => !n.type.includes('comment'));
            const single = codeNodes.length === 1 && codeNodes[0].type in NODE_TYPE_TO_SYMBOL_KIND ? codeNodes[0] : null;
            const symbolName = single ? this.extractSymbolName(single, splittableSet) : undefined;
            // The size floor is for fragments; a named declaration is kept however small.
            if (!symbolName && codeNodes.map((n) => n.text).join('').replace(/\s+/g, '').length < GAP_MIN_CHARS) { fragments.push(group); continue; }
            const referenced = new Set<string>();
            for (const n of group) for (const name of collectReferencedSymbols(n, referenceVocab, [symbolName])) referenced.add(name);
            const gapChunk: CodeChunk = {
                content: code.slice(first.startIndex, last.endIndex),
                metadata: {
                    startLine: first.startPosition.row + 1,
                    endLine: last.endPosition.row + 1,
                    language,
                    filePath,
                    content_type: 'code',
                    ...(symbolName ? { symbol_kind: NODE_TYPE_TO_SYMBOL_KIND[single!.type], symbol_name: symbolName } : {}),
                    ...(imports && imports.length > 0 ? { imports } : {}),
                    ...(referenced.size > 0 ? { mentioned_symbols: [...referenced] } : {}),
                },
            };
            chunks.push(gapChunk);
            if (single && splitOversizedMode() === 'syntax') this.declarationNodes.set(gapChunk, single);
        }
        this.attachFragments(fragments, code, covered, declarations);
        return chunks;
    }

    /**
     * A fragment under the size floor (`#ifdef X`, `template <typename T>`, `@:native(…)`,
     * `const X = {`, `#endif`, `} // namespace`) that sits directly against a declaration chunk —
     * only whitespace with at most one line break between them — joins it: a fragment directly
     * above moves the declaration's start up, one directly below moves its end down. Above is
     * tried first, fragments bottom-up, so a stack (`#ifdef` over `template <…>` over a class)
     * joins link by link; then below, top-down. A fragment touching no declaration stays out.
     * Of nested declarations sharing the edge, the outermost takes it. A join that would make
     * the chunk longer than the chunk size is skipped. A run touching declarations on both sides
     * joins the one below (`#endif` + `#ifdef Y` between two functions opens the second).
     */
    private attachFragments(fragments: Parser.SyntaxNode[][], code: string, covered: Array<[number, number]>, declarations: CodeChunk[]): void {
        // Where a node's text really ends: preprocessor nodes carry their line's newline.
        const endOf = (n: Parser.SyntaxNode) => n.startIndex + n.text.trimEnd().length;
        const lineOf = (offset: number) => (code.slice(0, offset).match(/\n/g) || []).length + 1;
        const touching = (a: number, b: number) => {
            const between = code.slice(a, b);
            return between.trim() === '' && (between.match(/\n/g) || []).length <= 1;
        };
        // A fragment is cut at blank lines first: only the run of nodes that actually touches a
        // declaration may join it (an `#include` two lines above an `#ifdef` stays out).
        const runs: Parser.SyntaxNode[][] = [];
        for (const group of fragments) {
            let run: Parser.SyntaxNode[] = [];
            for (const n of group) {
                if (run.length && (code.slice(endOf(run[run.length - 1]), n.startIndex).match(/\n/g) || []).length > 1) { runs.push(run); run = []; }
                run.push(n);
            }
            if (run.length) runs.push(run);
        }
        const candidates = runs.filter((run) => !run.every((n) => isCommentOrImport(n)));
        const left = new Set(candidates);
        for (const group of [...candidates].reverse()) {
            const start = group[0].startIndex, end = endOf(group[group.length - 1]);
            let best = -1;
            covered.forEach(([s, e], i) => {
                if (s >= end && touching(end, s) && (best < 0 || s < covered[best][0] || (s === covered[best][0] && e > covered[best][1]))) best = i;
            });
            // A join never pushes a declaration past the chunk size: that would cut it by lines.
            if (best < 0 || covered[best][1] - start > this.chunkSize) continue;
            const chunk = declarations[best];
            chunk.content = code.slice(start, covered[best][1]);
            chunk.metadata.startLine = group[0].startPosition.row + 1;
            covered[best][0] = start;
            left.delete(group);
        }
        for (const group of candidates) {
            if (!left.has(group)) continue;
            const start = group[0].startIndex, end = endOf(group[group.length - 1]);
            let best = -1;
            covered.forEach(([s, e], i) => {
                if (e <= start && touching(e, start) && (best < 0 || e > covered[best][1] || (e === covered[best][1] && s < covered[best][0]))) best = i;
            });
            if (best < 0 || end - covered[best][0] > this.chunkSize) continue;
            const chunk = declarations[best];
            chunk.content = code.slice(covered[best][0], end);
            chunk.metadata.endLine = lineOf(end);
            covered[best][1] = end;
        }
    }

    private extractSymbolName(node: Parser.SyntaxNode, splittable?: ReadonlySet<string>): string | undefined {
        return symbolNameFor(node, 0, splittable);
    }

    private async refineChunks(chunks: CodeChunk[], originalCode: string): Promise<CodeChunk[]> {
        const refined: CodeChunk[] = [];
        const syntax = splitOversizedMode() === 'syntax';
        let lineStarts: number[] | null = null;
        const skeletons = classBodyMode() === 'skeleton'
            ? this.buildSkeletons(chunks, originalCode, lineStarts ??= lineStartOffsets(originalCode))
            : null;
        for (const chunk of chunks) {
            const replaced = skeletons?.get(chunk);
            if (replaced) { refined.push(...replaced); continue; }
            if (chunk.content.length <= this.chunkSize) { refined.push(chunk); continue; }
            const node = syntax ? this.declarationNodes.get(chunk) : undefined;
            const pieces = node ? this.splitBySyntax(chunk, node, originalCode, lineStarts ??= lineStartOffsets(originalCode)) : null;
            refined.push(...(pieces ?? this.splitLargeChunk(chunk)));
        }
        return collapseIdenticalChunks(refined);
    }

    /**
     * index-class-skeletons-and-dedup-clones-across-paths, decisions 4–7. A type declaration
     * (a parent-scope node of the grammar: class, struct, interface, enum, abstract, module,
     * namespace) longer than the chunk size, with at least one
     * member that is a chunk of its own, is
     * replaced by its skeleton: every text of the declaration in document order, except that a
     * member chunk contributes only the first worded line of its doc block, the lines attached
     * above it, and its head up to its body, which is elided (`{ … }` for a braced body, `…`
     * otherwise). A member without a body contributes in full. The skeleton is packed in
     * member order into chunks of at most `chunkSize` (a single unit longer stays whole); each
     * chunk after the first opens with the declaration's first line. A wrapper chunk ending
     * with the declaration (`export class …`) goes with it and lends the skeleton its start.
     * Returns the chunks to replace, each mapped to what replaces it (empty: dropped).
     */
    private buildSkeletons(chunks: CodeChunk[], code: string, lineStarts: number[]): Map<CodeChunk, CodeChunk[]> {
        const out = new Map<CodeChunk, CodeChunk[]>();
        const located = chunks.map((chunk) => {
            const from = locateChunk(chunk, code, lineStarts);
            return { chunk, from, to: from < 0 ? -1 : from + chunk.content.length, node: this.declarationNodes.get(chunk) };
        }).filter((c) => c.from >= 0);
        const lineOf = (offset: number) => lineOfOffset(lineStarts, offset);
        for (const decl of located) {
            const node = decl.node;
            if (!node || !PARENT_SCOPE_NODE_TYPES.has(node.type) || decl.chunk.content.length <= this.chunkSize) continue;
            // Outermost member chunks inside the declaration, in document order.
            const members: typeof located = [];
            for (const m of located.filter((c) => c !== decl && c.node && c.from >= node.startIndex && c.to <= node.endIndex
                && !(c.from === decl.from && c.to === decl.to)).sort((a, b) => a.from - b.from || b.to - a.to)) {
                const prev = members[members.length - 1];
                if (!prev || m.from >= prev.to) members.push(m);
            }
            if (members.length === 0) continue;
            // A wrapper is the declaration's own parent that is no declaration with a body of its
            // own (`export …`, a decorator), never an enclosing class or function it ends.
            const wrapper = located.find((c) => c !== decl && c.node && node.parent && c.node.id === node.parent.id
                && !PARENT_SCOPE_NODE_TYPES.has(c.node.type) && c.node.childForFieldName('body') === null
                && c.chunk.content.length > this.chunkSize);
            const from = Math.min(decl.from, wrapper ? wrapper.from : decl.from);
            const to = Math.max(decl.to, wrapper ? wrapper.to : decl.to);

            // Units in document order: text outside every member, cut on node boundaries, and
            // one unit per member. Each unit's text runs from the end of the previous one, so
            // whitespace and anything no node holds is kept.
            type Unit = { start: number; end: number; member?: typeof located[number] };
            const units: Unit[] = [];
            const inMember = (s: number, e: number) => members.some((m) => m.from <= s && e <= m.to);
            const hitsMember = (s: number, e: number) => members.some((m) => m.from < e && s < m.to);
            const visit = (n: Parser.SyntaxNode) => {
                if (n.endIndex <= n.startIndex || inMember(n.startIndex, n.endIndex)) return;
                if (hitsMember(n.startIndex, n.endIndex) && n.childCount > 0) { for (const c of n.children) visit(c); return; }
                units.push({ start: n.startIndex, end: n.endIndex });
            };
            visit(node);
            for (const m of members) units.push({ start: m.from, end: m.to, member: m });
            units.sort((a, b) => a.start - b.start);
            // Units that overlap mean the parse does not describe the text (a recovered syntax error
            // across members): the declaration keeps its pieces.
            if (units.some((u, i) => i > 0 && u.start < units[i - 1].end)) continue;
            // The declaration header, up to and including its body opener, is one unit.
            const body = node.childForFieldName('body');
            const opener = body ?? node.children.find((c) => c.type === '{');
            const headerEnd = opener ? opener.startIndex + 1 : members[0].from;
            const headerCount = units.findIndex((u) => u.member || u.start >= headerEnd);
            if (headerCount > 1) units.splice(0, headerCount, { start: units[0].start, end: units[headerCount - 1].end });

            const texts: Array<{ text: string; first: number; last: number; start: number; end: number }> = [];
            // A unit longer than the chunk size that is not elided (a data table, a member without
            // a body) leaves the skeleton: a member is held by its own chunk, other text is cut as
            // with `pieces`, verbatim.
            const verbatimPieces: CodeChunk[] = [];
            let cursor = from;
            for (let k = 0; k < units.length; k++) {
                const u = units[k];
                let lead = code.slice(cursor, u.start);
                // The rest of the previous unit's last line stays with it, so a chunk never opens mid-line.
                const lineEnd = lead.indexOf('\n');
                if (texts.length && lineEnd > 0) {
                    const tail = lead.slice(0, lineEnd);
                    const prev = texts[texts.length - 1];
                    prev.text += tail;
                    if (tail.trim()) prev.last = cursor + tail.trimEnd().length - 1;
                    cursor += lineEnd;
                    lead = lead.slice(lineEnd);
                }
                let text: string;
                let last: number;
                if (u.member) {
                    const member = u.member.node!;
                    const pre = code.slice(u.start, Math.max(u.start, member.startIndex)).split('\n');
                    // Of the doc block: its opener, its first worded line and its closer, so it still
                    // reads as a comment. Every other line above the head is not a comment and stays.
                    // A comment line by the grammar: it lies inside a comment node above the head.
                    const spans: Array<[number, number]> = [];
                    for (let p = member.previousSibling; p && p.startIndex >= u.start; p = p.previousSibling) {
                        if (p.type.includes('comment')) spans.push([p.startIndex, p.endIndex]);
                    }
                    const lineAt: number[] = [];
                    pre.reduce((offset, line) => { lineAt.push(offset); return offset + line.length + 1; }, u.start);
                    const isComment = (_line: string, i: number) => pre[i].trim() !== ''
                        && spans.some(([a, b]) => a <= lineAt[i] + pre[i].length - pre[i].trimStart().length && lineAt[i] + pre[i].trimEnd().length <= b);
                    const commentAt = pre.map((line, i) => (isComment(line, i) ? i : -1)).filter((i) => i >= 0);
                    const worded = commentAt.find((i) => /[A-Za-z0-9]/.test(pre[i]));
                    const lastComment = commentAt[commentAt.length - 1];
                    const closes = lastComment !== undefined && /(\*\/|\*\))\s*$/.test(pre[lastComment]);
                    const kept = pre.filter((line, i) => !isComment(line, i)
                        || i === commentAt[0] || i === worded || (closes && i === lastComment));
                    const body = memberBody(member);
                    if (body && body.startIndex > member.startIndex) {
                        const head = code.slice(member.startIndex, body.startIndex).trimEnd();
                        const braced = body.text.startsWith('{') && body.text.endsWith('}');
                        text = lead + kept.join('\n') + head + (braced ? ` { ${SKELETON_ELISION} }` : ` ${SKELETON_ELISION}`);
                        last = member.startIndex + head.length - 1;
                    } else {
                        text = lead + kept.join('\n') + code.slice(member.startIndex, u.end);
                        last = u.end - 1;
                    }
                } else {
                    text = lead + code.slice(u.start, u.end);
                    last = u.end - 1;
                }
                if (u !== units[0] && u.end - u.start > this.chunkSize && !text.endsWith(SKELETON_ELISION) && !text.endsWith(`${SKELETON_ELISION} }`)) {
                    // What continues its last line (a `;`) goes with it.
                    let end = u.end;
                    while (units[k + 1] && !units[k + 1].member && lineOf(units[k + 1].start) === lineOf(end - 1)) end = units[++k].end;
                    // A member's own chunk ends at the member: what continues its line is cut on its own.
                    const cutFrom = u.member ? u.end + (code.slice(u.end, end).length - code.slice(u.end, end).trimStart().length) : u.start;
                    if (code.slice(cutFrom, end).trim()) {
                        verbatimPieces.push(...this.splitLargeChunk({ content: code.slice(cutFrom, end).trimEnd(), metadata: { ...decl.chunk.metadata, startLine: lineOf(cutFrom), endLine: lineOf(end - 1) } }));
                    }
                    cursor = end;
                    continue;
                }
                const firstInUnit = cursor + (lead.length - lead.trimStart().length);
                texts.push({ text, first: lead.trim() ? firstInUnit : u.start, last, start: u.start, end: u.end });
                cursor = u.end;
            }
            if (texts.length && cursor < to) {
                const tail = code.slice(cursor, to);
                texts[texts.length - 1].text += tail;
                if (tail.trim()) texts[texts.length - 1].last = cursor + tail.trimEnd().length - 1;
            }

            // Pack in member order; a chunk after the first opens with the declaration's first line.
            // The header line holds the name (in Java and C# the node starts at its annotations).
            const headerStart = lineStarts[(node.childForFieldName('name') ?? node.childForFieldName('body') ?? node).startPosition.row];
            const headerLine = code.slice(headerStart, code.indexOf('\n', headerStart) < 0 ? code.length : code.indexOf('\n', headerStart));
            const groups: Array<{ text: string; first: number; last: number }> = [];
            for (const [i, t] of texts.entries()) {
                const open = groups[groups.length - 1];
                // A unit that starts on the line the previous one ended on never opens a chunk.
                const sameLine = i > 0 && lineOf(t.start) === lineOf(texts[i - 1].end - 1);
                if (open && (sameLine || (open.text + t.text).trimEnd().length <= this.chunkSize)) { open.text += t.text; open.last = t.last; continue; }
                // The repeated header line lies above the range, which starts at the first unit held,
                // so ranges of one skeleton never nest.
                const opener = open ? headerLine.trim() + '\n' : '';
                groups.push({ text: opener + t.text.replace(/^\s*\n/, ''), first: t.first, last: t.last });
            }
            const names = decl.chunk.metadata.mentioned_symbols ?? [];
            const skeletons: CodeChunk[] = groups.map((g) => {
                const content = g.text.trim();
                const { mentioned_symbols: _unused, ...metadata } = decl.chunk.metadata;
                const held = names.filter((name) => new RegExp(`(^|[^A-Za-z0-9_$])${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}($|[^A-Za-z0-9_$])`).test(content));
                return {
                    content,
                    metadata: {
                        ...metadata,
                        startLine: lineOf(g.first),
                        endLine: lineOf(g.last),
                        skeleton: true,
                        ...(held.length > 0 ? { mentioned_symbols: held } : {}),
                    },
                };
            }).filter((c) => c.content.length > 0);
            // Every line of a skeleton, markers aside, is text of the declaration in order; a parse that
            // misreads a macro as a function breaks that, and the declaration then keeps its pieces.
            let at = from;
            const faithful = skeletons.every((c, i) => c.content.split('\n').every((line, j) => {
                if (i > 0 && j === 0) return true; // the repeated header line
                const part = line.split(SKELETON_ELISION)[0].replace(/\s*\{\s*$/, '').trim();
                if (!part) return true;
                const found = code.indexOf(part, at);
                if (found < 0 || found >= to) return false;
                at = found + part.length;
                return true;
            }));
            if (!faithful) continue;
            out.set(decl.chunk, [...skeletons, ...verbatimPieces]);
            if (wrapper && !out.has(wrapper.chunk)) out.set(wrapper.chunk, []);
        }
        return out;
    }

    /**
     * split-oversized-declarations-on-syntax — cAST inside one declaration. The declaration is
     * descended into, child by child, while a node does not fit together with the text it
     * carries; what the options attached above, tiny units and a descended node's head ride
     * forward, so a signature and its doc block open the piece with the body's first statements.
     * Units pack greedily while the piece stays within `chunkSize` (a glued unit or a leaf may
     * exceed it); a group with under GAP_MIN_CHARS non-whitespace characters joins its
     * neighbour. Pieces never overlap and inherit the declaration's metadata. Null when the
     * chunk's text cannot be located at its start line — the caller keeps the line split.
     */
    private splitBySyntax(chunk: CodeChunk, node: Parser.SyntaxNode, code: string, lineStarts: number[]): CodeChunk[] | null {
        const from = locateChunk(chunk, code, lineStarts);
        if (from < 0) return null;
        const to = from + chunk.content.length;

        // Units in document order. Text waiting to open the next unit is `carry`: what the options
        // attached above the node, a tiny unit (`{`, `private`), and the head of a node about to be
        // descended into (its signature). So a piece never ends on a signature or an opener, and a
        // doc block stays with the declaration's first statements. A node is descended into when
        // it would not fit TOGETHER with what it carries. A last tiny unit (`}`) joins the one
        // before it. A leaf too large for a piece stays whole.
        const folded: Array<[number, number]> = [];
        let carry: number | null = from < node.startIndex ? from : null;
        // Where the tiny units in the carry start: tininess is measured on them alone, never on
        // a doc block or a head riding in front of them.
        let tinyFrom: number | null = null;
        // Where the head in the carry starts: its bound is measured on the head alone, never on a
        // doc block in front of it (a long doc block must not cut a signature from its body).
        let headFrom: number | null = null;
        const small = (a: number, b: number) => code.slice(a, b).replace(/\s+/g, '').length < GAP_MIN_CHARS;
        const tooBig = (n: Parser.SyntaxNode) => n.childCount > 0 && n.endIndex - (carry ?? n.startIndex) > this.chunkSize;
        const visit = (n: Parser.SyntaxNode, last: boolean) => {
            if (n.endIndex <= n.startIndex) return;
            if (tooBig(n)) {
                const kids = n.children.filter((c) => c.endIndex > c.startIndex);
                kids.forEach((c, i) => {
                    const next = kids[i + 1];
                    const start = carry ?? c.startIndex;
                    // A head: the sibling after it will be descended even from this unit's start. Only
                    // a short one rides (a signature, not a run of list items), or the carry would
                    // grow without bound over a long enum or array.
                    if (next && next.childCount > 0 && next.endIndex - start > this.chunkSize
                        && c.endIndex - (headFrom ?? c.startIndex) < this.chunkSize / 4) {
                        carry = start; headFrom ??= c.startIndex; tinyFrom = null; return;
                    }
                    visit(c, last && i === kids.length - 1);
                });
                return;
            }
            const start = carry ?? n.startIndex;
            if (!last && small(tinyFrom ?? n.startIndex, n.endIndex)) { carry = start; tinyFrom ??= n.startIndex; return; }
            carry = null;
            tinyFrom = null;
            headFrom = null;
            folded.push([start, n.endIndex]);
        };
        visit(node, node.endIndex >= to);
        if (node.endIndex < to) folded.push([carry ?? node.endIndex, to]);
        else if (carry !== null) folded.push([carry, node.endIndex]);
        if (folded.length > 1 && small(...folded[folded.length - 1])) folded[folded.length - 2][1] = folded.pop()![1];

        const groups: Array<[number, number]> = [];
        for (const [s, e] of folded) {
            const open = groups[groups.length - 1];
            if (open && e - open[0] <= this.chunkSize) open[1] = e; else groups.push([s, e]);
        }
        const tiny = ([s, e]: [number, number]) => code.slice(s, e).replace(/\s+/g, '').length < GAP_MIN_CHARS;
        for (let i = 0; i < groups.length && groups.length > 1;) {
            if (!tiny(groups[i])) { i++; continue; }
            if (i > 0) { groups[i - 1][1] = groups[i][1]; groups.splice(i, 1); }
            else { groups[1][0] = groups[0][0]; groups.splice(0, 1); }
        }

        // Text between two groups (whitespace, a macro's line-continuation `\\` that no node
        // holds) goes to the earlier piece, so no character of the declaration is dropped.
        for (let i = 0; i + 1 < groups.length; i++) groups[i][1] = groups[i + 1][0];

        const lineOf = (offset: number) => lineOfOffset(lineStarts, offset);
        const pieces: CodeChunk[] = [];
        for (const [s, e] of groups) {
            const raw = code.slice(s, e);
            const content = raw.trim();
            if (!content) continue;
            const start = s + (raw.length - raw.trimStart().length);
            pieces.push({ content, metadata: { ...chunk.metadata, startLine: lineOf(start), endLine: lineOf(start + content.length) } });
        }
        return pieces.length ? pieces : null;
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

/**
 * The body a member declaration elides in a skeleton: its `body` field, or that of its only
 * named child (OCaml `value_definition > let_binding`), two levels deep at most.
 */
function memberBody(node: Parser.SyntaxNode, depth = 0): Parser.SyntaxNode | null {
    const body = node.childForFieldName('body');
    if (body) return body;
    if (depth >= 2) return null;
    // A wrapper hands its payload over as `definition` (a decorator) or as its only named child.
    const inner = node.childForFieldName('definition') ?? (node.namedChildren.length === 1 ? node.namedChildren[0] : null);
    return inner ? memberBody(inner, depth + 1) : null;
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
