import Parser from 'tree-sitter';
import { Splitter, CodeChunk } from './index';
import { MarkdownSplitter, MentionedVocabProvider } from './markdown-splitter';
import { extractStructural, extractClassStructural, extractTypeRelations } from './ast-structural-extractor';

// Language grammars, splittable node types, symbol-kind mapping and parent-scope
// set all come from the data-driven registry. Adding a language = one entry there.
import {
    getSplittableTypes,
    loadLanguage,
    NODE_TYPE_TO_SYMBOL_KIND,
    PARAMETER_LIST_NODE_TYPES,
    PARENT_SCOPE_NODE_TYPES,
    WRAPPER_NODE_TYPES,
} from './grammar-registry';

// Symbol kinds that declare a type (grammar-registry kinds); inside a function body they stay chunks.
const TYPE_SYMBOL_KINDS = new Set(['class', 'enum', 'interface', 'typedef', 'abstract']);

// Terminal identifier node types at the end of a C/C++ declarator chain.
const DECLARATOR_NAME_TYPES = new Set([
    'identifier', 'field_identifier', 'qualified_identifier',
    'destructor_name', 'operator_name',
]);

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
export function declaratorName(node: Parser.SyntaxNode): string | undefined {
    let cur: Parser.SyntaxNode | null = (node as any).childForFieldName?.('declarator') ?? null;
    for (let depth = 0; cur && depth < 8; depth++) {
        if (DECLARATOR_NAME_TYPES.has(cur.type)) return cur.text || undefined;
        cur = (cur as any).childForFieldName?.('declarator') ?? null;
    }
    return undefined;
}

export class AstCodeSplitter implements Splitter {
    private chunkSize: number = 2500;
    private chunkOverlap: number = 300;
    private parser: Parser;
    private langchainFallback: any; // LangChainCodeSplitter for fallback
    private markdownSplitter: MarkdownSplitter;

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

        // no-chunks-inside-function-bodies: inside a function body only a named type declaration
        // or a named function with a parameter list becomes a chunk — not `int i = 0;`,
        // `let n = … in` or a callback. Types stay because a class the parser misreads as a
        // function (`class CC_DLL Foo {…}`) holds its public enums and nested classes in that body.
        const traverse = (currentNode: Parser.SyntaxNode, parentScope?: string, insideBody = false) => {
            const isSplittable = splittableTypes.includes(currentNode.type);
            let scopeForChildren = parentScope;
            const kind = NODE_TYPE_TO_SYMBOL_KIND[currentNode.type];
            const emit = isSplittable && (!insideBody
                || (Boolean(this.extractSymbolName(currentNode)) && (TYPE_SYMBOL_KINDS.has(kind) || hasParameterList(currentNode))));
            const bodyForChildren = insideBody || (isSplittable && (kind === 'function' || kind === 'method') && !WRAPPER_NODE_TYPES.has(currentNode.type) && opensBody(currentNode, splittableTypes));

            if (emit) {
                const startLine = currentNode.startPosition.row + 1;
                const endLine = currentNode.endPosition.row + 1;
                const nodeText = code.slice(currentNode.startIndex, currentNode.endIndex);

                if (nodeText.trim().length > 0) {
                    const rawSymbolName = this.extractSymbolName(currentNode);
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

    /**
     * Find the identifier child of a tree-sitter node and return its text.
     * Tree-sitter conventions vary across grammars; we try a few common shapes.
     */
    private extractSymbolName(node: Parser.SyntaxNode): string | undefined {
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
                child.type === 'IDENTIFIER'
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
                    return this.extractSymbolName(child);
                }
            }
        }

        return undefined;
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
        return AstCodeSplitter.SUPPORTED_LANGUAGES.includes(language.toLowerCase());
    }

    /**
     * The languages this splitter actually supports. #130: this static was
     * called by `Context.getSplitterInfo()` through an untyped `require`, so a
     * call to a method that did not exist surfaced as a runtime `TypeError`
     * instead of a compile error. The list is the one `isLanguageSupported`
     * reads — one list, two readers — and the caller now binds this class
     * through a typed import, so the type checker holds the seam.
     */
    static getSupportedLanguages(): string[] {
        return [...AstCodeSplitter.SUPPORTED_LANGUAGES];
    }

    private static readonly SUPPORTED_LANGUAGES: string[] = [
        'javascript', 'js', 'typescript', 'ts', 'python', 'py',
        'java', 'cpp', 'c++', 'c', 'go', 'rust', 'rs', 'cs', 'csharp', 'scala',
        'haxe', 'hx', 'hxml'
    ];
}

/**
 * A function or method node's descendants are inside its body when it has a `body` field,
 * or a direct child that is not itself splittable has one (OCaml `value_definition` →
 * `let_binding`). A wrapper (`export_statement`, registered as one) opens no body, whatever
 * it wraps — `export namespace N {…}` holds declarations, not statements. A declaration whose
 * child is a type with a body (C++ `field_declaration` → `struct_specifier`) opens none either:
 * that child is splittable and decides for itself.
 */
function opensBody(node: Parser.SyntaxNode, splittableTypes: string[]): boolean {
    if (node.childForFieldName('body')) return true;
    return node.namedChildren.some((child) => !splittableTypes.includes(child.type) && child.childForFieldName('body') !== null);
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
