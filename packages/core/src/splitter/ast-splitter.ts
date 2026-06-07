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
    PARENT_SCOPE_NODE_TYPES,
} from './grammar-registry';

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

        // Initialize fallback splitter
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

        const traverse = (currentNode: Parser.SyntaxNode, parentScope?: string) => {
            const isSplittable = splittableTypes.includes(currentNode.type);
            let scopeForChildren = parentScope;

            if (isSplittable) {
                const startLine = currentNode.startPosition.row + 1;
                const endLine = currentNode.endPosition.row + 1;
                const nodeText = code.slice(currentNode.startIndex, currentNode.endIndex);

                if (nodeText.trim().length > 0) {
                    const symbolName = this.extractSymbolName(currentNode);
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
                            parent_symbol: parentScope,
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
                traverse(child, scopeForChildren);
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

    private async refineChunks(chunks: CodeChunk[], originalCode: string): Promise<CodeChunk[]> {
        const refinedChunks: CodeChunk[] = [];

        for (const chunk of chunks) {
            if (chunk.content.length <= this.chunkSize) {
                refinedChunks.push(chunk);
            } else {
                // Split large chunks using character-based splitting
                const subChunks = this.splitLargeChunk(chunk, originalCode);
                refinedChunks.push(...subChunks);
            }
        }

        return this.addOverlap(refinedChunks);
    }

    private splitLargeChunk(chunk: CodeChunk, originalCode: string): CodeChunk[] {
        const lines = chunk.content.split('\n');
        const subChunks: CodeChunk[] = [];
        let currentChunk = '';
        let currentStartLine = chunk.metadata.startLine;
        let currentLineCount = 0;

        // Sub-chunks inherit content_type / symbol info from the parent AST node so
        // every piece keeps its semantic tag after large-chunk splitting.
        const inheritedMetadata = (startLine: number, endLine: number) => ({
            ...chunk.metadata,
            startLine,
            endLine,
        });

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineWithNewline = i === lines.length - 1 ? line : line + '\n';

            if (currentChunk.length + lineWithNewline.length > this.chunkSize && currentChunk.length > 0) {
                subChunks.push({
                    content: currentChunk.trim(),
                    metadata: inheritedMetadata(currentStartLine, currentStartLine + currentLineCount - 1),
                });

                currentChunk = lineWithNewline;
                currentStartLine = chunk.metadata.startLine + i;
                currentLineCount = 1;
            } else {
                currentChunk += lineWithNewline;
                currentLineCount++;
            }
        }

        if (currentChunk.trim().length > 0) {
            subChunks.push({
                content: currentChunk.trim(),
                metadata: inheritedMetadata(currentStartLine, currentStartLine + currentLineCount - 1),
            });
        }

        return subChunks;
    }

    private addOverlap(chunks: CodeChunk[]): CodeChunk[] {
        if (chunks.length <= 1 || this.chunkOverlap <= 0) {
            return chunks;
        }

        const overlappedChunks: CodeChunk[] = [];

        for (let i = 0; i < chunks.length; i++) {
            let content = chunks[i].content;
            const metadata = { ...chunks[i].metadata };

            // Add overlap from previous chunk
            if (i > 0 && this.chunkOverlap > 0) {
                const prevChunk = chunks[i - 1];
                const overlapText = prevChunk.content.slice(-this.chunkOverlap);
                content = overlapText + '\n' + content;
                metadata.startLine = Math.max(1, metadata.startLine - this.getLineCount(overlapText));
            }

            overlappedChunks.push({
                content,
                metadata
            });
        }

        return overlappedChunks;
    }

    private getLineCount(text: string): number {
        return text.split('\n').length;
    }

    /**
     * Check if AST splitting is supported for the given language
     */
    static isLanguageSupported(language: string): boolean {
        const supportedLanguages = [
            'javascript', 'js', 'typescript', 'ts', 'python', 'py',
            'java', 'cpp', 'c++', 'c', 'go', 'rust', 'rs', 'cs', 'csharp', 'scala',
            'haxe', 'hx', 'hxml'
        ];
        return supportedLanguages.includes(language.toLowerCase());
    }
}
