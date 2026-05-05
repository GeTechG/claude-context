// Interface definitions
export interface CodeChunk {
    content: string;
    metadata: {
        startLine: number;
        endLine: number;
        language?: string;
        filePath?: string;
        // Phase 0: rich metadata for hybrid retrieval and per-domain filtering.
        // All optional for backward compatibility with chunks that predate the schema.
        content_type?: 'code' | 'doc' | 'docstring' | 'code_example';
        symbol_kind?: string;
        symbol_name?: string;
        parent_symbol?: string;
        heading_path?: string[];
        // rag-graph-layer Phase 1: structural fields harvested by the AST
        // splitter (code chunks) and the markdown splitter (doc / code_example
        // chunks). Empty arrays / undefined when not extractable.
        imports?: string[];
        extends?: string;
        implements?: string[];
        mentioned_symbols?: string[];
    };
}

// Splitter type enumeration
export enum SplitterType {
    LANGCHAIN = 'langchain',
    AST = 'ast'
}

// Splitter configuration interface
export interface SplitterConfig {
    type?: SplitterType;
    chunkSize?: number;
    chunkOverlap?: number;
}

export interface Splitter {
    /**
     * Split code into code chunks
     * @param code Code content
     * @param language Programming language
     * @param filePath File path
     * @returns Array of code chunks
     */
    split(code: string, language: string, filePath?: string): Promise<CodeChunk[]>;

    /**
     * Set chunk size
     * @param chunkSize Chunk size
     */
    setChunkSize(chunkSize: number): void;

    /**
     * Set chunk overlap size
     * @param chunkOverlap Chunk overlap size
     */
    setChunkOverlap(chunkOverlap: number): void;
}

// Implementation class exports
export * from './langchain-splitter';
export * from './ast-splitter';
export * from './markdown-splitter';