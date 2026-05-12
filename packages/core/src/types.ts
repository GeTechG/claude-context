export interface SearchQuery {
    term: string;
    includeContent?: boolean;
    limit?: number;
}

export interface SemanticSearchResult {
    content: string;
    relativePath: string;
    startLine: number;
    endLine: number;
    language: string;
    score: number;
    // Phase 0: rich metadata exposed for downstream filtering / eval / enrichment.
    // All optional — older chunks indexed before Phase 0 will not have them.
    content_type?: string;
    symbol_name?: string;
    symbol_kind?: string;
    parent_symbol?: string;
    heading_path?: string[];
    // Phase 3: candidate symbol names extracted from the full result pool.
    // Populated only on the first element of the result array (a denormalised
    // list that belongs to the response, not the chunk). MCP handlers lift it
    // out into the JSON response. Older clients ignore it without harm.
    candidateSymbols?: string[];
    // rag-graph-layer Phase 1.3: structural fields surfaced to search
    // pipeline so graph-expansion can resolve forward edges (imports /
    // extends / implements / mentioned_symbols) without an extra fetch.
    // chunk_id is the Milvus primary-key, exposed so graph-expansion can
    // look up neighbors by id.
    chunk_id?: string;
    imports?: string[];
    extends?: string;
    implements?: string[];
    mentioned_symbols?: string[];
    // rag-graph-comparison-bridge: which retrieval pool injected the chunk.
    // Set only by pools that need downstream identification (currently the
    // comparison bridge sets `comparisonBridge`). Older clients ignore it.
    pool?: string;
}
