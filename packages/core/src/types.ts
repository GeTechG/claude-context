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
}
