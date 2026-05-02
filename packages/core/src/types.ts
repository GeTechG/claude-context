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
}
