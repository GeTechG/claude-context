// Interface definitions
export interface VectorDocument {
    id: string;
    vector: number[];
    content: string;
    relativePath: string;
    startLine: number;
    endLine: number;
    fileExtension: string;
    metadata: Record<string, any>;
    // Phase 0: rich metadata as top-level columns so Milvus can filter on them
    // (filter exprs only see schema fields, not the JSON metadata blob).
    content_type?: string;
    symbol_kind?: string;
    symbol_name?: string;
    parent_symbol?: string;
    // heading_path is stored as a JSON-encoded string in Milvus to keep the
    // schema simple; consumers JSON.parse on read.
    heading_path?: string;
    // Phase 4: BGE-M3 learned-sparse channel. Populated only on v2 hybrid
    // collections; ignored on older schemas.
    sparse_learned?: { indices: number[]; values: number[] };
}

export interface SearchOptions {
    topK?: number;
    filter?: Record<string, any>;
    threshold?: number;
    filterExpr?: string;
}

// New interfaces for hybrid search
export interface HybridSearchRequest {
    // number[]   → dense vector channel (e.g. anns_field='vector')
    // string     → BM25 text query (e.g. anns_field='sparse_vector' with the
    //              attached BM25 function on the collection)
    // dict       → SPARSE_FLOAT_VECTOR (e.g. anns_field='sparse_learned',
    //              BGE-M3 learned sparse) as `{ "<index>": value }`
    data: number[] | string | Record<string, number>;
    anns_field: string;
    param: Record<string, any>;
    limit: number;
}

export interface HybridSearchOptions {
    rerank?: RerankStrategy;
    limit?: number;
    filterExpr?: string;
}

export interface HybridCollectionOptions {
    // Phase 4: add `sparse_learned` SPARSE_FLOAT_VECTOR column + IP index for
    // the BGE-M3 learned-sparse third channel. Default false to keep v1
    // collections (dense + BM25) on the same code path.
    enableLearnedSparse?: boolean;
}

export interface RerankStrategy {
    strategy: 'rrf' | 'weighted';
    params?: Record<string, any>;
}

export interface VectorSearchResult {
    document: VectorDocument;
    score: number;
}

export interface HybridSearchResult {
    document: VectorDocument;
    score: number;
}

export interface VectorDatabase {
    /**
     * Create collection
     * @param collectionName Collection name
     * @param dimension Vector dimension
     * @param description Collection description
     */
    createCollection(collectionName: string, dimension: number, description?: string): Promise<void>;

    /**
     * Create collection with hybrid search support
     * @param collectionName Collection name
     * @param dimension Dense vector dimension
     * @param description Collection description
     * @param options Optional schema flags (e.g. enableLearnedSparse for Phase 4)
     */
    createHybridCollection(collectionName: string, dimension: number, description?: string, options?: HybridCollectionOptions): Promise<void>;

    /**
     * Drop collection
     * @param collectionName Collection name
     */
    dropCollection(collectionName: string): Promise<void>;

    /**
     * Check if collection exists
     * @param collectionName Collection name
     */
    hasCollection(collectionName: string): Promise<boolean>;

    /**
     * List all collections
     */
    listCollections(): Promise<string[]>;

    /**
     * Insert vector documents
     * @param collectionName Collection name
     * @param documents Document array
     */
    insert(collectionName: string, documents: VectorDocument[]): Promise<void>;

    /**
     * Insert hybrid vector documents
     * @param collectionName Collection name
     * @param documents Document array
     */
    insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void>;

    /**
     * Search similar vectors
     * @param collectionName Collection name
     * @param queryVector Query vector
     * @param options Search options
     */
    search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]>;

    /**
     * Hybrid search with multiple vector fields
     * @param collectionName Collection name
     * @param searchRequests Array of search requests for different fields
     * @param options Hybrid search options including reranking
     */
    hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]>;

    /**
     * Delete documents
     * @param collectionName Collection name
     * @param ids Document ID array
     */
    delete(collectionName: string, ids: string[]): Promise<void>;

    /**
     * Query documents with filter conditions
     * @param collectionName Collection name
     * @param filter Filter expression
     * @param outputFields Fields to return
     * @param limit Maximum number of results
     */
    query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]>;

    /**
     * Get collection description
     * @param collectionName Collection name
     * @returns Collection description string
     */
    getCollectionDescription(collectionName: string): Promise<string>;

    /**
     * Check collection limit
     * Returns true if collection can be created, false if limit exceeded
     */
    checkCollectionLimit(): Promise<boolean>;

    /**
     * Get the number of entities (rows) in a collection.
     * Returns -1 if the count cannot be determined (query failed, collection missing, etc).
     * Callers should treat -1 as "unknown" and NOT as "empty".
     */
    getCollectionRowCount(collectionName: string): Promise<number>;
}

/**
 * Special error message for collection limit exceeded
 * This allows us to distinguish it from other errors across all Milvus implementations
 */
export const COLLECTION_LIMIT_MESSAGE = "[Error]: Your Zilliz Cloud account has hit its collection limit. To continue creating collections, you'll need to expand your capacity. We recommend visiting https://zilliz.com/pricing to explore options for dedicated or serverless clusters."; 