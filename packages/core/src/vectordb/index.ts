// Re-export types and interfaces
export {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
    RerankStrategy,
    DroppedRow,
    InsertResult,
    COLLECTION_LIMIT_MESSAGE
} from './types';

// #19: batch-level insert recovery (shared by the gRPC insert paths).
export {
    insertRowsWithRecovery,
    readInsertResponse,
    fieldLimitViolations,
    describeRowLocation,
    InsertRecoveryContext,
} from './insert-recovery';

// Implementation class exports
export { MilvusRestfulVectorDatabase, MilvusRestfulConfig } from './milvus-restful-vectordb';
export { MilvusVectorDatabase, MilvusConfig } from './milvus-vectordb';
export {
    ClusterManager,
    ZillizConfig,
    Project,
    Cluster,
    CreateFreeClusterRequest,
    CreateFreeClusterResponse,
    CreateFreeClusterWithDetailsResponse,
    DescribeClusterResponse
} from './zilliz-utils'; 