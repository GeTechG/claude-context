/**
 * Single rerank result. `index` refers to the position in the input
 * `documents` array passed to `rerank`. `score` is monotonic — higher means
 * more relevant — but its absolute scale is provider-specific.
 */
export interface RerankResult {
    index: number;
    score: number;
}

/**
 * Cross-encoder reranker. Phase 2 places one of these after the per-domain
 * weighted-RRF merge to arbitrate the top-N better than fixed guarantee
 * slots can.
 */
export abstract class Reranker {
    abstract rerank(query: string, documents: string[]): Promise<RerankResult[]>;
    abstract getProvider(): string;
}
