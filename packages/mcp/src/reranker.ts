import { InfinityReranker, Reranker } from "@zilliz/claude-context-core";
import { ContextMcpConfig } from "./config.js";

/**
 * Build a reranker instance from MCP config. Returns undefined when
 * `RERANKER_PROVIDER` is unset — Context falls back to Phase 0+
 * guarantee-slots in that case.
 */
export function createRerankerInstance(config: ContextMcpConfig): Reranker | undefined {
    if (!config.rerankerProvider) {
        return undefined;
    }

    switch (config.rerankerProvider) {
        case 'Infinity': {
            const baseURL = config.rerankerUrl || config.infinityUrl || 'http://localhost:7997';
            const model = config.rerankerModel || 'BAAI/bge-reranker-v2-m3';
            console.log(`[RERANKER] 🔧 Configuring Infinity reranker with model: ${model}, url: ${baseURL}`);
            const reranker = new InfinityReranker({ model, baseURL });
            console.log(`[RERANKER] ✅ Infinity reranker instance created successfully`);
            return reranker;
        }
        default:
            console.error(`[RERANKER] ❌ Unsupported reranker provider: ${config.rerankerProvider}`);
            throw new Error(`Unsupported reranker provider: ${config.rerankerProvider}`);
    }
}
