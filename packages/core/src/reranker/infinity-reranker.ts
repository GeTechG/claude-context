import { Reranker, RerankResult } from './base-reranker';
import { wrapFetchWithAutostart } from '../embedding/sidecar-autostart';

export interface InfinityRerankerConfig {
    model: string;
    baseURL: string;
    fetch?: typeof fetch;
}

interface InfinityRerankResponse {
    object?: string;
    results: Array<{
        index: number;
        relevance_score: number;
        document?: string | null;
    }>;
    model?: string;
    usage?: { prompt_tokens?: number; total_tokens?: number };
}

/**
 * Infinity sidecar reranker. Talks to the OpenAI-compatible `/rerank`
 * endpoint of an Infinity v2 server. The server runs the cross-encoder
 * model named in `config.model` (e.g. `BAAI/bge-reranker-v2-m3`).
 *
 * Phase 2 note: Qwen3-Reranker-0.6B is the design target, but
 * `michaelf34/infinity:0.0.77` (latest at 2026-05-03) ships transformers
 * 4.50.x and does not know the qwen3 architecture. The model name is
 * therefore configurable via env so we can swap to Qwen3-Reranker-0.6B
 * once a newer Infinity image lands.
 */
export class InfinityReranker extends Reranker {
    private config: InfinityRerankerConfig;
    private fetchImpl: typeof fetch;

    constructor(config: InfinityRerankerConfig) {
        super();
        this.config = { ...config };
        this.config.baseURL = (config.baseURL || 'http://localhost:7997').replace(/\/+$/, '');
        const baseFetch = config.fetch || (globalThis.fetch as typeof fetch);

        if (!baseFetch) {
            throw new Error(
                '[InfinityReranker] global fetch is not available; pass config.fetch or run on Node 18+',
            );
        }
        // Recover transparently when the sidecar is down (see sidecar-autostart).
        this.fetchImpl = wrapFetchWithAutostart(baseFetch);
    }

    async rerank(query: string, documents: string[]): Promise<RerankResult[]> {
        if (!documents.length) return [];

        const url = `${this.config.baseURL}/rerank`;
        const response = await this.fetchImpl(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: this.config.model,
                query,
                documents,
                return_documents: false,
            }),
        });

        if (!response.ok) {
            const errBody = await response.text().catch(() => '');
            throw new Error(
                `[InfinityReranker] ${response.status} ${response.statusText} from ${url}: ${errBody.slice(0, 500)}`,
            );
        }

        const json = (await response.json()) as InfinityRerankResponse;
        if (!Array.isArray(json?.results)) {
            throw new Error('[InfinityReranker] response missing results array');
        }
        return json.results.map((r) => ({ index: r.index, score: r.relevance_score }));
    }

    getProvider(): string {
        return 'Infinity';
    }
}
