import { Embedding, EmbeddingVector } from './base-embedding';

export interface InfinityEmbeddingConfig {
    model: string;
    baseURL: string;
    fetch?: typeof fetch;
    dimension?: number;
    maxTokens?: number;
}

interface InfinityEmbeddingsResponse {
    object: string;
    data: Array<{
        object: string;
        embedding: number[];
        index: number;
    }>;
    model: string;
    usage?: { prompt_tokens?: number; total_tokens?: number };
}

/**
 * Infinity sidecar embedding provider.
 *
 * Talks to the OpenAI-compatible `/embeddings` endpoint of an Infinity v2
 * server (https://github.com/michaelfeil/infinity). One forward pass returns
 * the dense vector; learned-sparse channel will be added in Phase 4.
 */
export class InfinityEmbedding extends Embedding {
    private config: InfinityEmbeddingConfig;
    private dimension: number = 1024;
    private dimensionDetected: boolean = false;
    private fetchImpl: typeof fetch;
    protected maxTokens: number = 8192;

    constructor(config: InfinityEmbeddingConfig) {
        super();
        this.config = { ...config };
        this.config.baseURL = (config.baseURL || 'http://localhost:7997').replace(/\/+$/, '');
        this.fetchImpl = config.fetch || (globalThis.fetch as typeof fetch);

        if (!this.fetchImpl) {
            throw new Error(
                '[InfinityEmbedding] global fetch is not available; pass config.fetch or run on Node 18+',
            );
        }

        if (config.dimension) {
            this.dimension = config.dimension;
            this.dimensionDetected = true;
        }
        if (config.maxTokens) {
            this.maxTokens = config.maxTokens;
        }
    }

    private async postEmbeddings(input: string | string[]): Promise<InfinityEmbeddingsResponse> {
        const url = `${this.config.baseURL}/embeddings`;
        const response = await this.fetchImpl(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: this.config.model, input }),
        });

        if (!response.ok) {
            const errBody = await response.text().catch(() => '');
            throw new Error(
                `[InfinityEmbedding] ${response.status} ${response.statusText} from ${url}: ${errBody.slice(0, 500)}`,
            );
        }

        const json = (await response.json()) as InfinityEmbeddingsResponse;
        if (!json?.data?.length) {
            throw new Error('[InfinityEmbedding] empty data array in response');
        }
        return json;
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processed = this.preprocessText(text);

        if (!this.dimensionDetected) {
            this.dimension = await this.detectDimension();
            this.dimensionDetected = true;
            console.log(
                `[InfinityEmbedding] 📏 Detected dimension: ${this.dimension} for model: ${this.config.model}`,
            );
        }

        const json = await this.postEmbeddings(processed);
        const vector = json.data[0].embedding;
        if (!Array.isArray(vector)) {
            throw new Error('[InfinityEmbedding] embedding is not an array');
        }
        return { vector, dimension: this.dimension };
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const processed = this.preprocessTexts(texts);

        if (!this.dimensionDetected) {
            this.dimension = await this.detectDimension();
            this.dimensionDetected = true;
            console.log(
                `[InfinityEmbedding] 📏 Detected dimension: ${this.dimension} for model: ${this.config.model}`,
            );
        }

        const json = await this.postEmbeddings(processed);
        json.data.sort((a, b) => a.index - b.index);
        return json.data.map((item) => ({ vector: item.embedding, dimension: this.dimension }));
    }

    getDimension(): number {
        return this.dimension;
    }

    getProvider(): string {
        return 'Infinity';
    }

    async detectDimension(testText: string = 'test'): Promise<number> {
        try {
            const json = await this.postEmbeddings(this.preprocessText(testText));
            const dim = json.data[0].embedding.length;
            return dim;
        } catch (error) {
            const msg = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`[InfinityEmbedding] Failed to detect dimension: ${msg}`);
        }
    }
}
