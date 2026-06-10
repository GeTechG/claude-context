import { Embedding, EmbeddingVector, SparseVector } from './base-embedding';
import { wrapFetchWithAutostart } from './sidecar-autostart';

export interface InfinityEmbeddingConfig {
    model: string;
    baseURL: string;
    fetch?: typeof fetch;
    dimension?: number;
    maxTokens?: number;
    // Phase 4: optional URL of the BGE-M3 learned-sparse sidecar (m3serve).
    // When set, embed/embedBatch fan out to /sparse in parallel with the
    // dense /embeddings call and attach `sparse: {indices, values}`.
    sparseURL?: string;
    sparseModel?: string;
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

interface SparseSidecarResponse {
    object: string;
    data: Array<{
        object: string;
        indices: number[];
        values: number[];
        index: number;
    }>;
    model: string;
}

/**
 * Infinity sidecar embedding provider.
 *
 * Talks to the OpenAI-compatible `/embeddings` endpoint of an Infinity v2
 * server (https://github.com/michaelfeil/infinity) for the dense channel.
 *
 * Phase 4: when `sparseURL` is configured, also POSTs to a separate sparse
 * sidecar (m3serve) at `/sparse` in parallel and attaches BGE-M3 learned
 * `lexical_weights` to each EmbeddingVector. Infinity 0.0.77 doesn't expose
 * BGE-M3 sparse, so we run a second container that loads the same model
 * via m3serve.Engine.
 */
export class InfinityEmbedding extends Embedding {
    private config: InfinityEmbeddingConfig;
    private dimension: number = 1024;
    private dimensionDetected: boolean = false;
    private fetchImpl: typeof fetch;
    protected maxTokens: number = 8192;
    private sparseURL?: string;
    private sparseModel: string;

    constructor(config: InfinityEmbeddingConfig) {
        super();
        this.config = { ...config };
        this.config.baseURL = (config.baseURL || 'http://localhost:7997').replace(/\/+$/, '');
        const baseFetch = config.fetch || (globalThis.fetch as typeof fetch);

        if (!baseFetch) {
            throw new Error(
                '[InfinityEmbedding] global fetch is not available; pass config.fetch or run on Node 18+',
            );
        }
        // Recover transparently when the sidecar is down (see sidecar-autostart).
        this.fetchImpl = wrapFetchWithAutostart(baseFetch);

        if (config.dimension) {
            this.dimension = config.dimension;
            this.dimensionDetected = true;
        }
        if (config.maxTokens) {
            this.maxTokens = config.maxTokens;
        }
        if (config.sparseURL) {
            this.sparseURL = config.sparseURL.replace(/\/+$/, '');
        }
        this.sparseModel = config.sparseModel || config.model;
    }

    private async postEmbeddings(
        input: string | string[],
        inputType: 'query' | 'passage',
    ): Promise<InfinityEmbeddingsResponse> {
        const url = `${this.config.baseURL}/embeddings`;
        // input_type is NVIDIA NIM convention: asymmetric retrieval models
        // (nemotron-embed, e5, etc) require `query: ` vs `passage: ` prefixes
        // applied server-side. Models without that requirement (bge-m3) ignore
        // the field, so passing it is always safe.
        const response = await this.fetchImpl(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: this.config.model, input, input_type: inputType }),
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

    private async postSparse(
        input: string | string[],
        inputType: 'query' | 'passage',
    ): Promise<SparseSidecarResponse> {
        if (!this.sparseURL) {
            throw new Error('[InfinityEmbedding] sparseURL not configured');
        }
        const url = `${this.sparseURL}/sparse`;
        // learned-sparse-swap: forward input_type to /sparse. bge-m3 sparse is
        // symmetric and ignores it (extra field), but doc-only learned-sparse
        // models (opensearch-neural-sparse-encoding-doc-v3) are ASYMMETRIC —
        // documents run the neural encoder while queries use a cheap
        // tokenizer + IDF weight-lookup. Without this hint the sparse sidecar
        // cannot tell a query from a passage on the wire.
        const response = await this.fetchImpl(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: this.sparseModel, input, input_type: inputType }),
        });

        if (!response.ok) {
            const errBody = await response.text().catch(() => '');
            throw new Error(
                `[InfinityEmbedding] sparse ${response.status} ${response.statusText} from ${url}: ${errBody.slice(0, 500)}`,
            );
        }

        const json = (await response.json()) as SparseSidecarResponse;
        if (!json?.data?.length) {
            throw new Error('[InfinityEmbedding] empty data array in sparse response');
        }
        return json;
    }

    private toSparseVector(item: SparseSidecarResponse['data'][number]): SparseVector {
        return { indices: item.indices ?? [], values: item.values ?? [] };
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

        if (this.sparseURL) {
            const [denseResp, sparseResp] = await Promise.all([
                this.postEmbeddings(processed, 'query'),
                this.postSparse(processed, 'query'),
            ]);
            const vector = denseResp.data[0].embedding;
            if (!Array.isArray(vector)) {
                throw new Error('[InfinityEmbedding] embedding is not an array');
            }
            return {
                vector,
                dimension: this.dimension,
                sparse: this.toSparseVector(sparseResp.data[0]),
            };
        }

        const json = await this.postEmbeddings(processed, 'query');
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

        if (this.sparseURL) {
            const [denseResp, sparseResp] = await Promise.all([
                this.postEmbeddings(processed, 'passage'),
                this.postSparse(processed, 'passage'),
            ]);
            denseResp.data.sort((a, b) => a.index - b.index);
            sparseResp.data.sort((a, b) => a.index - b.index);
            if (denseResp.data.length !== sparseResp.data.length) {
                throw new Error(
                    `[InfinityEmbedding] dense/sparse length mismatch: ${denseResp.data.length} vs ${sparseResp.data.length}`,
                );
            }
            return denseResp.data.map((item, i) => ({
                vector: item.embedding,
                dimension: this.dimension,
                sparse: this.toSparseVector(sparseResp.data[i]),
            }));
        }

        const json = await this.postEmbeddings(processed, 'passage');
        json.data.sort((a, b) => a.index - b.index);
        return json.data.map((item) => ({ vector: item.embedding, dimension: this.dimension }));
    }

    getDimension(): number {
        return this.dimension;
    }

    getProvider(): string {
        return 'Infinity';
    }

    /**
     * prose-embedding-swap: expose the dense model id so callers can record
     * per-collection dense-model metadata (which model built which Milvus
     * collection) for query-side routing.
     */
    getModel(): string {
        return this.config.model;
    }

    hasSparse(): boolean {
        return Boolean(this.sparseURL);
    }

    async detectDimension(testText: string = 'test'): Promise<number> {
        try {
            // Probing call — indexer-side default ('passage') is fine because
            // for nemotron-embed both prefixes return the same dim (MRL trim
            // is applied identically) and bge-m3 ignores the field.
            const json = await this.postEmbeddings(this.preprocessText(testText), 'passage');
            const dim = json.data[0].embedding.length;
            return dim;
        } catch (error) {
            const msg = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`[InfinityEmbedding] Failed to detect dimension: ${msg}`);
        }
    }
}
