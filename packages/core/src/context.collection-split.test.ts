// code-collection-split: unit tests for the split-aware collection resolver
// helpers. Covers:
//   1) Legacy (SPLIT_COLLECTIONS=false / unset) → single-collection mode is
//      byte-stable (prose === code === legacy).
//   2) Split (SPLIT_COLLECTIONS=true) → two v6 collections by content_type.
//   3) `resolveChunkCollection` routes by content_type per spec mapping.
//   4) `getCollectionName` returns the code-collection in split mode (back-
//      compat hook for callers that have not been split-ified).

import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { VectorDatabase } from './vectordb';

class TestEmbedding extends Embedding {
    protected maxTokens = 8192;
    async detectDimension(): Promise<number> { return 3; }
    async embed(_text: string): Promise<EmbeddingVector> { return { vector: [1, 0, 0], dimension: 3 }; }
    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> { return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 })); }
    getDimension(): number { return 3; }
    getProvider(): string { return 'test'; }
}

function makeVectorDb(): jest.Mocked<VectorDatabase> {
    return {
        createCollection: jest.fn().mockResolvedValue(undefined),
        createHybridCollection: jest.fn().mockResolvedValue(undefined),
        dropCollection: jest.fn().mockResolvedValue(undefined),
        hasCollection: jest.fn().mockResolvedValue(false),
        listCollections: jest.fn().mockResolvedValue([]),
        insert: jest.fn().mockResolvedValue(undefined),
        insertHybrid: jest.fn().mockResolvedValue(undefined),
        search: jest.fn().mockResolvedValue([]),
        hybridSearch: jest.fn().mockResolvedValue([]),
        delete: jest.fn().mockResolvedValue(undefined),
        query: jest.fn().mockResolvedValue([]),
        getCollectionDescription: jest.fn().mockResolvedValue(''),
        checkCollectionLimit: jest.fn().mockResolvedValue(true),
        getCollectionRowCount: jest.fn().mockResolvedValue(0),
    };
}

function makeCtx(): Context {
    return new Context({ embedding: new TestEmbedding(), vectorDatabase: makeVectorDb() });
}

describe('code-collection-split: getCollectionAddress / resolveChunkCollection', () => {
    let savedSplit: string | undefined;
    let savedVersion: string | undefined;
    let savedHybrid: string | undefined;

    beforeEach(() => {
        savedSplit = process.env.SPLIT_COLLECTIONS;
        savedVersion = process.env.COLLECTION_VERSION;
        savedHybrid = process.env.HYBRID_MODE;
        delete process.env.SPLIT_COLLECTIONS;
        delete process.env.COLLECTION_VERSION;
        delete process.env.HYBRID_MODE;
    });

    afterEach(() => {
        const restore = (name: string, value: string | undefined) => {
            if (value === undefined) delete process.env[name];
            else process.env[name] = value;
        };
        restore('SPLIT_COLLECTIONS', savedSplit);
        restore('COLLECTION_VERSION', savedVersion);
        restore('HYBRID_MODE', savedHybrid);
    });

    it('legacy mode (SPLIT_COLLECTIONS unset): isSplit=false; prose === code === legacy', () => {
        process.env.COLLECTION_VERSION = 'v3';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');
        expect(addr.isSplit).toBe(false);
        expect(addr.legacy).toMatch(/^hybrid_v3_code_chunks_[0-9a-f]{8}$/);
        expect(addr.prose).toBe(addr.legacy);
        expect(addr.code).toBe(addr.legacy);
        expect(ctx.getCollectionName('/path/to/codebase')).toBe(addr.legacy);
    });

    it('legacy mode (SPLIT_COLLECTIONS=false): identical to unset', () => {
        process.env.COLLECTION_VERSION = 'v3';
        process.env.SPLIT_COLLECTIONS = 'false';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');
        expect(addr.isSplit).toBe(false);
        expect(addr.prose).toBe(addr.code);
    });

    it('split mode (SPLIT_COLLECTIONS=true, v6): isSplit=true; prose / code differ', () => {
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COLLECTION_VERSION = 'v6';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');
        expect(addr.isSplit).toBe(true);
        expect(addr.prose).toMatch(/^hybrid_v6_prose_[0-9a-f]{8}$/);
        expect(addr.code).toMatch(/^hybrid_v6_code_[0-9a-f]{8}$/);
        expect(addr.prose).not.toBe(addr.code);
        // Per the back-compat convention, getCollectionName returns the
        // code-collection in split mode.
        expect(ctx.getCollectionName('/path/to/codebase')).toBe(addr.code);
    });

    it('split + same path hash: prose / code share the <hash> suffix', () => {
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COLLECTION_VERSION = 'v6';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');
        const proseHash = addr.prose.split('_').pop();
        const codeHash = addr.code.split('_').pop();
        expect(proseHash).toBe(codeHash);
        expect(proseHash).toMatch(/^[0-9a-f]{8}$/);
    });

    it('split mode is gated by HYBRID_MODE=true (semantic-search only is not split)', () => {
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COLLECTION_VERSION = 'v6';
        process.env.HYBRID_MODE = 'false';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');
        // Non-hybrid mode collapses to legacy regardless of SPLIT_COLLECTIONS,
        // because legacy semantic-search has no per-domain pools and the
        // v6 naming convention only makes sense in the hybrid pipeline.
        expect(addr.isSplit).toBe(false);
    });

    it('resolveChunkCollection: code/docstring → code; doc/code_example → prose', () => {
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COLLECTION_VERSION = 'v6';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');

        expect(ctx.resolveChunkCollection('code', addr)).toBe(addr.code);
        expect(ctx.resolveChunkCollection('docstring', addr)).toBe(addr.code);
        expect(ctx.resolveChunkCollection('doc', addr)).toBe(addr.prose);
        expect(ctx.resolveChunkCollection('code_example', addr)).toBe(addr.prose);
        // Defensive default: unknown / undefined content_type → code-side.
        expect(ctx.resolveChunkCollection(undefined, addr)).toBe(addr.code);
        expect(ctx.resolveChunkCollection('mystery', addr)).toBe(addr.code);
    });

    it('resolveChunkCollection in legacy mode: any content_type → legacy', () => {
        process.env.COLLECTION_VERSION = 'v3';
        const ctx = makeCtx();
        const addr = ctx.getCollectionAddress('/path/to/codebase');
        for (const ct of ['code', 'docstring', 'doc', 'code_example', undefined, 'mystery']) {
            expect(ctx.resolveChunkCollection(ct as any, addr)).toBe(addr.legacy);
        }
    });

    it('legacy hash byte-stable: same path / same env → same name across calls', () => {
        process.env.COLLECTION_VERSION = 'v3';
        const ctx = makeCtx();
        const a = ctx.getCollectionName('/some/path/abc');
        const b = ctx.getCollectionName('/some/path/abc');
        expect(a).toBe(b);
    });
});

// prose-embedding-swap: per-pool dense embedder resolution. The default
// invariant — when no distinct prose embedder is wired (or the same instance
// is passed) every pool resolves to the code embedder — guarantees the
// PROSE_DENSE_MODEL=bge-m3 path is byte-identical to v6.
describe('prose-embedding-swap: embeddingForPool / hasDistinctProseEmbedding', () => {
    function ctxWith(prose?: Embedding): { ctx: Context; code: Embedding; prose?: Embedding } {
        const code = new TestEmbedding();
        const ctx = new Context({
            embedding: code,
            ...(prose && { proseEmbedding: prose }),
            vectorDatabase: makeVectorDb(),
        });
        return { ctx, code, prose };
    }

    it('no prose embedder → prose pool resolves to the code embedder', () => {
        const { ctx, code } = ctxWith();
        expect((ctx as any).hasDistinctProseEmbedding()).toBe(false);
        expect((ctx as any).embeddingForPool('prose')).toBe(code);
        expect((ctx as any).embeddingForPool('code')).toBe(code);
    });

    it('same instance passed as proseEmbedding collapses to no-distinct (default invariant)', () => {
        const code = new TestEmbedding();
        const ctx = new Context({ embedding: code, proseEmbedding: code, vectorDatabase: makeVectorDb() });
        expect((ctx as any).hasDistinctProseEmbedding()).toBe(false);
        expect((ctx as any).embeddingForPool('prose')).toBe(code);
    });

    it('distinct prose embedder → prose pool uses it, code pool stays on bge-m3 embedder', () => {
        const prose = new TestEmbedding();
        const { ctx, code } = ctxWith(prose);
        expect((ctx as any).hasDistinctProseEmbedding()).toBe(true);
        expect((ctx as any).embeddingForPool('prose')).toBe(prose);
        expect((ctx as any).embeddingForPool('code')).toBe(code);
        expect((ctx as any).embeddingForPool('code')).not.toBe(prose);
    });
});
