import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { VectorDatabase } from './vectordb';
import { Reranker, RerankResult } from './reranker';

class TestEmbedding extends Embedding {
    protected maxTokens = 8192;
    async detectDimension(): Promise<number> { return 3; }
    async embed(_text: string): Promise<EmbeddingVector> {
        return { vector: [1, 0, 0], dimension: 3 };
    }
    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 }));
    }
    getDimension(): number { return 3; }
    getProvider(): string { return 'test'; }
}

class StubReranker implements Reranker {
    public calls: Array<{ query: string; documents: string[] }> = [];
    async rerank(query: string, documents: string[]): Promise<RerankResult[]> {
        this.calls.push({ query, documents });
        return documents.map((_d, i) => ({ index: i, score: 1.0 - i / 100 }));
    }
    getProvider(): string { return 'stub'; }
}

const createVectorDatabase = (): jest.Mocked<VectorDatabase> => ({
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
});

function makeCtx(reranker?: Reranker): Context {
    const prevHybrid = process.env.HYBRID_MODE;
    process.env.HYBRID_MODE = 'false';
    const ctx = new Context({
        embedding: new TestEmbedding(),
        vectorDatabase: createVectorDatabase(),
        ...(reranker && { reranker }),
    });
    if (prevHybrid === undefined) delete process.env.HYBRID_MODE;
    else process.env.HYBRID_MODE = prevHybrid;
    return ctx;
}

describe('shouldBypassReranker (Phase R)', () => {
    let savedBypass: string | undefined;

    beforeEach(() => {
        savedBypass = process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME;
        delete process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME;
    });

    afterEach(() => {
        if (savedBypass === undefined) delete process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME;
        else process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME = savedBypass;
    });

    it('returns false when env not set (default off, no behavior change)', () => {
        const ctx = makeCtx();
        const out = (ctx as any).shouldBypassReranker('Std.parseInt', { codeSignal: true, docSignal: false });
        expect(out).toBe(false);
    });

    it('returns true when env=true AND code-only AND qualified-name', () => {
        process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME = 'true';
        const ctx = makeCtx();
        const out = (ctx as any).shouldBypassReranker('Std.parseInt', { codeSignal: true, docSignal: false });
        expect(out).toBe(true);
    });

    it('returns false for NL-only queries even when env=true', () => {
        process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME = 'true';
        const ctx = makeCtx();
        const out = (ctx as any).shouldBypassReranker('how to read a file', { codeSignal: false, docSignal: true });
        expect(out).toBe(false);
    });

    it('returns false for embedded NL even if both signals are on (parser anchored)', () => {
        process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME = 'true';
        const ctx = makeCtx();
        // parseQualifiedName is anchored — surrounding NL prose makes it return null.
        const out = (ctx as any).shouldBypassReranker('Lambda.fold reduce list to single value', { codeSignal: true, docSignal: true });
        expect(out).toBe(false);
    });

    it('returns true for multi-component qualified names that the classifier flags docSignal=true', () => {
        process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME = 'true';
        const ctx = makeCtx();
        // `haxe.io.Path.join` has 4 identifier-shaped words >=3 chars, so the
        // classifier marks docSignal=true; the parser still recognises it as a
        // pure qualified name and bypass should engage.
        const out = (ctx as any).shouldBypassReranker('haxe.io.Path.join', { codeSignal: true, docSignal: true });
        expect(out).toBe(true);
    });

    it('returns false for single identifier (parser yields null)', () => {
        process.env.RERANKER_BYPASS_FOR_QUALIFIED_NAME = 'true';
        const ctx = makeCtx();
        const out = (ctx as any).shouldBypassReranker('parseInt', { codeSignal: true, docSignal: false });
        expect(out).toBe(false);
    });
});
