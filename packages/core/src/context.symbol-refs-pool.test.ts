// rag-symbol-refs-lsp-pool: integration tests for the Context.runSymbolRefsPoolForSubject
// wiring + getter shapes. Mocks the Serena LSP client and the Milvus query
// surface — exercises only the activation gates, env getters and the
// Context-side glue.

import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { VectorDatabase } from './vectordb';

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

function makeCtx(): Context {
    return new Context({
        embedding: new TestEmbedding(),
        vectorDatabase: createVectorDatabase(),
    });
}

const SAVED_KEYS = [
    'SYMBOL_REFS_POOL',
    'SYMBOL_REFS_POOL_WEIGHT',
    'SYMBOL_REFS_LSP_BASE_URL',
    'SYMBOL_REFS_LSP_TIMEOUT_MS',
    'SYMBOL_REFS_MAX_REFERENCES',
    'SYMBOL_REFS_MAX_IMPLEMENTATIONS',
];

describe('Context — symbol-refs-pool env getters', () => {
    let saved: Record<string, string | undefined> = {};
    beforeEach(() => {
        saved = {};
        for (const k of SAVED_KEYS) { saved[k] = process.env[k]; delete process.env[k]; }
    });
    afterEach(() => {
        for (const k of SAVED_KEYS) {
            if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k];
        }
    });

    it('defaults: pool off, weight 1.0, timeout 1500, refs/impls 20/10', () => {
        const ctx = makeCtx() as any;
        expect(ctx.getSymbolRefsPool()).toBe(false);
        expect(ctx.getSymbolRefsPoolWeight()).toBe(1.0);
        expect(ctx.getSymbolRefsLspTimeoutMs()).toBe(1500);
        expect(ctx.getSymbolRefsMaxReferences()).toBe(20);
        expect(ctx.getSymbolRefsMaxImplementations()).toBe(10);
        expect(ctx.getSymbolRefsLspBaseUrl()).toBeUndefined();
    });

    it('reads SYMBOL_REFS_POOL=true and clamps weight to [0, 3]', () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        process.env.SYMBOL_REFS_POOL_WEIGHT = '5.0';
        const ctx = makeCtx() as any;
        expect(ctx.getSymbolRefsPool()).toBe(true);
        expect(ctx.getSymbolRefsPoolWeight()).toBe(3.0);
    });

    it('rejects negative weight by falling back to default', () => {
        process.env.SYMBOL_REFS_POOL_WEIGHT = '-1';
        const ctx = makeCtx() as any;
        expect(ctx.getSymbolRefsPoolWeight()).toBe(1.0);
    });

    it('honours SYMBOL_REFS_LSP_BASE_URL override', () => {
        process.env.SYMBOL_REFS_LSP_BASE_URL = 'http://daemon:9999';
        const ctx = makeCtx() as any;
        expect(ctx.getSymbolRefsLspBaseUrl()).toBe('http://daemon:9999');
    });
});

describe('Context.runSymbolRefsPoolForSubject — activation gates', () => {
    let saved: Record<string, string | undefined> = {};
    beforeEach(() => {
        saved = {};
        for (const k of SAVED_KEYS) { saved[k] = process.env[k]; delete process.env[k]; }
    });
    afterEach(() => {
        for (const k of SAVED_KEYS) {
            if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k];
        }
    });

    function withVocab(ctx: any, vocab: Set<string>): void {
        ctx.symbolVocabCache.set('/codebase', vocab as ReadonlySet<string>);
    }

    function stubLsp(ctx: any, lsp: { findSymbol: any; findReferencingSymbols: any; findImplementations: any }): void {
        ctx.symbolRefsLspClient = lsp;
    }

    it('returns [] when SYMBOL_REFS_POOL is unset (default off)', async () => {
        const ctx = makeCtx() as any;
        const out = await ctx.runSymbolRefsPoolForSubject('Bytes.toString', { codeSignal: true, docSignal: false }, '/codebase', 'col');
        expect(out).toEqual([]);
    });

    it('returns [] when codeSignal is false', async () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        const ctx = makeCtx() as any;
        const out = await ctx.runSymbolRefsPoolForSubject('how can i open a file', { codeSignal: false, docSignal: true }, '/codebase', 'col');
        expect(out).toEqual([]);
    });

    it('returns [] when query parses neither as qualified name nor as single-vocab symbol', async () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        const ctx = makeCtx() as any;
        withVocab(ctx, new Set(['Bytes']));
        const out = await ctx.runSymbolRefsPoolForSubject('do thing now', { codeSignal: true, docSignal: false }, '/codebase', 'col');
        expect(out).toEqual([]);
    });

    it('returns [] when qualified-name className is missing from vocab', async () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        const ctx = makeCtx() as any;
        withVocab(ctx, new Set(['SomethingElse']));
        const out = await ctx.runSymbolRefsPoolForSubject('Bytes.toString', { codeSignal: true, docSignal: false }, '/codebase', 'col');
        expect(out).toEqual([]);
    });

    it('invokes runSymbolRefsPool when qualified-name + vocab + codeSignal align', async () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        const ctx = makeCtx() as any;
        withVocab(ctx, new Set(['Bytes', 'toString']));
        const lsp = {
            findSymbol: jest.fn(async () => []),
            findReferencingSymbols: jest.fn(async () => []),
            findImplementations: jest.fn(async () => []),
        };
        stubLsp(ctx, lsp);
        const dbQuery = (ctx.vectorDatabase.query as jest.Mock);
        // Declaration lookup → 1 chunk; hydration → same chunk.
        dbQuery.mockImplementation(async (_col: string, filter: string) => {
            if (/symbol_name == "Bytes"/.test(filter)) return [{ id: 'c1', relativePath: 'std/Bytes.hx' }];
            if (filter.startsWith('id in')) return [{ id: 'c1', relativePath: 'std/Bytes.hx', startLine: 0, endLine: 5, content: 'x', metadata: '{}' }];
            return [];
        });
        const out = await ctx.runSymbolRefsPoolForSubject('Bytes.toString', { codeSignal: true, docSignal: false }, '/codebase', 'col');
        expect(out.length).toBe(1);
        expect(lsp.findReferencingSymbols).toHaveBeenCalledWith('Bytes/toString', 'std/Bytes.hx', 20);
    });

    it('invokes runSymbolRefsPool with single-symbol parse when no qualifier present', async () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        const ctx = makeCtx() as any;
        withVocab(ctx, new Set(['BytesBuffer']));
        const lsp = {
            findSymbol: jest.fn(async () => []),
            findReferencingSymbols: jest.fn(async () => []),
            findImplementations: jest.fn(async () => []),
        };
        stubLsp(ctx, lsp);
        const dbQuery = (ctx.vectorDatabase.query as jest.Mock);
        dbQuery.mockImplementation(async (_col: string, filter: string) => {
            if (/symbol_name == "BytesBuffer"/.test(filter)) return [{ id: 'cb', relativePath: 'std/BytesBuffer.hx' }];
            if (filter.startsWith('id in')) return [{ id: 'cb', relativePath: 'std/BytesBuffer.hx', metadata: '{}' }];
            return [];
        });
        await ctx.runSymbolRefsPoolForSubject('how to use BytesBuffer', { codeSignal: true, docSignal: true }, '/codebase', 'col');
        expect(lsp.findReferencingSymbols).toHaveBeenCalledWith('BytesBuffer', 'std/BytesBuffer.hx', 20);
    });

    it('returns [] without throwing when runSymbolRefsPool internals reject', async () => {
        process.env.SYMBOL_REFS_POOL = 'true';
        const ctx = makeCtx() as any;
        withVocab(ctx, new Set(['Bytes', 'toString']));
        stubLsp(ctx, {
            findSymbol: jest.fn(async () => { throw new Error('LSP totally down'); }),
            findReferencingSymbols: jest.fn(async () => []),
            findImplementations: jest.fn(async () => []),
        });
        const out = await ctx.runSymbolRefsPoolForSubject('Bytes.toString', { codeSignal: true, docSignal: false }, '/codebase', 'col');
        expect(out).toEqual([]);
    });
});
