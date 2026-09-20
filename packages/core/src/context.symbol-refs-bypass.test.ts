// reserve-slots-for-reference-chunks: unit tests for the reserved-slot
// partition as the symbol-references pool uses it, and for its env bound.
// The partition itself is the one the comparison bridge already used; these
// tests pin that it reserves the right chunks for the new marker, that it is a
// no-op at the default, and that the bridge's own behaviour is unchanged by
// the parameter (its 14 tests stay where they are).

import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { VectorDatabase } from './vectordb';
import { SemanticSearchResult } from './types';

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

function makeCtx(): any {
    return new Context({
        embedding: new TestEmbedding(),
        vectorDatabase: createVectorDatabase(),
    }) as any;
}

function row(id: string, pool?: string): SemanticSearchResult {
    return {
        content: `content of ${id}`,
        relativePath: `src/${id}.ts`,
        startLine: 1,
        endLine: 10,
        language: 'typescript',
        score: 0.01,
        chunk_id: id,
        ...(pool ? { pool } : {}),
    } as SemanticSearchResult;
}

describe('Context — reserved slots for reference chunks', () => {
    const SAVED = ['SYMBOL_REFS_BYPASS_SLOTS', 'SYMBOL_REFS_LSP'];
    let saved: Record<string, string | undefined> = {};
    beforeEach(() => {
        saved = {};
        for (const k of SAVED) { saved[k] = process.env[k]; delete process.env[k]; }
    });
    afterEach(() => {
        for (const k of SAVED) {
            if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k];
        }
    });

    it('defaults to 0 slots, and clamps nonsense and excess', () => {
        const ctx = makeCtx();
        expect(ctx.getSymbolRefsBypassSlots()).toBe(0);
        process.env.SYMBOL_REFS_BYPASS_SLOTS = '2';
        expect(ctx.getSymbolRefsBypassSlots()).toBe(2);
        process.env.SYMBOL_REFS_BYPASS_SLOTS = '-3';
        expect(ctx.getSymbolRefsBypassSlots()).toBe(0);
        process.env.SYMBOL_REFS_BYPASS_SLOTS = 'two';
        expect(ctx.getSymbolRefsBypassSlots()).toBe(0);
        process.env.SYMBOL_REFS_BYPASS_SLOTS = '99';
        expect(ctx.getSymbolRefsBypassSlots()).toBe(10);
    });

    it('reserves nothing at 0 slots and hands the list through untouched', () => {
        const ctx = makeCtx();
        const candidates = [row('a'), row('r1', 'symbolRefs'), row('b')];
        const out = ctx.partitionForBridgeBypass(candidates, 10, 0, 'symbolRefs');
        expect(out.reserved).toEqual([]);
        expect(out.rerankInput).toBe(candidates);
    });

    it('reserves nothing when the pool did not fire', () => {
        const ctx = makeCtx();
        const candidates = [row('a'), row('b')];
        const out = ctx.partitionForBridgeBypass(candidates, 10, 2, 'symbolRefs');
        expect(out.reserved).toEqual([]);
        expect(out.rerankInput).toBe(candidates);
    });

    it('reserves the pool\'s own top chunks and keeps the rest for the reranker', () => {
        const ctx = makeCtx();
        const candidates = [row('a'), row('r1', 'symbolRefs'), row('b'), row('r2', 'symbolRefs'), row('r3', 'symbolRefs')];
        const out = ctx.partitionForBridgeBypass(candidates, 10, 2, 'symbolRefs');
        expect(out.reserved.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['r1', 'r2']);
        expect(out.rerankInput.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['a', 'b', 'r3']);
    });

    it('reserves no more than exist, and no more than the slots left', () => {
        const ctx = makeCtx();
        const candidates = [row('r1', 'symbolRefs'), row('a')];
        expect(ctx.partitionForBridgeBypass(candidates, 10, 5, 'symbolRefs').reserved).toHaveLength(1);
        // topK here is what is left after the bridge took its own slots.
        expect(ctx.partitionForBridgeBypass([row('r1', 'symbolRefs'), row('r2', 'symbolRefs')], 1, 2, 'symbolRefs').reserved).toHaveLength(1);
    });

    it('reserves in the POOL\'s order, not in the merged order (D2)', () => {
        // The finding this test exists for: collecting the pool's chunks into a
        // Set threw away their rank, and the partition then took whichever
        // reference the merged RRF order happened to like — that is, the one
        // the reranker was least likely to cut anyway, while the pool's rank-1
        // declaration stayed in the rerank input and got cut. The merged order
        // below is deliberately the reverse of the pool's.
        const ctx = makeCtx();
        const candidates = [row('r3', 'symbolRefs'), row('a'), row('r2', 'symbolRefs'), row('r1', 'symbolRefs')];
        const poolRank: Record<string, number> = { r1: 0, r2: 1, r3: 2 };
        const rankOf = (r: SemanticSearchResult) => poolRank[r.chunk_id as string] ?? Number.MAX_SAFE_INTEGER;
        const out = ctx.partitionForBridgeBypass(candidates, 10, 2, 'symbolRefs', rankOf);
        expect(out.reserved.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['r1', 'r2']);
        // Whatever is not reserved keeps its merged order for the reranker.
        expect(out.rerankInput.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['r3', 'a']);
    });

    it('without a rank function the merged order decides, as the bridge has always done', () => {
        const ctx = makeCtx();
        const candidates = [row('b1', 'comparisonBridge'), row('a'), row('b2', 'comparisonBridge')];
        const out = ctx.partitionForBridgeBypass(candidates, 10, 1);
        expect(out.reserved.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['b1']);
    });

    it('a chunk the pool never contributed sorts last rather than jumping the queue', () => {
        const ctx = makeCtx();
        const candidates = [row('unranked', 'symbolRefs'), row('r1', 'symbolRefs')];
        const rankOf = (r: SemanticSearchResult) => (r.chunk_id === 'r1' ? 0 : Number.MAX_SAFE_INTEGER);
        const out = ctx.partitionForBridgeBypass(candidates, 10, 1, 'symbolRefs', rankOf);
        expect(out.reserved.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['r1']);
    });

    it('SYMBOL_REFS_LSP defaults to on, and reads the three ways of saying off', () => {
        const ctx = makeCtx();
        expect(ctx.getSymbolRefsLspEnabled()).toBe(true);
        for (const off of ['false', 'FALSE', '0', 'off', 'Off']) {
            process.env.SYMBOL_REFS_LSP = off;
            expect(ctx.getSymbolRefsLspEnabled()).toBe(false);
        }
        for (const on of ['true', '1', 'yes', '']) {
            process.env.SYMBOL_REFS_LSP = on;
            expect(ctx.getSymbolRefsLspEnabled()).toBe(true);
        }
    });

    it('does not touch chunks another pool has claimed', () => {
        const ctx = makeCtx();
        const candidates = [row('bridge1', 'comparisonBridge'), row('r1', 'symbolRefs')];
        const refs = ctx.partitionForBridgeBypass(candidates, 10, 2, 'symbolRefs');
        expect(refs.reserved.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['r1']);
        // And the bridge's own partition, called without a marker, still picks
        // its own chunks — the parameter defaults to the bridge.
        const bridge = ctx.partitionForBridgeBypass(candidates, 10, 2);
        expect(bridge.reserved.map((r: SemanticSearchResult) => r.chunk_id)).toEqual(['bridge1']);
    });
});
