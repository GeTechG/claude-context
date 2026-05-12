// rag-comparison-bridge-reranker-bypass: unit tests for the pool-aware
// reranker bypass partition logic. Targets `Context.partitionForBridgeBypass`
// — a pure helper extracted from `searchHybrid` so the reservation behavior
// can be validated without spinning up Milvus / Infinity / the bridge module.
//
// Integration coverage (bridge → reranker wiring inside `searchHybrid`) lives
// in the bake-off harness; these tests cover only the partition contract.

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

function makeCtx(): Context {
    const prevHybrid = process.env.HYBRID_MODE;
    process.env.HYBRID_MODE = 'false';
    const ctx = new Context({
        embedding: new TestEmbedding(),
        vectorDatabase: createVectorDatabase(),
    });
    if (prevHybrid === undefined) delete process.env.HYBRID_MODE;
    else process.env.HYBRID_MODE = prevHybrid;
    return ctx;
}

function makeResult(over: Partial<SemanticSearchResult> & { id: string }): SemanticSearchResult {
    return {
        content: `content-${over.id}`,
        relativePath: `path/${over.id}.hx`,
        startLine: 1,
        endLine: 10,
        language: 'haxe',
        score: 1,
        chunk_id: over.id,
        ...over,
    };
}

function bridge(id: string): SemanticSearchResult {
    return makeResult({ id, pool: 'comparisonBridge' });
}

function rest(id: string): SemanticSearchResult {
    return makeResult({ id });
}

function getEnvGetterValue(ctx: Context, key: string): number {
    const saved = process.env[key];
    try {
        process.env[key] = String(saved);
        return (ctx as any).getComparisonBridgeBypassSlots();
    } finally {
        if (saved === undefined) delete process.env[key];
        else process.env[key] = saved;
    }
}

describe('Context.partitionForBridgeBypass (rag-comparison-bridge-reranker-bypass)', () => {
    let savedSlots: string | undefined;

    beforeEach(() => {
        savedSlots = process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS;
        delete process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS;
    });

    afterEach(() => {
        if (savedSlots === undefined) delete process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS;
        else process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS = savedSlots;
    });

    it('bypassSlots=0 → no reservation; all candidates go to reranker', () => {
        const ctx = makeCtx();
        const candidates = [bridge('b1'), bridge('b2'), rest('r1'), rest('r2')];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 0);
        expect(out.reserved).toEqual([]);
        expect(out.rerankInput).toEqual(candidates);
        expect(out.rerankInput).toHaveLength(4);
    });

    it('bypassSlots=3 + 0 bridge chunks → fall-through; identical to bypassSlots=0', () => {
        const ctx = makeCtx();
        const candidates = [rest('r1'), rest('r2'), rest('r3')];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 3);
        expect(out.reserved).toEqual([]);
        expect(out.rerankInput).toEqual(candidates);
    });

    it('bypassSlots=3 + 5 bridge chunks → 3 reserved (in order), 2 leftover + rest go to reranker', () => {
        const ctx = makeCtx();
        const candidates = [
            bridge('b1'), bridge('b2'), bridge('b3'), bridge('b4'), bridge('b5'),
            rest('r1'), rest('r2'),
        ];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 3);
        expect(out.reserved.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['b1', 'b2', 'b3']);
        expect(out.rerankInput.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['b4', 'b5', 'r1', 'r2']);
        expect(out.reserved.length + out.rerankInput.length).toBe(candidates.length);
    });

    it('bypassSlots=5 + 3 bridge chunks → 3 reserved (no padding for missing 2)', () => {
        const ctx = makeCtx();
        const candidates = [
            bridge('b1'), bridge('b2'), bridge('b3'),
            rest('r1'), rest('r2'), rest('r3'), rest('r4'),
        ];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 5);
        expect(out.reserved.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['b1', 'b2', 'b3']);
        // bypass does NOT pad reserved with non-bridge chunks; missing slots
        // fall to the reranker so the final composition gets `topK -
        // reserved.length = 12` reranked candidates.
        expect(out.reserved).toHaveLength(3);
        expect(out.rerankInput.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['r1', 'r2', 'r3', 'r4']);
    });

    it('preserves pool RRF order in reservation (highest-rank bridge chunks first)', () => {
        const ctx = makeCtx();
        // Bridge chunks intentionally interleaved with rest at non-trivial
        // positions; partition must pick the first N bridge chunks in the
        // order they appear in the input (which mirrors weighted-RRF rank).
        const candidates = [
            rest('r1'),
            bridge('b1'),
            rest('r2'),
            bridge('b2'),
            rest('r3'),
            bridge('b3'),
            bridge('b4'),
        ];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 2);
        expect(out.reserved.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['b1', 'b2']);
        expect(out.rerankInput.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['r1', 'r2', 'r3', 'b3', 'b4']);
    });

    it('chunks without `pool === "comparisonBridge"` marker are not eligible for reservation', () => {
        const ctx = makeCtx();
        // Mix of pool markers and missing pool field. Only `comparisonBridge`
        // is eligible; `symbolRefs` / undefined chunks fall through.
        const candidates = [
            makeResult({ id: 's1', pool: 'symbolRefs' }),
            rest('r1'),
            bridge('b1'),
            makeResult({ id: 's2', pool: 'symbolRefs' }),
            rest('r2'),
            bridge('b2'),
            makeResult({ id: 'x1', pool: 'someOtherPool' }),
        ];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 5);
        expect(out.reserved.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['b1', 'b2']);
        expect(out.rerankInput).toHaveLength(5);
        for (const r of out.rerankInput) {
            expect(r.pool).not.toBe('comparisonBridge');
        }
    });

    it('isComparisonShape false (= no bridge chunks emitted by upstream) → bypass no-op even with bypassSlots > 0', () => {
        const ctx = makeCtx();
        // Simulates the integration path where `isComparisonShape(query) ===
        // false` → bridge pool never built → no chunks carry the marker.
        // bypassSlots > 0 is non-actionable; reservation stays empty.
        const candidates = [
            rest('r1'),
            rest('r2'),
            rest('r3'),
            rest('r4'),
        ];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 5);
        expect(out.reserved).toEqual([]);
        expect(out.rerankInput).toEqual(candidates);
    });

    it('reservation count + rerankerSlots = topK (final top-K size invariant)', () => {
        const ctx = makeCtx();
        const candidates = [
            bridge('b1'), bridge('b2'), bridge('b3'),
            rest('r1'), rest('r2'), rest('r3'), rest('r4'), rest('r5'),
            rest('r6'), rest('r7'), rest('r8'), rest('r9'), rest('r10'),
            rest('r11'), rest('r12'), rest('r13'),
        ];
        const topK = 15;
        for (const slots of [0, 2, 3, 5]) {
            const out = (ctx as any).partitionForBridgeBypass(candidates, topK, slots);
            const rerankerSlots = Math.max(topK - out.reserved.length, 0);
            // The integration call site then asks the reranker to return at
            // most `rerankerSlots` chunks. Final composition size therefore
            // capped at `reserved.length + rerankerSlots = topK`.
            expect(out.reserved.length + rerankerSlots).toBe(topK);
        }
    });

    it('reservationCount also clamped by topK (does not exceed final size)', () => {
        const ctx = makeCtx();
        // 8 bridge chunks available, bypassSlots=10 — but topK=5 caps the
        // reservation. (Wouldn't normally happen in production — topK is
        // typically 15 — but the partition contract should be robust.)
        const candidates = Array.from({ length: 8 }, (_, i) => bridge(`b${i + 1}`));
        const out = (ctx as any).partitionForBridgeBypass(candidates, 5, 10);
        expect(out.reserved).toHaveLength(5);
        expect(out.reserved.map((c: SemanticSearchResult) => c.chunk_id)).toEqual(['b1', 'b2', 'b3', 'b4', 'b5']);
    });

    it('reserved bridge chunks retain `pool: "comparisonBridge"` marker in output (downstream diagnostic relies on it)', () => {
        const ctx = makeCtx();
        const candidates = [bridge('b1'), bridge('b2'), rest('r1')];
        const out = (ctx as any).partitionForBridgeBypass(candidates, 15, 2);
        expect(out.reserved).toHaveLength(2);
        for (const r of out.reserved) {
            expect(r.pool).toBe('comparisonBridge');
        }
    });
});

describe('Context.getComparisonBridgeBypassSlots (env getter)', () => {
    let savedSlots: string | undefined;

    beforeEach(() => {
        savedSlots = process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS;
        delete process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS;
    });

    afterEach(() => {
        if (savedSlots === undefined) delete process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS;
        else process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS = savedSlots;
    });

    it('returns 0 when env not set (default-off invariant)', () => {
        const ctx = makeCtx();
        expect(getEnvGetterValue(ctx, 'COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS')).toBe(0);
    });

    it('returns the integer value when set in valid range', () => {
        process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS = '3';
        const ctx = makeCtx();
        expect((ctx as any).getComparisonBridgeBypassSlots()).toBe(3);
    });

    it('clamps values above 10 to 10', () => {
        process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS = '999';
        const ctx = makeCtx();
        expect((ctx as any).getComparisonBridgeBypassSlots()).toBe(10);
    });

    it('returns 0 for negative or non-numeric values', () => {
        const ctx = makeCtx();
        process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS = '-3';
        expect((ctx as any).getComparisonBridgeBypassSlots()).toBe(0);
        process.env.COMPARISON_BRIDGE_RERANKER_BYPASS_SLOTS = 'abc';
        expect((ctx as any).getComparisonBridgeBypassSlots()).toBe(0);
    });
});
