// sweep-symbol-refs-pool-weight: the pre-rerank candidate dump has to answer
// two questions the sweep is about — which pool put a chunk into the
// reranker's input, and whether it survived the cut. These tests cover the
// provenance the merge records, the dump the two halves write, and the dormant
// path when CANDIDATE_LOG_DIR is unset.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

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

function makeCtx(): any {
    return new Context({
        embedding: new TestEmbedding(),
        vectorDatabase: createVectorDatabase(),
    }) as any;
}

function hybridRow(id: string, score: number): any {
    return {
        document: {
            id,
            content: `content of ${id}`,
            relativePath: `src/${id}.ts`,
            startLine: 1,
            endLine: 10,
            content_type: 'code',
            symbol_name: id,
            metadata: {},
        },
        score,
    };
}

function semanticRow(id: string, score: number): any {
    return {
        content: `content of ${id}`,
        relativePath: `src/${id}.ts`,
        startLine: 1,
        endLine: 10,
        language: 'typescript',
        score,
        content_type: 'code',
        symbol_name: id,
        chunk_id: id,
    };
}

describe('Context — pre-rerank candidate dump', () => {
    let dir = '';
    let savedDir: string | undefined;
    let savedQid: string | undefined;

    beforeEach(() => {
        savedDir = process.env.CANDIDATE_LOG_DIR;
        savedQid = process.env.CANDIDATE_LOG_QID;
        dir = fs.mkdtempSync(path.join(os.tmpdir(), 'candidate-dump-'));
    });

    afterEach(() => {
        if (savedDir === undefined) delete process.env.CANDIDATE_LOG_DIR; else process.env.CANDIDATE_LOG_DIR = savedDir;
        if (savedQid === undefined) delete process.env.CANDIDATE_LOG_QID; else process.env.CANDIDATE_LOG_QID = savedQid;
        fs.rmSync(dir, { recursive: true, force: true });
    });

    it('arms the collector only when CANDIDATE_LOG_DIR is set', async () => {
        const ctx = makeCtx();

        process.env.CANDIDATE_LOG_DIR = dir;
        await ctx.semanticSearch('/no/such/codebase', 'query');
        expect(ctx.candidateProvenance).toBeInstanceOf(Map);

        delete process.env.CANDIDATE_LOG_DIR;
        await ctx.semanticSearch('/no/such/codebase', 'query');
        expect(ctx.candidateProvenance).toBeNull();
    });

    it('the merge records which pool contributed a chunk, and at what rank', () => {
        const ctx = makeCtx();
        ctx.candidateProvenance = new Map();

        ctx.weightedRrfMerge(
            [
                { results: [hybridRow('a', 0.9), hybridRow('b', 0.8)], weight: 1.0, name: 'code' },
                { results: [hybridRow('b', 0.5), hybridRow('r', 0.4)], weight: 0.6, name: 'symbolRefs' },
            ],
            10,
            60,
        );

        expect(ctx.candidateProvenance.get('a')).toEqual([{ pool: 'code', rank: 1 }]);
        expect(ctx.candidateProvenance.get('r')).toEqual([{ pool: 'symbolRefs', rank: 2 }]);
        // Found by both pools: one entry per pool, each at its own rank.
        expect(ctx.candidateProvenance.get('b')).toEqual([
            { pool: 'code', rank: 2 },
            { pool: 'symbolRefs', rank: 1 },
        ]);
    });

    it('an unnamed pool contributes no provenance', () => {
        const ctx = makeCtx();
        ctx.candidateProvenance = new Map();
        ctx.weightedRrfMerge([{ results: [hybridRow('a', 0.9)], weight: 1.0 }], 10, 60);
        expect(ctx.candidateProvenance.size).toBe(0);
    });

    it('a chunk seen twice in one pool keeps its best rank', () => {
        const ctx = makeCtx();
        ctx.candidateProvenance = new Map();
        // Two subjects of a comparison split both merge a 'code' pool.
        ctx.weightedRrfMerge([{ results: [hybridRow('x', 0.9), hybridRow('a', 0.1)], weight: 1.0, name: 'code' }], 10, 60);
        ctx.weightedRrfMerge([{ results: [hybridRow('a', 0.9), hybridRow('x', 0.1)], weight: 1.0, name: 'code' }], 10, 60);
        expect(ctx.candidateProvenance.get('a')).toEqual([{ pool: 'code', rank: 1 }]);
        expect(ctx.candidateProvenance.get('x')).toEqual([{ pool: 'code', rank: 1 }]);
    });

    it('writes pools and survived, keyed by chunk_id', () => {
        const ctx = makeCtx();
        process.env.CANDIDATE_LOG_DIR = dir;
        process.env.CANDIDATE_LOG_QID = 'q42';
        ctx.candidateProvenance = new Map([
            ['a', [{ pool: 'code', rank: 1 }]],
            ['r', [{ pool: 'symbolRefs', rank: 1 }]],
        ]);

        ctx.maybeStagePreRerankCandidates('who calls X', [semanticRow('a', 0.02), semanticRow('r', 0.01), semanticRow('z', 0.005)]);
        expect(ctx.pendingCandidateDump.rows).toHaveLength(3);

        // Only 'a' and 'r' come back from the reranker, and 'r' outranks 'a'
        // there — survival follows the returned list, not the staged order.
        ctx.flushPreRerankCandidateDump([semanticRow('r', 9), semanticRow('a', 8)], {
            topK: 10,
            rerankerBypassed: false,
            reservedBridgeSlots: 0,
            reservedRefSlots: 0,
            multiQuery: true,
        });

        const payload = JSON.parse(fs.readFileSync(path.join(dir, 'q42.json'), 'utf8'));
        expect(payload.qid).toBe('q42');
        expect(payload.count).toBe(3);
        expect(payload.survived).toBe(2);
        const byId: Record<string, any> = Object.fromEntries(payload.candidates.map((c: any) => [c.chunk_id, c]));
        expect(byId.r.pools).toEqual([{ pool: 'symbolRefs', rank: 1 }]);
        expect(byId.r.survived).toBe(true);
        // Staged second, returned first: the rank recorded is the one in the
        // list the caller receives.
        expect(byId.r.rank).toBe(2);
        expect(byId.r.finalRank).toBe(1);
        expect(byId.z.pools).toEqual([]);
        expect(byId.z.survived).toBe(false);
        expect(byId.z.finalRank).toBeNull();
        // The staged rows are consumed, so a second flush cannot write again.
        expect(ctx.pendingCandidateDump).toBeNull();
    });

    it('states the cut survival was measured at, and whether the reranker ran', () => {
        const ctx = makeCtx();
        process.env.CANDIDATE_LOG_DIR = dir;
        process.env.CANDIDATE_LOG_QID = 'bypassed';
        ctx.candidateProvenance = new Map();

        ctx.maybeStagePreRerankCandidates('haxe.Json.parse', [semanticRow('a', 0.02)]);
        ctx.flushPreRerankCandidateDump([semanticRow('a', 9)], {
            topK: 10,
            rerankerBypassed: true,
            reservedBridgeSlots: 2,
            reservedRefSlots: 1,
            multiQuery: true,
        });

        const payload = JSON.parse(fs.readFileSync(path.join(dir, 'bypassed.json'), 'utf8'));
        // topK is what `survived` means; RERANKER_OUTPUT_K is a different
        // number and is recorded separately so the two cannot be confused.
        expect(payload.topK).toBe(10);
        expect(payload.rerankerBypassed).toBe(true);
        expect(payload.reservedBridgeSlots).toBe(2);
        expect(payload.reservedRefSlots).toBe(1);
        expect(payload.multiQuery).toBe(true);
        expect(payload).toHaveProperty('rerankerOutputK');
        expect(payload).toHaveProperty('graphExpand');
        expect(payload).toHaveProperty('proseGraphExpand');
    });

    it('stages nothing and writes nothing when the dump is off', () => {
        const ctx = makeCtx();
        delete process.env.CANDIDATE_LOG_DIR;
        ctx.candidateProvenance = null;

        ctx.maybeStagePreRerankCandidates('who calls X', [semanticRow('a', 0.02)]);
        expect(ctx.pendingCandidateDump).toBeNull();

        ctx.flushPreRerankCandidateDump([semanticRow('a', 9)], {
            topK: 10,
            rerankerBypassed: false,
            reservedBridgeSlots: 0,
            reservedRefSlots: 0,
            multiQuery: true,
        });
        expect(fs.readdirSync(dir)).toEqual([]);
    });
});
