import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { SemanticSearchResult } from './types';
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

function makeResult(relativePath: string, symbol_name: string, opts: Partial<SemanticSearchResult> = {}): SemanticSearchResult {
    return {
        content: 'x',
        relativePath,
        startLine: opts.startLine ?? 1,
        endLine: opts.endLine ?? 10,
        language: 'haxe',
        score: opts.score ?? 0.5,
        content_type: opts.content_type ?? 'code',
        symbol_name,
        symbol_kind: opts.symbol_kind,
        parent_symbol: opts.parent_symbol,
    };
}

function makeCtx(): Context {
    const prevHybrid = process.env.HYBRID_MODE;
    process.env.HYBRID_MODE = 'false';
    const ctx = new Context({ embedding: new TestEmbedding(), vectorDatabase: createVectorDatabase() });
    if (prevHybrid === undefined) delete process.env.HYBRID_MODE;
    else process.env.HYBRID_MODE = prevHybrid;
    return ctx;
}

describe('applyCanonicalDedup (Phase B)', () => {
    let savedCanonicalDedup: string | undefined;
    let savedMarkers: string | undefined;

    beforeEach(() => {
        savedCanonicalDedup = process.env.CANONICAL_DEDUP;
        savedMarkers = process.env.PATH_DEMOTE_MARKERS;
        delete process.env.CANONICAL_DEDUP;
        delete process.env.PATH_DEMOTE_MARKERS;
    });

    afterEach(() => {
        if (savedCanonicalDedup === undefined) delete process.env.CANONICAL_DEDUP;
        else process.env.CANONICAL_DEDUP = savedCanonicalDedup;
        if (savedMarkers === undefined) delete process.env.PATH_DEMOTE_MARKERS;
        else process.env.PATH_DEMOTE_MARKERS = savedMarkers;
    });

    it('keeps canonical and drops clones with demote-markers in the same (symbol, basename) cluster', () => {
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/php/_std/Std.hx', 'parseInt'),
            makeResult('haxe/std/Std.hx', 'parseInt'),
            makeResult('haxe/std/lua/_std/Std.hx', 'parseInt'),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => r.relativePath)).toEqual(['haxe/std/Std.hx']);
    });

    it('leaves cluster intact when only clones are present (no canonical)', () => {
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/php/_std/Std.hx', 'parseInt'),
            makeResult('haxe/std/lua/_std/Std.hx', 'parseInt'),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => r.relativePath)).toEqual([
            'haxe/std/php/_std/Std.hx',
            'haxe/std/lua/_std/Std.hx',
        ]);
    });

    it('among multiple canonicals keeps the shortest path', () => {
        const ctx = makeCtx();
        const results = [
            makeResult('packages/lib/src/Foo.ts', 'bar'),
            makeResult('src/Foo.ts', 'bar'),
            makeResult('apps/web/src/Foo.ts', 'bar'),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => r.relativePath)).toEqual(['src/Foo.ts']);
    });

    it('CANONICAL_DEDUP=false → no path-cluster pruning', () => {
        process.env.CANONICAL_DEDUP = 'false';
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/php/_std/Std.hx', 'parseInt'),
            makeResult('haxe/std/Std.hx', 'parseInt'),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out).toHaveLength(2);
    });

    it('PATH_DEMOTE_MARKERS=foo,bar overrides defaults', () => {
        process.env.PATH_DEMOTE_MARKERS = 'foo,bar';
        const ctx = makeCtx();
        const results = [
            makeResult('node_modules/lib/Util.ts', 'doThing'),  // no longer a marker — competes as canonical
            makeResult('foo/lib/Util.ts', 'doThing'),           // marker → demoted
            makeResult('src/Util.ts', 'doThing'),               // canonical, shortest path
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        // Two canonicals (node_modules, src) collapse to shortest path. foo/ is dropped.
        expect(out.map((r) => r.relativePath)).toEqual(['src/Util.ts']);
    });

    it('PATH_DEMOTE_MARKERS empty (custom override) → no clustering', () => {
        process.env.PATH_DEMOTE_MARKERS = '';
        // Empty env falls back to defaults per getPathDemoteMarkers; verify by setting a single nonsense marker.
        process.env.PATH_DEMOTE_MARKERS = 'no_such_segment_xyz';
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/php/_std/Std.hx', 'parseInt'),  // _std no longer a marker
            makeResult('haxe/std/Std.hx', 'parseInt'),           // shorter path wins
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => r.relativePath)).toEqual(['haxe/std/Std.hx']);
    });

    it('different basenames are treated as distinct clusters', () => {
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/Std.hx', 'parseInt'),
            makeResult('haxe/std/StringTools.hx', 'parseInt'),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out).toHaveLength(2);
    });

    it('by default keeps one row per cluster, even of one path', () => {
        const ctx = makeCtx();
        const results = [
            makeResult('src/Context.hx', 'Context', { startLine: 855, endLine: 926 }),
            makeResult('src/Context.hx', 'Context', { startLine: 39, endLine: 123 }),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => r.startLine)).toEqual([855]);
    });

    it('CANONICAL_DEDUP=path keeps every piece of one class in one file', () => {
        process.env.CANONICAL_DEDUP = 'path';
        const ctx = makeCtx();
        const results = [
            makeResult('src/Context.hx', 'Context', { startLine: 855, endLine: 926 }),
            makeResult('src/Context.hx', 'Context', { startLine: 39, endLine: 123 }),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => r.startLine)).toEqual([855, 39]);
    });

    it('CANONICAL_DEDUP=path keeps every row of the canonical path and drops every row of a vendored copy', () => {
        process.env.CANONICAL_DEDUP = 'path';
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/php/_std/Std.hx', 'Std', { startLine: 1, endLine: 80 }),
            makeResult('haxe/std/Std.hx', 'Std', { startLine: 100, endLine: 180 }),
            makeResult('haxe/std/php/_std/Std.hx', 'Std', { startLine: 90, endLine: 160 }),
            makeResult('haxe/std/Std.hx', 'Std', { startLine: 1, endLine: 90 }),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out.map((r) => `${r.relativePath}:${r.startLine}`)).toEqual([
            'haxe/std/Std.hx:100',
            'haxe/std/Std.hx:1',
        ]);
    });

    it('CANONICAL_DEDUP=path leaves an all-clone group with several rows per path intact', () => {
        process.env.CANONICAL_DEDUP = 'path';
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/php/_std/Std.hx', 'Std', { startLine: 1, endLine: 80 }),
            makeResult('haxe/std/lua/_std/Std.hx', 'Std', { startLine: 1, endLine: 80 }),
            makeResult('haxe/std/php/_std/Std.hx', 'Std', { startLine: 90, endLine: 160 }),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out).toHaveLength(3);
    });

    it('results without symbol_name are not clustered', () => {
        const ctx = makeCtx();
        const results = [
            makeResult('haxe/std/Std.hx', '', { symbol_name: undefined } as any),
            makeResult('haxe/std/php/_std/Std.hx', '', { symbol_name: undefined } as any),
        ];
        const out: SemanticSearchResult[] = (ctx as any).applyCanonicalDedup(results);
        expect(out).toHaveLength(2);
    });

    it('overlap dedup: a skeleton row never covers the member rows its range spans', () => {
        const ctx = makeCtx();
        const skeleton = { ...makeResult('src/Widget.ts', 'Widget', { startLine: 1, endLine: 80 }), skeleton: true };
        const member = makeResult('src/Widget.ts', 'draw', { startLine: 30, endLine: 40 });
        const piece = makeResult('src/Widget.ts', 'Widget', { startLine: 1, endLine: 80 });
        const out: SemanticSearchResult[] = (ctx as any).deduplicateResults([skeleton, member]);
        expect(out.map((r) => r.symbol_name)).toEqual(['Widget', 'draw']);
        const memberFirst: SemanticSearchResult[] = (ctx as any).deduplicateResults([
            makeResult('src/Widget.ts', 'big', { startLine: 7, endLine: 70 }), skeleton]);
        expect(memberFirst.map((r) => r.symbol_name)).toEqual(['big', 'Widget']);
        const verbatim: SemanticSearchResult[] = (ctx as any).deduplicateResults([piece, member]);
        expect(verbatim.map((r) => r.symbol_name)).toEqual(['Widget']);
    });
});
