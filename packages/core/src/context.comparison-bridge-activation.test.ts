// audit-evidence-matching-and-bridge-reachability (2.1): the comparison bridge could not
// reach the served split index — `searchHybrid` gated it on a legacy COLLECTION_VERSION
// allowlist, and the v8 generations fail that test before the graph is even loaded. These
// tests drive the REAL `semanticSearch` branch (no Milvus, no Infinity: a mocked vector
// database and a temp-file graph) and pin four things:
//
//   1. a split address activates the bridge whatever the generation is called;
//   2. a non-split address at the same generation still does not (the legacy gate is
//      unchanged where it was the only signal);
//   3. GRAPH_EXPAND is NOT relaxed by the same edit;
//   4. `SYMBOL_GRAPH_FILE` selects which graph the loader reads.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

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

// One seed chunk (`EReg`, package `haxe.std`) and two partners the graph knows about.
const SEED = {
    id: 'chunk_seed0000000001',
    content: 'class EReg { function match(s:String):Bool; }',
    relativePath: 'haxe/std/EReg.hx',
    startLine: 39, endLine: 123,
    content_type: 'code', symbol_name: 'EReg', parent_symbol: 'haxe.std',
    metadata: JSON.stringify({ language: 'haxe' }),
};
const PARTNERS: Record<string, any> = {
    chunk_partner000001: { id: 'chunk_partner000001', content: 'class EReg helper A', relativePath: 'haxe/std/StringTools.hx', startLine: 1, endLine: 9, content_type: 'code', symbol_name: 'StringTools', metadata: '{}' },
    chunk_partner000002: { id: 'chunk_partner000002', content: 'class EReg helper B', relativePath: 'haxe/std/Std.hx', startLine: 1, endLine: 9, content_type: 'code', symbol_name: 'Std', metadata: '{}' },
};

const graphPayload = (partnerSymbols: string[]) => JSON.stringify({
    version: 'v3-3',
    by_symbol: {
        EReg: { canonical_chunk_ids: [SEED.id], mentioned_by_chunk_ids: [] },
        StringTools: { canonical_chunk_ids: ['chunk_partner000001'], mentioned_by_chunk_ids: [] },
        Std: { canonical_chunk_ids: ['chunk_partner000002'], mentioned_by_chunk_ids: [] },
    },
    by_package: { 'haxe.std': ['EReg', ...partnerSymbols] },
    by_supertype: { IMap: ['StringMap'] },
});

/** Every `query` call the search made, so an `id in [...]` fetch can be told from the probe. */
let queries: Array<{ collection: string; filter: string }> = [];

function makeVectorDb(): jest.Mocked<VectorDatabase> {
    return {
        createCollection: jest.fn().mockResolvedValue(undefined),
        createHybridCollection: jest.fn().mockResolvedValue(undefined),
        dropCollection: jest.fn().mockResolvedValue(undefined),
        hasCollection: jest.fn().mockResolvedValue(true),
        listCollections: jest.fn().mockResolvedValue([]),
        insert: jest.fn().mockResolvedValue(undefined),
        insertHybrid: jest.fn().mockResolvedValue(undefined),
        search: jest.fn().mockResolvedValue([]),
        hybridSearch: jest.fn().mockImplementation(async (collection: string) =>
            (collection.includes('prose') ? [] : [{ document: { ...SEED }, score: 0.9 }])),
        delete: jest.fn().mockResolvedValue(undefined),
        query: jest.fn().mockImplementation(async (collection: string, filter: string) => {
            queries.push({ collection, filter });
            const ids = [...String(filter || '').matchAll(/"([A-Za-z0-9_-]+)"/g)].map((m) => m[1]);
            return ids.map((id) => PARTNERS[id]).filter(Boolean);
        }),
        getCollectionDescription: jest.fn().mockResolvedValue(''),
        checkCollectionLimit: jest.fn().mockResolvedValue(true),
        getCollectionRowCount: jest.fn().mockResolvedValue(0),
    };
}

const QUERY = 'EReg.match vs EReg.replace difference';

describe('comparison-bridge activation on a split index', () => {
    const saved: Record<string, string | undefined> = {};
    const NAMES = [
        'COLLECTION_VERSION', 'SPLIT_COLLECTIONS', 'HYBRID_MODE', 'MULTI_QUERY',
        'COMPARISON_BRIDGE_ENABLED', 'COMPARISON_BRIDGE_MAX_PACKAGE_FANOUT',
        'GRAPH_EXPAND', 'SYMBOL_GRAPH_FILE', 'RERANKER_PROVIDER', 'SYMBOL_ROUTING_ENABLED',
        'SYMBOL_REFS_ENABLED', 'SYMBOL_REFS_LSP',
    ];
    let dir = '';

    beforeEach(() => {
        for (const n of NAMES) { saved[n] = process.env[n]; delete process.env[n]; }
        dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bridge-activation-'));
        fs.writeFileSync(path.join(dir, '.symbols-graph.json'), graphPayload(['StringTools']));
        process.env.HYBRID_MODE = 'true';
        process.env.MULTI_QUERY = 'true';
        process.env.RERANKER_PROVIDER = '';
        process.env.SYMBOL_REFS_LSP = 'false';
        queries = [];
    });

    afterEach(() => {
        for (const n of NAMES) { if (saved[n] === undefined) delete process.env[n]; else process.env[n] = saved[n]; }
        fs.rmSync(dir, { recursive: true, force: true });
    });

    const search = async () => {
        const ctx = new Context({ embedding: new TestEmbedding(), vectorDatabase: makeVectorDb() });
        return ctx.semanticSearch(dir, QUERY, 10, 0.0, undefined, 'comparison');
    };
    const bridged = (results: any[]) => results.filter((r) => r.pool === 'comparisonBridge').map((r) => r.chunk_id);

    it('activates on a v8 SPLIT address — the generation name alone no longer refuses it', async () => {
        process.env.COLLECTION_VERSION = 'v8p0';
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COMPARISON_BRIDGE_ENABLED = '1';
        expect(bridged(await search())).toEqual(['chunk_partner000001']);
    });

    it('still refuses a v8 NON-split address: the legacy gate is unchanged where it is the only signal', async () => {
        process.env.COLLECTION_VERSION = 'v8p0';
        process.env.SPLIT_COLLECTIONS = 'false';
        process.env.COMPARISON_BRIDGE_ENABLED = '1';
        expect(bridged(await search())).toEqual([]);
    });

    it('stays dormant on a split address while the flag is off', async () => {
        process.env.COLLECTION_VERSION = 'v8p0';
        process.env.SPLIT_COLLECTIONS = 'true';
        expect(bridged(await search())).toEqual([]);
        expect(queries.filter((q) => q.filter.startsWith('id in ['))).toEqual([]);
    });

    it('does NOT relax GRAPH_EXPAND: no hop expansion is fetched on a v8 split address', async () => {
        process.env.COLLECTION_VERSION = 'v8p0';
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.GRAPH_EXPAND = '1';
        await search();
        expect(queries.filter((q) => q.filter.startsWith('id in ['))).toEqual([]);
    });

    it('SYMBOL_GRAPH_FILE selects the graph the loader reads; unset keeps the codebase file', async () => {
        process.env.COLLECTION_VERSION = 'v8p0';
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COMPARISON_BRIDGE_ENABLED = '1';

        const side = path.join(dir, 'side-graph.json');
        fs.writeFileSync(side, graphPayload(['Std']));
        process.env.SYMBOL_GRAPH_FILE = side;
        expect(bridged(await search())).toEqual(['chunk_partner000002']);

        delete process.env.SYMBOL_GRAPH_FILE;
        expect(bridged(await search())).toEqual(['chunk_partner000001']);
    });

    it('a selected graph that is missing degrades to no bridge and never touches the index', async () => {
        process.env.COLLECTION_VERSION = 'v8p0';
        process.env.SPLIT_COLLECTIONS = 'true';
        process.env.COMPARISON_BRIDGE_ENABLED = '1';
        process.env.SYMBOL_GRAPH_FILE = path.join(dir, 'no-such-graph.json');
        expect(bridged(await search())).toEqual([]);
        expect(queries.filter((q) => q.filter.startsWith('id in ['))).toEqual([]);
    });
});
