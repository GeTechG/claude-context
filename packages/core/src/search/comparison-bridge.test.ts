// rag-graph-comparison-bridge: unit tests for the 5th-pool builder.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { GraphIndex } from './graph-expansion';
import { buildComparisonBridgePool } from './comparison-bridge';
import { SemanticSearchResult } from '../types';

function tmpFile(content: string): string {
    const f = path.join(os.tmpdir(), `cb-graph-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);
    fs.writeFileSync(f, content, 'utf-8');
    return f;
}

function makeSeed(over: Partial<SemanticSearchResult>): SemanticSearchResult {
    return {
        content: '',
        relativePath: '',
        startLine: 0,
        endLine: 0,
        language: 'haxe',
        score: 1,
        content_type: 'code',
        ...over,
    };
}

const V32_PAYLOAD = {
    version: 'v3-2',
    by_symbol: {
        Bytes: { canonical_chunk_ids: ['c_bytes'], mentioned_by_chunk_ids: [] },
        BytesBuffer: { canonical_chunk_ids: ['c_bb'], mentioned_by_chunk_ids: [] },
        Input: { canonical_chunk_ids: ['c_input'], mentioned_by_chunk_ids: [] },
        Output: { canonical_chunk_ids: ['c_output'], mentioned_by_chunk_ids: [] },
        IntMap: { canonical_chunk_ids: ['c_im'], mentioned_by_chunk_ids: [] },
        ObjectMap: { canonical_chunk_ids: ['c_om'], mentioned_by_chunk_ids: [] },
        StringMap: { canonical_chunk_ids: ['c_sm'], mentioned_by_chunk_ids: [] },
        // mentions-only entry → bridge should skip (no canonical)
        AbstractRef: { canonical_chunk_ids: [], mentioned_by_chunk_ids: ['c_doc'] },
    },
    by_package: {
        'haxe.io': ['Bytes', 'BytesBuffer', 'Input', 'Output'],
        'haxe.ds': ['IntMap', 'ObjectMap', 'StringMap'],
        'big.pkg': Array.from({ length: 25 }, (_, i) => `Sym${i}`),
    },
    by_supertype: {
        IMap: ['IntMap', 'ObjectMap', 'StringMap'],
    },
};

// rag-graph-abstract-typedef-edges: v3-3 payload with both new top-level
// indices populated and the matching per-symbol forward attributes on
// `by_symbol`. Layout mirrors the production `persistGraphSideIndex`
// writer: every chunk declaring `abstract Foo(T) from A to B` contributes
// to `by_abstract_underlying[T|A|B]` with value `Foo`, and Foo's own
// `by_symbol` entry carries the union of T/A/B in `abstract_underlying`.
const V33_PAYLOAD = {
    version: 'v3-3',
    by_symbol: {
        Bytes: {
            canonical_chunk_ids: ['c_bytes'],
            mentioned_by_chunk_ids: [],
            abstract_underlying: ['Array', 'BytesData'],
        },
        BytesData: { canonical_chunk_ids: ['c_bd'], mentioned_by_chunk_ids: [] },
        ReadOnlyArray: {
            canonical_chunk_ids: ['c_roa'],
            mentioned_by_chunk_ids: [],
            abstract_underlying: ['Array', 'Iterable'],
        },
        Array: { canonical_chunk_ids: ['c_arr'], mentioned_by_chunk_ids: [] },
        Iterable: { canonical_chunk_ids: ['c_iter'], mentioned_by_chunk_ids: [] },
        Null: {
            canonical_chunk_ids: ['c_null'],
            mentioned_by_chunk_ids: [],
            typedef_alias: 'T',
        },
        Maybe: {
            canonical_chunk_ids: ['c_maybe'],
            mentioned_by_chunk_ids: [],
            typedef_alias: 'T',
        },
        T: { canonical_chunk_ids: ['c_t'], mentioned_by_chunk_ids: [] },
    },
    by_package: {},
    by_supertype: {},
    by_abstract_underlying: {
        Array: ['Bytes', 'ReadOnlyArray'],
        BytesData: ['Bytes'],
        Iterable: ['ReadOnlyArray'],
    },
    by_typedef_alias: {
        T: ['Null', 'Maybe'],
    },
};

function loadGraph(): GraphIndex {
    const idx = GraphIndex.load(tmpFile(JSON.stringify(V32_PAYLOAD)));
    if (!idx) throw new Error('graph load failed in test setup');
    return idx;
}

describe('buildComparisonBridgePool', () => {
    it('returns empty when graph version is v3-1 (no comparison-bridge support)', () => {
        const v31 = GraphIndex.load(tmpFile(JSON.stringify({
            version: 'v3-1',
            by_symbol: { Bytes: { canonical_chunk_ids: ['c_bytes'] } },
        })));
        const seeds = [makeSeed({ symbol_name: 'Bytes', relativePath: 'haxe/io/Bytes.hx', chunk_id: 's1' })];
        const res = buildComparisonBridgePool(seeds, v31!, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).toEqual([]);
    });

    it('returns empty when no seeds provided', () => {
        const graph = loadGraph();
        const res = buildComparisonBridgePool([], graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).toEqual([]);
    });

    it('package partners derived from seed relativePath dir chain', () => {
        const graph = loadGraph();
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', relativePath: 'haxe/io/Bytes.hx', chunk_id: 'seed_bytes' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        // seed pkg derives to "haxe.io"; bucket = [Bytes, BytesBuffer, Input, Output]
        // self filtered → partners = BytesBuffer, Input, Output → canonical chunk_ids c_bb, c_input, c_output
        expect(res.chunkIds.sort()).toEqual(['c_bb', 'c_input', 'c_output']);
        expect(res.packageHits).toBe(3);
        expect(res.supertypeHits).toBe(0);
    });

    it('parent_symbol takes precedence over path-derived package', () => {
        const graph = GraphIndex.load(tmpFile(JSON.stringify({
            version: 'v3-2',
            by_symbol: {
                parseInt: { canonical_chunk_ids: ['c_pi'] },
                parseFloat: { canonical_chunk_ids: ['c_pf'] },
            },
            by_package: { Std: ['parseInt', 'parseFloat'] },
        })))!;
        const seeds = [
            makeSeed({
                symbol_name: 'parseInt',
                parent_symbol: 'Std',
                relativePath: 'unrelated/path/somefile.hx',
                chunk_id: 's_pi',
            }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).toEqual(['c_pf']);
    });

    it('skips package lookup when bucket size > maxPackageFanout', () => {
        const graph = loadGraph();
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', relativePath: 'big/pkg/Bytes.hx', chunk_id: 's1' }),
        ];
        // "big.pkg" has 25 entries; with maxPackageFanout=15 the lookup must
        // be skipped entirely (no partners pulled even though bucket exists).
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).toEqual([]);
    });

    it('expands package lookup when maxPackageFanout >= bucket size', () => {
        const graph = loadGraph();
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', relativePath: 'big/pkg/Bytes.hx', chunk_id: 's1' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 30 });
        // big.pkg = 25 sym; maxPartners=8 → 8 partners (none have canonicals → 0 chunks)
        expect(res.chunkIds.length).toBe(0);
        // Partners considered but skipped due to missing canonical → 0 packageHits
        // (only counts successful pushes).
    });

    it('skips partners with empty canonical_chunk_ids (mentions-only entries)', () => {
        const graph = GraphIndex.load(tmpFile(JSON.stringify({
            version: 'v3-2',
            by_symbol: {
                Real: { canonical_chunk_ids: ['c_real'] },
                MentionsOnly: { canonical_chunk_ids: [], mentioned_by_chunk_ids: ['c_doc'] },
            },
            by_package: { pkg: ['Real', 'MentionsOnly'] },
        })))!;
        const seeds = [
            makeSeed({ symbol_name: 'Real', relativePath: 'pkg/Real.hx', chunk_id: 's1' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        // Only MentionsOnly is a partner, but has no canonical chunk → 0 pool.
        expect(res.chunkIds).toEqual([]);
    });

    it('supertype partners injected even when extends/implements set on seed', () => {
        const graph = loadGraph();
        const seeds = [
            makeSeed({
                symbol_name: 'IntMap',
                relativePath: 'haxe/ds/IntMap.hx',
                chunk_id: 'seed_im',
                implements: ['IMap'],
            }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 30 });
        // rag-graph-abstract-typedef-edges: edge precedence now puts
        // supertype before package, so IMap → [ObjectMap, StringMap] is the
        // first axis to push (supertypeHits=2). Package lookup then yields
        // the same two, dedup-skipped (packageHits=0). Union unchanged.
        expect(res.chunkIds.sort()).toEqual(['c_om', 'c_sm']);
        expect(res.supertypeHits).toBe(2);
        expect(res.packageHits).toBe(0);
    });

    it('drops chunk_ids that are already in primary top-K', () => {
        const graph = loadGraph();
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', relativePath: 'haxe/io/Bytes.hx', chunk_id: 'c_bb' }),
            // Note: chunk_id 'c_bb' is also BytesBuffer's canonical — the bridge
            // must not surface a chunk_id already in primary top-K.
            makeSeed({ symbol_name: 'Input', relativePath: 'haxe/io/Input.hx', chunk_id: 'seed_in' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).not.toContain('c_bb');
    });

    it('non-code seeds are skipped', () => {
        const graph = loadGraph();
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', content_type: 'doc', relativePath: 'haxe/io/Bytes.hx' }),
            makeSeed({ symbol_name: 'BytesBuffer', content_type: 'docstring', relativePath: 'haxe/io/BytesBuffer.hx' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).toEqual([]);
        expect(res.seedsCount).toBe(0);
    });

    it('seeds without symbol_name are skipped', () => {
        const graph = loadGraph();
        const seeds = [makeSeed({ relativePath: 'haxe/io/something.hx', chunk_id: 's1' })];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.chunkIds).toEqual([]);
        expect(res.seedsCount).toBe(0);
    });

    it('caps the pool size at the provided cap', () => {
        // Build a graph with one huge bucket and several seeds.
        const payload: any = {
            version: 'v3-2',
            by_symbol: {},
            by_package: { 'big.pkg': [] },
            by_supertype: {},
        };
        for (let i = 0; i < 100; i++) {
            const name = `Sym${i.toString().padStart(3, '0')}`;
            payload.by_symbol[name] = { canonical_chunk_ids: [`c_${name}`] };
            payload.by_package['big.pkg'].push(name);
        }
        const graph = GraphIndex.load(tmpFile(JSON.stringify(payload)))!;
        const seeds = [
            makeSeed({ symbol_name: 'Sym000', relativePath: 'big/pkg/foo.hx', chunk_id: 's1' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, {
            maxPartners: 100, maxPackageFanout: 1000, cap: 12,
        });
        expect(res.chunkIds.length).toBe(12);
    });

    // rag-graph-abstract-typedef-edges: v3-3 axes.
    it('bridge surfaces the underlying-type partner for an abstract seed (forward)', () => {
        const graph = GraphIndex.load(tmpFile(JSON.stringify(V33_PAYLOAD)))!;
        // Seed declares `abstract Bytes(BytesData) from Array<UInt8> to BytesData`.
        // Forward path: lookup by_symbol[BytesData] / by_symbol[Array] →
        // c_bd, c_arr land in the pool.
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', relativePath: 'haxe/io/Bytes.hx', chunk_id: 'seed_bytes' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 30 });
        expect(res.chunkIds).toContain('c_bd');  // BytesData (forward)
        expect(res.chunkIds).toContain('c_arr'); // Array (forward)
        // Sibling: ReadOnlyArray also has Array in its abstract_underlying →
        // by_abstract_underlying[Array] = [Bytes (self), ReadOnlyArray] →
        // ReadOnlyArray (c_roa) lands too.
        expect(res.chunkIds).toContain('c_roa');
        expect(res.abstractUnderlyingHits).toBeGreaterThan(0);
    });

    it('bridge surfaces the alias-target partner for a typedef seed (forward + sibling)', () => {
        const graph = GraphIndex.load(tmpFile(JSON.stringify(V33_PAYLOAD)))!;
        // Seed declares `typedef Null<T> = T`.
        // Forward: by_symbol[T] → c_t.
        // Sibling: by_typedef_alias[T] = [Null (self), Maybe] → c_maybe.
        const seeds = [
            makeSeed({ symbol_name: 'Null', relativePath: 'StdTypes.hx', chunk_id: 'seed_null' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 30 });
        expect(res.chunkIds).toContain('c_t');     // T (forward)
        expect(res.chunkIds).toContain('c_maybe'); // Maybe (sibling typedef)
        expect(res.typedefAliasHits).toBeGreaterThan(0);
    });

    it('v3-3 axes silently degrade on a v3-2 graph (no abstract/typedef hits)', () => {
        const graph = loadGraph(); // v3-2 fixture
        const seeds = [
            makeSeed({ symbol_name: 'Bytes', relativePath: 'haxe/io/Bytes.hx', chunk_id: 'seed_bytes' }),
        ];
        const res = buildComparisonBridgePool(seeds, graph, { maxPartners: 8, maxPackageFanout: 15 });
        expect(res.abstractUnderlyingHits).toBe(0);
        expect(res.typedefAliasHits).toBe(0);
    });

    it('debug log emitted under queryId/debug', () => {
        const spy = jest.spyOn(console, 'log').mockImplementation(() => undefined);
        try {
            const graph = loadGraph();
            const seeds = [
                makeSeed({ symbol_name: 'Bytes', relativePath: 'haxe/io/Bytes.hx', chunk_id: 's1' }),
            ];
            buildComparisonBridgePool(seeds, graph, {
                maxPartners: 8, maxPackageFanout: 15, debug: true, queryId: 'q44',
            });
            const logged = spy.mock.calls.map((c) => c.join(' '));
            const match = logged.find((line) => line.startsWith('[comparison-bridge] q=q44'));
            expect(match).toBeDefined();
            // v3-3 log adds abs_hits / td_hits between sup_hits and pool_size.
            expect(match).toMatch(/seeds=1 pkg_hits=3 sup_hits=0 abs_hits=0 td_hits=0 pool_size=3/);
        } finally {
            spy.mockRestore();
        }
    });
});
