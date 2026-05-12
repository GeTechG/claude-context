// rag-graph-layer Phase 3.3: tests for GraphIndex + expandGraphPool.
// rag-graph-comparison-bridge: tests for v3-2 schema + derivePackageFromPath.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { GraphIndex, expandGraphPool, derivePackageFromPath } from './graph-expansion';
import { SemanticSearchResult } from '../types';

function tmpFile(content: string): string {
    const file = path.join(os.tmpdir(), `graph-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);
    fs.writeFileSync(file, content, 'utf-8');
    return file;
}

function makeChunk(over: Partial<SemanticSearchResult>): SemanticSearchResult {
    return {
        content: '',
        relativePath: '',
        startLine: 0,
        endLine: 0,
        language: 'haxe',
        score: 1,
        ...over,
    };
}

describe('GraphIndex.load', () => {
    it('parses a v3-1 side-file', () => {
        const file = tmpFile(JSON.stringify({
            version: 'v3-1',
            by_symbol: {
                Bytes: {
                    canonical_chunk_ids: ['chunk_code_bytes'],
                    mentioned_by_chunk_ids: ['chunk_doc_a', 'chunk_doc_b'],
                },
                'Bytes.alloc': {
                    canonical_chunk_ids: ['chunk_method_alloc'],
                    mentioned_by_chunk_ids: [],
                },
            },
        }));
        const idx = GraphIndex.load(file);
        expect(idx).not.toBeNull();
        expect(idx!.version).toBe('v3-1');
        expect(idx!.symbolCount).toBe(2);
        expect(idx!.lookupForward('Bytes')).toEqual(['chunk_code_bytes']);
        expect(idx!.lookupReverse('Bytes')).toEqual(['chunk_doc_a', 'chunk_doc_b']);
        expect(idx!.lookupForward('Bytes.alloc')).toEqual(['chunk_method_alloc']);
        expect(idx!.lookupReverse('NotInIndex')).toEqual([]);
    });

    it('returns null on missing file', () => {
        expect(GraphIndex.load('/nonexistent/path.json')).toBeNull();
    });

    it('returns null on malformed JSON', () => {
        const file = tmpFile('{not json');
        expect(GraphIndex.load(file)).toBeNull();
    });

    it('skips entries with both lists empty', () => {
        const file = tmpFile(JSON.stringify({
            version: 'v3-1',
            by_symbol: {
                EmptyEntry: { canonical_chunk_ids: [], mentioned_by_chunk_ids: [] },
                Real: { canonical_chunk_ids: ['c1'] },
            },
        }));
        const idx = GraphIndex.load(file);
        expect(idx!.symbolCount).toBe(1);
        expect(idx!.lookupForward('Real')).toEqual(['c1']);
    });
});

describe('expandGraphPool', () => {
    function makeIndex(payload: any): GraphIndex {
        const file = tmpFile(JSON.stringify(payload));
        const idx = GraphIndex.load(file);
        expect(idx).not.toBeNull();
        return idx!;
    }

    function fetcherFor(map: Record<string, SemanticSearchResult>) {
        return async (id: string) => map[id] || null;
    }

    it('forward-edge: code seed with extends → canonical chunk in neighbours', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                BaseClass: { canonical_chunk_ids: ['chunk_base'], mentioned_by_chunk_ids: [] },
            },
        });
        const seed = makeChunk({
            chunk_id: 'seed_a',
            content_type: 'code',
            symbol_name: 'A',
            extends: 'BaseClass',
        });
        const target = makeChunk({ chunk_id: 'chunk_base', symbol_name: 'BaseClass', content_type: 'code' });
        const neighbours = await expandGraphPool([seed], idx, fetcherFor({ chunk_base: target }));
        expect(neighbours).toHaveLength(1);
        expect(neighbours[0].chunk_id).toBe('chunk_base');
    });

    it('forward-edge: doc seed with mentioned_symbols → canonical code chunks', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                X: { canonical_chunk_ids: ['code_x'], mentioned_by_chunk_ids: [] },
                Y: { canonical_chunk_ids: ['code_y'], mentioned_by_chunk_ids: [] },
            },
        });
        const seed = makeChunk({
            chunk_id: 'seed_doc',
            content_type: 'doc',
            mentioned_symbols: ['X', 'Y'],
        });
        const fetched = await expandGraphPool([seed], idx, fetcherFor({
            code_x: makeChunk({ chunk_id: 'code_x', content_type: 'code', symbol_name: 'X' }),
            code_y: makeChunk({ chunk_id: 'code_y', content_type: 'code', symbol_name: 'Y' }),
        }));
        const ids = fetched.map((c) => c.chunk_id).sort();
        expect(ids).toEqual(['code_x', 'code_y']);
    });

    it('reverse-edge: code seed → doc chunks that mention its symbol', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                Foo: { canonical_chunk_ids: ['code_foo'], mentioned_by_chunk_ids: ['doc_1', 'doc_2'] },
            },
        });
        const seed = makeChunk({
            chunk_id: 'code_foo',
            content_type: 'code',
            symbol_name: 'Foo',
        });
        const fetched = await expandGraphPool([seed], idx, fetcherFor({
            doc_1: makeChunk({ chunk_id: 'doc_1', content_type: 'doc' }),
            doc_2: makeChunk({ chunk_id: 'doc_2', content_type: 'doc' }),
        }));
        const ids = fetched.map((c) => c.chunk_id).sort();
        expect(ids).toEqual(['doc_1', 'doc_2']);
    });

    it('docstring seeds are excluded from edge construction', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                Bar: { canonical_chunk_ids: ['code_bar'], mentioned_by_chunk_ids: [] },
            },
        });
        const seed = makeChunk({
            chunk_id: 'doc_string_seed',
            content_type: 'docstring',
            symbol_name: 'Bar',
        });
        const fetched = await expandGraphPool([seed], idx, fetcherFor({}));
        expect(fetched).toEqual([]);
    });

    it('dedupes chunk_ids across multiple seeds', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                Shared: { canonical_chunk_ids: ['code_shared'], mentioned_by_chunk_ids: [] },
            },
        });
        const seedA = makeChunk({ chunk_id: 'a', content_type: 'code', symbol_name: 'A', extends: 'Shared' });
        const seedB = makeChunk({ chunk_id: 'b', content_type: 'code', symbol_name: 'B', extends: 'Shared' });
        const target = makeChunk({ chunk_id: 'code_shared', content_type: 'code', symbol_name: 'Shared' });
        const fetched = await expandGraphPool([seedA, seedB], idx, fetcherFor({ code_shared: target }));
        expect(fetched).toHaveLength(1);
    });

    it('drops chunk_ids that are already in seeds', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                Base: { canonical_chunk_ids: ['code_base'], mentioned_by_chunk_ids: [] },
            },
        });
        const seedExt = makeChunk({ chunk_id: 'code_a', content_type: 'code', symbol_name: 'A', extends: 'Base' });
        const seedBase = makeChunk({ chunk_id: 'code_base', content_type: 'code', symbol_name: 'Base' });
        const fetched = await expandGraphPool([seedExt, seedBase], idx, fetcherFor({
            code_base: seedBase,
        }));
        expect(fetched).toEqual([]);
    });

    it('caps the neighbour pool at 50 by default', async () => {
        const cans = Array.from({ length: 200 }, (_, i) => `c${i}`);
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                Big: { canonical_chunk_ids: cans, mentioned_by_chunk_ids: [] },
            },
        });
        const seed = makeChunk({ chunk_id: 'seed', content_type: 'doc', mentioned_symbols: ['Big'] });
        const map: Record<string, SemanticSearchResult> = {};
        for (const id of cans) map[id] = makeChunk({ chunk_id: id, content_type: 'code' });
        const fetched = await expandGraphPool([seed], idx, fetcherFor(map));
        expect(fetched.length).toBeLessThanOrEqual(50);
    });

    it('fetcher errors on individual ids do not abort the whole expansion', async () => {
        const idx = makeIndex({
            version: 'v3-1',
            by_symbol: {
                Z: { canonical_chunk_ids: ['ok', 'broken'], mentioned_by_chunk_ids: [] },
            },
        });
        const seed = makeChunk({ chunk_id: 'seed', content_type: 'doc', mentioned_symbols: ['Z'] });
        const fetcher = async (id: string) => {
            if (id === 'broken') throw new Error('stale id');
            return makeChunk({ chunk_id: id });
        };
        const fetched = await expandGraphPool([seed], idx, fetcher);
        expect(fetched.map((c) => c.chunk_id)).toEqual(['ok']);
    });
});

describe('derivePackageFromPath (rag-graph-comparison-bridge)', () => {
    it('joins the file directory chain with dots', () => {
        expect(derivePackageFromPath('haxe/io/Bytes.hx')).toBe('haxe.io');
        expect(derivePackageFromPath('python/lib/foo.py')).toBe('python.lib');
        expect(derivePackageFromPath('src/main/java/com/example/Foo.java'))
            .toBe('src.main.java.com.example');
    });

    it('returns undefined for files with no enclosing directory', () => {
        expect(derivePackageFromPath('Map.hx')).toBeUndefined();
        expect(derivePackageFromPath('foo.py')).toBeUndefined();
    });

    it('returns undefined for empty or undefined input', () => {
        expect(derivePackageFromPath(undefined)).toBeUndefined();
        expect(derivePackageFromPath('')).toBeUndefined();
    });

    it('handles windows-style separators', () => {
        expect(derivePackageFromPath('haxe\\io\\Bytes.hx')).toBe('haxe.io');
    });
});

describe('GraphIndex v3-2 — by_package / by_supertype', () => {
    it('loads by_package and by_supertype top-level keys', () => {
        const file = tmpFile(JSON.stringify({
            version: 'v3-2',
            by_symbol: {
                Bytes: { canonical_chunk_ids: ['c_bytes'], mentioned_by_chunk_ids: [] },
                BytesBuffer: { canonical_chunk_ids: ['c_bb'], mentioned_by_chunk_ids: [] },
                IntMap: { canonical_chunk_ids: ['c_im'], mentioned_by_chunk_ids: [] },
                StringMap: { canonical_chunk_ids: ['c_sm'], mentioned_by_chunk_ids: [] },
            },
            by_package: {
                'haxe.io': ['Bytes', 'BytesBuffer', 'Input', 'Output'],
                'haxe.ds': ['IntMap', 'ObjectMap', 'StringMap'],
            },
            by_supertype: {
                IMap: ['IntMap', 'ObjectMap', 'StringMap'],
            },
        }));
        const idx = GraphIndex.load(file);
        expect(idx).not.toBeNull();
        expect(idx!.version).toBe('v3-2');
        expect(idx!.packageCount).toBe(2);
        expect(idx!.supertypeCount).toBe(1);
        expect(idx!.lookupPackage('haxe.io')).toEqual(['Bytes', 'BytesBuffer', 'Input', 'Output']);
        expect(idx!.lookupSupertype('IMap')).toEqual(['IntMap', 'ObjectMap', 'StringMap']);
        expect(idx!.lookupPackage('absent')).toEqual([]);
        expect(idx!.lookupSupertype('absent')).toEqual([]);
        expect(idx!.supportsComparisonBridge()).toBe(true);
        expect(idx!.lookupSymbol('Bytes')?.canonical_chunk_ids).toEqual(['c_bytes']);
        expect(idx!.lookupSymbol('absent')).toBeUndefined();
    });

    it('v3-1 payload exposes empty package/supertype lookups', () => {
        const file = tmpFile(JSON.stringify({
            version: 'v3-1',
            by_symbol: { Bytes: { canonical_chunk_ids: ['c'], mentioned_by_chunk_ids: [] } },
        }));
        const idx = GraphIndex.load(file);
        expect(idx).not.toBeNull();
        expect(idx!.version).toBe('v3-1');
        expect(idx!.packageCount).toBe(0);
        expect(idx!.supertypeCount).toBe(0);
        expect(idx!.lookupPackage('haxe.io')).toEqual([]);
        expect(idx!.supportsComparisonBridge()).toBe(false);
    });

    it('rejects unrecognized schema versions', () => {
        const file = tmpFile(JSON.stringify({
            version: 'v9-99-future',
            by_symbol: { X: { canonical_chunk_ids: ['c'] } },
        }));
        expect(GraphIndex.load(file)).toBeNull();
    });

    it('ignores non-string entries in by_package / by_supertype', () => {
        const file = tmpFile(JSON.stringify({
            version: 'v3-2',
            by_symbol: { Foo: { canonical_chunk_ids: ['c'], mentioned_by_chunk_ids: [] } },
            by_package: {
                'pkg.a': ['Foo', 42, null, 'Bar'],
                'pkg.empty': [],
            },
            by_supertype: { Iface: ['Foo', null] },
        }));
        const idx = GraphIndex.load(file);
        expect(idx).not.toBeNull();
        expect(idx!.lookupPackage('pkg.a')).toEqual(['Foo', 'Bar']);
        expect(idx!.lookupPackage('pkg.empty')).toEqual([]);
        expect(idx!.lookupSupertype('Iface')).toEqual(['Foo']);
    });
});
