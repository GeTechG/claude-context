// find-references-as-a-tool: the tool answers from the index's scalar fields,
// groups by relation and file, orders deterministically, and lets the frequency
// bound answer instead of truncating.
import { findReferences } from './find-references';
import { referenceFilter } from './symbol-index-refs';
import { createSymbolFrequencyGate } from './symbol-frequency';

// The row carries the relation's own field, because `like` is a wildcard match
// and the code confirms the name exactly after the query.
const row = (id: string, relativePath: string, startLine = 10, extra: Record<string, any> = {}) => ({
    id, relativePath, startLine, endLine: startLine + 5,
    content_type: 'code', symbol_kind: 'class', symbol_name: id,
    mentioned_symbols: '["Bytes","Other"]', implements: '["Bytes"]', extends: 'Bytes',
    ...extra,
});

function makeVectorDb(byRelation: { mentions?: any[]; extends?: any[]; implements?: any[] }) {
    const filters: string[] = [];
    return {
        filters,
        query: jest.fn(async (_c: string, filter: string) => {
            filters.push(filter);
            if (/mentioned_symbols like/.test(filter)) return byRelation.mentions ?? [];
            if (/extends == /.test(filter)) return byRelation.extends ?? [];
            if (/implements like/.test(filter)) return byRelation.implements ?? [];
            return [];
        }),
    } as any;
}

const base = { collection: 'col', symbol: 'Bytes' };

describe('findReferences', () => {
    it('groups by relation and by file, chunks ordered by line', async () => {
        const vectorDatabase = makeVectorDb({
            mentions: [row('b', 'z/two.hx', 40), row('a', 'a/one.hx', 30), row('c', 'a/one.hx', 10)],
            implements: [row('i', 'a/impl.hx', 1)],
        });
        const out = await findReferences({ ...base, vectorDatabase });
        const mentions = out.groups.find((g) => g.relation === 'mentions')!;
        expect(mentions.files.map((f) => f.relativePath)).toEqual(['a/one.hx', 'z/two.hx']);
        expect(mentions.files[0].spans.map((sp) => sp.startLine)).toEqual([10, 30]);
        expect(mentions.files[0].spans[0].chunkId).toBe('c');
        expect(mentions.totalFiles).toBe(2);
        expect(out.groups.find((g) => g.relation === 'extends')!.files).toEqual([]);
        expect(out.groups.find((g) => g.relation === 'implements')!.files).toHaveLength(1);
    });

    it('is ordered the same however Milvus returns the rows', async () => {
        const rows = [row('b', 'z/two.hx', 40), row('a', 'a/one.hx', 30), row('c', 'a/one.hx', 10)];
        const first = await findReferences({ ...base, vectorDatabase: makeVectorDb({ mentions: rows }) });
        const second = await findReferences({ ...base, vectorDatabase: makeVectorDb({ mentions: [...rows].reverse() }) });
        expect(JSON.stringify(first.groups)).toEqual(JSON.stringify(second.groups));
    });

    it('matches the quoted name, so Bytes never matches BytesBuffer', async () => {
        const vectorDatabase = makeVectorDb({});
        await findReferences({ ...base, vectorDatabase });
        expect(vectorDatabase.filters).toContain('mentioned_symbols like "%\\"Bytes\\"%"');
        expect(vectorDatabase.filters.some((f: string) => f.includes('%Bytes%'))).toBe(false);
        // the same rule the pool queries with, from the same helper
        expect(referenceFilter('implements', 'Bytes')).toBe('implements like "%\\"Bytes\\"%"');
    });

    it('escapes a name that carries quotes or backslashes, and cannot break out of the literal', () => {
        // The tool is the first caller to put an agent-supplied string into a
        // Milvus filter, and there was no test for this anywhere.
        // An unescaped `"` would end the literal; count how many survive.
        const delimiters = (f: string) => (f.match(/(?<!\\)"/g) || []).length;
        for (const name of ['a"b', 'a\\b', 'Bytes" or 1==1 or "x', '\\"']) {
            for (const relation of ['mentions', 'extends', 'implements'] as const) {
                expect(delimiters(referenceFilter(relation, name))).toBe(2);
            }
        }
        expect(referenceFilter('extends', 'Plain')).toBe('extends == "Plain"');
    });

    it('ANDs the category scope onto every relation', async () => {
        const vectorDatabase = makeVectorDb({});
        await findReferences({ ...base, vectorDatabase, scopeFilter: 'relativePath like "godot/%"' });
        expect(vectorDatabase.filters).toHaveLength(3);
        for (const f of vectorDatabase.filters) expect(f).toContain('and (relativePath like "godot/%")');
    });

    it('merges a symbol\'s touching chunks into one span, and does not merge across symbols', async () => {
        // The shape the first end-to-end call hit: one chunked-up class printed
        // as 35 rows, with the two rows that answered the question buried in it.
        const mentions = [
            row('k1', 'core_bind.cpp', 47, { endLine: 112, symbol_name: 'CoreBind' }),
            row('k2', 'core_bind.cpp', 102, { endLine: 154, symbol_name: 'CoreBind' }),
            row('k3', 'core_bind.cpp', 152, { endLine: 211, symbol_name: 'CoreBind' }),
            row('add', 'core_bind.cpp', 188, { endLine: 190, symbol_name: 'add_resource_format_saver', symbol_kind: 'function' }),
        ];
        const out = await findReferences({ ...base, vectorDatabase: makeVectorDb({ mentions }) });
        const file = out.groups.find((g) => g.relation === 'mentions')!.files[0];
        expect(file.totalChunks).toBe(4);
        expect(file.spans).toHaveLength(2);
        expect(file.spans[0]).toMatchObject({ chunkId: 'k1', startLine: 47, endLine: 211, symbolName: 'CoreBind', chunks: 3 });
        expect(file.spans[1]).toMatchObject({ chunkId: 'add', startLine: 188, endLine: 190, chunks: 1 });
    });

    it('does not merge chunks of one symbol that are a gap apart', async () => {
        const mentions = [
            row('a', 'f.hx', 10, { endLine: 20, symbol_name: 'Same' }),
            row('b', 'f.hx', 40, { endLine: 50, symbol_name: 'Same' }),
        ];
        const out = await findReferences({ ...base, vectorDatabase: makeVectorDb({ mentions }) });
        expect(out.groups.find((g) => g.relation === 'mentions')!.files[0].spans).toHaveLength(2);
    });

    it('limits FILES per relation and keeps the true total', async () => {
        const mentions = Array.from({ length: 5 }, (_, i) => row(`m${i}`, `f${i}.hx`));
        const out = await findReferences({ ...base, vectorDatabase: makeVectorDb({ mentions }), limit: 2 });
        const g = out.groups.find((x) => x.relation === 'mentions')!;
        expect(g.files).toHaveLength(2);
        expect(g.totalFiles).toBe(5);
    });

    it('drops a wildcard match that does not really name the symbol', async () => {
        // `Nod%` / `get_singleton` both widen their own `like` match: Milvus
        // treats `%` and `_` as wildcards and a backslash escapes neither.
        const mentions = [
            row('real', 'a.hx', 10, { mentioned_symbols: '["Bytes"]' }),
            row('wide', 'b.hx', 10, { mentioned_symbols: '["BytesBuffer","Nodes"]' }),
            row('nofield', 'c.hx', 10, { mentioned_symbols: undefined }),
        ];
        const out = await findReferences({ ...base, vectorDatabase: makeVectorDb({ mentions }) });
        const g = out.groups.find((x) => x.relation === 'mentions')!;
        expect(g.files.map((f) => f.relativePath)).toEqual(['a.hx']);
        expect(g.totalFiles).toBe(1);
    });

    it('a failed query is a failure in the answer, not an empty relation', async () => {
        const vectorDatabase = {
            query: jest.fn(async (_c: string, filter: string) => {
                if (/mentioned_symbols like/.test(filter)) throw new Error('CollectionNotExists');
                return [];
            }),
        } as any;
        const out = await findReferences({ ...base, vectorDatabase });
        const g = out.groups.find((x) => x.relation === 'mentions')!;
        expect(g.failed).toMatch(/CollectionNotExists/);
        expect(out.groups.find((x) => x.relation === 'extends')!.failed).toBeNull();
    });

    it('reports the limit it applied and what it clamped', async () => {
        const vectorDatabase = makeVectorDb({});
        expect((await findReferences({ ...base, vectorDatabase })).limit).toBe(30);
        expect(await findReferences({ ...base, vectorDatabase, limit: 500 })).toMatchObject({ limit: 100, limitClampedFrom: 500 });
        expect(await findReferences({ ...base, vectorDatabase, limit: 0 })).toMatchObject({ limit: 30, limitClampedFrom: 0 });
    });

    it('lets the frequency bound answer instead of returning an arbitrary slice', async () => {
        const vectorDatabase = makeVectorDb({ mentions: [row('m', 'f.hx')] });
        const frequency = createSymbolFrequencyGate({
            schema: 'symbol-frequency/v1',
            documents: 2000,
            distribution: { identifiers: 2, histogram: { '1': 1, '1164': 1 } },
            frequencies: { get_singleton: 1164, Bytes: 1 },
        } as any, 0.99);
        const out = await findReferences({ ...base, symbol: 'get_singleton', vectorDatabase, frequency });
        expect(out.bounded).not.toBeNull();
        expect(out.bounded!.documentFrequency).toBe(1164);
        expect(out.bounded!.identifier).toBe('get_singleton');
        expect(out.groups).toEqual([]);
        expect(vectorDatabase.query).not.toHaveBeenCalled();
    });
});
