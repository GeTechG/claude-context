// reference-edges-from-the-index: the pool resolves referencing chunks from the
// INDEX, and the language server is what may add to it. Measured 2026-09-19:
// over 98 Haxe std declarations Serena returned >=3 external referencing files
// for 5 symbols and implementations for none, and no cross-file references at
// all for C++ or JS/TS — a pool whose only source answers that rarely runs empty
// whatever its weight.
import { runSymbolRefsPool } from './symbol-refs-pool';
import { createSymbolFrequencyGate } from './symbol-frequency';

interface Row { id: string; relativePath?: string; startLine?: number; endLine?: number; content?: string; symbol_name?: string }

function makeVectorDb(stubs: { decl?: Row[]; refs?: Row[]; impls?: Row[]; hydrate?: Map<string, Row> }): any {
    const seen: string[] = [];
    const db = {
        filters: seen,
        query: jest.fn(async (_c: string, filter: string) => {
            seen.push(filter);
            if (/symbol_name == "/.test(filter) && filter.includes('content_type')) return stubs.decl ?? [];
            if (filter.startsWith('id in')) {
                const ids = Array.from(filter.matchAll(/"([^"]+)"/g)).map((m) => m[1]);
                return ids.map((id) => stubs.hydrate?.get(id)).filter((r): r is Row => !!r);
            }
            if (/mentioned_symbols like/.test(filter)) return stubs.refs ?? [];
            if (/extends == |implements like/.test(filter)) return stubs.impls ?? [];
            return [];
        }),
    };
    return db;
}

// A language server that answers nothing, which is what this corpus measured.
const silentLsp = () => ({
    findSymbol: jest.fn(async () => []),
    findReferencingSymbols: jest.fn(async () => []),
    findImplementations: jest.fn(async () => []),
});

const hydrationFor = (ids: string[]) => new Map(ids.map((id) => [id, { id, relativePath: `${id}.hx`, startLine: 1, endLine: 9, content: `body of ${id}`, symbol_name: id }]));

const base = { query: 'what calls Bytes', codebasePath: '/codebase', collection: 'col', maxRefs: 20, maxImpls: 10 };

describe('symbol-refs pool — the index is the source', () => {
    it('returns referencing chunks when the language server answers nothing', async () => {
        const decl = [{ id: 'decl1', relativePath: 'std/Bytes.hx', startLine: 1, endLine: 9 }];
        const refs = [{ id: 'ref1' }, { id: 'ref2' }];
        const vectorDatabase = makeVectorDb({ decl, refs, hydrate: hydrationFor(['decl1', 'ref1', 'ref2']) });
        const out = await runSymbolRefsPool({ ...base, parsed: { symbolName: 'Bytes' }, vectorDatabase, lspClient: silentLsp() } as any);
        const ids = out.map((r: any) => r.chunk_id ?? r.id ?? r.document?.id);
        expect(ids).toContain('ref1');
        expect(ids).toContain('ref2');
    });

    it('matches the name with its quotes, so Bytes does not match BytesBuffer', async () => {
        const vectorDatabase = makeVectorDb({ decl: [{ id: 'd', relativePath: 'a.hx', startLine: 1, endLine: 2 }], hydrate: hydrationFor(['d']) });
        await runSymbolRefsPool({ ...base, parsed: { symbolName: 'Bytes' }, vectorDatabase, lspClient: silentLsp() } as any);
        const refFilter = (vectorDatabase.filters as string[]).find((f) => f.includes('mentioned_symbols like'));
        // The quotes are escaped for Milvus, so the literal in the filter is \"Bytes\".
        // What matters is that they are THERE: a bare `%Bytes%` would match
        // "BytesBuffer" and call every one of its users a user of Bytes.
        expect(refFilter).toContain('\\"Bytes\\"');
    });

    it('reads implementations from extends and implements', async () => {
        const vectorDatabase = makeVectorDb({
            decl: [{ id: 'd', relativePath: 'IThing.hx', startLine: 1, endLine: 2 }],
            impls: [{ id: 'impl1' }],
            hydrate: hydrationFor(['d', 'impl1']),
        });
        const out = await runSymbolRefsPool({ ...base, parsed: { symbolName: 'IThing' }, vectorDatabase, lspClient: silentLsp() } as any);
        expect(out.map((r: any) => r.chunk_id ?? r.id ?? r.document?.id)).toContain('impl1');
        const implFilter = (vectorDatabase.filters as string[]).find((f) => f.includes('implements like'));
        expect(implFilter).toContain('extends == "IThing"');
    });

    it('the frequency bound gates the index source too', async () => {
        // A subject the corpus uses everywhere returns an arbitrary slice of the
        // whole corpus, and an arbitrary slice is not evidence — whichever source
        // produced it.
        const vectorDatabase = makeVectorDb({
            decl: [{ id: 'd', relativePath: 'a.hx', startLine: 1, endLine: 2 }],
            refs: [{ id: 'shouldNotAppear' }],
            hydrate: hydrationFor(['d', 'shouldNotAppear']),
        });
        const table = {
            schema: 'local-rag-symbol-frequency-v1',
            signature: 'sig',
            documents: 100,
            distribution: { identifiers: 2, histogram: { '1': 1, '90': 1 } },
            frequencies: { Everywhere: 90, Rare: 1 },
        };
        const symbolFrequency = createSymbolFrequencyGate(table as any, 0.999);
        const out = await runSymbolRefsPool({ ...base, parsed: { symbolName: 'Everywhere' }, vectorDatabase, lspClient: silentLsp(), symbolFrequency } as any);
        expect(out.map((r: any) => r.chunk_id ?? r.id ?? r.document?.id)).not.toContain('shouldNotAppear');
        expect((vectorDatabase.filters as string[]).some((f) => f.includes('mentioned_symbols like'))).toBe(false);
    });

    it('index rows rank ahead of language-server rows', async () => {
        const decl = [{ id: 'decl1', relativePath: 'std/Bytes.hx', startLine: 1, endLine: 9 }];
        const lsp = {
            findSymbol: jest.fn(async () => []),
            findReferencingSymbols: jest.fn(async () => [{ filePath: '/codebase/lsp.hx', range: { start: { line: 0, character: 0 }, end: { line: 5, character: 0 } } }]),
            findImplementations: jest.fn(async () => []),
        };
        const vectorDatabase: any = {
            filters: [] as string[],
            query: jest.fn(async (_c: string, filter: string) => {
                (vectorDatabase.filters as string[]).push(filter);
                if (/symbol_name == "/.test(filter) && filter.includes('content_type')) return decl;
                if (filter.startsWith('id in')) {
                    const ids = Array.from(filter.matchAll(/"([^"]+)"/g)).map((m) => m[1]);
                    return ids.map((id) => hydrationFor(['decl1', 'fromIndex', 'fromLsp']).get(id)).filter(Boolean);
                }
                if (/mentioned_symbols like/.test(filter)) return [{ id: 'fromIndex' }];
                if (/relativePath == "lsp.hx"/.test(filter)) return [{ id: 'fromLsp' }];
                return [];
            }),
        };
        const out = await runSymbolRefsPool({ ...base, parsed: { symbolName: 'Bytes' }, vectorDatabase, lspClient: lsp } as any);
        const ids = out.map((r: any) => r.chunk_id ?? r.id ?? r.document?.id);
        expect(ids.indexOf('fromIndex')).toBeGreaterThan(-1);
        expect(ids.indexOf('fromLsp')).toBeGreaterThan(-1);
        expect(ids.indexOf('fromIndex')).toBeLessThan(ids.indexOf('fromLsp'));
    });
});
