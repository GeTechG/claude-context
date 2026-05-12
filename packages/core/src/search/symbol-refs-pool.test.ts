// rag-symbol-refs-lsp-pool: tests for runSymbolRefsPool. Declaration lookup
// goes through Milvus (`symbol_name == "X"`); refs/impls go through the
// (mocked) Serena LSP client.

import { runSymbolRefsPool } from './symbol-refs-pool';
import { Location } from './serena-lsp-client';

function loc(filePath: string, startLine = 0, endLine = 100): Location {
    return {
        filePath,
        range: { start: { line: startLine, character: 0 }, end: { line: endLine, character: 0 } },
    };
}

interface FakeRow { id: string; relativePath?: string; startLine?: number; endLine?: number; content?: string; metadata?: string }

function makeVectorDb(stubs: {
    declRows: Map<string, FakeRow[]>;          // keyed by symbol_name literal
    locationRows: Map<string, FakeRow[]>;      // keyed by relativePath literal
    hydrations: Map<string, FakeRow>;          // keyed by id
}): any {
    return {
        query: jest.fn(async (_collection: string, filter: string, _fields: string[], _limit?: number) => {
            // Declaration lookup: symbol_name == "X" and content_type in [...]
            const declMatch = filter.match(/symbol_name == "([^"]+)"/);
            if (declMatch && filter.includes('content_type')) {
                return stubs.declRows.get(declMatch[1]) ?? [];
            }
            // Hydration: id in [...]
            if (filter.startsWith('id in')) {
                const ids = Array.from(filter.matchAll(/"([^"]+)"/g)).map((m) => m[1]);
                return ids.map((id) => stubs.hydrations.get(id)).filter((r): r is FakeRow => !!r);
            }
            // Location lookup: relativePath == "X" AND ...
            const relMatch = filter.match(/relativePath == "([^"]+)"/);
            if (relMatch) {
                return stubs.locationRows.get(relMatch[1]) ?? [];
            }
            return [];
        }),
    };
}

function makeLsp(impl: { refs: Location[]; impls: Location[] }): any {
    return {
        findSymbol: jest.fn(async () => []),                                  // unused — Milvus seeds the decl
        findReferencingSymbols: jest.fn(async () => impl.refs),
        findImplementations: jest.fn(async () => impl.impls),
    };
}

describe('runSymbolRefsPool', () => {
    const baseOpts = {
        query: 'how to use Bytes',
        codebasePath: '/codebase',
        collection: 'col',
        maxRefs: 20,
        maxImpls: 10,
    };

    it('returns chunks ranked decl-from-Milvus < refs < impls and hydrates them', async () => {
        const lsp = makeLsp({
            refs: [loc('std/Input.hx', 10, 10), loc('std/Output.hx', 5, 5)],
            impls: [loc('std/UInt8Array.hx', 1, 5)],
        });
        const declRows = new Map([
            ['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]],
        ]);
        const locationRows = new Map<string, FakeRow[]>([
            ['std/Input.hx', [{ id: 'c_input' }]],
            ['std/Output.hx', [{ id: 'c_output' }]],
            ['std/UInt8Array.hx', [{ id: 'c_uint8' }]],
        ]);
        const hydrations = new Map<string, FakeRow>([
            ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}' }],
            ['c_input', { id: 'c_input', metadata: '{}' }],
            ['c_output', { id: 'c_output', metadata: '{}' }],
            ['c_uint8', { id: 'c_uint8', metadata: '{}' }],
        ]);
        const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
        const out = await runSymbolRefsPool({
            ...baseOpts,
            parsed: { symbolName: 'Bytes' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(out.map((r) => r.document.id)).toEqual(['c_bytes', 'c_input', 'c_output', 'c_uint8']);
        expect(out[0].score).toBeGreaterThan(out[1].score);
    });

    it('uses Class/method form for qualified-name parses (Serena name_path syntax)', async () => {
        const lsp = makeLsp({ refs: [], impls: [] });
        const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
        const vectorDatabase = makeVectorDb({ declRows, locationRows: new Map(), hydrations: new Map([['c_bytes', { id: 'c_bytes', metadata: '{}' }]]) });
        await runSymbolRefsPool({
            ...baseOpts,
            parsed: { className: 'Bytes', methodName: 'toString', fullyQualified: 'Bytes.toString' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(lsp.findReferencingSymbols).toHaveBeenCalledWith('Bytes/toString', 'std/Bytes.hx', 20);
        expect(lsp.findImplementations).toHaveBeenCalledWith('Bytes/toString', 'std/Bytes.hx', 10);
    });

    it('returns [] when Milvus has no declaration chunk (nothing to seed refs from)', async () => {
        const lsp = makeLsp({ refs: [loc('std/Input.hx', 0, 5)], impls: [] });
        const vectorDatabase = makeVectorDb({ declRows: new Map(), locationRows: new Map(), hydrations: new Map() });
        const out = await runSymbolRefsPool({
            ...baseOpts,
            parsed: { symbolName: 'Nope' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(out).toEqual([]);
        expect(lsp.findReferencingSymbols).not.toHaveBeenCalled();
    });

    it('survives one of refs/impls rejecting (Promise.allSettled)', async () => {
        const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
        const lsp: any = {
            findSymbol: jest.fn(async () => []),
            findReferencingSymbols: jest.fn(async () => [loc('std/Input.hx', 10, 10)]),
            findImplementations: jest.fn(async () => { throw new Error('LSP down'); }),
        };
        const locationRows = new Map<string, FakeRow[]>([['std/Input.hx', [{ id: 'c_input' }]]]);
        const hydrations = new Map<string, FakeRow>([
            ['c_bytes', { id: 'c_bytes', metadata: '{}' }],
            ['c_input', { id: 'c_input', metadata: '{}' }],
        ]);
        const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
        const out = await runSymbolRefsPool({
            ...baseOpts,
            parsed: { symbolName: 'Bytes' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(out.map((r) => r.document.id)).toEqual(['c_bytes', 'c_input']);
    });

    it('skips locations whose Milvus lookup returns nothing', async () => {
        const lsp = makeLsp({ refs: [loc('vendored/External.hx', 0, 0)], impls: [] });
        const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
        // vendored path -> no entry in locationRows
        const hydrations = new Map<string, FakeRow>([['c_bytes', { id: 'c_bytes', metadata: '{}' }]]);
        const vectorDatabase = makeVectorDb({ declRows, locationRows: new Map(), hydrations });
        const out = await runSymbolRefsPool({
            ...baseOpts,
            parsed: { symbolName: 'Bytes' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(out.map((r) => r.document.id)).toEqual(['c_bytes']);
    });

    it('respects maxRefs/maxImpls (forwarded to LSP client)', async () => {
        const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
        const lsp = makeLsp({ refs: [], impls: [] });
        const vectorDatabase = makeVectorDb({ declRows, locationRows: new Map(), hydrations: new Map([['c_bytes', { id: 'c_bytes', metadata: '{}' }]]) });
        await runSymbolRefsPool({
            ...baseOpts,
            maxRefs: 7,
            maxImpls: 3,
            parsed: { className: 'Bytes', methodName: 'toString', fullyQualified: 'Bytes.toString' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(lsp.findReferencingSymbols).toHaveBeenCalledWith('Bytes/toString', 'std/Bytes.hx', 7);
        expect(lsp.findImplementations).toHaveBeenCalledWith('Bytes/toString', 'std/Bytes.hx', 3);
    });

    it('dedupes chunk_ids and caps total at TOTAL_CHUNK_CAP (30)', async () => {
        const lsp = makeLsp({ refs: Array.from({ length: 40 }, (_, i) => loc(`r${i}.hx`, 0, 5)), impls: [] });
        const declRows = new Map([['Bytes', [{ id: 'c_decl', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
        const locationRows = new Map<string, FakeRow[]>();
        for (let i = 0; i < 40; i++) locationRows.set(`r${i}.hx`, [{ id: `c_r${i}` }]);
        const hydrations = new Map<string, FakeRow>([['c_decl', { id: 'c_decl', metadata: '{}' }]]);
        for (let i = 0; i < 40; i++) hydrations.set(`c_r${i}`, { id: `c_r${i}`, metadata: '{}' });
        const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
        const out = await runSymbolRefsPool({
            ...baseOpts,
            parsed: { symbolName: 'Bytes' },
            lspClient: lsp,
            vectorDatabase,
        });
        expect(out.length).toBe(30);
    });
});
