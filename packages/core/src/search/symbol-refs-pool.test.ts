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

interface FakeRow { id: string; relativePath?: string; startLine?: number; endLine?: number; content?: string; metadata?: string; symbol_name?: string }

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

    // rag-symbol-refs-multi-hop: hop-2 expansion tests ------------------

    describe('hop-2 expansion (maxHops=2)', () => {
        function makeBytesPoolSetup(refs: Location[], impls: Location[] = []) {
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
                ['std/Input.hx', [{ id: 'c_input' }]],
                ['std/Output.hx', [{ id: 'c_output' }]],
                ['std/Crypto.hx', [{ id: 'c_crypto' }]],
                ['std/Sha1.hx', [{ id: 'c_sha1' }]],
            ]);
            // Hydration rows return symbol_name so they double as hop-2 seed metadata.
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: 'BytesBuffer' }],
                ['c_input', { id: 'c_input', relativePath: 'std/Input.hx', metadata: '{}', symbol_name: 'Input' }],
                ['c_output', { id: 'c_output', relativePath: 'std/Output.hx', metadata: '{}', symbol_name: 'Output' }],
                ['c_crypto', { id: 'c_crypto', relativePath: 'std/Crypto.hx', metadata: '{}', symbol_name: 'Crypto' }],
                ['c_sha1', { id: 'c_sha1', relativePath: 'std/Sha1.hx', metadata: '{}', symbol_name: 'Sha1' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            const lsp = makeLsp({ refs, impls });
            return { vectorDatabase, lsp };
        }

        it('does NOT call hop-2 when maxHops=1 (default)', async () => {
            const { vectorDatabase, lsp } = makeBytesPoolSetup([loc('std/BytesBuffer.hx', 1, 5)]);
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                // maxHops omitted → defaults to 1
            });
            expect(lsp.findReferencingSymbols).toHaveBeenCalledTimes(1);
        });

        it('calls hop-2 for each eligible hop-1 seed (up to maxHop1Seeds)', async () => {
            // 3 hop-1 refs → 3 hop-2 calls (maxHop1Seeds=3 default)
            const { vectorDatabase, lsp } = makeBytesPoolSetup([
                loc('std/BytesBuffer.hx', 1, 5),
                loc('std/Input.hx', 1, 5),
                loc('std/Output.hx', 1, 5),
            ]);
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            // 1 (hop-1) + 3 (hop-2 seeds) = 4 invocations
            expect(lsp.findReferencingSymbols).toHaveBeenCalledTimes(4);
        });

        it('respects maxHop1Seeds cap (caps at 2 even when more hop-1 chunks exist)', async () => {
            const { vectorDatabase, lsp } = makeBytesPoolSetup([
                loc('std/BytesBuffer.hx', 1, 5),
                loc('std/Input.hx', 1, 5),
                loc('std/Output.hx', 1, 5),
            ]);
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
                maxHop1Seeds: 2,
            });
            // 1 (hop-1) + 2 (hop-2 seeds capped) = 3 invocations
            expect(lsp.findReferencingSymbols).toHaveBeenCalledTimes(3);
        });

        it('forwards maxHop2Refs to LSP client as the per-seed cap', async () => {
            const { vectorDatabase, lsp } = makeBytesPoolSetup([loc('std/BytesBuffer.hx', 1, 5)]);
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
                maxHop1Seeds: 1,
                maxHop2Refs: 4,
            });
            // First call is hop-1 with maxRefs=20; second is hop-2 seed with cap=4.
            expect(lsp.findReferencingSymbols).toHaveBeenNthCalledWith(2, 'BytesBuffer', 'std/BytesBuffer.hx', 4);
        });

        it('appends hop-2 chunks AFTER hop-1 in pool-rank order', async () => {
            // hop-1: c_bb (from BytesBuffer)
            // hop-2 (BytesBuffer's refs): Crypto.hx → c_crypto
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
                ['std/Crypto.hx', [{ id: 'c_crypto' }]],
            ]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: 'BytesBuffer' }],
                ['c_crypto', { id: 'c_crypto', relativePath: 'std/Crypto.hx', metadata: '{}', symbol_name: 'Crypto' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            // First call (hop-1) returns BytesBuffer; second call (hop-2) returns Crypto.
            const refsByCall = [
                [loc('std/BytesBuffer.hx', 0, 10)],
                [loc('std/Crypto.hx', 0, 10)],
            ];
            let callIdx = 0;
            const lsp: any = {
                findSymbol: jest.fn(async () => []),
                findReferencingSymbols: jest.fn(async () => refsByCall[callIdx++] ?? []),
                findImplementations: jest.fn(async () => []),
            };
            const out = await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
                maxHop1Seeds: 1,
            });
            expect(out.map((r) => r.document.id)).toEqual(['c_bytes', 'c_bb', 'c_crypto']);
            // RRF score is strictly decreasing with rank.
            expect(out[0].score).toBeGreaterThan(out[1].score);
            expect(out[1].score).toBeGreaterThan(out[2].score);
        });

        it('skips hop-2 when hop-1 returned no new chunks', async () => {
            // Decl found but refs/impls empty → no hop-1 seeds → no hop-2.
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows: new Map(), hydrations });
            const lsp = makeLsp({ refs: [], impls: [] });
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            // Only the hop-1 attempt (which returned empty); no hop-2 calls.
            expect(lsp.findReferencingSymbols).toHaveBeenCalledTimes(1);
        });

        it('skips hop-2 when all hop-1 chunks have empty symbol_name', async () => {
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
            ]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                // c_bb has empty symbol_name → ineligible seed.
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: '' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            const lsp = makeLsp({ refs: [loc('std/BytesBuffer.hx', 1, 5)], impls: [] });
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            expect(lsp.findReferencingSymbols).toHaveBeenCalledTimes(1);
        });

        it('skips hop-2 seed when symbol_name fails the stop-word guard (length < 4)', async () => {
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
                ['std/Input.hx', [{ id: 'c_input' }]],
            ]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: 'BytesBuffer' }],
                // length-3 symbol_name → ineligible.
                ['c_input', { id: 'c_input', relativePath: 'std/Input.hx', metadata: '{}', symbol_name: 'abc' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            const lsp = makeLsp({
                refs: [loc('std/BytesBuffer.hx', 1, 5), loc('std/Input.hx', 1, 5)],
                impls: [],
            });
            await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            // 1 hop-1 + 1 hop-2 (only the BytesBuffer seed passes guard).
            expect(lsp.findReferencingSymbols).toHaveBeenCalledTimes(2);
            expect(lsp.findReferencingSymbols).toHaveBeenNthCalledWith(2, 'BytesBuffer', 'std/BytesBuffer.hx', 3);
        });

        it('survives hop-2 partial failure — pool returns hop-1 + successful hop-2 results', async () => {
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
                ['std/Input.hx', [{ id: 'c_input' }]],
                ['std/Output.hx', [{ id: 'c_output' }]],
                ['std/Crypto.hx', [{ id: 'c_crypto' }]],
                ['std/Sha1.hx', [{ id: 'c_sha1' }]],
            ]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: 'BytesBuffer' }],
                ['c_input', { id: 'c_input', relativePath: 'std/Input.hx', metadata: '{}', symbol_name: 'Input' }],
                ['c_output', { id: 'c_output', relativePath: 'std/Output.hx', metadata: '{}', symbol_name: 'Output' }],
                ['c_crypto', { id: 'c_crypto', relativePath: 'std/Crypto.hx', metadata: '{}', symbol_name: 'Crypto' }],
                ['c_sha1', { id: 'c_sha1', relativePath: 'std/Sha1.hx', metadata: '{}', symbol_name: 'Sha1' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            // call 1 (hop-1): returns 3 refs
            // call 2 (hop-2 seed BytesBuffer): returns Crypto
            // call 3 (hop-2 seed Input): rejects
            // call 4 (hop-2 seed Output): returns Sha1
            const callPlan: Array<{ kind: 'ok'; v: Location[] } | { kind: 'fail'; e: Error }> = [
                { kind: 'ok', v: [loc('std/BytesBuffer.hx', 1, 5), loc('std/Input.hx', 1, 5), loc('std/Output.hx', 1, 5)] },
                { kind: 'ok', v: [loc('std/Crypto.hx', 0, 5)] },
                { kind: 'fail', e: new Error('hop-2 LSP timeout') },
                { kind: 'ok', v: [loc('std/Sha1.hx', 0, 5)] },
            ];
            let idx = 0;
            const lsp: any = {
                findSymbol: jest.fn(async () => []),
                findReferencingSymbols: jest.fn(async () => {
                    const step = callPlan[idx++];
                    if (!step) return [];
                    if (step.kind === 'fail') throw step.e;
                    return step.v;
                }),
                findImplementations: jest.fn(async () => []),
            };
            const out = await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
                maxHop1Seeds: 3,
            });
            const ids = out.map((r) => r.document.id);
            // hop-0 + 3 hop-1 + 2 successful hop-2 (Crypto, Sha1; Input rejected)
            expect(ids).toEqual(['c_bytes', 'c_bb', 'c_input', 'c_output', 'c_crypto', 'c_sha1']);
        });

        it('survives complete hop-2 failure (all rejected) — returns hop-1 only', async () => {
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
            ]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: 'BytesBuffer' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            let callCount = 0;
            const lsp: any = {
                findSymbol: jest.fn(async () => []),
                findReferencingSymbols: jest.fn(async () => {
                    callCount++;
                    if (callCount === 1) return [loc('std/BytesBuffer.hx', 1, 5)];
                    throw new Error(`hop-2 down call#${callCount}`);
                }),
                findImplementations: jest.fn(async () => []),
            };
            const out = await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            expect(out.map((r) => r.document.id)).toEqual(['c_bytes', 'c_bb']);
        });

        it('dedupes hop-2 chunks that overlap with hop-1', async () => {
            // hop-2 ref points back at a location that maps to c_bb (already in hop-1).
            const declRows = new Map([['Bytes', [{ id: 'c_bytes', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>([
                ['std/BytesBuffer.hx', [{ id: 'c_bb' }]],
            ]);
            const hydrations = new Map<string, FakeRow>([
                ['c_bytes', { id: 'c_bytes', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
                ['c_bb', { id: 'c_bb', relativePath: 'std/BytesBuffer.hx', metadata: '{}', symbol_name: 'BytesBuffer' }],
            ]);
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            const callPlan: Location[][] = [
                [loc('std/BytesBuffer.hx', 1, 5)],   // hop-1
                [loc('std/BytesBuffer.hx', 1, 5)],   // hop-2 returns the same file → dedup hits
            ];
            let idx = 0;
            const lsp: any = {
                findSymbol: jest.fn(async () => []),
                findReferencingSymbols: jest.fn(async () => callPlan[idx++] ?? []),
                findImplementations: jest.fn(async () => []),
            };
            const out = await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            // c_bb appears exactly once (deduped).
            expect(out.map((r) => r.document.id)).toEqual(['c_bytes', 'c_bb']);
        });

        it('does not add hop-2 chunks once TOTAL_CHUNK_CAP=30 is filled by hop-1', async () => {
            // Build 30 hop-1 refs to saturate the cap before hop-2 runs.
            const refs = Array.from({ length: 30 }, (_, i) => loc(`r${i}.hx`, 0, 5));
            const declRows = new Map([['Bytes', [{ id: 'c_decl', relativePath: 'std/Bytes.hx' } as FakeRow]]]);
            const locationRows = new Map<string, FakeRow[]>();
            const hydrations = new Map<string, FakeRow>([
                ['c_decl', { id: 'c_decl', relativePath: 'std/Bytes.hx', metadata: '{}', symbol_name: 'Bytes' }],
            ]);
            for (let i = 0; i < 30; i++) {
                locationRows.set(`r${i}.hx`, [{ id: `c_r${i}` }]);
                hydrations.set(`c_r${i}`, { id: `c_r${i}`, relativePath: `r${i}.hx`, metadata: '{}', symbol_name: `Sym${i}` });
            }
            // Plus a "hop-2-only" target that would land at c_extra if cap weren't binding.
            locationRows.set('std/Extra.hx', [{ id: 'c_extra' }]);
            hydrations.set('c_extra', { id: 'c_extra', relativePath: 'std/Extra.hx', metadata: '{}', symbol_name: 'Extra' });
            const vectorDatabase = makeVectorDb({ declRows, locationRows, hydrations });
            const callPlan: Location[][] = [
                refs,                                // hop-1: 30 refs → cap fills
                [loc('std/Extra.hx', 0, 5)],         // hop-2 seed 1: would add c_extra if cap weren't binding
                [loc('std/Extra.hx', 0, 5)],         // hop-2 seed 2
                [loc('std/Extra.hx', 0, 5)],         // hop-2 seed 3
            ];
            let idx = 0;
            const lsp: any = {
                findSymbol: jest.fn(async () => []),
                findReferencingSymbols: jest.fn(async () => callPlan[idx++] ?? []),
                findImplementations: jest.fn(async () => []),
            };
            const out = await runSymbolRefsPool({
                ...baseOpts,
                parsed: { symbolName: 'Bytes' },
                lspClient: lsp,
                vectorDatabase,
                maxHops: 2,
            });
            expect(out.length).toBe(30);
            expect(out.map((r) => r.document.id)).not.toContain('c_extra');
        });
    });
});
