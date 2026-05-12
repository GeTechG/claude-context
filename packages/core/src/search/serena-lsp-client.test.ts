// rag-symbol-refs-lsp-pool: tests for SerenaLspClient. All daemon I/O is
// stubbed via the test seams (clientFactory + healthProbe + stateFileResolver).

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
    SerenaLspClient,
    parseFindSymbolResponse,
    parseReferencesResponse,
    parseImplementationsResponse,
    WHOLE_FILE_END_LINE,
} from './serena-lsp-client';

function tmpStateFile(payload: any): string {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'serena-state-'));
    const file = path.join(dir, 'state.json');
    fs.writeFileSync(file, JSON.stringify(payload), 'utf-8');
    return file;
}

function fakeClient(impl: { callTool: (req: any) => Promise<any>; close?: () => Promise<void> }): any {
    return {
        connect: jest.fn().mockResolvedValue(undefined),
        callTool: jest.fn(impl.callTool),
        close: jest.fn(impl.close ?? (() => Promise.resolve())),
    };
}

describe('SerenaLspClient.detectBaseUrl', () => {
    it('returns the URL when state.json exists and the daemon answers /', async () => {
        const stateFile = tmpStateFile({ pid: 42, port: 12345, project: '/x' });
        const client = new SerenaLspClient('/irrelevant', {
            stateFileResolver: () => stateFile,
            healthProbe: async () => true,
        });
        await expect(client.detectBaseUrl()).resolves.toBe('http://127.0.0.1:12345');
    });

    it('returns null when state.json is missing', async () => {
        const client = new SerenaLspClient('/irrelevant', {
            stateFileResolver: () => '/no/such/file.json',
            healthProbe: async () => true,
        });
        await expect(client.detectBaseUrl()).resolves.toBeNull();
    });

    it('returns null when health probe fails', async () => {
        const stateFile = tmpStateFile({ pid: 42, port: 12345 });
        const client = new SerenaLspClient('/irrelevant', {
            stateFileResolver: () => stateFile,
            healthProbe: async () => false,
        });
        await expect(client.detectBaseUrl()).resolves.toBeNull();
    });

    it('caches the URL within TTL and re-resolves on forceRefresh', async () => {
        const stateFile = tmpStateFile({ port: 1111 });
        let probeCount = 0;
        const client = new SerenaLspClient('/irrelevant', {
            stateFileResolver: () => stateFile,
            healthProbe: async () => { probeCount++; return true; },
        });
        await client.detectBaseUrl();
        await client.detectBaseUrl();
        expect(probeCount).toBe(1);
        await client.detectBaseUrl(true);
        expect(probeCount).toBe(2);
    });

    it('honours baseUrlOverride and never reads state.json', async () => {
        const client = new SerenaLspClient('/irrelevant', {
            baseUrlOverride: 'http://override:9999',
            stateFileResolver: () => { throw new Error('should not be called'); },
        });
        await expect(client.detectBaseUrl()).resolves.toBe('http://override:9999');
    });
});

describe('SerenaLspClient.findSymbol / findReferencingSymbols / findImplementations', () => {
    function clientWithFake(impl: { callTool: (req: any) => Promise<any> }): {
        client: SerenaLspClient;
        fake: ReturnType<typeof fakeClient>;
    } {
        const fake = fakeClient(impl);
        const stateFile = tmpStateFile({ port: 1234 });
        const client = new SerenaLspClient('/x', {
            stateFileResolver: () => stateFile,
            healthProbe: async () => true,
            clientFactory: () => fake,
            transportFactory: () => ({}),
        });
        return { client, fake };
    }

    it('parses find_symbol response into Location[]', async () => {
        const { client } = clientWithFake({
            callTool: async () => ({
                content: [{
                    type: 'text',
                    text: JSON.stringify([{
                        name_path: 'Bytes',
                        kind: 'Class',
                        relative_path: 'haxe/std/haxe/io/Bytes.hx',
                        body_location: { start_line: 27, end_line: 631 },
                    }]),
                }],
            }),
        });
        const locs = await client.findSymbol('Bytes');
        expect(locs).toEqual([
            { filePath: 'haxe/std/haxe/io/Bytes.hx', range: { start: { line: 27, character: 0 }, end: { line: 631, character: 0 } } },
        ]);
    });

    it('parses find_referencing_symbols normal response and respects max', async () => {
        const refs = {
            'a.hx': { Method: [{ name_path: 'X/foo', reference_line: 10 }, { name_path: 'X/bar', reference_line: 20 }] },
            'b.hx': { Variable: [{ name_path: 'Y/baz', reference_line: 5 }] },
        };
        const { client } = clientWithFake({
            callTool: async () => ({
                content: [{ type: 'text', text: 'References without surrounding lines: ' + JSON.stringify(refs) }],
            }),
        });
        const got = await client.findReferencingSymbols('X', 'a.hx', 2);
        expect(got.length).toBe(2);
        expect(got[0].filePath).toBe('a.hx');
        expect(got[0].range.start.line).toBe(10);
    });

    it('handles oversized find_referencing_symbols summary by widening to whole file', async () => {
        const counts = { 'big.hx': 12, 'huge.hx': 30 };
        const { client } = clientWithFake({
            callTool: async () => ({
                content: [{
                    type: 'text',
                    text: 'The answer is too long ... Reference counts per file:\n' + JSON.stringify(counts),
                }],
            }),
        });
        const got = await client.findReferencingSymbols('X', 'big.hx', 10);
        expect(got.length).toBe(2);
        expect(got[0].range.start.line).toBe(0);
        expect(got[0].range.end.line).toBe(WHOLE_FILE_END_LINE);
    });

    it('returns empty array when callTool throws (timeout / network)', async () => {
        const { client } = clientWithFake({
            callTool: async () => { throw new Error('AbortError: timed out'); },
        });
        const got = await client.findReferencingSymbols('X', 'a.hx', 10);
        expect(got).toEqual([]);
    });

    it('returns empty array when Serena marks the response isError', async () => {
        const { client } = clientWithFake({
            callTool: async () => ({
                isError: true,
                content: [{ type: 'text', text: 'Error executing tool: ...' }],
            }),
        });
        const got = await client.findSymbol('X');
        expect(got).toEqual([]);
    });

    it('parses find_implementations response into Location[]', async () => {
        const impls = [
            { name_path: 'Foo', kind: 'Class', relative_path: 'foo.hx', body_location: { start_line: 1, end_line: 5 } },
            { name_path: 'Bar', kind: 'Class', relative_path: 'bar.hx', body_location: { start_line: 10, end_line: 20 } },
        ];
        const { client } = clientWithFake({
            callTool: async () => ({ content: [{ type: 'text', text: JSON.stringify(impls) }] }),
        });
        const got = await client.findImplementations('IFace', 'iface.hx', 5);
        expect(got).toEqual([
            { filePath: 'foo.hx', range: { start: { line: 1, character: 0 }, end: { line: 5, character: 0 } } },
            { filePath: 'bar.hx', range: { start: { line: 10, character: 0 }, end: { line: 20, character: 0 } } },
        ]);
    });
});

describe('parser primitives', () => {
    it('parseFindSymbolResponse returns [] on garbage', () => {
        expect(parseFindSymbolResponse(null)).toEqual([]);
        expect(parseFindSymbolResponse({})).toEqual([]);
        expect(parseFindSymbolResponse({ content: [{ type: 'text', text: 'not json' }] })).toEqual([]);
        expect(parseFindSymbolResponse({ content: [{ type: 'text', text: 'Error executing tool: x' }] })).toEqual([]);
    });

    it('parseReferencesResponse handles an empty file map', () => {
        expect(parseReferencesResponse({ content: [{ type: 'text', text: 'References without surrounding lines: {}' }] })).toEqual([]);
    });

    it('parseImplementationsResponse drops entries without relative_path', () => {
        const text = JSON.stringify([{ name_path: 'X' }, { name_path: 'Y', relative_path: 'y.hx', body_location: { start_line: 3, end_line: 4 } }]);
        const got = parseImplementationsResponse({ content: [{ type: 'text', text }] });
        expect(got.length).toBe(1);
        expect(got[0].filePath).toBe('y.hx');
    });
});
