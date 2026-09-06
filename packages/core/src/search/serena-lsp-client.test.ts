// rag-symbol-refs-lsp-pool: tests for SerenaLspClient. All service I/O is
// stubbed via the test seams (clientFactory + healthProbe + baseUrlOverride).

import {
    SerenaLspClient,
    parseFindSymbolResponse,
    parseReferencesResponse,
    parseImplementationsResponse,
    WHOLE_FILE_END_LINE,
} from './serena-lsp-client';

function fakeClient(impl: { callTool: (req: any) => Promise<any>; close?: () => Promise<void> }): any {
    return {
        connect: jest.fn().mockResolvedValue(undefined),
        callTool: jest.fn(impl.callTool),
        close: jest.fn(impl.close ?? (() => Promise.resolve())),
    };
}

describe('SerenaLspClient.detectBaseUrl', () => {
    const ENV_KEY = 'SERENA_BASE_URL';
    const savedEnv = process.env[ENV_KEY];
    afterEach(() => {
        if (savedEnv === undefined) delete process.env[ENV_KEY];
        else process.env[ENV_KEY] = savedEnv;
    });

    it('defaults to the compose service port when SERENA_BASE_URL is unset', async () => {
        delete process.env[ENV_KEY];
        const client = new SerenaLspClient({ healthProbe: async () => true });
        await expect(client.detectBaseUrl()).resolves.toBe('http://127.0.0.1:18948');
    });

    it('honours SERENA_BASE_URL and strips a trailing slash', async () => {
        process.env[ENV_KEY] = 'http://serena.internal:9999/';
        const client = new SerenaLspClient({ healthProbe: async () => true });
        await expect(client.detectBaseUrl()).resolves.toBe('http://serena.internal:9999');
    });

    it('returns null when health probe fails', async () => {
        const client = new SerenaLspClient({ healthProbe: async () => false });
        await expect(client.detectBaseUrl()).resolves.toBeNull();
    });

    it('caches the URL within TTL and re-resolves on forceRefresh', async () => {
        let probeCount = 0;
        const client = new SerenaLspClient({
            healthProbe: async () => { probeCount++; return true; },
        });
        await client.detectBaseUrl();
        await client.detectBaseUrl();
        expect(probeCount).toBe(1);
        await client.detectBaseUrl(true);
        expect(probeCount).toBe(2);
    });

    it('honours baseUrlOverride and never probes', async () => {
        const client = new SerenaLspClient({
            baseUrlOverride: 'http://override:9999',
            healthProbe: async () => { throw new Error('should not be called'); },
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
        const client = new SerenaLspClient({
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

// local-rag #61 / #62: session hygiene.
describe('SerenaLspClient session hygiene', () => {
    const ENV_KEY = 'SERENA_BASE_URL';
    const savedEnv = process.env[ENV_KEY];
    const savedFetch = (globalThis as any).fetch;
    afterEach(() => {
        if (savedEnv === undefined) delete process.env[ENV_KEY];
        else process.env[ENV_KEY] = savedEnv;
        (globalThis as any).fetch = savedFetch;
    });

    it('the default liveness probe asks the server root, never /mcp, and any status counts as alive', async () => {
        delete process.env[ENV_KEY];
        const asked: string[] = [];
        (globalThis as any).fetch = jest.fn(async (url: string) => { asked.push(String(url)); return { status: 404 }; });
        const client = new SerenaLspClient();
        await expect(client.detectBaseUrl(true)).resolves.toBe('http://127.0.0.1:18948');
        expect(asked).toEqual(['http://127.0.0.1:18948/']);
        expect(asked.some((url) => url.endsWith('/mcp'))).toBe(false);
    });

    it('serialises calls on one instance: a sibling never overlaps, and a failing call cannot dispose the connection under it', async () => {
        let inFlight = 0;
        let peak = 0;
        const order: string[] = [];
        let connects = 0;
        const fake = fakeClient({
            callTool: async (req: any) => {
                inFlight += 1; peak = Math.max(peak, inFlight);
                order.push(`start:${req.name}`);
                await new Promise((resolve) => setTimeout(resolve, 5));
                inFlight -= 1;
                order.push(`end:${req.name}`);
                if (req.name === 'find_implementations') throw new Error('request timed out');
                return { content: [{ type: 'text', text: 'References without surrounding lines: ' + JSON.stringify({ 'a.hx': { Method: [{ name_path: 'X/foo', reference_line: 10 }] } }) }] };
            },
        });
        const client = new SerenaLspClient({
            baseUrlOverride: 'http://stub',
            clientFactory: () => { connects += 1; return fake; },
            transportFactory: () => ({ terminateSession: jest.fn(), close: jest.fn() }),
        });
        const [refs, impls] = await Promise.all([
            client.findReferencingSymbols('X/y', 'a.hx', 5),
            client.findImplementations('X/y', 'a.hx', 5),
        ]);
        expect(peak).toBe(1);
        // refs ran to completion before impls started, and impls' failure
        // (dispose + one retry on a fresh connection) touched only itself.
        expect(order.slice(0, 2)).toEqual(['start:find_referencing_symbols', 'end:find_referencing_symbols']);
        expect(refs).toHaveLength(1);
        expect(impls).toEqual([]);
        expect(connects).toBe(2);
        expect(fake.close).toHaveBeenCalledTimes(1);
    });

    it('a session DELETE that hangs is bounded, and the socket is closed regardless', async () => {
        jest.useFakeTimers();
        try {
            const transport = { terminateSession: jest.fn(() => new Promise(() => undefined)), close: jest.fn() };
            const fake = fakeClient({ callTool: async () => ({ content: [{ type: 'text', text: '[]' }] }) });
            const client = new SerenaLspClient({ baseUrlOverride: 'http://stub', clientFactory: () => fake, transportFactory: () => transport });
            await client.findSymbol('X');
            const closing = client.close();
            await jest.advanceTimersByTimeAsync(1100);
            await closing;
            expect(transport.terminateSession).toHaveBeenCalledTimes(1);
            expect(fake.close).toHaveBeenCalledTimes(1);
        } finally {
            jest.useRealTimers();
        }
    });

    it('a connect that fails after opening a session terminates that session', async () => {
        const transport = { terminateSession: jest.fn(async () => undefined), close: jest.fn() };
        const fake = { connect: jest.fn().mockRejectedValue(new Error('initialize failed')), callTool: jest.fn(), close: jest.fn() };
        const client = new SerenaLspClient({ baseUrlOverride: 'http://stub', clientFactory: () => fake, transportFactory: () => transport });
        await expect(client.findSymbol('X')).resolves.toEqual([]);
        // connect is attempted, then retried once on a fresh transport: both
        // failed sessions are terminated, none leaks.
        expect(fake.connect).toHaveBeenCalledTimes(2);
        expect(transport.terminateSession).toHaveBeenCalledTimes(2);
    });

    it('a call that never settles is abandoned by the chain, and the next call still runs', async () => {
        jest.useFakeTimers();
        try {
            let calls = 0;
            const fake = fakeClient({
                callTool: async (req: any) => {
                    calls += 1;
                    if (req.name === 'find_implementations') return new Promise(() => undefined);
                    return { content: [{ type: 'text', text: 'References without surrounding lines: ' + JSON.stringify({ 'a.hx': { Method: [{ name_path: 'X/foo', reference_line: 10 }] } }) }] };
                },
            });
            const client = new SerenaLspClient({ baseUrlOverride: 'http://stub', timeoutMs: 100, clientFactory: () => fake, transportFactory: () => ({ terminateSession: jest.fn(), close: jest.fn() }) });
            const stuck = client.findImplementations('X/y', 'a.hx', 5);
            const next = client.findReferencingSymbols('X/y', 'a.hx', 5);
            await jest.advanceTimersByTimeAsync(100 * 2 + 5000 + 10);
            await expect(stuck).resolves.toEqual([]);
            await expect(next).resolves.toHaveLength(1);
            expect(calls).toBe(2);
        } finally {
            jest.useRealTimers();
        }
    });

    it('terminates the server session before closing a disposed connection', async () => {
        const transport = { terminateSession: jest.fn(async () => undefined), close: jest.fn() };
        const fake = fakeClient({ callTool: async () => ({ content: [{ type: 'text', text: '[]' }] }) });
        const client = new SerenaLspClient({ baseUrlOverride: 'http://stub', clientFactory: () => fake, transportFactory: () => transport });
        await client.findSymbol('X');
        await client.close();
        expect(transport.terminateSession).toHaveBeenCalledTimes(1);
        expect(fake.close).toHaveBeenCalledTimes(1);
        expect(transport.terminateSession.mock.invocationCallOrder[0]).toBeLessThan(fake.close.mock.invocationCallOrder[0]);
        // Idempotent, and a transport without the method is closed as before.
        await client.close();
        expect(fake.close).toHaveBeenCalledTimes(1);
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

    it('parseReferencesResponse accepts Serena 1.7 body locations', () => {
        const text = JSON.stringify({
            'node.cpp': { Method: [{ name_path: 'Node/update', body_location: { start_line: 105, end_line: 132 } }] },
        });
        expect(parseReferencesResponse({ content: [{ type: 'text', text }] })).toEqual([{
            filePath: 'node.cpp',
            range: { start: { line: 105, character: 0 }, end: { line: 132, character: 0 } },
        }]);
    });

    it('parseImplementationsResponse drops entries without relative_path', () => {
        const text = JSON.stringify([{ name_path: 'X' }, { name_path: 'Y', relative_path: 'y.hx', body_location: { start_line: 3, end_line: 4 } }]);
        const got = parseImplementationsResponse({ content: [{ type: 'text', text }] });
        expect(got.length).toBe(1);
        expect(got[0].filePath).toBe('y.hx');
    });
});

// local-rag #64: a call the service did not answer, told from a call it answered
// with nothing, by what the service itself said.
describe('SerenaLspClient — getLastToolError (local-rag #64)', () => {
    const make = (): any => new (SerenaLspClient as any)({ baseUrlOverride: 'http://serena:1', timeoutMs: 50 });

    it('records a tool result flagged isError — the service declining an over-long answer is not "no references"', async () => {
        const client = make();
        client.ensureClient = jest.fn(async () => ({ callTool: jest.fn(async () => ({ isError: true, content: [] })) }));
        const out = await (client as any).callTool('find_referencing_symbols', {});
        expect(out).toBeNull();
        expect(client.getLastToolError('find_referencing_symbols')).toBeTruthy();
        expect(client.getLastToolError('find_referencing_symbols').reason).toMatch(/isError/);
    });

    it('records nothing when the service answered, even with an empty result', async () => {
        const client = make();
        client.ensureClient = jest.fn(async () => ({ callTool: jest.fn(async () => ({ content: [{ type: 'text', text: '[]' }] })) }));
        await (client as any).callTool('find_referencing_symbols', {});
        expect(client.getLastToolError('find_referencing_symbols')).toBeUndefined();
    });

    it('a call the chain ABANDONS is recorded as unanswered, not left looking answered', async () => {
        // The loudest case of the service failing to answer: the call outlived
        // two per-call budgets plus slack, `withDeadline` gave up on it, and
        // `callToolSerial` is still running with nothing recorded. A caller that
        // saw `null` with no record would read it as "nothing matches" — the
        // exact hole mechanism 3 exists to close. Driven through the real
        // `callTool` path, with a call that genuinely never settles.
        const client = make();
        client.ensureClient = jest.fn(async () => ({ callTool: jest.fn(() => new Promise(() => { /* never settles */ })) }));
        const out = await (client as any).callTool('find_referencing_symbols', {});
        expect(out).toBeNull();
        const recorded = client.getLastToolError('find_referencing_symbols');
        expect(recorded).toBeTruthy();
        expect(recorded.reason).toMatch(/abandoned after \d+ms without an answer/);
    }, 20000);

    it('an abandoned predecessor cannot write onto the record of a successor that overlaps it', async () => {
        // A real overlap: the first call never settles and is abandoned, the
        // second runs while the first is still pending and answers, and the
        // first's own `record()` then fires late. The successor's entry must
        // survive it.
        //
        // local-rag #78: the zombie settles with a FAILURE, and that is the
        // whole point of the test. Settling it successfully returns at
        // `if (result !== undefined) return result` — `record()` is never
        // reached, the test writes nothing, and it passes just as well with the
        // sequence guard deleted. A service declining the call (`isError`) is
        // the exact shape #64 exists for, and it takes the zombie through
        // `record()` with the successor's entry already in the slot.
        const client = make();
        let settleFirst: ((value: any) => void) | null = null;
        let call = 0;
        client.ensureClient = jest.fn(async () => ({
            callTool: jest.fn(() => {
                call += 1;
                if (call === 1) return new Promise((resolve) => { settleFirst = resolve; });
                return Promise.resolve({ content: [{ type: 'text', text: '[]' }] });
            }),
        }));
        const first = (client as any).callTool('find_referencing_symbols', {});
        await expect(first).resolves.toBeNull();
        // The successor claims the slot and is answered.
        await (client as any).callTool('find_referencing_symbols', {});
        expect(client.getLastToolError('find_referencing_symbols')).toBeUndefined();
        // Now the zombie settles, as one does around two timeouts later — with
        // the service declining, so it runs `record()` against the guard.
        if (settleFirst) (settleFirst as any)({ isError: true, content: [] });
        await new Promise((resolve) => setTimeout(resolve, 20));
        expect(client.getLastToolError('find_referencing_symbols')).toBeUndefined();
    }, 20000);

    it('the same late failure IS recorded when no successor claimed the slot', async () => {
        // The control for the test above: it proves the zombie's late failure
        // really does reach `record()`, so that "the successor's entry survived"
        // is evidence about the guard and not about a path nothing walks.
        const client = make();
        let settleFirst: ((value: any) => void) | null = null;
        client.ensureClient = jest.fn(async () => ({
            callTool: jest.fn(() => new Promise((resolve) => { settleFirst = resolve; })),
        }));
        const first = (client as any).callTool('find_referencing_symbols', {});
        await expect(first).resolves.toBeNull();
        expect(client.getLastToolError('find_referencing_symbols').reason).toMatch(/abandoned after \d+ms/);
        if (settleFirst) (settleFirst as any)({ isError: true, content: [] });
        await new Promise((resolve) => setTimeout(resolve, 20));
        const recorded = client.getLastToolError('find_referencing_symbols');
        expect(recorded.reason).toMatch(/isError/);
        expect(recorded.attempts).toBe(1);
    }, 20000);

    it('a zombie\'s attempt error is not recorded as the successor\'s reason', async () => {
        // local-rag #78 (b): `lastAttemptError` used to be one field shared by
        // every call, so a predecessor settling late could put its message into
        // the successor's record and `trace.reference_error` would name a
        // failure that call never had. Here the successor gets no connection at
        // all — it has no message of its own — and the zombie fails in the exact
        // window between the successor's last attempt and what it records.
        const client = make();
        let rejectZombie: ((error: any) => void) | null = null;
        let ensured = 0;
        client.ensureClient = jest.fn(async () => {
            ensured += 1;
            if (ensured === 1) return { callTool: jest.fn(() => new Promise((_resolve, reject) => { rejectZombie = reject; })) };
            // The successor's attempts: no connection, so nothing of its own is
            // ever written. On its retry the zombie's own attempt fails.
            if (ensured === 3 && rejectZombie) {
                (rejectZombie as any)(new Error('the zombie could not reach the service'));
                for (let tick = 0; tick < 5; tick += 1) await Promise.resolve();
            }
            return null;
        });
        const zombie = (client as any).callTool('find_referencing_symbols', {});
        await expect(zombie).resolves.toBeNull();
        await expect((client as any).callTool('find_referencing_symbols', {})).resolves.toBeNull();
        const recorded = client.getLastToolError('find_referencing_symbols');
        expect(recorded.reason).toBe('both attempts failed');
        expect(recorded.attempts).toBe(2);
    }, 20000);

    it('an abandoned call that settles late does not tear down the connection its successor is on', async () => {
        // local-rag #78 (a): the retry path used to call `disposeClient()`, which
        // reads whatever connection the client holds NOW. A call abandoned around
        // two budgets earlier reaches that line while its successor is mid-call
        // on the shared connection, and closes it under it.
        jest.useFakeTimers();
        try {
            const transport = { terminateSession: jest.fn(async () => undefined), close: jest.fn() };
            let rejectFirst: ((error: any) => void) | null = null;
            let calls = 0;
            const fake = fakeClient({
                callTool: () => {
                    calls += 1;
                    // The zombie's attempt, then the successor's, which is still
                    // in flight when the zombie finally fails.
                    if (calls === 1) return new Promise((_resolve, reject) => { rejectFirst = reject; });
                    return new Promise(() => undefined);
                },
            });
            const client = new SerenaLspClient({ baseUrlOverride: 'http://stub', timeoutMs: 100, clientFactory: () => fake, transportFactory: () => transport });
            const zombie = client.findReferencingSymbols('X/y', 'a.hx', 5);
            const successor = client.findImplementations('X/y', 'a.hx', 5);
            await jest.advanceTimersByTimeAsync(100 * 2 + 5000 + 10);
            await expect(zombie).resolves.toEqual([]);
            expect(calls).toBe(2);
            // The successor is on the connection; now the zombie's own attempt
            // fails, which is what used to send it into the retry path.
            (rejectFirst as any)(new Error('MCP error -32001: Request timed out'));
            await jest.advanceTimersByTimeAsync(100);
            expect(fake.close).not.toHaveBeenCalled();
            expect(transport.terminateSession).not.toHaveBeenCalled();
            // And the zombie sent no second attempt: nobody was waiting for it.
            expect(calls).toBe(2);
            await jest.advanceTimersByTimeAsync(100 * 2 + 5000 + 10);
            await expect(successor).resolves.toEqual([]);
        } finally {
            jest.useRealTimers();
        }
    });
});
