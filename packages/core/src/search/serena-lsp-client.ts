// rag-symbol-refs-lsp-pool: thin Serena MCP/SSE client used by the
// symbol-refs pool. Exposes three LSP-backed lookups (find_symbol,
// find_referencing_symbols, find_implementations) over Serena's
// stdio-less SSE transport. All errors fold to empty arrays so the pool
// degrades to a no-op when the daemon is missing or the LSP can't compile.
//
// scope-serena-to-compose: Serena runs as the `serena` service in
// infra/docker-compose.yml on a fixed port, so there is nothing to discover —
// the base URL is SERENA_BASE_URL (default http://127.0.0.1:18948). Any HTTP
// response from the server counts as "alive"; Serena exposes no /health
// endpoint.
//
// local-rag #61 / #62 (serena-session-hygiene):
//   - the liveness probe asks `/` (a 404 from the same server), NOT `/mcp`:
//     Serena's streamable-HTTP manager opens a new server-side session for
//     EVERY request to `/mcp` that carries no session id, whatever the method
//     or the status it answers with, and nothing ever closes it — one leaked
//     session per probe, per operation, per healthcheck (937 transports for
//     1075 POSTs in one 59-row build);
//   - calls on one client instance are serialised. Serena executes tool calls
//     on a single queue anyway, so concurrency bought nothing, and the failure
//     path below disposes the shared connection — which, with a sibling call
//     in flight on it, rejected the sibling too (the -32001 then -32000 pair
//     of every pre-fix build). With one call in flight per instance a dispose
//     can only ever hit the call that failed;
//   - a disposed connection terminates its server session (DELETE) before it
//     closes, so the client leaks nothing either.

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

export interface Location {
    filePath: string;
    range: {
        start: { line: number; character: number };
        end: { line: number; character: number };
    };
}

export interface SerenaLspClientOptions {
    /** Bypass SERENA_BASE_URL — used by tests and edge-case deployments. */
    baseUrlOverride?: string;
    /** Per-call timeout for LSP RPCs. */
    timeoutMs?: number;
    /**
     * Test seam — inject a custom transport factory so unit tests can run
     * the client without spawning a real Serena daemon.
     */
    transportFactory?: (baseUrl: string) => unknown;
    /**
     * Test seam — inject a custom client factory. Mutually exclusive with
     * transportFactory in practice (factories build different layers).
     */
    clientFactory?: () => unknown;
    /** Test seam — override the default health probe. */
    healthProbe?: (baseUrl: string) => Promise<boolean>;
}

const DEFAULT_BASE_URL = 'http://127.0.0.1:18948';
const URL_TTL_MS = 30_000;
const DEFAULT_TIMEOUT_MS = 1500;
const HEALTH_PROBE_TIMEOUT_MS = 500;
// The session DELETE on dispose is best-effort and bounded: it runs inside the
// per-instance chain, so an unbounded DELETE against a wedged server would
// stall every later call on the client (#61 review).
const SESSION_TERMINATE_TIMEOUT_MS = 1000;
// A chained call that never settles — the SDK timeout did not fire — is
// abandoned after two per-call timeouts plus slack so the chain advances; the
// zombie keeps its connection until it settles (mirrors infra's call guard).
const CHAIN_STEP_ATTEMPT_FACTOR = 2;
const CHAIN_STEP_SLACK_MS = 5000;

function withDeadline<T>(promise: Promise<T>, ms: number, onTimeout: () => T): Promise<T> {
    if (!(ms > 0) || !Number.isFinite(ms)) return promise;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const deadline = new Promise<T>((resolve) => {
        timer = setTimeout(() => resolve(onTimeout()), ms);
        if (timer && typeof (timer as any).unref === 'function') (timer as any).unref();
    });
    promise.catch(() => undefined);
    return Promise.race([promise, deadline]).finally(() => { if (timer) clearTimeout(timer); });
}
// Sentinel "end of file" line used when Serena returns the oversized
// reference summary without per-line positions. Picked far above any
// realistic source-file length so the chunk-mapper's line-overlap filter
// matches every chunk in the affected file instead of dropping them.
const WHOLE_FILE_END_LINE = 1_000_000;
export { WHOLE_FILE_END_LINE };

async function defaultHealthProbe(baseUrl: string): Promise<boolean> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), HEALTH_PROBE_TIMEOUT_MS);
    try {
        // ANY status proves the port is bound to a working server (the root
        // answers 404). Connection refused / abort => service down. Never
        // `/mcp`: a bare request there opens a server session nothing closes
        // (#61).
        const res = await fetch(`${baseUrl}/`, { method: 'GET', signal: ctrl.signal });
        return res.status > 0;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

export class SerenaLspClient {
    private readonly opts: SerenaLspClientOptions;
    private readonly timeoutMs: number;
    private cachedBaseUrl: string | null = null;
    private cachedAt = 0;
    private mcpClient: any | null = null;
    private mcpClientUrl: string | null = null;
    private mcpTransport: any | null = null;
    // One call in flight per instance (#62). The chain never rejects: a
    // failed call settles the chain so the next call still runs.
    private queue: Promise<unknown> = Promise.resolve();
    // local-rag #64: the service's own error for the last call of each tool
    // that it did NOT answer, cleared when that tool is called again. A `null`
    // result with an entry here is a call the service never answered; a `null`
    // with no entry is the service answering that nothing matched.
    // Keyed to the CALL, not to the tool name: a predecessor the chain abandoned
    // settles on its own around two timeouts later — while its successor is
    // running — and a write from it would otherwise land on the successor's
    // entry and report a failure that call never had. Each call takes a
    // monotonic sequence and may only write while it is still the latest.
    private lastToolError = new Map<string, { seq: number; error: { reason: string; attempts: number; at: number } | null }>();
    private lastAttemptError: string | null = null;
    private callSequence = 0;

    constructor(opts: SerenaLspClientOptions = {}) {
        this.opts = opts;
        this.timeoutMs = opts.timeoutMs && opts.timeoutMs > 0 ? opts.timeoutMs : DEFAULT_TIMEOUT_MS;
    }

    /**
     * Resolve the Serena service's base URL, confirming it answers. Returns
     * null when the service is down, so the pool degrades to a no-op.
     * The positive result is cached for 30 seconds.
     */
    async detectBaseUrl(forceRefresh = false): Promise<string | null> {
        if (this.opts.baseUrlOverride) {
            return this.opts.baseUrlOverride;
        }
        const now = Date.now();
        if (!forceRefresh && this.cachedBaseUrl && now - this.cachedAt < URL_TTL_MS) {
            return this.cachedBaseUrl;
        }
        const configured = process.env.SERENA_BASE_URL;
        const baseUrl = configured && configured.length > 0 ? configured.replace(/\/+$/, '') : DEFAULT_BASE_URL;
        const healthy = await (this.opts.healthProbe ?? defaultHealthProbe)(baseUrl);
        if (!healthy) {
            return null;
        }
        this.cachedBaseUrl = baseUrl;
        this.cachedAt = now;
        return baseUrl;
    }

    private invalidateCache(): void {
        this.cachedBaseUrl = null;
        this.cachedAt = 0;
    }

    private async ensureClient(baseUrl: string): Promise<any | null> {
        if (this.mcpClient && this.mcpClientUrl === baseUrl) {
            return this.mcpClient;
        }
        await this.disposeClient();
        try {
            const client = this.opts.clientFactory
                ? (this.opts.clientFactory() as any)
                : new Client({ name: 'symbol-refs-lsp-pool', version: '0.1.0' }, { capabilities: {} });
            const transport = this.opts.transportFactory
                ? this.opts.transportFactory(baseUrl)
                : new StreamableHTTPClientTransport(new URL(`${baseUrl}/mcp`));
            // Kept BEFORE connect: a connect that fails after its initialize
            // POST has already opened a session, and only the transport can
            // close it (#61 review).
            this.mcpTransport = transport;
            await client.connect(transport as any);
            this.mcpClient = client;
            this.mcpClientUrl = baseUrl;
            return client;
        } catch (err) {
            console.warn(`[SerenaLspClient] connect failed: ${err instanceof Error ? err.message : err}`);
            await this.terminateTransport(this.mcpTransport);
            this.mcpTransport = null;
            return null;
        }
    }

    private async terminateTransport(transport: any | null): Promise<void> {
        if (!transport) return;
        try {
            if (typeof transport.terminateSession === 'function') {
                await withDeadline(Promise.resolve(transport.terminateSession()), SESSION_TERMINATE_TIMEOUT_MS, () => undefined);
            }
        } catch {
            /* swallow — best effort */
        }
    }

    private async disposeClient(): Promise<void> {
        if (!this.mcpClient) return;
        const client = this.mcpClient;
        const transport = this.mcpTransport;
        this.mcpClient = null;
        this.mcpClientUrl = null;
        this.mcpTransport = null;
        // Close the server session before the socket (#61): `close()` alone
        // aborts the stream and leaves the session alive on the server. The
        // DELETE is bounded, and the close runs whatever it did.
        try {
            await this.terminateTransport(transport);
        } finally {
            try {
                await client.close();
            } catch {
                /* swallow — best effort */
            }
        }
    }

    /** Tear down any cached MCP connection. Idempotent. */
    async close(): Promise<void> {
        await this.disposeClient();
    }

    private callTool(name: string, args: Record<string, unknown>): Promise<any | null> {
        // Serialised per instance (#62): the retry path below disposes the
        // shared connection, which must never happen under a sibling call.
        // Each step is bounded so a call that never settles cannot hold the
        // chain: the step yields null, the chain advances, and the zombie is
        // left to settle on its own (the outer guard defers the close).
        const deadline = this.timeoutMs * CHAIN_STEP_ATTEMPT_FACTOR + CHAIN_STEP_SLACK_MS;
        const seq = ++this.callSequence;
        const run = this.queue.then(() => withDeadline(this.callToolSerial(name, args, seq), deadline, () => {
            console.warn(`[SerenaLspClient] ${name} abandoned after ${deadline}ms; the chain moves on`);
            // local-rag #64: the abandonment is itself the service failing to
            // answer, and it is the LOUDEST such case — the call outlived two
            // per-call budgets plus slack. `callToolSerial` is still running and
            // has not recorded anything, so without this the caller sees a
            // `null` with no error record and reads it as "the service answered
            // that nothing matches", which is the exact hole mechanism 3 exists
            // to close. Guarded by the same sequence rule: a later call that has
            // already claimed the slot is not overwritten.
            const held = this.lastToolError.get(name);
            if (!held || held.seq <= seq) {
                this.lastToolError.set(name, { seq, error: { reason: `abandoned after ${deadline}ms without an answer`, attempts: 0, at: Date.now() } });
            }
            return null;
        }));
        this.queue = run.then(() => undefined, () => undefined);
        return run;
    }

    private async callToolSerial(name: string, args: Record<string, unknown>, seq = ++this.callSequence): Promise<any | null> {
        // local-rag #64: a call that the service never answered and a call the
        // service answered with nothing are both `null` to every caller of this
        // method, and telling them apart from the outside by elapsed time cannot
        // work — this method spends up to TWO budgets, and the guard's queue
        // wait is not part of either. So the transport records the service's own
        // error for the last call of each tool, which is the better evidence the
        // design asks for. The RETURN CONTRACT IS UNCHANGED: nothing here alters
        // what the retrieval pool or any existing caller receives.
        // This call claims the slot; an older, abandoned one can no longer write.
        const current = this.lastToolError.get(name);
        if (!current || current.seq <= seq) this.lastToolError.set(name, { seq, error: null });
        this.lastAttemptError = null;
        const record = (reason: string, attempts: number): void => {
            const held = this.lastToolError.get(name);
            if (held && held.seq > seq) return;
            this.lastToolError.set(name, { seq, error: { reason, attempts, at: Date.now() } });
        };
        const baseUrl = await this.detectBaseUrl();
        if (!baseUrl) { record('no base URL', 0); return null; }
        const result = await this.callToolOnce(baseUrl, name, args);
        // A tool RESULT flagged `isError` is the service declining to answer —
        // Serena's refusal of an over-long reference enumeration arrives this
        // way, as a result and not as an exception, and that refusal is the
        // 487 048-character case this whole change exists for. It is not an
        // answer of "nothing references this" and must not be recorded as one.
        if (result === null && this.lastAttemptError) { record(this.lastAttemptError, 1); return null; }
        if (result !== undefined) return result;
        // Retry once against a freshly probed URL — handles the service being
        // restarted between cache load and call.
        this.invalidateCache();
        await this.disposeClient();
        const refreshed = await this.detectBaseUrl(true);
        if (!refreshed) { record(this.lastAttemptError || 'no base URL on retry', 1); return null; }
        const retry = await this.callToolOnce(refreshed, name, args);
        // Same rule as the first attempt: a `null` with an attempt error behind
        // it is the service declining (an `isError` result), not an answer of
        // nothing.
        if (retry === null && this.lastAttemptError) { record(this.lastAttemptError, 2); return null; }
        if (retry === undefined) { record(this.lastAttemptError || 'both attempts failed', 2); return null; }
        return retry;
    }

    /**
     * local-rag #64: what the service said about the last call of `name`, when
     * it did not answer it. `undefined` means the last call of that tool was
     * answered (including answered with nothing), which is the distinction a
     * caller cannot otherwise make.
     */
    getLastToolError(name: string): { reason: string; attempts: number; at: number } | undefined {
        return this.lastToolError.get(name)?.error ?? undefined;
    }

    private async callToolOnce(
        baseUrl: string,
        name: string,
        args: Record<string, unknown>,
    ): Promise<any | null | undefined> {
        const client = await this.ensureClient(baseUrl);
        if (!client) return undefined;
        try {
            const response = await client.callTool({ name, arguments: args }, undefined, {
                timeout: this.timeoutMs,
            });
            if (response?.isError) {
                // Surfaced to `callToolSerial` so it is recorded as a call the
                // service did not answer, rather than swallowed into the same
                // `null` an unreferenced symbol returns.
                this.lastAttemptError = `${name} returned isError`;
                console.warn(`[SerenaLspClient] ${name} returned isError`);
                return null;
            }
            return response;
        } catch (err) {
            const reason = err instanceof Error ? err.message : String(err);
            // Kept so `callToolSerial` can record what the service actually said
            // (`MCP error -32001: Request timed out`) rather than a guess.
            this.lastAttemptError = reason;
            console.warn(`[SerenaLspClient] ${name} failed: ${reason}`);
            return undefined;
        }
    }

    /** Call Serena `find_symbol` to discover declaration locations of `namePath`. */
    async findSymbol(namePath: string, relativePath?: string, maxMatches = 5): Promise<Location[]> {
        const args: Record<string, unknown> = {
            name_path_pattern: namePath,
            max_matches: maxMatches,
        };
        if (relativePath) args.relative_path = relativePath;
        const response = await this.callTool('find_symbol', args);
        return parseFindSymbolResponse(response);
    }

    /** Call Serena `find_referencing_symbols` and project the response onto Location[]. */
    async findReferencingSymbols(
        namePath: string,
        relativePath: string,
        max: number,
    ): Promise<Location[]> {
        // max_answer_chars sized to keep the per-file/per-line breakdown
        // path active (Serena downgrades to file-counts-only when text
        // exceeds ~40000 chars; that mode only gives file paths without
        // line positions, which still works for chunk mapping but loses
        // resolution).
        const response = await this.callTool('find_referencing_symbols', {
            name_path: namePath,
            relative_path: relativePath,
            max_answer_chars: 60000,
        });
        const all = parseReferencesResponse(response);
        return all.slice(0, Math.max(0, max));
    }

    /** Call Serena `find_implementations` and project the response onto Location[]. */
    async findImplementations(
        namePath: string,
        relativePath: string,
        max: number,
    ): Promise<Location[]> {
        const response = await this.callTool('find_implementations', {
            name_path: namePath,
            relative_path: relativePath,
            max_answer_chars: 30000,
        });
        const all = parseImplementationsResponse(response);
        return all.slice(0, Math.max(0, max));
    }
}

// --- response parsing helpers ---------------------------------------------

function extractTextPayload(response: any): string | null {
    if (!response) return null;
    const content = response.content;
    if (Array.isArray(content)) {
        for (const c of content) {
            if (c?.type === 'text' && typeof c.text === 'string') {
                return c.text;
            }
        }
    }
    const structured = response.structuredContent?.result;
    if (typeof structured === 'string') return structured;
    return null;
}

function tryParseJsonBlob(text: string): any {
    // Serena returns mixed text + JSON ("References without surrounding
    // lines: {...}"). Pick the trailing JSON object/array.
    const trimmed = text.trim();
    if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
        try { return JSON.parse(trimmed); } catch { /* fallthrough */ }
    }
    const idx = trimmed.search(/[{[]/);
    if (idx >= 0) {
        try { return JSON.parse(trimmed.slice(idx)); } catch { /* fallthrough */ }
    }
    return null;
}

function pointFromLine(line: number | undefined): { line: number; character: number } {
    const safe = typeof line === 'number' && Number.isFinite(line) && line >= 0 ? Math.floor(line) : 0;
    return { line: safe, character: 0 };
}

function locationFromBodyLocation(filePath: string, body: any): Location | null {
    if (!filePath) return null;
    const startLine = typeof body?.start_line === 'number' ? body.start_line : 0;
    const endLine = typeof body?.end_line === 'number' ? body.end_line : startLine;
    return {
        filePath,
        range: {
            start: pointFromLine(startLine),
            end: pointFromLine(endLine),
        },
    };
}

export function parseFindSymbolResponse(response: any): Location[] {
    const text = extractTextPayload(response);
    if (!text) return [];
    if (/^Error executing tool/i.test(text)) return [];
    const data = tryParseJsonBlob(text);
    if (!Array.isArray(data)) return [];
    const out: Location[] = [];
    for (const entry of data) {
        const filePath = entry?.relative_path;
        if (typeof filePath !== 'string' || filePath.length === 0) continue;
        const loc = locationFromBodyLocation(filePath, entry?.body_location);
        if (loc) out.push(loc);
    }
    return out;
}

export function parseReferencesResponse(response: any): Location[] {
    const text = extractTextPayload(response);
    if (!text) return [];
    if (/^Error executing tool/i.test(text)) return [];
    const data = tryParseJsonBlob(text);
    if (!data || typeof data !== 'object') return [];
    const out: Location[] = [];
    if (looksLikeOversizedSummary(text)) {
        // Fallback path: Serena returned only file→count map. Synthesise
        // a whole-file range (line 0 .. very-large) so the chunk-mapper's
        // line-overlap filter widens to "any chunk in this file" instead
        // of nothing.
        for (const [filePath, count] of Object.entries(data)) {
            if (typeof filePath !== 'string' || typeof count !== 'number') continue;
            out.push({
                filePath,
                range: {
                    start: pointFromLine(0),
                    end: pointFromLine(WHOLE_FILE_END_LINE),
                },
            });
        }
        return out;
    }
    for (const [filePath, perKind] of Object.entries(data)) {
        if (typeof filePath !== 'string' || !perKind || typeof perKind !== 'object') continue;
        for (const refs of Object.values(perKind as Record<string, unknown>)) {
            if (!Array.isArray(refs)) continue;
            for (const r of refs) {
                const body = r?.body_location;
                const startLine = typeof r?.reference_line === 'number'
                    ? r.reference_line
                    : typeof body?.start_line === 'number' ? body.start_line : undefined;
                const endLine = typeof r?.reference_line === 'number'
                    ? r.reference_line
                    : typeof body?.end_line === 'number' ? body.end_line : startLine;
                out.push({
                    filePath,
                    range: {
                        start: pointFromLine(startLine),
                        end: pointFromLine(endLine),
                    },
                });
            }
        }
    }
    return out;
}

export function parseImplementationsResponse(response: any): Location[] {
    const text = extractTextPayload(response);
    if (!text) return [];
    if (/^Error executing tool/i.test(text)) return [];
    const data = tryParseJsonBlob(text);
    if (!Array.isArray(data)) return [];
    const out: Location[] = [];
    for (const entry of data) {
        const filePath = entry?.relative_path;
        if (typeof filePath !== 'string' || filePath.length === 0) continue;
        const loc = locationFromBodyLocation(filePath, entry?.body_location);
        if (loc) out.push(loc);
    }
    return out;
}

function looksLikeOversizedSummary(text: string): boolean {
    return /Reference counts per file/i.test(text) || /answer is too long/i.test(text);
}
