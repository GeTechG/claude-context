// rag-symbol-refs-lsp-pool: thin Serena MCP/SSE client used by the
// symbol-refs pool. Exposes three LSP-backed lookups (find_symbol,
// find_referencing_symbols, find_implementations) over Serena's
// stdio-less SSE transport. All errors fold to empty arrays so the pool
// degrades to a no-op when the daemon is missing or the LSP can't compile.
//
// Daemon discovery follows configs/claude-plugin/scripts/serena-shared.sh:
// the per-project state.json lives at
//   ${XDG_CACHE_HOME:-$HOME/.cache}/serena-daemons/<KEY>/state.json
// where KEY = sha256(realpath(<indexed codebase>))[:12]. The state file
// only stores `{pid, port, ...}` — we construct the base URL ourselves
// (`http://127.0.0.1:<port>`) and treat ANY HTTP response on `/` as
// "alive" because Serena's MCP/SSE server doesn't expose `/health`.

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';

export interface Location {
    filePath: string;
    range: {
        start: { line: number; character: number };
        end: { line: number; character: number };
    };
}

export interface SerenaLspClientOptions {
    /** Bypass `state.json` discovery — used by tests and edge-case deployments. */
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
    /** Test seam — override the default state-file resolver. */
    stateFileResolver?: (indexPath: string) => string;
    /** Test seam — override the default health probe. */
    healthProbe?: (baseUrl: string) => Promise<boolean>;
}

const URL_TTL_MS = 30_000;
const DEFAULT_TIMEOUT_MS = 1500;
const HEALTH_PROBE_TIMEOUT_MS = 500;
// Sentinel "end of file" line used when Serena returns the oversized
// reference summary without per-line positions. Picked far above any
// realistic source-file length so the chunk-mapper's line-overlap filter
// matches every chunk in the affected file instead of dropping them.
const WHOLE_FILE_END_LINE = 1_000_000;
export { WHOLE_FILE_END_LINE };

function defaultStateFile(indexPath: string): string {
    const cacheRoot = process.env.XDG_CACHE_HOME && process.env.XDG_CACHE_HOME.length > 0
        ? process.env.XDG_CACHE_HOME
        : path.join(os.homedir(), '.cache');
    const real = fs.realpathSync(indexPath);
    const key = crypto.createHash('sha256').update(real).digest('hex').slice(0, 12);
    return path.join(cacheRoot, 'serena-daemons', key, 'state.json');
}

async function defaultHealthProbe(baseUrl: string): Promise<boolean> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), HEALTH_PROBE_TIMEOUT_MS);
    try {
        // Serena's MCP/SSE server (uvicorn) returns 404 on `/` but is
        // alive — any HTTP response confirms the port is bound to a
        // working server. Connection refused / abort => dead daemon.
        const res = await fetch(`${baseUrl}/`, { method: 'GET', signal: ctrl.signal });
        return res.status > 0;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

export class SerenaLspClient {
    private readonly indexPath: string;
    private readonly opts: SerenaLspClientOptions;
    private readonly timeoutMs: number;
    private cachedBaseUrl: string | null = null;
    private cachedAt = 0;
    private mcpClient: any | null = null;
    private mcpClientUrl: string | null = null;

    constructor(indexPath: string, opts: SerenaLspClientOptions = {}) {
        this.indexPath = indexPath;
        this.opts = opts;
        this.timeoutMs = opts.timeoutMs && opts.timeoutMs > 0 ? opts.timeoutMs : DEFAULT_TIMEOUT_MS;
    }

    /**
     * Resolve the daemon's base URL. Returns null when the daemon is not
     * running or its state.json is missing/stale. Cached for 30 seconds.
     */
    async detectBaseUrl(forceRefresh = false): Promise<string | null> {
        if (this.opts.baseUrlOverride) {
            return this.opts.baseUrlOverride;
        }
        const now = Date.now();
        if (!forceRefresh && this.cachedBaseUrl && now - this.cachedAt < URL_TTL_MS) {
            return this.cachedBaseUrl;
        }
        const stateFile = (this.opts.stateFileResolver ?? defaultStateFile)(this.indexPath);
        let state: any;
        try {
            const raw = await fs.promises.readFile(stateFile, 'utf-8');
            state = JSON.parse(raw);
        } catch {
            return null;
        }
        const port = state?.port;
        if (typeof port !== 'number' || !Number.isFinite(port) || port <= 0) {
            return null;
        }
        const baseUrl = `http://127.0.0.1:${port}`;
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
                : new SSEClientTransport(new URL(`${baseUrl}/sse`));
            await client.connect(transport as any);
            this.mcpClient = client;
            this.mcpClientUrl = baseUrl;
            return client;
        } catch (err) {
            console.warn(`[SerenaLspClient] connect failed: ${err instanceof Error ? err.message : err}`);
            return null;
        }
    }

    private async disposeClient(): Promise<void> {
        if (!this.mcpClient) return;
        try {
            await this.mcpClient.close();
        } catch {
            /* swallow — best effort */
        }
        this.mcpClient = null;
        this.mcpClientUrl = null;
    }

    /** Tear down any cached MCP connection. Idempotent. */
    async close(): Promise<void> {
        await this.disposeClient();
    }

    private async callTool(name: string, args: Record<string, unknown>): Promise<any | null> {
        const baseUrl = await this.detectBaseUrl();
        if (!baseUrl) return null;
        const result = await this.callToolOnce(baseUrl, name, args);
        if (result !== undefined) return result;
        // Retry once with a fresh discovery — handles port reassignment after
        // a daemon restart between cache load and call.
        this.invalidateCache();
        await this.disposeClient();
        const refreshed = await this.detectBaseUrl(true);
        if (!refreshed) return null;
        const retry = await this.callToolOnce(refreshed, name, args);
        return retry === undefined ? null : retry;
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
                console.warn(`[SerenaLspClient] ${name} returned isError`);
                return null;
            }
            return response;
        } catch (err) {
            const reason = err instanceof Error ? err.message : String(err);
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
                const line = typeof r?.reference_line === 'number' ? r.reference_line : undefined;
                out.push({
                    filePath,
                    range: {
                        start: pointFromLine(line),
                        end: pointFromLine(line),
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
