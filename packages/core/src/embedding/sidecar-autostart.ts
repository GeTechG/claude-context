/**
 * On-demand sidecar autostart.
 *
 * The embedding/reranker sidecars (dense, sparse, reranker containers) are
 * frequently stopped to free GPU VRAM. When that happens a retrieval query
 * fails with a connection error because there is no server to encode it.
 *
 * This module lets a fetch to one of those sidecars recover transparently:
 * on a connection error it runs a configured command to bring the sidecars
 * back up, waits for their health endpoints, then the caller retries once.
 *
 * Entirely opt-in and deployment-specific: with `SIDECAR_AUTOSTART_CMD`
 * unset (the default), {@link wrapFetchWithAutostart} returns the fetch
 * untouched and there is zero behavior change — the generic provider stays
 * generic. The local-rag MCP config wires the env to a `docker compose up`.
 *
 * Env:
 *   SIDECAR_AUTOSTART_CMD          shell command that brings the sidecars up
 *                                  (e.g. `docker compose -f … up -d …`).
 *                                  When unset, autostart is disabled.
 *   SIDECAR_AUTOSTART_HEALTH_URLS  comma-separated health URLs to poll until
 *                                  HTTP 200 before retrying (optional).
 *   SIDECAR_AUTOSTART_TIMEOUT_MS   health-poll budget, default 180000.
 */

// Shared single-flight bring-up: many embedding/reranker calls fail at once
// when the backend is down; they must all await ONE `docker compose up`, not
// stampede it. Reset to null once it settles so a later down-event re-triggers.
let inflight: Promise<boolean> | null = null;

/**
 * True when a thrown fetch error looks like the server is simply not there
 * (container down / port closed) — the case autostart can fix. HTTP error
 * responses (4xx/5xx) are NOT connection errors and are left to the caller.
 */
function isConnectionError(err: unknown): boolean {
    if (!(err instanceof Error)) return false;
    // Node's undici throws `TypeError: fetch failed` with the real cause nested.
    const cause = (err as { cause?: { code?: string } }).cause;
    const code = cause?.code ?? (err as { code?: string }).code;
    const connCodes = [
        'ECONNREFUSED',
        'ECONNRESET',
        'ETIMEDOUT',
        'EHOSTUNREACH',
        'ENOTFOUND',
        'UND_ERR_SOCKET',
        'UND_ERR_CONNECT_TIMEOUT',
    ];
    if (code && connCodes.includes(code)) return true;
    return /fetch failed|ECONNREFUSED|socket hang up|other side closed/i.test(err.message);
}

async function pollHealth(urls: string[], timeoutMs: number, fetchImpl: typeof fetch): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    const pending = new Set(urls);
    while (Date.now() < deadline) {
        for (const url of [...pending]) {
            try {
                const resp = await fetchImpl(url, { method: 'GET' });
                if (resp.ok) pending.delete(url);
            } catch {
                // not up yet — keep waiting
            }
        }
        if (pending.size === 0) return true;
        await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    return pending.size === 0;
}

async function bringUp(cmd: string, fetchImpl: typeof fetch): Promise<boolean> {
    try {
        const { exec } = await import('node:child_process');
        const { promisify } = await import('node:util');
        const execAsync = promisify(exec);

        // stderr so it never pollutes MCP stdio (the protocol channel).
        console.error(`[sidecar-autostart] embedding backend unreachable — running: ${cmd}`);
        await execAsync(cmd, { timeout: 120000 });

        const healthUrls = (process.env.SIDECAR_AUTOSTART_HEALTH_URLS || '')
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean);
        if (healthUrls.length) {
            const timeoutMs = Number(process.env.SIDECAR_AUTOSTART_TIMEOUT_MS || '180000');
            const ok = await pollHealth(healthUrls, timeoutMs, fetchImpl);
            if (!ok) {
                console.error('[sidecar-autostart] timed out waiting for sidecar health');
                return false;
            }
        }
        console.error('[sidecar-autostart] sidecars are up');
        return true;
    } catch (err) {
        console.error(`[sidecar-autostart] failed to bring up sidecars: ${err}`);
        return false;
    }
}

/** Bring the sidecars up (single-flight). Resolves false if disabled or it failed. */
export function ensureSidecarsUp(fetchImpl: typeof fetch): Promise<boolean> {
    const cmd = process.env.SIDECAR_AUTOSTART_CMD;
    if (!cmd) return Promise.resolve(false);
    if (!inflight) {
        inflight = bringUp(cmd, fetchImpl).finally(() => {
            inflight = null;
        });
    }
    return inflight;
}

/**
 * Wrap a fetch so a connection error to a sidecar triggers an autostart and a
 * single retry. No-op (returns the original fetch) when autostart is disabled.
 */
export function wrapFetchWithAutostart(fetchImpl: typeof fetch): typeof fetch {
    if (!process.env.SIDECAR_AUTOSTART_CMD) return fetchImpl;
    const wrapped = (async (...args: Parameters<typeof fetch>) => {
        try {
            return await fetchImpl(...args);
        } catch (err) {
            if (isConnectionError(err) && (await ensureSidecarsUp(fetchImpl))) {
                return await fetchImpl(...args);
            }
            throw err;
        }
    }) as typeof fetch;
    return wrapped;
}
