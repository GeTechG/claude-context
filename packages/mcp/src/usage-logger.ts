// usage-logging-dataset: non-blocking JSONL sink that captures real RAG usage
// (every search_code / expand_context call and the agent's recorded answers)
// into an append-only training/analysis dataset. The MCP server is the only
// seam where *real* agent traffic passes — the eval harness calls core
// semanticSearch directly and bypasses this layer, so capture here is real
// usage by construction.
//
// Hard requirements (see design.md):
//  - Default ON; kill-switch via RAG_USAGE_LOG (0/false/off → disabled).
//  - Non-blocking, best-effort: never awaited on the response path, every
//    failure swallowed to stderr as `[USAGE-LOG] ...`. A logging error, full
//    disk, or unwritable dir can never reach a tool result.
//  - Append-only JSONL, one record per line, rotated per UTC day by filename.
//  - Absolute knowledge-root / repo paths relativised so the dataset is
//    portable and free of machine-specific paths.
//
// Pure helpers (parseEnabled, dayBucket, relativisePath, serializeRecord,
// newRequestId, newAnswerId) are exported and unit-tested without any Milvus
// or filesystem dependency.

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as crypto from "crypto";
import { fileURLToPath } from "url";

// ---------------------------------------------------------------------------
// Kill-switch
// ---------------------------------------------------------------------------

/**
 * Parse the RAG_USAGE_LOG env value. Default ON: only the explicit falsey
 * tokens `0` / `false` / `off` (case-insensitive) disable capture. Pure so the
 * kill-switch semantics are unit-testable.
 */
export function parseEnabled(val: string | undefined | null): boolean {
    if (val === undefined || val === null) return true;
    const v = String(val).trim().toLowerCase();
    return !(v === "0" || v === "false" || v === "off");
}

/** Whether usage capture is currently enabled (reads the env lazily). */
export function isUsageLogEnabled(): boolean {
    return parseEnabled(process.env.RAG_USAGE_LOG);
}

// ---------------------------------------------------------------------------
// Id generation — no Math.random / Date.now dependency
// ---------------------------------------------------------------------------

let idCounter = 0;

function genId(prefix: string): string {
    try {
        return `${prefix}_${crypto.randomUUID()}`;
    } catch {
        // Fallback for runtimes without crypto.randomUUID: a per-process
        // counter salted with the pid. Ids only need dataset uniqueness;
        // ordering comes from the stamped ISO timestamp.
        idCounter += 1;
        return `${prefix}_${process.pid}_${idCounter}`;
    }
}

/** Fresh request_id for a retrieval record / Request-ID response handle. */
export function newRequestId(): string {
    return genId("req");
}

/** Fresh record_id for an answer record. */
export function newAnswerId(): string {
    return genId("ans");
}

// ---------------------------------------------------------------------------
// Day bucketing
// ---------------------------------------------------------------------------

/**
 * Derive the UTC day bucket (YYYY-MM-DD) from a record's ISO-8601 timestamp.
 * Pure; the common case is a plain prefix slice so it needs no Date at all.
 */
export function dayBucket(iso: string): string {
    const m = /^(\d{4})-(\d{2})-(\d{2})T/.exec(iso);
    if (m) return `${m[1]}-${m[2]}-${m[3]}`;
    // Fallback for non-ISO input: normalise via Date in UTC.
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "unknown-date";
    const y = d.getUTCFullYear();
    const mo = String(d.getUTCMonth() + 1).padStart(2, "0");
    const da = String(d.getUTCDate()).padStart(2, "0");
    return `${y}-${mo}-${da}`;
}

// ---------------------------------------------------------------------------
// Path relativisation
// ---------------------------------------------------------------------------

function expandTilde(v: string): string {
    if (v === "~") return os.homedir();
    if (v.startsWith("~/")) return path.join(os.homedir(), v.slice(2));
    return v;
}

function isUnder(p: string, root: string): boolean {
    const rel = path.relative(root, p);
    return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
}

export interface RootOpts {
    repoRoot?: string | null;
    knowledgeRoot?: string | null;
}

/**
 * Rewrite an absolute path to a portable, machine-independent form:
 *   - under the repo root  → repo-relative (e.g. `knowledge/foo/bar.hx`)
 *   - under an external knowledge root → `<knowledgeBasename>/<rel>`
 *   - otherwise            → unchanged
 * Relative paths (e.g. a result's `relativePath`) are returned untouched.
 * Pure so relativisation is unit-testable with injected roots.
 */
export function relativisePath(p: unknown, opts: RootOpts = {}): unknown {
    if (typeof p !== "string" || p.length === 0) return p;
    if (!path.isAbsolute(p)) return p;
    const { repoRoot, knowledgeRoot } = opts;
    if (repoRoot && isUnder(p, repoRoot)) {
        return path.relative(repoRoot, p) || ".";
    }
    if (knowledgeRoot && isUnder(p, knowledgeRoot)) {
        const rel = path.relative(knowledgeRoot, p);
        return rel ? `${path.basename(knowledgeRoot)}/${rel}` : path.basename(knowledgeRoot);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Default root resolution (runtime only; tests inject roots explicitly)
// ---------------------------------------------------------------------------

let cachedRepoRoot: string | null | undefined;
let cachedKnowledgeRoot: string | null | undefined;

function defaultRepoRoot(): string {
    if (cachedRepoRoot !== undefined && cachedRepoRoot !== null) return cachedRepoRoot;
    // This module lives at <repo>/patches/claude-context/packages/mcp/{src,dist}/
    // so the repo root is five levels up from the file's directory (same depth
    // whether running from compiled dist or tsx-loaded src).
    const here = path.dirname(fileURLToPath(import.meta.url));
    cachedRepoRoot = path.resolve(here, "..", "..", "..", "..", "..");
    return cachedRepoRoot;
}

function defaultKnowledgeRoot(repoRoot: string): string | null {
    if (cachedKnowledgeRoot !== undefined) return cachedKnowledgeRoot;
    const env = process.env.LOCAL_RAG_KNOWLEDGE_ROOT;
    if (typeof env === "string" && env.trim().length > 0) {
        cachedKnowledgeRoot = expandTilde(env.trim());
        return cachedKnowledgeRoot;
    }
    const inTree = path.join(repoRoot, "knowledge");
    try {
        if (fs.statSync(inTree).isDirectory()) {
            cachedKnowledgeRoot = inTree;
            return cachedKnowledgeRoot;
        }
    } catch {
        // not present — fall through
    }
    cachedKnowledgeRoot = null;
    return null;
}

/** Relativise a path using the runtime-resolved repo / knowledge roots. */
export function relativise(p: unknown): unknown {
    const repoRoot = defaultRepoRoot();
    return relativisePath(p, { repoRoot, knowledgeRoot: defaultKnowledgeRoot(repoRoot) });
}

/**
 * Resolve the single knowledge-base root this server serves, so the query
 * tools can default `path` when the agent omits it (there is only ever one
 * base in a local-rag deployment). Mirrors infra/lib/knowledge-root.js — the
 * resolver the indexer uses — so the resolved path, and therefore the Milvus
 * collection hash, is identical to what was indexed:
 *   1. LOCAL_RAG_KNOWLEDGE_ROOT env (tilde-expanded)
 *   2. <repo>/local-rag.config.json → knowledgeRoot (v1, tilde-expanded, absolute)
 *   3. in-tree <repo>/knowledge
 * Returns null when none resolve — the caller must then require an explicit path.
 */
export function resolveKnowledgeRoot(): string | null {
    const env = process.env.LOCAL_RAG_KNOWLEDGE_ROOT;
    if (typeof env === "string" && env.trim().length > 0) {
        return path.resolve(expandTilde(env.trim()));
    }
    const repoRoot = defaultRepoRoot();
    // Optional external corpus location (same field the indexer reads).
    try {
        const raw = fs.readFileSync(path.join(repoRoot, "local-rag.config.json"), "utf8");
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed.knowledgeRoot === "string" && parsed.knowledgeRoot.trim().length > 0) {
            const expanded = expandTilde(parsed.knowledgeRoot.trim());
            if (path.isAbsolute(expanded)) return path.resolve(expanded);
        }
    } catch {
        // missing / unparseable → fall through to in-tree default
    }
    const inTree = path.join(repoRoot, "knowledge");
    try {
        if (fs.statSync(inTree).isDirectory()) return inTree;
    } catch {
        // not present
    }
    return null;
}

// ---------------------------------------------------------------------------
// Output directory
// ---------------------------------------------------------------------------

/** Resolve the dataset directory: RAG_USAGE_LOG_DIR or <repo>/infra/usage-dataset. */
export function logDir(): string {
    const env = process.env.RAG_USAGE_LOG_DIR;
    if (typeof env === "string" && env.trim().length > 0) {
        return path.resolve(expandTilde(env.trim()));
    }
    return path.join(defaultRepoRoot(), "infra", "usage-dataset");
}

// ---------------------------------------------------------------------------
// Serialization + append
// ---------------------------------------------------------------------------

/** Serialize one record to a single JSONL line (no trailing newline). Pure. */
export function serializeRecord(record: unknown): string {
    return JSON.stringify(record);
}

/**
 * Fire-and-forget append of one record to `<prefix>-YYYY-MM-DD.jsonl`. Stamps a
 * UTC ISO timestamp when the record lacks one, creates the dir lazily, and
 * swallows every failure to stderr. Returns immediately — the actual write
 * happens on async fs callbacks and is never awaited on the response path.
 */
function appendRecord(prefix: "retrieval" | "answers", record: any): void {
    if (!isUsageLogEnabled()) return;
    try {
        if (typeof record.ts !== "string") {
            record.ts = new Date().toISOString();
        }
        const day = dayBucket(record.ts);
        const dir = logDir();
        const file = path.join(dir, `${prefix}-${day}.jsonl`);
        const line = serializeRecord(record) + "\n";
        fs.mkdir(dir, { recursive: true }, (mkErr) => {
            if (mkErr) {
                console.error(`[USAGE-LOG] mkdir ${dir} failed: ${mkErr.message}`);
                return;
            }
            fs.appendFile(file, line, (apErr) => {
                if (apErr) console.error(`[USAGE-LOG] append ${file} failed: ${apErr.message}`);
            });
        });
    } catch (err: any) {
        console.error(`[USAGE-LOG] ${err?.message ?? err}`);
    }
}

/** Append one retrieval record (search_code / expand_context). Non-blocking. */
export function logRetrieval(record: any): void {
    appendRecord("retrieval", record);
}

/** Append one answer record (record_answer). Non-blocking. */
export function logAnswer(record: any): void {
    appendRecord("answers", record);
}
