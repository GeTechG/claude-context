import { test } from "node:test";
import assert from "node:assert/strict";
import {
    parseEnabled,
    dayBucket,
    relativisePath,
    serializeRecord,
    newRequestId,
    newAnswerId,
} from "./usage-logger.js";

// ---- kill-switch parsing -------------------------------------------------

test("parseEnabled defaults ON when unset", () => {
    assert.equal(parseEnabled(undefined), true);
    assert.equal(parseEnabled(null), true);
    assert.equal(parseEnabled(""), true);
});

test("parseEnabled disables only on 0 / false / off (case-insensitive)", () => {
    for (const v of ["0", "false", "off", "FALSE", "Off", " 0 ", "  OFF  "]) {
        assert.equal(parseEnabled(v), false, `expected ${JSON.stringify(v)} to disable`);
    }
});

test("parseEnabled stays ON for any other token", () => {
    for (const v of ["1", "true", "on", "yes", "anything"]) {
        assert.equal(parseEnabled(v), true, `expected ${JSON.stringify(v)} to enable`);
    }
});

// ---- day bucketing -------------------------------------------------------

test("dayBucket slices the UTC day from an ISO timestamp", () => {
    assert.equal(dayBucket("2026-06-06T16:45:21.643Z"), "2026-06-06");
    assert.equal(dayBucket("2026-01-02T00:00:00.000Z"), "2026-01-02");
});

test("dayBucket normalises non-prefix input via Date and flags garbage", () => {
    assert.equal(dayBucket("2026-06-06"), "2026-06-06"); // Date-parsed UTC
    assert.equal(dayBucket("not-a-date"), "unknown-date");
});

// ---- path relativisation -------------------------------------------------

const ROOTS = {
    repoRoot: "/home/sergey/local-rag",
    knowledgeRoot: "/home/sergey/local-rag/knowledge",
};

test("relativisePath rewrites repo-root paths to repo-relative", () => {
    assert.equal(relativisePath("/home/sergey/local-rag/knowledge", ROOTS), "knowledge");
    assert.equal(
        relativisePath("/home/sergey/local-rag/knowledge/haxe/std/StringTools.hx", ROOTS),
        "knowledge/haxe/std/StringTools.hx",
    );
});

test("relativisePath rewrites an external knowledge root with a basename prefix", () => {
    const ext = { repoRoot: "/repo", knowledgeRoot: "/data/kb" };
    assert.equal(relativisePath("/data/kb/python/asyncio.py", ext), "kb/python/asyncio.py");
    assert.equal(relativisePath("/data/kb", ext), "kb");
});

test("relativisePath leaves relative and out-of-root paths untouched", () => {
    assert.equal(relativisePath("haxe/std/Std.hx", ROOTS), "haxe/std/Std.hx");
    assert.equal(relativisePath("/etc/passwd", ROOTS), "/etc/passwd");
    assert.equal(relativisePath(undefined as any, ROOTS), undefined);
    assert.equal(relativisePath(123 as any, ROOTS), 123);
});

// ---- serialization shape -------------------------------------------------

test("serializeRecord produces one parseable JSON line preserving the record", () => {
    const rec = {
        request_id: "req_abc",
        ts: "2026-06-06T16:45:21.643Z",
        tool: "search_code",
        codebase: "knowledge/haxe",
        query: "string replace",
        result_count: 2,
        results: [{ rank: 1, score: 0.9, chunk_id: "c1", relativePath: "std/StringTools.hx" }],
    };
    const line = serializeRecord(rec);
    assert.equal(line.includes("\n"), false, "serialized record must be a single line");
    assert.deepEqual(JSON.parse(line), rec);
});

// ---- id generation -------------------------------------------------------

test("newRequestId / newAnswerId are prefixed and unique", () => {
    const a = newRequestId();
    const b = newRequestId();
    const c = newAnswerId();
    assert.match(a, /^req_/);
    assert.match(c, /^ans_/);
    assert.notEqual(a, b);
});
