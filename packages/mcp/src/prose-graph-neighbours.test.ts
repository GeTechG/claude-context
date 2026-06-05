import { test } from "node:test";
import assert from "node:assert/strict";
import type { ProseGraphEdge, SemanticSearchResult } from "@zilliz/claude-context-core";
import {
    selectNeighbours,
    normalizeEdgeTypes,
    clampLimit,
    formatExpansion,
    DEFAULT_EXPAND_LIMIT,
    MAX_EXPAND_LIMIT,
} from "./prose-graph-neighbours.js";

function edge(to: string, type: ProseGraphEdge["type"], weight: number): ProseGraphEdge {
    return { to, type, weight };
}

function result(id: string, over: Partial<SemanticSearchResult> = {}): SemanticSearchResult {
    return {
        content: `content of ${id}`,
        relativePath: `docs/${id}.md`,
        startLine: 1,
        endLine: 5,
        language: "markdown",
        score: 1,
        chunk_id: id,
        ...over,
    } as SemanticSearchResult;
}

test("clampLimit: defaults, caps, and rejects bad input", () => {
    assert.equal(clampLimit(undefined), DEFAULT_EXPAND_LIMIT);
    assert.equal(clampLimit(0), DEFAULT_EXPAND_LIMIT);
    assert.equal(clampLimit(-3), DEFAULT_EXPAND_LIMIT);
    assert.equal(clampLimit("7" as unknown), DEFAULT_EXPAND_LIMIT);
    assert.equal(clampLimit(3), 3);
    assert.equal(clampLimit(3.9), 3);
    assert.equal(clampLimit(1000), MAX_EXPAND_LIMIT);
});

test("normalizeEdgeTypes: keeps recognized, drops unknown, null on empty", () => {
    assert.equal(normalizeEdgeTypes(undefined), null);
    assert.equal(normalizeEdgeTypes([]), null);
    assert.equal(normalizeEdgeTypes(["nope"]), null);
    const s = normalizeEdgeTypes(["heading", "bogus", "link"]);
    assert.ok(s);
    assert.deepEqual([...(s as Set<string>)].sort(), ["heading", "link"]);
});

test("selectNeighbours: orders by weight desc with id tie-break", () => {
    const edges = [
        edge("c", "heading", 0.5),
        edge("a", "co_mention", 0.9),
        edge("b", "co_mention", 0.9),
    ];
    const out = selectNeighbours(edges, null, 10);
    assert.deepEqual(out.map((e) => e.to), ["a", "b", "c"]);
});

test("selectNeighbours: edge_types filter", () => {
    const edges = [
        edge("a", "heading", 0.9),
        edge("b", "co_mention", 0.8),
        edge("c", "code_example", 0.7),
    ];
    const out = selectNeighbours(edges, new Set(["heading", "code_example"]), 10);
    assert.deepEqual(out.map((e) => e.to).sort(), ["a", "c"]);
});

test("selectNeighbours: limit cap", () => {
    const edges = [
        edge("a", "heading", 0.9),
        edge("b", "co_mention", 0.8),
        edge("c", "code_example", 0.7),
    ];
    const out = selectNeighbours(edges, null, 2);
    assert.equal(out.length, 2);
    assert.deepEqual(out.map((e) => e.to), ["a", "b"]);
});

test("selectNeighbours: dedups by neighbour id keeping strongest edge", () => {
    const edges = [
        edge("a", "co_mention", 0.4),
        edge("a", "heading", 0.95),
        edge("b", "link", 0.6),
    ];
    const out = selectNeighbours(edges, null, 10);
    assert.equal(out.length, 2);
    assert.equal(out[0].to, "a");
    assert.equal(out[0].type, "heading"); // strongest edge wins
    assert.equal(out[0].weight, 0.95);
});

test("selectNeighbours: empty / absent edges", () => {
    assert.deepEqual(selectNeighbours([], null, 10), []);
    assert.deepEqual(selectNeighbours(undefined as unknown as ProseGraphEdge[], null, 10), []);
});

test("formatExpansion: header counts by type, renders content blocks", () => {
    const selected = [
        edge("a", "heading", 0.9),
        edge("b", "code_example", 0.8),
    ];
    const contentById = new Map<string, SemanticSearchResult>([
        ["a", result("a", { heading_path: ["Guide", "Null safety"] })],
        ["b", result("b", { content_type: "code_example", language: "haxe" })],
    ]);
    const text = formatExpansion("seed1", selected, contentById, 2);
    assert.match(text, /neighbours of chunk `seed1`/);
    assert.match(text, /returning 2 of 2 1-hop neighbour\(s\)/);
    assert.match(text, /heading=1, code_example=1/);
    assert.match(text, /\[heading\] weight=0\.900/);
    assert.match(text, /Chunk-ID: a/);
    assert.match(text, /Heading: Guide › Null safety/);
    assert.match(text, /content of b/);
});

test("formatExpansion: stale neighbours skipped and noted", () => {
    const selected = [
        edge("a", "heading", 0.9),
        edge("stale", "co_mention", 0.8),
    ];
    const contentById = new Map<string, SemanticSearchResult>([
        ["a", result("a")],
        // "stale" intentionally absent from the fetch map
    ]);
    const text = formatExpansion("seed1", selected, contentById, 2);
    assert.match(text, /Chunk-ID: a/);
    assert.doesNotMatch(text, /Chunk-ID: stale/);
    assert.match(text, /1 neighbour\(s\) could not be fetched/);
    assert.match(text, /infra\/build-prose-graph\.js/);
});
