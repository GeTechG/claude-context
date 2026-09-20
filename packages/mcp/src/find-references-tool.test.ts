// find-references-as-a-tool: the switch (D6) and the text the bound answers
// with (D3). No Milvus, no model call.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
    FIND_REFERENCES_TOOL_SPEC,
    findReferencesEnabled,
    findReferencesTools,
    formatReferences,
} from "./find-references-tool.js";

test("the tool is absent from the tool list unless the switch is on", () => {
    for (const env of [{}, { FIND_REFERENCES_TOOL: '' }, { FIND_REFERENCES_TOOL: 'false' }, { FIND_REFERENCES_TOOL: 'off' }, { FIND_REFERENCES_TOOL: '0' }]) {
        assert.equal(findReferencesEnabled(env), false);
        assert.deepEqual(findReferencesTools(env), []);
    }
    for (const env of [{ FIND_REFERENCES_TOOL: 'true' }, { FIND_REFERENCES_TOOL: 'TRUE' }, { FIND_REFERENCES_TOOL: '1' }, { FIND_REFERENCES_TOOL: 'on' }]) {
        assert.equal(findReferencesEnabled(env), true);
        assert.deepEqual(findReferencesTools(env), [FIND_REFERENCES_TOOL_SPEC]);
    }
});

test("the frequency bound answers with the count and no rows, and does not send the agent to `categories`", () => {
    const text = formatReferences({
        symbol: 'get_singleton',
        groups: [], limit: 30, limitClampedFrom: null,
        bounded: { identifier: 'get_singleton', documentFrequency: 1164, boundDocuments: 196, boundQuantile: 0.99 },
    } as any, undefined);
    assert.match(text, /1164 documents/);
    assert.match(text, /bound of 196/);
    // The bound is corpus-wide and is read BEFORE any scope, so "narrow by
    // category" would be an instruction that cannot work.
    assert.match(text, /corpus-wide, so a `categories` scope will not lift it/);
    assert.doesNotMatch(text, /1164 of 1164/);
    assert.doesNotMatch(text, /Chunk-ID/);
});

test("a bound on a name path says which identifier the count belongs to", () => {
    const text = formatReferences({
        symbol: 'Foo.bar', groups: [], limit: 30, limitClampedFrom: null,
        bounded: { identifier: 'bar', documentFrequency: 900, boundDocuments: 196, boundQuantile: 0.99 },
    } as any, undefined);
    assert.match(text, /`bar` \(the leaf of `Foo\.bar`\)/);
});

test("a failed query is reported as unknown, never as an empty index", () => {
    const allFailed = formatReferences({
        symbol: 'Bytes', bounded: null, limit: 30, limitClampedFrom: null,
        groups: [{ relation: 'mentions', totalFiles: 0, atFetchCeiling: false, files: [], failed: 'CollectionNotExists' }],
    } as any, undefined);
    assert.match(allFailed, /every index query failed/);
    assert.match(allFailed, /CollectionNotExists/);
    assert.match(allFailed, /NOT "there are no such edges"/);
    assert.doesNotMatch(allFailed, /No index edges/);

    const partly = formatReferences({
        symbol: 'Bytes', bounded: null, limit: 30, limitClampedFrom: null,
        groups: [
            { relation: 'mentions', totalFiles: 1, atFetchCeiling: false, failed: null, files: [{ relativePath: 'a.hx', totalChunks: 1, spans: [{ chunkId: 'c', startLine: 1, endLine: 2, symbolName: 'A', symbolKind: 'class', contentType: 'code', chunks: 1 }] }] },
            { relation: 'extends', totalFiles: 0, atFetchCeiling: false, files: [], failed: 'timeout' },
        ],
    } as any, undefined);
    assert.match(partly, /INCOMPLETE: the `extends` query failed \(timeout\)/);
});

test("a clamped limit is stated, and the answer stops advising a limit that cannot rise", () => {
    const group = { relation: 'mentions', totalFiles: 150, atFetchCeiling: false, failed: null, files: [{ relativePath: 'a.hx', totalChunks: 1, spans: [{ chunkId: 'c', startLine: 1, endLine: 2, symbolName: 'A', symbolKind: 'class', contentType: 'code', chunks: 1 }] }] };
    const text = formatReferences({ symbol: 'Bytes', bounded: null, limit: 100, limitClampedFrom: 500, groups: [group] } as any, undefined);
    assert.match(text, /`limit` 500 is outside the allowed range; 100 files per relation were used/);
    assert.match(text, /already at the maximum `limit`/);
    assert.doesNotMatch(text, /raise `limit`/);
});

test("a grouped answer names the relation, the file, the range and the Chunk-ID", () => {
    const text = formatReferences({
        symbol: 'Bytes',
        bounded: null, limit: 30, limitClampedFrom: null,
        groups: [
            {
                relation: 'mentions',
                totalFiles: 3,
                atFetchCeiling: false,
                failed: null,
                files: [{ relativePath: 'haxe/BytesBuffer.hx', totalChunks: 3, spans: [{ chunkId: 'c1', startLine: 12, endLine: 48, symbolName: 'BytesBuffer', symbolKind: 'class', contentType: 'code', chunks: 3 }] }],
            },
            { relation: 'extends', totalFiles: 0, atFetchCeiling: false, files: [], failed: null },
        ],
    } as any, ['haxe']);
    assert.match(text, /scoped to haxe/);
    assert.match(text, /## mentions — 3 files, showing 1/);
    assert.match(text, /haxe\/BytesBuffer\.hx/);
    assert.match(text, /L12-48 BytesBuffer \(class\) \[code\] 3 chunks Chunk-ID: c1/);
    // a relation with nothing in it is not a heading with an empty body
    assert.doesNotMatch(text, /## extends/);
});

test("nothing found says so, and says it did not search", () => {
    const text = formatReferences({ symbol: 'Nope', bounded: null, limit: 30, limitClampedFrom: null, groups: [{ relation: 'mentions', totalFiles: 0, atFetchCeiling: false, files: [], failed: null }] } as any, undefined);
    assert.match(text, /No index edges for `Nope`/);
    assert.match(text, /does not search/);
});

test("the description says what the tool is not for, and states the measured ceiling", () => {
    assert.match(FIND_REFERENCES_TOOL_SPEC.description, /not a search/i);
    assert.match(FIND_REFERENCES_TOOL_SPEC.description, /0\.95/);
    assert.match(FIND_REFERENCES_TOOL_SPEC.description, /0\.47/);
    // the attribution corrected on 2026-09-20 must not come back as an
    // explanation the agent reads as fact
    assert.doesNotMatch(FIND_REFERENCES_TOOL_SPEC.description, /extractor/i);
});
