// prose-graph-deterministic §4: tests for ProseGraphIndex + expansion.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
    ProseGraphIndex,
    collectProseGraphCandidateIds,
    expandProseGraphPool,
} from './prose-graph-expansion';
import { buildProseGraph, ProseChunkRecord } from './prose-graph-builder';
import { SemanticSearchResult } from '../types';

function tmpFile(content: string): string {
    const file = path.join(os.tmpdir(), `prose-graph-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);
    fs.writeFileSync(file, content, 'utf-8');
    return file;
}

function seed(chunk_id: string): SemanticSearchResult {
    return { content: '', relativePath: '', startLine: 0, endLine: 0, language: 'markdown', score: 1, chunk_id };
}

function rec(over: Partial<ProseChunkRecord> & { chunk_id: string }): ProseChunkRecord {
    return { relativePath: 'a.md', content_type: 'doc', heading_path: [], startLine: 0, endLine: 0, mentioned_symbols: [], ...over };
}

describe('ProseGraphIndex.load', () => {
    it('loads a valid prose-v1 payload', () => {
        const payload = buildProseGraph([
            rec({ chunk_id: 'h1', heading_path: ['A'], startLine: 1, endLine: 2 }),
            rec({ chunk_id: 'h2', heading_path: ['A', 'B'], startLine: 3, endLine: 4 }),
        ], { generatedAt: 'X' });
        const file = tmpFile(JSON.stringify(payload));
        const idx = ProseGraphIndex.load(file);
        expect(idx).not.toBeNull();
        expect(idx!.version).toBe('prose-v1');
        expect(idx!.nodeCount).toBe(2);
        // h1↔h2 are connected by both a heading and a sequence edge.
        expect(Array.from(new Set(idx!.neighbours('h1').map((e) => e.to)))).toEqual(['h2']);
        expect(idx!.neighbours('h1').map((e) => e.type).sort()).toEqual(['heading', 'sequence']);
    });

    it('rejects an unrecognized version → null (graceful off)', () => {
        const file = tmpFile(JSON.stringify({ version: 'prose-v99', adjacency: {} }));
        expect(ProseGraphIndex.load(file)).toBeNull();
    });

    it('returns null on malformed JSON', () => {
        const file = tmpFile('{ not json');
        expect(ProseGraphIndex.load(file)).toBeNull();
    });

    it('returns null on a missing file', () => {
        expect(ProseGraphIndex.load('/no/such/prose-graph.json')).toBeNull();
    });
});

describe('collectProseGraphCandidateIds', () => {
    const payload = buildProseGraph([
        rec({ chunk_id: 'a', relativePath: 'f.md', mentioned_symbols: ['X', 'Y'], startLine: 1, endLine: 2 }),
        rec({ chunk_id: 'b', relativePath: 'g.md', mentioned_symbols: ['X', 'Y'], startLine: 1, endLine: 2 }),
        rec({ chunk_id: 'c', relativePath: 'h.md', mentioned_symbols: ['X'], startLine: 1, endLine: 2 }),
    ], { generatedAt: 'X' });
    const idx = ProseGraphIndex.load(tmpFile(JSON.stringify(payload)))!;

    it('collects 1-hop neighbours, dropping the seeds themselves', () => {
        const out = collectProseGraphCandidateIds([seed('a')], idx);
        expect(out).toContain('b');
        expect(out).toContain('c');
        expect(out).not.toContain('a');
    });

    it('orders stronger (higher-weight) neighbours first', () => {
        const out = collectProseGraphCandidateIds([seed('a')], idx);
        // a↔b shares 2 symbols, a↔c shares 1 → b before c.
        expect(out.indexOf('b')).toBeLessThan(out.indexOf('c'));
    });

    it('returns [] for empty seeds', () => {
        expect(collectProseGraphCandidateIds([], idx)).toEqual([]);
    });
});

describe('expandProseGraphPool', () => {
    const payload = buildProseGraph([
        rec({ chunk_id: 'a', relativePath: 'f.md', heading_path: ['A'], startLine: 1, endLine: 2 }),
        rec({ chunk_id: 'b', relativePath: 'f.md', heading_path: ['A', 'B'], startLine: 3, endLine: 4 }),
    ], { generatedAt: 'X' });
    const idx = ProseGraphIndex.load(tmpFile(JSON.stringify(payload)))!;

    it('fetches neighbour chunks', async () => {
        const fetcher = async (id: string) => ({ ...seed(id) });
        const out = await expandProseGraphPool([seed('a')], idx, fetcher);
        expect(out.map((r) => r.chunk_id)).toEqual(['b']);
    });

    it('skips chunks the fetcher cannot resolve (stale id)', async () => {
        const fetcher = async (_id: string) => null;
        const out = await expandProseGraphPool([seed('a')], idx, fetcher);
        expect(out).toEqual([]);
    });

    it('survives a throwing fetcher without error', async () => {
        const fetcher = async (_id: string) => { throw new Error('missing'); };
        await expect(expandProseGraphPool([seed('a')], idx, fetcher)).resolves.toEqual([]);
    });
});
