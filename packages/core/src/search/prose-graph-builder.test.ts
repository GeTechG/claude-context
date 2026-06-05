// prose-graph-deterministic §2–3: tests for the deterministic prose-graph
// builder — five edge types over existing prose-collection columns.

import {
    buildProseGraph,
    ProseChunkRecord,
    slugifyHeading,
    extractMarkdownLinkTargets,
    resolveLinkTarget,
    PROSE_GRAPH_VERSION,
} from './prose-graph-builder';

function rec(over: Partial<ProseChunkRecord> & { chunk_id: string }): ProseChunkRecord {
    return {
        relativePath: 'a.md',
        content_type: 'doc',
        heading_path: [],
        startLine: 0,
        endLine: 0,
        mentioned_symbols: [],
        ...over,
    };
}

/** Collect neighbour ids of `id` filtered by edge type. */
function neighbours(payload: ReturnType<typeof buildProseGraph>, id: string, type?: string): string[] {
    return (payload.adjacency[id] || [])
        .filter((e) => !type || e.type === type)
        .map((e) => e.to)
        .sort();
}

describe('buildProseGraph — heading hierarchy', () => {
    it('links parent↔child by one-level heading_path prefix within a file', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'h1', heading_path: ['Macros'], startLine: 1, endLine: 5 }),
            rec({ chunk_id: 'h2', heading_path: ['Macros', 'Build'], startLine: 6, endLine: 10 }),
            rec({ chunk_id: 'h3', heading_path: ['Macros', 'Build', 'Lifecycle'], startLine: 11, endLine: 15 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'h1', 'heading')).toEqual(['h2']);
        expect(neighbours(g, 'h2', 'heading').sort()).toEqual(['h1', 'h3']);
        expect(neighbours(g, 'h3', 'heading')).toEqual(['h2']);
    });

    it('does not link two-level jumps or different files', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'a', relativePath: 'x.md', heading_path: ['A'], startLine: 1, endLine: 2 }),
            rec({ chunk_id: 'b', relativePath: 'x.md', heading_path: ['A', 'B', 'C'], startLine: 3, endLine: 4 }),
            rec({ chunk_id: 'c', relativePath: 'y.md', heading_path: ['A', 'B'], startLine: 1, endLine: 2 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'a', 'heading')).toEqual([]); // two-level jump
        expect(neighbours(g, 'c', 'heading')).toEqual([]); // different file
    });
});

describe('buildProseGraph — code_example↔doc adjacency', () => {
    it('links a code_example to the surrounding doc by shared heading + adjacent lines', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'doc', content_type: 'doc', heading_path: ['Sys', 'exec'], startLine: 1, endLine: 8 }),
            rec({ chunk_id: 'ex', content_type: 'code_example', heading_path: ['Sys', 'exec'], startLine: 9, endLine: 14 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'doc', 'code_example')).toEqual(['ex']);
        expect(neighbours(g, 'ex', 'code_example')).toEqual(['doc']);
    });

    it('does not link when line ranges are far apart', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'doc', content_type: 'doc', heading_path: ['Sys'], startLine: 1, endLine: 8 }),
            rec({ chunk_id: 'ex', content_type: 'code_example', heading_path: ['Sys'], startLine: 80, endLine: 90 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'doc', 'code_example')).toEqual([]);
    });

    it('does not link two docs or two code_examples', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'd1', content_type: 'doc', heading_path: ['S'], startLine: 1, endLine: 4 }),
            rec({ chunk_id: 'd2', content_type: 'doc', heading_path: ['S'], startLine: 5, endLine: 8 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'd1', 'code_example')).toEqual([]);
    });
});

describe('buildProseGraph — co-mention', () => {
    it('links chunks sharing ≥1 mentioned_symbol with weight ∝ shared count', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'a', relativePath: 'a.md', mentioned_symbols: ['Bytes', 'Sys'] }),
            rec({ chunk_id: 'b', relativePath: 'b.md', mentioned_symbols: ['Bytes', 'Sys', 'Path'] }),
            rec({ chunk_id: 'c', relativePath: 'c.md', mentioned_symbols: ['Path'] }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        const ab = (g.adjacency['a'] || []).find((e) => e.to === 'b' && e.type === 'co_mention');
        expect(ab).toBeDefined();
        expect(ab!.weight).toBe(2); // Bytes + Sys
        const bc = (g.adjacency['b'] || []).find((e) => e.to === 'c' && e.type === 'co_mention');
        expect(bc!.weight).toBe(1); // Path
    });

    it('skips over-generic symbols above maxCoMentionBucket', () => {
        const records: ProseChunkRecord[] = [];
        for (let i = 0; i < 5; i++) {
            records.push(rec({ chunk_id: `n${i}`, relativePath: `${i}.md`, mentioned_symbols: ['Common'] }));
        }
        const g = buildProseGraph(records, { generatedAt: 'X', maxCoMentionBucket: 3 });
        expect(neighbours(g, 'n0', 'co_mention')).toEqual([]);
    });
});

describe('buildProseGraph — intra-file sequence', () => {
    it('links consecutive-by-startLine chunks of the same file', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'c', relativePath: 'f.md', startLine: 20, endLine: 25 }),
            rec({ chunk_id: 'a', relativePath: 'f.md', startLine: 1, endLine: 5 }),
            rec({ chunk_id: 'b', relativePath: 'f.md', startLine: 6, endLine: 10 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'a', 'sequence')).toEqual(['b']);
        expect(neighbours(g, 'b', 'sequence').sort()).toEqual(['a', 'c']);
    });
});

describe('extractMarkdownLinkTargets', () => {
    it('extracts inline and reference-style targets', () => {
        const md = [
            'See [the macro guide](macros/build.md#lifecycle) for details.',
            'Also [this][ref] and [shorthand][].',
            '',
            '[ref]: ../other/page.md',
            '[shorthand]: shorthand.md#top',
        ].join('\n');
        const targets = extractMarkdownLinkTargets(md);
        expect(targets).toContain('macros/build.md#lifecycle');
        expect(targets).toContain('../other/page.md');
        expect(targets).toContain('shorthand.md#top');
    });

    it('ignores undefined references', () => {
        expect(extractMarkdownLinkTargets('[x][missing]')).toEqual([]);
    });
});

describe('resolveLinkTarget', () => {
    const byRelPath = new Map<string, ProseChunkRecord[]>([
        ['macros/build.md', [
            rec({ chunk_id: 'b_top', relativePath: 'macros/build.md', heading_path: ['Build'], startLine: 1 }),
            rec({ chunk_id: 'b_life', relativePath: 'macros/build.md', heading_path: ['Build', 'Lifecycle'], startLine: 20 }),
        ]],
    ]);

    it('resolves a relative path + anchor to the matching heading chunk', () => {
        expect(resolveLinkTarget('build.md#lifecycle', 'macros/intro.md', byRelPath)).toBe('b_life');
    });

    it('resolves a path with no anchor to the lowest-startLine chunk', () => {
        expect(resolveLinkTarget('macros/build.md', 'index.md', byRelPath)).toBe('b_top');
    });

    it('resolves `..` segments', () => {
        expect(resolveLinkTarget('../macros/build.md', 'guide/page.md', byRelPath)).toBe('b_top');
    });

    it('returns null for unresolvable and external links', () => {
        expect(resolveLinkTarget('missing.md', 'a.md', byRelPath)).toBeNull();
        expect(resolveLinkTarget('https://example.com', 'a.md', byRelPath)).toBeNull();
    });
});

describe('buildProseGraph — markdown link edges', () => {
    it('writes a link edge to the resolved target chunk', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'src', relativePath: 'guide/intro.md', heading_path: ['Intro'], startLine: 1, endLine: 3,
                content: 'Read [the API](../api/bytes.md#alloc).' }),
            rec({ chunk_id: 'tgt', relativePath: 'api/bytes.md', heading_path: ['Bytes', 'alloc'], startLine: 10, endLine: 14 }),
            rec({ chunk_id: 'tgt_top', relativePath: 'api/bytes.md', heading_path: ['Bytes'], startLine: 1, endLine: 9 }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(neighbours(g, 'src', 'link')).toEqual(['tgt']);
        expect(neighbours(g, 'tgt', 'link')).toEqual(['src']); // stored both ends
    });

    it('skips unresolvable links without error', () => {
        const records: ProseChunkRecord[] = [
            rec({ chunk_id: 'src', relativePath: 'a.md', content: 'Broken [link](nope.md).' }),
        ];
        const g = buildProseGraph(records, { generatedAt: 'X' });
        expect(g.adjacency['src']).toBeUndefined();
    });
});

describe('buildProseGraph — determinism & shape', () => {
    it('is order-independent apart from generatedAt', () => {
        const a: ProseChunkRecord[] = [
            rec({ chunk_id: 'x', relativePath: 'f.md', heading_path: ['A'], startLine: 1, endLine: 2 }),
            rec({ chunk_id: 'y', relativePath: 'f.md', heading_path: ['A', 'B'], startLine: 3, endLine: 4 }),
        ];
        const g1 = buildProseGraph(a, { generatedAt: 'X' });
        const g2 = buildProseGraph(a.slice().reverse(), { generatedAt: 'X' });
        expect(JSON.stringify(g1)).toBe(JSON.stringify(g2));
        expect(g1.version).toBe(PROSE_GRAPH_VERSION);
        expect(g1.stats.nodes).toBe(2);
        expect(g1.stats.heading).toBe(1);
    });
});

describe('slugifyHeading', () => {
    it('produces github-style anchors', () => {
        expect(slugifyHeading('Build Lifecycle')).toBe('build-lifecycle');
        expect(slugifyHeading('Sys.exec()')).toBe('sysexec');
    });
});
