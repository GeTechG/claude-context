// prose-graph-deterministic §2–3: deterministic builder for the prose-graph
// side-index (`knowledge/.prose-graph.json`).
//
// Unlike `.symbols-graph.json` (AST/symbol edges over the code collection),
// this builds *narrative* edges over the prose collection
// (`hybrid_v6_prose_<hash>`, content_type IN ('doc','code_example')) from the
// columns Milvus already stores — NO re-ingestion, NO LLM-extraction. Five
// edge types (design D1):
//
//   heading       — parent↔child by one-level heading_path prefix (same file)
//   code_example  — code_example↔doc adjacency (same file + shared/prefix
//                   heading_path + adjacent/overlapping [startLine,endLine])
//   co_mention    — chunks sharing ≥1 mentioned_symbols (weight ∝ # shared)
//   sequence      — chunks adjacent by startLine within a file
//   link          — internal markdown links resolved by a regex pass over
//                   `content` (inline + reference-style), resolving to a
//                   target relativePath (+ heading anchor)
//
// The first four are pure reads of `heading_path` / `content_type` /
// `relativePath` / `startLine` / `endLine` / `mentioned_symbols`; the fifth is
// a deterministic regex over `content`. The whole build is pure and
// deterministic given the chunk records (offline; not on the hot path).

import * as path from 'path';

export type ProseEdgeType = 'heading' | 'code_example' | 'co_mention' | 'sequence' | 'link';

export const PROSE_GRAPH_VERSION = 'prose-v1';

/** A single prose-collection chunk record, read from Milvus output_fields. */
export interface ProseChunkRecord {
    chunk_id: string;
    relativePath: string;
    content_type?: string;
    heading_path?: string[];
    startLine: number;
    endLine: number;
    mentioned_symbols?: string[];
    content?: string;
}

export interface ProseGraphEdge {
    to: string;
    type: ProseEdgeType;
    weight: number;
}

export interface ProseGraphStats {
    nodes: number;
    edges: number;
    heading: number;
    code_example: number;
    co_mention: number;
    sequence: number;
    link: number;
}

export interface ProseGraphPayload {
    version: string;
    generatedAt: string;
    stats: ProseGraphStats;
    /** chunk_id → 1-hop neighbours (every edge stored on both endpoints). */
    adjacency: Record<string, ProseGraphEdge[]>;
}

export interface BuildProseGraphOptions {
    /** Max line gap for code_example↔doc / sequence adjacency. Default 2. */
    adjacencyGap?: number;
    /** Skip co-mention symbols appearing in more than this many chunks
     * (too-generic terms create dense, low-signal cliques). Default 50. */
    maxCoMentionBucket?: number;
    /** Cap co-mention neighbours kept per node (highest weight first). Default 30. */
    maxCoMentionPerNode?: number;
    /** ISO timestamp stamped into the payload. Defaults to now() at call time.
     * Pass explicitly to keep builds byte-reproducible in tests. */
    generatedAt?: string;
}

const DEFAULTS = {
    adjacencyGap: 2,
    maxCoMentionBucket: 50,
    maxCoMentionPerNode: 30,
};

// --- helpers ---------------------------------------------------------------

function normalizeRel(p: string | undefined): string {
    if (!p) return '';
    return p.replace(/\\/g, '/');
}

/** True when `a` is a strict prefix of `b` (array prefix), i.e. b extends a. */
function isHeadingPrefix(a: string[], b: string[]): boolean {
    if (a.length >= b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/** Equal-or-prefix in either direction (for code_example↔doc heading match). */
function headingsRelated(a: string[], b: string[]): boolean {
    const min = Math.min(a.length, b.length);
    for (let i = 0; i < min; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true; // one is a prefix of (or equal to) the other
}

/** Two line ranges are adjacent when they overlap or the gap is ≤ `gap`. */
function rangesAdjacent(
    aStart: number,
    aEnd: number,
    bStart: number,
    bEnd: number,
    gap: number,
): boolean {
    // overlap
    if (aStart <= bEnd && bStart <= aEnd) return true;
    // gap below/above
    if (bStart > aEnd && bStart - aEnd <= gap + 1) return true;
    if (aStart > bEnd && aStart - bEnd <= gap + 1) return true;
    return false;
}

/** GitHub-style heading-anchor slug. */
export function slugifyHeading(heading: string): string {
    return (heading || '')
        .toLowerCase()
        .trim()
        .replace(/[^\w\s-]/g, '')   // drop punctuation
        .replace(/\s+/g, '-')        // spaces → hyphens
        .replace(/-+/g, '-')         // collapse repeats
        .replace(/^-+|-+$/g, '');
}

/**
 * Extract internal markdown link targets from a chunk's `content`.
 * Returns raw link targets (`path`, `path#anchor`, or `#anchor`) for both
 * inline `[text](target)` and reference-style (`[text][ref]` + `[ref]: target`).
 * Pure regex — no LLM, no re-crawl.
 */
export function extractMarkdownLinkTargets(content: string | undefined): string[] {
    if (!content) return [];
    const targets: string[] = [];

    // Inline: [text](target)  — target up to whitespace or closing paren.
    const inlineRe = /\[(?:[^\]]*)\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)/g;
    let m: RegExpExecArray | null;
    while ((m = inlineRe.exec(content)) !== null) {
        if (m[1]) targets.push(m[1]);
    }

    // Reference definitions: [ref]: target
    const refDefRe = /^\s*\[([^\]]+)\]:\s*(\S+)/gm;
    const refMap = new Map<string, string>();
    while ((m = refDefRe.exec(content)) !== null) {
        refMap.set(m[1].toLowerCase(), m[2]);
    }
    // Reference uses: [text][ref] (and shorthand [ref][]). Only emit if defined.
    if (refMap.size > 0) {
        const refUseRe = /\[([^\]]+)\]\[([^\]]*)\]/g;
        while ((m = refUseRe.exec(content)) !== null) {
            const ref = (m[2] || m[1]).toLowerCase();
            const target = refMap.get(ref);
            if (target) targets.push(target);
        }
    }

    return targets;
}

/**
 * Resolve a raw markdown link target to a chunk_id in the index.
 *
 * - Splits off the `#anchor`.
 * - Resolves the path part relative to the source chunk's directory
 *   (posix semantics; leading `/` is treated as corpus-root relative).
 * - Picks the target chunk: when an anchor is present, the chunk of that
 *   relativePath whose last heading slugifies to the anchor; otherwise the
 *   lowest-startLine chunk of that relativePath.
 * - Pure-anchor links (`#anchor`, no path) resolve within the source file.
 * Returns null when nothing resolves (caller skips silently).
 */
export function resolveLinkTarget(
    rawTarget: string,
    sourceRelPath: string,
    byRelPath: Map<string, ProseChunkRecord[]>,
): string | null {
    if (!rawTarget) return null;
    // External links / mailto / etc. never resolve to a local chunk.
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(rawTarget) || rawTarget.startsWith('mailto:')) {
        return null;
    }
    const hashIdx = rawTarget.indexOf('#');
    let pathPart = hashIdx >= 0 ? rawTarget.slice(0, hashIdx) : rawTarget;
    const anchor = hashIdx >= 0 ? rawTarget.slice(hashIdx + 1) : '';

    let targetRel: string;
    if (pathPart === '') {
        // Pure-anchor link → same file.
        targetRel = normalizeRel(sourceRelPath);
    } else {
        pathPart = pathPart.replace(/\\/g, '/');
        if (pathPart.startsWith('/')) {
            // Corpus-root relative.
            targetRel = path.posix.normalize(pathPart.replace(/^\/+/, ''));
        } else {
            const srcDir = path.posix.dirname(normalizeRel(sourceRelPath));
            targetRel = path.posix.normalize(path.posix.join(srcDir, pathPart));
        }
    }

    const candidates = byRelPath.get(targetRel);
    if (!candidates || candidates.length === 0) return null;

    if (anchor) {
        const wantSlug = anchor.toLowerCase();
        let best: ProseChunkRecord | null = null;
        for (const c of candidates) {
            const hp = c.heading_path;
            if (hp && hp.length > 0 && slugifyHeading(hp[hp.length - 1]) === wantSlug) {
                if (!best || c.startLine < best.startLine) best = c;
            }
        }
        if (best) return best.chunk_id;
        // Anchor didn't match any heading → fall through to file-level resolve.
    }

    // Lowest-startLine chunk of the target file.
    let head: ProseChunkRecord | null = null;
    for (const c of candidates) {
        if (!head || c.startLine < head.startLine) head = c;
    }
    return head ? head.chunk_id : null;
}

// --- builder ---------------------------------------------------------------

/**
 * Build the prose-graph payload from the prose-collection chunk records.
 * Deterministic: same input records (in any order) produce the same payload
 * apart from `generatedAt`. Edges are stored on both endpoints.
 */
export function buildProseGraph(
    records: ProseChunkRecord[],
    options: BuildProseGraphOptions = {},
): ProseGraphPayload {
    const adjacencyGap = options.adjacencyGap ?? DEFAULTS.adjacencyGap;
    const maxCoMentionBucket = options.maxCoMentionBucket ?? DEFAULTS.maxCoMentionBucket;
    const maxCoMentionPerNode = options.maxCoMentionPerNode ?? DEFAULTS.maxCoMentionPerNode;

    // Keep only well-formed records with a chunk_id; index by id and by file.
    const chunks: ProseChunkRecord[] = [];
    const byId = new Map<string, ProseChunkRecord>();
    const byRelPath = new Map<string, ProseChunkRecord[]>();
    for (const r of records) {
        if (!r || typeof r.chunk_id !== 'string' || r.chunk_id.length === 0) continue;
        if (byId.has(r.chunk_id)) continue;
        const rec: ProseChunkRecord = {
            chunk_id: r.chunk_id,
            relativePath: normalizeRel(r.relativePath),
            content_type: r.content_type,
            heading_path: Array.isArray(r.heading_path) ? r.heading_path : [],
            startLine: Number.isFinite(r.startLine) ? r.startLine : 0,
            endLine: Number.isFinite(r.endLine) ? r.endLine : 0,
            mentioned_symbols: Array.isArray(r.mentioned_symbols) ? r.mentioned_symbols : [],
            content: r.content,
        };
        chunks.push(rec);
        byId.set(rec.chunk_id, rec);
        const bucket = byRelPath.get(rec.relativePath) || [];
        bucket.push(rec);
        byRelPath.set(rec.relativePath, bucket);
    }

    // Edge accumulator: id → (to → (type → weight)). Symmetric storage.
    const adj = new Map<string, Map<string, Map<ProseEdgeType, number>>>();
    const stats: ProseGraphStats = {
        nodes: chunks.length, edges: 0,
        heading: 0, code_example: 0, co_mention: 0, sequence: 0, link: 0,
    };

    const bump = (a: string, b: string, type: ProseEdgeType, weight: number): boolean => {
        if (a === b) return false;
        let an = adj.get(a);
        if (!an) { an = new Map(); adj.set(a, an); }
        let am = an.get(b);
        if (!am) { am = new Map(); an.set(b, am); }
        const isNew = !am.has(type);
        am.set(type, weight); // last write wins; co-mention sets final weight
        return isNew;
    };
    // Add an undirected typed edge, counting it once for stats.
    const addEdge = (a: string, b: string, type: ProseEdgeType, weight = 1): void => {
        if (a === b) return;
        const isNew = bump(a, b, type, weight);
        bump(b, a, type, weight);
        if (isNew) {
            stats[type]++;
            stats.edges++;
        }
    };

    // 1) heading hierarchy + 2) code_example↔doc + 4) sequence — per file.
    for (const [, bucket] of byRelPath) {
        // Stable order by startLine then chunk_id for determinism.
        const file = bucket.slice().sort((x, y) =>
            x.startLine - y.startLine || (x.chunk_id < y.chunk_id ? -1 : 1));

        // heading hierarchy: parent↔child by one-level prefix.
        for (let i = 0; i < file.length; i++) {
            for (let j = 0; j < file.length; j++) {
                if (i === j) continue;
                const a = file[i].heading_path || [];
                const b = file[j].heading_path || [];
                if (b.length === a.length + 1 && isHeadingPrefix(a, b)) {
                    addEdge(file[i].chunk_id, file[j].chunk_id, 'heading', 1);
                }
            }
        }

        // code_example↔doc adjacency.
        for (let i = 0; i < file.length; i++) {
            for (let j = i + 1; j < file.length; j++) {
                const x = file[i], y = file[j];
                const isPair =
                    (x.content_type === 'code_example' && y.content_type === 'doc') ||
                    (x.content_type === 'doc' && y.content_type === 'code_example');
                if (!isPair) continue;
                if (!headingsRelated(x.heading_path || [], y.heading_path || [])) continue;
                if (!rangesAdjacent(x.startLine, x.endLine, y.startLine, y.endLine, adjacencyGap)) continue;
                addEdge(x.chunk_id, y.chunk_id, 'code_example', 1);
            }
        }

        // intra-file sequence: consecutive-by-startLine chunks.
        for (let i = 0; i + 1 < file.length; i++) {
            addEdge(file[i].chunk_id, file[i + 1].chunk_id, 'sequence', 1);
        }
    }

    // 3) co-mention via inverted index symbol → chunk_ids.
    const bySymbol = new Map<string, string[]>();
    for (const c of chunks) {
        const seen = new Set<string>();
        for (const sym of c.mentioned_symbols || []) {
            if (!sym || seen.has(sym)) continue;
            seen.add(sym);
            const b = bySymbol.get(sym) || [];
            b.push(c.chunk_id);
            bySymbol.set(sym, b);
        }
    }
    // Accumulate shared-symbol counts per unordered pair.
    const pairWeight = new Map<string, number>();
    const pairKey = (a: string, b: string) => (a < b ? `${a} ${b}` : `${b} ${a}`);
    for (const [, ids] of bySymbol) {
        if (ids.length < 2 || ids.length > maxCoMentionBucket) continue;
        const sorted = ids.slice().sort();
        for (let i = 0; i < sorted.length; i++) {
            for (let j = i + 1; j < sorted.length; j++) {
                const k = pairKey(sorted[i], sorted[j]);
                pairWeight.set(k, (pairWeight.get(k) || 0) + 1);
            }
        }
    }
    // Per-node top-K by weight (then chunk_id) so dense nodes stay bounded.
    const coNeighbours = new Map<string, { to: string; weight: number }[]>();
    for (const [k, w] of pairWeight) {
        const sep = k.indexOf(' ');
        const a = k.slice(0, sep), b = k.slice(sep + 1);
        (coNeighbours.get(a) || coNeighbours.set(a, []).get(a)!).push({ to: b, weight: w });
        (coNeighbours.get(b) || coNeighbours.set(b, []).get(b)!).push({ to: a, weight: w });
    }
    // Keep only mutually top-K edges to preserve symmetry: collect a kept set.
    const keptCoPairs = new Set<string>();
    for (const [node, list] of coNeighbours) {
        list.sort((p, q) => q.weight - p.weight || (p.to < q.to ? -1 : 1));
        for (const e of list.slice(0, maxCoMentionPerNode)) {
            keptCoPairs.add(pairKey(node, e.to));
        }
    }
    for (const k of keptCoPairs) {
        const sep = k.indexOf(' ');
        const a = k.slice(0, sep), b = k.slice(sep + 1);
        addEdge(a, b, 'co_mention', pairWeight.get(k) || 1);
    }

    // 5) internal markdown links (regex over content).
    for (const c of chunks) {
        const raws = extractMarkdownLinkTargets(c.content);
        for (const raw of raws) {
            const targetId = resolveLinkTarget(raw, c.relativePath, byRelPath);
            if (!targetId || targetId === c.chunk_id) continue;
            addEdge(c.chunk_id, targetId, 'link', 1);
        }
    }

    // Finalize: deterministic adjacency lists (sorted by to, then type).
    const adjacency: Record<string, ProseGraphEdge[]> = {};
    const ids = Array.from(adj.keys()).sort();
    for (const id of ids) {
        const edges: ProseGraphEdge[] = [];
        const neigh = adj.get(id)!;
        for (const [to, typeMap] of neigh) {
            for (const [type, weight] of typeMap) {
                edges.push({ to, type, weight });
            }
        }
        edges.sort((e1, e2) =>
            (e1.to < e2.to ? -1 : e1.to > e2.to ? 1 : 0) ||
            (e1.type < e2.type ? -1 : e1.type > e2.type ? 1 : 0));
        adjacency[id] = edges;
    }

    return {
        version: PROSE_GRAPH_VERSION,
        generatedAt: options.generatedAt ?? new Date().toISOString(),
        stats,
        adjacency,
    };
}
