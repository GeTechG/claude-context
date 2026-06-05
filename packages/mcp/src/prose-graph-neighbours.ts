// prose-graph-mcp-tool: pure neighbour-selection + formatting helpers for the
// agent-callable `expand_context` MCP tool. Kept free of any Milvus / Context
// dependency so the selection and rendering logic is unit-testable in
// isolation (node:test). The only I/O — loading the ProseGraphIndex and
// fetching neighbour chunk content — lives in the handler.

import type { ProseGraphEdge, ProseEdgeType, SemanticSearchResult } from "@zilliz/claude-context-core";
import { truncateContent } from "./utils.js";

/** All edge relationship types the deterministic prose-graph emits. */
export const PROSE_EDGE_TYPES: ProseEdgeType[] = [
    "heading",
    "code_example",
    "co_mention",
    "sequence",
    "link",
];

export const DEFAULT_EXPAND_LIMIT = 10;
export const MAX_EXPAND_LIMIT = 50;

/** Per-neighbour content budget in the formatted response. */
const NEIGHBOUR_CONTENT_TRUNCATE = 1200;

/**
 * Normalize the optional `limit` argument: a positive integer capped at
 * MAX_EXPAND_LIMIT, falling back to DEFAULT_EXPAND_LIMIT for absent / invalid
 * / non-positive input. Never throws on bad input.
 */
export function clampLimit(limit: unknown): number {
    const n = typeof limit === "number" && Number.isFinite(limit) ? Math.floor(limit) : NaN;
    if (!Number.isFinite(n) || n <= 0) return DEFAULT_EXPAND_LIMIT;
    return Math.min(n, MAX_EXPAND_LIMIT);
}

/**
 * Normalize the optional `edge_types[]` filter to the set of recognized edge
 * types. Returns null (no filtering) when the input is absent/empty or carries
 * no recognized value.
 */
export function normalizeEdgeTypes(edgeTypes: unknown): Set<string> | null {
    if (!Array.isArray(edgeTypes)) return null;
    const allowed = new Set<string>(PROSE_EDGE_TYPES as string[]);
    const picked = edgeTypes.filter((t): t is string => typeof t === "string" && allowed.has(t));
    return picked.length > 0 ? new Set(picked) : null;
}

/**
 * Pure neighbour selection for a seed chunk's 1-hop edges:
 *   1. filter to the requested `edge_types` (when provided),
 *   2. dedup by neighbour chunk_id, keeping the strongest edge (tie-break on
 *      edge type lexicographically) so a neighbour reachable by several edge
 *      types contributes one entry,
 *   3. order by edge weight descending with a deterministic tie-break on the
 *      neighbour chunk_id,
 *   4. cap at `limit`.
 */
export function selectNeighbours(
    edges: ProseGraphEdge[],
    edgeTypes: Set<string> | null,
    limit: number,
): ProseGraphEdge[] {
    if (!Array.isArray(edges) || edges.length === 0) return [];

    const filtered = edgeTypes ? edges.filter((e) => e && edgeTypes.has(e.type)) : edges;

    // Dedup by neighbour id, keeping the strongest edge.
    const bestByTo = new Map<string, ProseGraphEdge>();
    for (const e of filtered) {
        if (!e || !e.to) continue;
        const cur = bestByTo.get(e.to);
        const w = e.weight || 0;
        if (!cur) {
            bestByTo.set(e.to, e);
            continue;
        }
        const cw = cur.weight || 0;
        if (w > cw || (w === cw && e.type < cur.type)) {
            bestByTo.set(e.to, e);
        }
    }

    const ordered = Array.from(bestByTo.values()).sort(
        (a, b) =>
            (b.weight || 0) - (a.weight || 0) ||
            (a.to < b.to ? -1 : a.to > b.to ? 1 : 0),
    );

    return ordered.slice(0, limit);
}

/**
 * Render the `expand_context` response for a seed chunk: a header line (seed
 * id + neighbour counts by relationship type), then one block per neighbour
 * that has fetchable content. Neighbours whose chunk_id is missing from the
 * current prose collection (stale after a reindex) are skipped and counted in
 * a trailing note.
 */
export function formatExpansion(
    seedId: string,
    selected: ProseGraphEdge[],
    contentById: Map<string, SemanticSearchResult>,
    totalNeighbours: number,
): string {
    const byType = new Map<string, number>();
    for (const e of selected) byType.set(e.type, (byType.get(e.type) || 0) + 1);
    const typeSummary = PROSE_EDGE_TYPES.filter((t) => byType.has(t))
        .map((t) => `${t}=${byType.get(t)}`)
        .join(", ");

    const header =
        `Prose-graph neighbours of chunk \`${seedId}\` — ` +
        `returning ${selected.length} of ${totalNeighbours} 1-hop neighbour(s)` +
        `${typeSummary ? ` (${typeSummary})` : ""}.`;

    const blocks: string[] = [];
    let rendered = 0;
    let unfetched = 0;
    for (const edge of selected) {
        const r = contentById.get(edge.to);
        if (!r) {
            unfetched++;
            continue;
        }
        rendered++;
        const headingPath =
            Array.isArray(r.heading_path) && r.heading_path.length > 0
                ? r.heading_path.join(" › ")
                : "(none)";
        const language = r.language && r.language !== "unknown" ? r.language : "text";
        const content = truncateContent(r.content || "", NEIGHBOUR_CONTENT_TRUNCATE);
        const weight = Number.isFinite(edge.weight) ? edge.weight.toFixed(3) : "1.000";
        blocks.push(
            `${rendered}. [${edge.type}] weight=${weight}\n` +
                `   Chunk-ID: ${edge.to}\n` +
                `   Location: ${r.relativePath}:${r.startLine}-${r.endLine}\n` +
                `   Heading: ${headingPath}\n` +
                `   Content-Type: ${r.content_type ?? "unknown"}\n` +
                `   Content:\n\`\`\`${language}\n${content}\n\`\`\``,
        );
    }

    let out = header;
    if (blocks.length > 0) out += `\n\n${blocks.join("\n\n")}`;
    if (unfetched > 0) {
        out +=
            `\n\n(${unfetched} neighbour(s) could not be fetched — likely stale after a ` +
            `reindex; rebuild the side-index with infra/build-prose-graph.js)`;
    }
    return out;
}
