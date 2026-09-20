// find-references-as-a-tool: the tool's switch, its declaration and the text it
// renders. Kept out of index.ts and handlers.ts so all three are unit-testable
// without a Milvus connection (same shape as prose-graph-neighbours.ts).

import type { FindReferencesAnswer, ReferenceRelation } from "@zilliz/claude-context-core";

/**
 * D6: off by default. The agent-visible tool surface does not change until the
 * owner flips this in `.mcp.json` in chat, and rollback is the switch off.
 */
export function findReferencesEnabled(env: Record<string, string | undefined> = process.env): boolean {
    const raw = (env.FIND_REFERENCES_TOOL || '').trim().toLowerCase();
    return raw === 'true' || raw === '1' || raw === 'on';
}

/** The tool-list fragment: present only behind the switch. */
export function findReferencesTools(env?: Record<string, string | undefined>): unknown[] {
    return findReferencesEnabled(env) ? [FIND_REFERENCES_TOOL_SPEC] : [];
}

export const RELATION_LABELS: Record<ReferenceRelation, string> = {
    mentions: 'mentions',
    extends: 'extends',
    implements: 'implements',
};

// The description states what was MEASURED on the served index, not why. The
// causes behind a missing edge were separated on 2026-09-20 (vocabulary gate 57
// of 281 labelled pairs, extractor 2) and belong in the run record, not in a
// tool description an agent reads as fact.
export const FIND_REFERENCES_DESCRIPTION = `Ask the index directly which code rows reference a symbol, instead of ranking for them. Given a symbol NAME (not a question), returns the rows whose \`mentioned_symbols\` include it, whose \`extends\` is it, or whose \`implements\` include it — grouped by relation, one entry per file, each with line ranges and a \`Chunk-ID\` you can pass to expand_context for the body.

Use it for "what depends on / uses / extends / implements X" once you know the exact symbol name — search_code ranks passages by how much they LOOK like an answer, and a file that merely uses a symbol does not, which is why those questions retrieve badly.

What it is NOT: not a search. It does no embedding, no ranking and no scoring, it does not accept natural language, and it returns nothing for a name the index does not carry — use search_code to find the name first.

Measured on the served index (2026-09-20, \`infra/cross-corpus-runs/find-references-as-a-tool-2026-09-20\`): for a symbol declared as a TYPE (class / interface / typedef) 0.95 of its known dependents are present in these fields; for a FUNCTION or METHOD, 0.47 — so "what calls this function" is answered only partly on this build, and on JavaScript and C++ often not at all. Type-level questions are what this tool is good at.

A symbol the corpus uses everywhere gets the document count and a suggestion to narrow, not a truncated list: picking which rows to show would be the ranking decision this tool exists to avoid.`;

export const FIND_REFERENCES_TOOL_SPEC = {
    name: "find_references",
    description: FIND_REFERENCES_DESCRIPTION,
    inputSchema: {
        type: "object",
        properties: {
            path: {
                type: "string",
                description: `Optional. ABSOLUTE path to the indexed codebase. Omit it for the local knowledge base (resolved automatically); only pass it for a different indexed codebase.`
            },
            symbol: {
                type: "string",
                description: "The exact symbol name to look up (e.g. 'ResourceFormatSaver', 'BytesBuffer'). Not a question, not a qualified path — the bare declared name, matched exactly."
            },
            relations: {
                type: "array",
                items: { type: "string", enum: ["mentions", "extends", "implements"] },
                description: "Optional: restrict to these relations. 'mentions' = the row names the symbol; 'extends' / 'implements' = the row declares a type derived from it. Omit for all three."
            },
            categories: {
                type: "array",
                items: { type: "string" },
                description: "Optional: restrict to knowledge-base categories (the names list_categories returns, e.g. ['godot']). Omit to look across every category."
            },
            limit: {
                type: "integer",
                description: "Optional: maximum number of FILES per relation (default 30, max 100).",
                default: 30,
                maximum: 100
            }
        },
        required: ["symbol"]
    }
};

function plural(n: number, one: string, many: string): string {
    return `${n} ${n === 1 ? one : many}`;
}

/**
 * D3: the bound is the answer. No rows, the count, and how to narrow — never a
 * silent top-N.
 */
export function formatReferences(answer: FindReferencesAnswer, scope: string[] | undefined): string {
    const scopeLine = scope && scope.length > 0 ? ` (scoped to ${scope.join(', ')})` : '';

    if (answer.bounded) {
        const { identifier, documentFrequency, boundDocuments, boundQuantile } = answer.bounded;
        // The count belongs to the IDENTIFIER the bound prices — the leaf of a
        // name path — so the message names it rather than implying the count is
        // the whole string's.
        const named = identifier === answer.symbol ? `\`${identifier}\`` : `\`${identifier}\` (the leaf of \`${answer.symbol}\`)`;
        return `${named} is named in ${plural(documentFrequency, 'document', 'documents')}, at or above this corpus's bound of ${boundDocuments} (quantile ${boundQuantile ?? 'n/a'}).\n`
            + `That is too common to be a useful reference question, so no rows are returned: choosing which of those ${documentFrequency} to show would be a ranking decision, and this tool makes none.\n`
            // NOT "narrow by category": the bound is a corpus-wide document
            // frequency and is evaluated before any scope, so `categories`
            // cannot move it. Telling the agent otherwise is a loop.
            + `The bound is corpus-wide, so a \`categories\` scope will not lift it. Ask about a more specific symbol instead — the type that owns this member, or the subclass you actually mean.`;
    }

    const failed = answer.groups.filter((g) => g.failed);
    if (failed.length === answer.groups.length && failed.length > 0) {
        return `find_references could not answer for \`${answer.symbol}\`${scopeLine}: every index query failed.\n`
            + failed.map((g) => `  ${RELATION_LABELS[g.relation]}: ${g.failed}`).join('\n')
            + `\nThis is NOT "there are no such edges" — the index was not successfully asked.`;
    }

    const totals = answer.groups.map((g) => `${RELATION_LABELS[g.relation]} ${g.totalFiles}`).join(', ');
    const found = answer.groups.reduce((n, g) => n + g.totalFiles, 0);

    if (found === 0 && failed.length === 0) {
        return `No index edges for \`${answer.symbol}\`${scopeLine}.\n`
            + `Either the name is not the one the code declares (check the spelling with search_code), or this build's index does not carry the edge. `
            + `This tool reads the index's own fields only — it does not search.`;
    }

    let text = `find_references \`${answer.symbol}\`${scopeLine} — ${totals} (files)\n`;
    text += `Index edges only: no ranking, no score. Pass a Chunk-ID to expand_context for the body.\n`;
    if (answer.limitClampedFrom !== null) {
        text += `Note: \`limit\` ${answer.limitClampedFrom} is outside the allowed range; ${answer.limit} files per relation were used.\n`;
    }
    for (const g of failed) {
        text += `INCOMPLETE: the \`${RELATION_LABELS[g.relation]}\` query failed (${g.failed}) — that relation is unknown here, not empty.\n`;
    }

    for (const group of answer.groups) {
        if (group.files.length === 0) continue;
        const shown = group.files.length;
        const more = group.totalFiles > shown
            ? `, showing ${shown} — ${answer.limit >= 100 ? 'already at the maximum `limit`; narrow with `categories`' : 'raise `limit` or narrow with `categories`'}`
            : '';
        const floor = group.atFetchCeiling ? ' (at the scan ceiling; the count is a floor)' : '';
        text += `\n## ${RELATION_LABELS[group.relation]} — ${plural(group.totalFiles, 'file', 'files')}${more}${floor}\n`;
        for (const file of group.files) {
            text += `${file.relativePath}\n`;
            for (const span of file.spans) {
                const sym = span.symbolName ? ` ${span.symbolName}` : '';
                const kind = span.symbolKind ? ` (${span.symbolKind})` : '';
                // The Chunk-ID is the span's FIRST chunk: that is what
                // expand_context takes, and a span is one edge however many
                // chunks the splitter cut it into.
                const merged = span.chunks > 1 ? ` ${span.chunks} chunks` : '';
                text += `  L${span.startLine}-${span.endLine}${sym}${kind} [${span.contentType || 'code'}]${merged} Chunk-ID: ${span.chunkId}\n`;
            }
        }
    }
    return text.trimEnd();
}
