import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

// pilot-chunk-context-headers (D1): an index-time option that indexes every
// chunk as a short header plus its text. `content`, the chunk id and every
// returned text stay the chunk text; only what the dense, learned-sparse and
// BM25 channels read moves. Every header value comes from the chunk's own
// stored fields or from the context file — nothing corpus-specific.

export type ChunkContextHeaderMode = 'off' | 'metadata' | 'generated';

// Milvus 2.5 caps a VarChar at 65535 bytes, and `index_text` holds header +
// content. A chunk already at the `content` ceiling loses its tail from
// `index_text` only, and the build counts it.
export const INDEX_TEXT_MAX_BYTES = 65535;

// A `generated` build refuses a context file covering fewer chunk ids than this.
export const CHUNK_CONTEXT_MIN_COVERAGE = 0.99;

// First line of a context file (written by infra/generate-chunk-contexts.js).
export const CHUNK_CONTEXT_FILE_KIND = 'chunk-context-file';

const DESCRIPTION_KEY = 'chunk_context_header';

export function resolveChunkContextHeaderMode(raw: string | undefined | null): ChunkContextHeaderMode {
    const value = (raw ?? '').trim().toLowerCase();
    if (value === '' || value === 'off') return 'off';
    if (value === 'metadata' || value === 'generated') return value;
    throw new Error(`CHUNK_CONTEXT_HEADER must be one of off, metadata, generated; got ${JSON.stringify(raw)}`);
}

// The mode is recorded on the collection description, and only when it is not
// `off`, so an `off` collection's description is byte-identical to before.
export function describeWithHeaderMode(description: string, mode: ChunkContextHeaderMode): string {
    return mode === 'off' ? description : `${description};${DESCRIPTION_KEY}:${mode}`;
}

export function headerModeFromDescription(description: string | undefined | null): string {
    const match = /(?:^|;)chunk_context_header:([a-z]+)/.exec(description || '');
    return match ? match[1] : 'off';
}

export interface ChunkHeaderFields {
    relativePath?: string;
    heading_path?: string[];
    symbol_name?: string;
    parent_symbol?: string;
    symbol_kind?: string;
}

/**
 * The header lines, each present only when its field is non-empty:
 *   File: <relativePath>
 *   Section: <heading_path joined " > ">
 *   Symbol: <parent_symbol>.<symbol_name> (<symbol_kind>)
 *   Context: <generated sentence>
 * A symbol line needs a name or a parent; the kind alone names nothing.
 */
export function buildChunkContextHeader(fields: ChunkHeaderFields, context?: string): string {
    const lines: string[] = [];
    if (fields.relativePath) lines.push(`File: ${fields.relativePath}`);
    const section = (fields.heading_path || []).filter((segment) => segment).join(' > ');
    if (section) lines.push(`Section: ${section}`);
    const symbol = [fields.parent_symbol, fields.symbol_name].filter((part) => part).join('.');
    if (symbol) lines.push(`Symbol: ${symbol}${fields.symbol_kind ? ` (${fields.symbol_kind})` : ''}`);
    const sentence = (context || '').replace(/\s+/g, ' ').trim();
    if (sentence) lines.push(`Context: ${sentence}`);
    return lines.join('\n');
}

export function withChunkContextHeader(header: string, content: string): string {
    return header ? `${header}\n\n${content}` : content;
}

export function truncateUtf8(text: string, maxBytes: number): string {
    if (Buffer.byteLength(text, 'utf8') <= maxBytes) return text;
    let out = Buffer.from(text, 'utf8').subarray(0, maxBytes).toString('utf8');
    while (out.length && Buffer.byteLength(out, 'utf8') > maxBytes) out = out.slice(0, -1);
    return out;
}

// The chunk set a context file was generated over: sha256 of the sorted
// distinct chunk ids, one per line. The generator and the build both call this.
export function chunkSetSha256(ids: Iterable<string>): string {
    const sorted = Array.from(new Set(ids)).sort();
    return crypto.createHash('sha256').update(sorted.join('\n'), 'utf8').digest('hex');
}

export interface ChunkContextFile {
    header: { kind: string; chunk_set_sha256: string; [key: string]: any };
    contexts: Map<string, string>;
}

/** Header line, then one JSON line per completed batch carrying `contexts: { id: sentence }`. */
export function loadChunkContextFile(file: string | undefined | null): ChunkContextFile {
    if (!file) throw new Error('CHUNK_CONTEXT_HEADER=generated needs CHUNK_CONTEXT_FILE');
    let text: string;
    try {
        text = fs.readFileSync(file, 'utf8');
    } catch (error: any) {
        throw new Error(`CHUNK_CONTEXT_FILE ${file} cannot be read: ${error?.message || error}`);
    }
    const lines = text.split('\n').filter((line) => line.trim());
    const header = lines.length ? JSON.parse(lines[0]) : null;
    if (!header || header.kind !== CHUNK_CONTEXT_FILE_KIND || typeof header.chunk_set_sha256 !== 'string') {
        throw new Error(`CHUNK_CONTEXT_FILE ${file} does not start with a ${CHUNK_CONTEXT_FILE_KIND} header`);
    }
    const contexts = new Map<string, string>();
    for (const line of lines.slice(1)) {
        const record = JSON.parse(line);
        for (const [id, sentence] of Object.entries(record.contexts || {})) {
            if (typeof sentence === 'string' && sentence.trim()) contexts.set(id, sentence);
        }
    }
    return { header, contexts };
}

/** Throws, naming the mismatch, when the file was generated over another chunk set or covers too few ids. */
export function assertChunkContextCoverage(file: ChunkContextFile, ids: Set<string>, source = 'CHUNK_CONTEXT_FILE', minCoverage = CHUNK_CONTEXT_MIN_COVERAGE): { covered: number; total: number } {
    const sha = chunkSetSha256(ids);
    if (file.header.chunk_set_sha256 !== sha) {
        // The generator records the collections it read and their row counts, so
        // the refusal can say whether the corpus moved or a collection was short.
        const counts = file.header.collection_row_counts && typeof file.header.collection_row_counts === 'object'
            ? ` from ${Object.entries(file.header.collection_row_counts).map(([name, rows]) => `${name} (${rows} rows)`).join(', ')}, ${file.header.chunk_count ?? '?'} chunks`
            : '';
        throw new Error(`${source} was generated over chunk set ${file.header.chunk_set_sha256}${counts}; this build indexes chunk set ${sha} (${ids.size} chunks). Was the corpus updated between the two builds?`);
    }
    let covered = 0;
    for (const id of ids) if (file.contexts.has(id)) covered++;
    if (ids.size > 0 && covered / ids.size < minCoverage) {
        throw new Error(`${source} covers ${covered} of ${ids.size} chunk ids (${(100 * covered / ids.size).toFixed(2)}%), below ${minCoverage * 100}%`);
    }
    return { covered, total: ids.size };
}

/**
 * serve-generated-chunk-contexts: a relative CHUNK_CONTEXT_FILE names a file under the
 * knowledge root (LOCAL_RAG_KNOWLEDGE_ROOT when set, else the corpus being indexed), so
 * one `.mcp.json` value works for the CLI, the MCP server and the panel container.
 */
export function resolveChunkContextFilePath(raw: string | undefined | null, knowledgeRoot: string | undefined | null, codebasePath: string): string | undefined {
    if (!raw) return undefined;
    return path.isAbsolute(raw) ? raw : path.join(knowledgeRoot || codebasePath, raw);
}
