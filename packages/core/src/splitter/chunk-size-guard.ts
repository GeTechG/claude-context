import { CodeChunk } from './index';

/**
 * #19 — a chunk whose `content` overflows the Milvus schema kills its batch.
 *
 * Every collection this engine creates declares `content` as
 * `VarChar(max_length: 65535)` (see `createCollection` / `createHybridCollection`
 * in `vectordb/milvus-vectordb.ts`), and Milvus counts that limit in **bytes**,
 * not characters. Neither splitter guarantees it:
 *
 *   - the AST splitter emits one chunk per top-level node, and a minified
 *     bundle is a single node — `examples/jsm/libs/draco/draco_decoder.js` in
 *     three.js is one 520 KB line;
 *   - the langchain fallback splits on separators, so a file with no separator
 *     inside 65535 bytes (that same minified line) comes back whole;
 *   - the markdown splitter splits on headings, with the same failure mode for
 *     a heading-less blob.
 *
 * Rather than patch each splitter, the limit is enforced once on the chunk
 * stream (`Context.processFileList`), so it holds for every language, every
 * splitter and every collection. Oversized chunks are split into byte-bounded
 * parts on line boundaries where possible; only a single line that is itself
 * too long is cut mid-line, and only a chunk that would need more than
 * `maxParts` parts loses content — that tail is dropped with a marker and
 * `truncated: true`.
 */

/** `content` VarChar max_length on every collection this engine creates. */
export const MILVUS_CONTENT_MAX_BYTES = 65535;

/**
 * Ceiling on the parts one oversized chunk may become. A 520 KB minified line
 * is 8 parts; the cap only bites on pathological input (a multi-MB generated
 * blob), where indexing the head and saying so beats emitting hundreds of
 * meaningless fragments.
 */
export const DEFAULT_MAX_CHUNK_PARTS = 64;

export interface OversizedChunkInfo {
    filePath?: string;
    startLine?: number;
    bytes: number;
    parts: number;
    /** Bytes that did NOT make it into any part (0 unless maxParts was hit). */
    truncatedBytes: number;
}

export interface ChunkSizeGuardOptions {
    /** Byte ceiling for a single chunk's content. Defaults to the schema limit. */
    maxBytes?: number;
    /** Ceiling on parts per oversized chunk. Defaults to DEFAULT_MAX_CHUNK_PARTS. */
    maxParts?: number;
    /** Called once per chunk that had to be split. */
    onOversized?: (info: OversizedChunkInfo) => void;
}

export function utf8Length(text: string): number {
    return Buffer.byteLength(text, 'utf8');
}

/**
 * Cut `text` to at most `maxBytes` UTF-8 bytes without splitting a multi-byte
 * sequence (a half-written code point would be re-encoded as U+FFFD and could
 * itself grow past the limit).
 */
function cutToBytes(text: string, maxBytes: number): string {
    if (maxBytes <= 0) return '';
    const buf = Buffer.from(text, 'utf8');
    if (buf.length <= maxBytes) return text;
    let end = maxBytes;
    // Back up over UTF-8 continuation bytes (0b10xxxxxx) so `end` lands on a
    // code-point boundary.
    while (end > 0 && (buf[end] & 0xc0) === 0x80) end--;
    return buf.subarray(0, end).toString('utf8');
}

interface Part {
    text: string;
    startLine: number;
    endLine: number;
}

/**
 * Break one chunk's text into ≤ maxBytes pieces, preferring line boundaries.
 * `startLine` is the absolute line number of the text's first line.
 */
function partition(text: string, startLine: number, maxBytes: number): Part[] {
    const parts: Part[] = [];
    const lines = text.split('\n');
    let buffer: string[] = [];
    let bufferBytes = 0;
    let bufferStart = startLine;

    const flush = (endLine: number) => {
        if (buffer.length === 0) return;
        parts.push({ text: buffer.join('\n'), startLine: bufferStart, endLine });
        buffer = [];
        bufferBytes = 0;
    };

    for (let i = 0; i < lines.length; i++) {
        const lineNumber = startLine + i;
        const line = lines[i];
        const lineBytes = utf8Length(line);

        if (lineBytes > maxBytes) {
            // A single line too long for the field: the only case where a cut
            // lands mid-line. Every segment keeps the line's own number.
            flush(lineNumber - 1);
            let rest = line;
            while (rest.length > 0) {
                const segment = cutToBytes(rest, maxBytes);
                if (segment.length === 0) break; // defensive: cannot make progress
                parts.push({ text: segment, startLine: lineNumber, endLine: lineNumber });
                rest = rest.slice(segment.length);
            }
            bufferStart = lineNumber + 1;
            continue;
        }

        // +1 for the '\n' that will re-join this line to the buffered ones.
        const withSeparator = buffer.length === 0 ? lineBytes : bufferBytes + 1 + lineBytes;
        if (withSeparator > maxBytes) {
            flush(lineNumber - 1);
            bufferStart = lineNumber;
        }
        buffer.push(line);
        bufferBytes = buffer.length === 1 ? lineBytes : bufferBytes + 1 + lineBytes;
    }
    flush(startLine + lines.length - 1);
    return parts;
}

/**
 * Split a single chunk so that every returned chunk's content fits `maxBytes`.
 * Returns `[chunk]` untouched when it already fits — the overwhelmingly common
 * case, so this is a byte-length check and nothing else.
 */
export function splitOversizedChunk(
    chunk: CodeChunk,
    options: ChunkSizeGuardOptions = {},
): CodeChunk[] {
    const maxBytes = options.maxBytes ?? MILVUS_CONTENT_MAX_BYTES;
    // A single code point is up to 4 bytes: below that no cut can make progress
    // and the function would loop or emit empty parts. A limit that small is a
    // caller mistake, not something to paper over.
    if (!Number.isFinite(maxBytes) || maxBytes < 4) {
        throw new RangeError(`chunk-size-guard: maxBytes must be at least 4, got ${maxBytes}`);
    }
    const maxParts = Math.max(1, options.maxParts ?? DEFAULT_MAX_CHUNK_PARTS);
    const totalBytes = utf8Length(chunk.content);
    if (totalBytes <= maxBytes) return [chunk];

    const startLine = chunk.metadata.startLine ?? 1;
    let parts = partition(chunk.content, startLine, maxBytes);
    if (parts.length === 0) return [chunk];

    // Last resort: too many parts → keep the head, truncate with a marker so a
    // reader of the chunk (and of `metadata.truncated`) knows content is gone.
    let truncatedBytes = 0;
    if (parts.length > maxParts) {
        const kept = parts.slice(0, maxParts);
        const last = kept[kept.length - 1];
        // The marker's own width depends on the number it prints, so reserve
        // the width of the largest number it could print (the whole chunk).
        const marker = (missing: number) => `\n… [truncated: ${missing} more bytes of this chunk were not indexed]`;
        const reserve = utf8Length(marker(totalBytes));
        const fits = reserve + 4 <= maxBytes;
        const head = cutToBytes(last.text, fits ? maxBytes - reserve : maxBytes);
        // Everything not in a kept part is lost: the dropped parts, the bytes
        // cut off this last part, and the separators between them.
        const retained = kept.slice(0, -1).reduce((sum, part) => sum + utf8Length(part.text), 0) + utf8Length(head);
        truncatedBytes = totalBytes - retained;
        // A limit too small to hold the marker keeps the content instead of the
        // explanation; `metadata.truncated` still says the tail is missing.
        last.text = fits ? head + marker(truncatedBytes) : head;
        last.endLine = parts[parts.length - 1].endLine;
        parts = kept;
    }

    options.onOversized?.({
        filePath: chunk.metadata.filePath,
        startLine,
        bytes: totalBytes,
        parts: parts.length,
        truncatedBytes,
    });

    return parts.map((part, index) => ({
        content: part.text,
        metadata: {
            ...chunk.metadata,
            startLine: part.startLine,
            endLine: part.endLine,
            oversized_split: true,
            part: `${index + 1}/${parts.length}`,
            part_index: index + 1,
            part_count: parts.length,
            ...(truncatedBytes > 0 && index === parts.length - 1 ? { truncated: true } : {}),
        },
    }));
}

/**
 * Enforce the byte ceiling over a whole chunk list. Order is preserved and
 * chunks that already fit are passed through by reference.
 */
export function enforceChunkByteLimit(
    chunks: CodeChunk[],
    options: ChunkSizeGuardOptions = {},
): CodeChunk[] {
    const maxBytes = options.maxBytes ?? MILVUS_CONTENT_MAX_BYTES;
    if (!Number.isFinite(maxBytes) || maxBytes < 4) {
        throw new RangeError(`chunk-size-guard: maxBytes must be at least 4, got ${maxBytes}`);
    }
    let oversized = false;
    for (const chunk of chunks) {
        if (utf8Length(chunk.content) > maxBytes) {
            oversized = true;
            break;
        }
    }
    if (!oversized) return chunks;

    const out: CodeChunk[] = [];
    for (const chunk of chunks) out.push(...splitOversizedChunk(chunk, options));
    return out;
}
