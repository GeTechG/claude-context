import { Splitter, CodeChunk } from './index';
import { extractMentionedSymbolsFromText } from '../enrichment/symbol-extractor';

// rag-graph-layer Phase 1.2: provider for the vocabulary used to filter
// `mentioned_symbols` at split time. The splitter calls `getMentionedVocab()`
// per `split()` invocation; if it returns null/undefined the filter is
// skipped (raw extraction). Indexing-pipeline wiring sets this provider on
// each splitter instance to point at `<codebasePath>/.symbols-vocab.json`.
export type MentionedVocabProvider = () => ReadonlySet<string> | null | undefined;

/**
 * Section-aware splitter for markdown (.md) and reStructuredText (.rst).
 *
 * Behavior:
 *   - Splits on H1-H3 (markdown `#`/`##`/`###`; rst over/underline `=`/`-`/`~`).
 *   - Each prose section becomes a chunk with `content_type='doc'` and a
 *     `heading_path` of all ancestor headings (H1..current).
 *   - Fenced code blocks (``` ```...``` ```) are emitted as separate chunks
 *     with `content_type='code_example'`, `language` taken from the info string
 *     when present, and the same `heading_path` as the surrounding section.
 *   - Sections that exceed `chunkSize` are split further by character count
 *     (heading_path is preserved on each sub-chunk).
 */
export class MarkdownSplitter implements Splitter {
    private chunkSize: number;
    private chunkOverlap: number;
    // rag-graph-layer Phase 1.2: per-instance vocab provider — populated by
    // the indexing pipeline so split-time can filter mentioned_symbols
    // through `.symbols-vocab.json` when available.
    private mentionedVocabProvider?: MentionedVocabProvider;

    constructor(chunkSize?: number, chunkOverlap?: number) {
        this.chunkSize = chunkSize ?? 2500;
        this.chunkOverlap = chunkOverlap ?? 300;
    }

    setChunkSize(chunkSize: number): void {
        this.chunkSize = chunkSize;
    }

    setChunkOverlap(chunkOverlap: number): void {
        this.chunkOverlap = chunkOverlap;
    }

    setMentionedVocabProvider(provider: MentionedVocabProvider | undefined): void {
        this.mentionedVocabProvider = provider;
    }

    async split(code: string, language: string, filePath?: string): Promise<CodeChunk[]> {
        const lang = language.toLowerCase();
        const isRst = lang === 'rst' || lang === 'restructuredtext';
        const sections = isRst
            ? this.parseRstSections(code)
            : this.parseMarkdownSections(code);

        // rag-graph-layer Phase 1.2: resolve vocab once per file so the
        // provider isn't re-read per chunk.
        const vocab = this.mentionedVocabProvider?.() ?? undefined;

        const chunks: CodeChunk[] = [];
        for (const section of sections) {
            for (const block of section.blocks) {
                if (block.content.trim().length === 0) continue;
                const isCode = block.kind === 'code';
                const blockLang = isCode ? (block.language || 'text') : (isRst ? 'rst' : 'markdown');

                const subChunks = this.maybeSplitOversized(block.content, this.chunkSize);
                let lineCursor = block.startLine;
                for (const piece of subChunks) {
                    const lineCount = piece.split('\n').length;
                    const mentioned = extractMentionedSymbolsFromText(piece, vocab);
                    chunks.push({
                        content: piece,
                        metadata: {
                            startLine: lineCursor,
                            endLine: lineCursor + lineCount - 1,
                            language: blockLang,
                            filePath,
                            content_type: isCode ? 'code_example' : 'doc',
                            heading_path: this.compactHeadings(section.headingPath),
                            mentioned_symbols: mentioned,
                        },
                    });
                    lineCursor += lineCount;
                }
            }
        }

        if (chunks.length === 0) {
            // Empty or heading-only document — emit one whole-file doc chunk so
            // the file still gets indexed.
            const lineCount = code.split('\n').length;
            chunks.push({
                content: code,
                metadata: {
                    startLine: 1,
                    endLine: Math.max(1, lineCount),
                    language: isRst ? 'rst' : 'markdown',
                    filePath,
                    content_type: 'doc',
                    mentioned_symbols: extractMentionedSymbolsFromText(code, vocab),
                },
            });
        }

        return chunks;
    }

    // ---------- markdown parsing ----------

    private parseMarkdownSections(code: string): Section[] {
        const lines = code.split('\n');
        const headingStack: string[] = []; // path from H1 down
        const sections: Section[] = [];
        let current: Section = this.makeSection([], 1);

        let inFence = false;
        let fenceMarker = '';
        let fenceLang = '';
        let fenceStartLine = 0;
        let fenceBuffer: string[] = [];
        let proseBuffer: string[] = [];
        let proseStartLine = 1;

        const flushProse = (endLineExclusive: number) => {
            if (proseBuffer.length === 0) return;
            const text = proseBuffer.join('\n');
            if (text.trim().length > 0) {
                current.blocks.push({
                    kind: 'doc',
                    content: text,
                    startLine: proseStartLine,
                    endLine: endLineExclusive,
                });
            }
            proseBuffer = [];
        };

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNo = i + 1;

            if (inFence) {
                // Inside a fenced code block — collect verbatim until matching closer.
                fenceBuffer.push(line);
                if (line.trimStart().startsWith(fenceMarker)) {
                    current.blocks.push({
                        kind: 'code',
                        content: fenceBuffer.join('\n'),
                        startLine: fenceStartLine,
                        endLine: lineNo,
                        language: fenceLang || undefined,
                    });
                    inFence = false;
                    fenceBuffer = [];
                    fenceMarker = '';
                    fenceLang = '';
                    proseStartLine = lineNo + 1;
                }
                continue;
            }

            // Detect a fence opener (``` or ~~~ with at least 3 chars).
            const fenceMatch = /^(\s*)(```+|~~~+)(.*)$/.exec(line);
            if (fenceMatch) {
                flushProse(lineNo - 1);
                inFence = true;
                fenceMarker = fenceMatch[2];
                fenceLang = fenceMatch[3].trim().split(/\s+/)[0] || '';
                fenceStartLine = lineNo;
                fenceBuffer = [line];
                continue;
            }

            // Detect ATX heading (#..###### with required space after).
            const headingMatch = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
            if (headingMatch) {
                const level = headingMatch[1].length;
                const text = headingMatch[2].trim();
                if (level <= 3) {
                    flushProse(lineNo - 1);
                    if (current.blocks.length > 0) sections.push(current);
                    headingStack.length = level - 1;
                    headingStack[level - 1] = text;
                    current = this.makeSection([...headingStack], lineNo);
                    proseStartLine = lineNo + 1;
                    continue;
                }
                // H4..H6 — keep as prose so the body still ranks under its H1-H3 ancestors.
            }

            // Detect Setext heading (line followed by === or ---).
            if (i + 1 < lines.length) {
                const next = lines[i + 1];
                if (line.trim().length > 0 && /^=+\s*$/.test(next)) {
                    flushProse(lineNo - 1);
                    if (current.blocks.length > 0) sections.push(current);
                    headingStack.length = 0;
                    headingStack[0] = line.trim();
                    current = this.makeSection([...headingStack], lineNo);
                    proseStartLine = lineNo + 2;
                    i++; // skip underline
                    continue;
                }
                if (line.trim().length > 0 && /^-+\s*$/.test(next)) {
                    flushProse(lineNo - 1);
                    if (current.blocks.length > 0) sections.push(current);
                    headingStack.length = 1;
                    headingStack[1] = line.trim();
                    current = this.makeSection([...headingStack], lineNo);
                    proseStartLine = lineNo + 2;
                    i++;
                    continue;
                }
            }

            if (proseBuffer.length === 0) proseStartLine = lineNo;
            proseBuffer.push(line);
        }

        if (inFence && fenceBuffer.length > 0) {
            // Unterminated fence — emit what we have as a code block.
            current.blocks.push({
                kind: 'code',
                content: fenceBuffer.join('\n'),
                startLine: fenceStartLine,
                endLine: lines.length,
                language: fenceLang || undefined,
            });
        }
        flushProse(lines.length);
        if (current.blocks.length > 0) sections.push(current);

        return sections;
    }

    // ---------- rst parsing ----------
    // RST headings are a title line followed by a punctuation underline (and
    // optionally a matching overline). Convention is shaky, so we use the
    // hierarchy implied by the order in which underline characters appear.
    private parseRstSections(code: string): Section[] {
        const lines = code.split('\n');
        const charLevels = new Map<string, number>(); // first-seen punctuation char -> 1-based level
        const headingStack: string[] = [];
        const sections: Section[] = [];
        let current: Section = this.makeSection([], 1);

        let inDirective = false;
        let directiveLang = '';
        let directiveStartLine = 0;
        let directiveBuffer: string[] = [];
        let directiveIndent = 0;
        let proseBuffer: string[] = [];
        let proseStartLine = 1;

        const flushProse = (endLineExclusive: number) => {
            if (proseBuffer.length === 0) return;
            const text = proseBuffer.join('\n');
            if (text.trim().length > 0) {
                current.blocks.push({
                    kind: 'doc',
                    content: text,
                    startLine: proseStartLine,
                    endLine: endLineExclusive,
                });
            }
            proseBuffer = [];
        };

        const flushDirective = (endLine: number) => {
            if (directiveBuffer.length === 0) return;
            current.blocks.push({
                kind: 'code',
                content: directiveBuffer.join('\n'),
                startLine: directiveStartLine,
                endLine,
                language: directiveLang || undefined,
            });
            directiveBuffer = [];
            directiveLang = '';
        };

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineNo = i + 1;

            if (inDirective) {
                if (line.trim().length === 0 || line.search(/\S/) >= directiveIndent) {
                    directiveBuffer.push(line);
                    continue;
                }
                flushDirective(lineNo - 1);
                inDirective = false;
                proseStartLine = lineNo;
            }

            // .. code-block:: lang   /  .. sourcecode:: lang
            const directiveMatch = /^(\s*)\.\.\s+(code-block|sourcecode|code)::\s*(\S+)?/.exec(line);
            if (directiveMatch) {
                flushProse(lineNo - 1);
                inDirective = true;
                directiveLang = (directiveMatch[3] || '').trim();
                directiveStartLine = lineNo;
                directiveBuffer = [line];
                directiveIndent = (directiveMatch[1] || '').length + 3; // body must be indented relative
                continue;
            }

            // Heading detection: title + underline of identical punctuation chars.
            if (i + 1 < lines.length && line.trim().length > 0) {
                const next = lines[i + 1];
                const trimmedNext = next.trimEnd();
                const headingChar = trimmedNext.length > 0 ? trimmedNext[0] : '';
                const isUnderline =
                    headingChar.length === 1 &&
                    /[!-/:-@[-`{-~]/.test(headingChar) &&
                    trimmedNext.length >= line.trim().length &&
                    trimmedNext.split('').every(c => c === headingChar);
                if (isUnderline) {
                    let level = charLevels.get(headingChar);
                    if (level === undefined) {
                        level = charLevels.size + 1;
                        charLevels.set(headingChar, level);
                    }
                    if (level <= 3) {
                        flushProse(lineNo - 1);
                        if (current.blocks.length > 0) sections.push(current);
                        headingStack.length = level - 1;
                        headingStack[level - 1] = line.trim();
                        current = this.makeSection([...headingStack], lineNo);
                        proseStartLine = lineNo + 2;
                        i++;
                        continue;
                    }
                }
            }

            if (proseBuffer.length === 0) proseStartLine = lineNo;
            proseBuffer.push(line);
        }

        if (inDirective) flushDirective(lines.length);
        flushProse(lines.length);
        if (current.blocks.length > 0) sections.push(current);

        return sections;
    }

    // ---------- helpers ----------

    private makeSection(headingPath: string[], startLine: number): Section {
        return { headingPath, startLine, blocks: [] };
    }

    private compactHeadings(stack: string[]): string[] | undefined {
        // Drop sparse holes (e.g. file starts at H2 with no H1) so heading_path
        // is always a dense list of strings.
        const cleaned = stack.filter((h): h is string => typeof h === 'string' && h.length > 0);
        return cleaned.length > 0 ? cleaned : undefined;
    }

    private maybeSplitOversized(text: string, maxSize: number): string[] {
        if (text.length <= maxSize) return [text];
        const out: string[] = [];
        const lines = text.split('\n');
        let buf: string[] = [];
        let size = 0;
        for (const line of lines) {
            const lineSize = line.length + 1;
            if (size + lineSize > maxSize && buf.length > 0) {
                out.push(buf.join('\n'));
                buf = [];
                size = 0;
            }
            buf.push(line);
            size += lineSize;
        }
        if (buf.length > 0) out.push(buf.join('\n'));
        return out;
    }
}

interface Section {
    headingPath: string[];
    startLine: number;
    blocks: Block[];
}

interface Block {
    kind: 'doc' | 'code';
    content: string;
    startLine: number;
    endLine: number;
    language?: string;
}
