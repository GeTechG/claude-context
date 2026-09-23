// Class-reference XML → Markdown, applied while indexing (context.processFileList).
//
// Godot ships its API documentation as one XML file per class under
// `doc/classes/`, NOT as doc comments: scene/main/node.cpp is 3% comments and
// none of them is the API description. Measured over the 912 files in the
// reference corpus: 2.6M characters of prose, 9 743 documented methods, 5 437
// documented properties and 1 503 GDScript/C# examples that have no counterpart
// anywhere in the C++ sources. Indexing the raw XML would bury that prose in
// markup, so it is rewritten as Markdown and handed to the Markdown splitter,
// which lands it in the prose pool with a usable `heading_path`.
//
// Regex rather than an XML parser: the input is single-schema, machine-emitted
// output of Godot's own doc tool, and `classDocXmlToMarkdown` returns null for
// anything that is not a `<class …>` document, so a mismatch degrades to "skip
// this file", never to garbage chunks.

const CLASS_OPEN = /<class\s+([^>]*?)>/;

/** Sections in emission order; each maps to a `## ` heading. */
const ENTRY_SECTIONS = ['constructors', 'methods', 'operators'] as const;

function decodeEntities(text: string): string {
    return text
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&apos;/g, "'")
        .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
        .replace(/&amp;/g, '&'); // last: an escaped entity must not be re-expanded
}

function parseAttributes(raw: string): Record<string, string> {
    const attrs: Record<string, string> = {};
    for (const m of raw.matchAll(/([a-z_]+)="([^"]*)"/g)) {
        attrs[m[1]] = decodeEntities(m[2]);
    }
    return attrs;
}

/**
 * Godot's BBCode-ish markup → Markdown. Cross-references ([method X], [Node], …)
 * become inline code: the link target is meaningless outside the doc site, but
 * the symbol name is exactly what a retrieval query matches on.
 */
function bbcodeToMarkdown(text: string): string {
    let out = decodeEntities(text);
    out = out.replace(/\[(gdscript|csharp)(?:\s[^\]]*)?\]([\s\S]*?)\[\/\1\]/g,
        (_, lang, body) => `\n\`\`\`${lang}\n${dedent(body)}\n\`\`\`\n`);
    // ONLY the plural wrapper. `codeblocks?` would also match the singular
    // `[codeblock]`, stripping its markers before the handler below could fence
    // it — the example body then leaks out as prose, and a GDScript `# comment`
    // line inside it becomes a Markdown H1 that captures the whole file's
    // heading_path.
    out = out.replace(/\[codeblocks\]|\[\/codeblocks\]/g, '');
    out = out.replace(/\[codeblock(?:\s[^\]]*)?\]([\s\S]*?)\[\/codeblock\]/g,
        (_, body) => `\n\`\`\`\n${dedent(body)}\n\`\`\`\n`);
    out = out.replace(/\[code(?:\s[^\]]*)?\]([\s\S]*?)\[\/code\]/g, (_, body) => `\`${body.trim()}\``);
    out = out.replace(/\[b\]([\s\S]*?)\[\/b\]/g, '**$1**');
    out = out.replace(/\[i\]([\s\S]*?)\[\/i\]/g, '*$1*');
    out = out.replace(/\[url=([^\]]+)\]([\s\S]*?)\[\/url\]/g, '[$2]($1)');
    out = out.replace(/\[br\]\s*/g, '\n');
    // [method X] / [member X] / [constant X] / [param x] / [Node] → `X`
    out = out.replace(/\[(?:method|member|constant|signal|param|enum|theme_item|annotation|constructor|operator)\s+([^\]]+)\]/g, '`$1`');
    out = out.replace(/\[([A-Z][A-Za-z0-9_.]*)\]/g, '`$1`');
    return stripProseIndent(out).replace(/[ \t]+$/gm, '').replace(/\n{3,}/g, '\n\n').trim();
}

/**
 * Drop the XML element indentation from prose lines. Without this, a
 * continuation line still carrying its two source tabs is an indented code
 * block to every Markdown parser, and the description turns into a code
 * listing. Lines inside fenced blocks keep their indentation — the fences were
 * emitted above with their bodies already dedented.
 */
function stripProseIndent(text: string): string {
    let inFence = false;
    return text.split('\n').map(line => {
        if (/^\s*```/.test(line)) {
            inFence = !inFence;
            return line.trim();
        }
        return inFence ? line : line.replace(/^[ \t]+/, '');
    }).join('\n');
}

/** Strip the XML indentation shared by every line of a code block. */
function dedent(body: string): string {
    const lines = body.replace(/^\n+|\s+$/g, '').split('\n');
    const indents = lines.filter(l => l.trim()).map(l => l.match(/^[ \t]*/)![0].length);
    const cut = indents.length ? Math.min(...indents) : 0;
    return lines.map(l => l.slice(cut)).join('\n');
}

/** Inner text of the first `<tag …>…</tag>` in `scope`, BBCode already converted. */
function sectionText(scope: string, tag: string): string {
    const m = new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)</${tag}>`).exec(scope);
    return m ? bbcodeToMarkdown(m[1]) : '';
}

function signature(name: string, block: string, attrs: Record<string, string>): string {
    const params = [...block.matchAll(/<param\s+([^>]*?)\/?>/g)]
        .map(m => parseAttributes(m[1]))
        .map(p => `${p.name}: ${p.type}${p.default ? ` = ${p.default}` : ''}`);
    const ret = /<return\s+([^>]*?)\/?>/.exec(block);
    const retType = ret ? parseAttributes(ret[1]).type : '';
    const qualifiers = attrs.qualifiers ? ` ${attrs.qualifiers}` : '';
    return `${name}(${params.join(', ')})${retType ? ` -> ${retType}` : ''}${qualifiers}`;
}

/** An XML line span `[from, to]`, 1-based and inclusive. */
export type XmlLineSpan = [number, number];

/**
 * Convert one class-reference XML document to Markdown.
 * Returns null when the input is not such a document — the caller skips it.
 */
export function classDocXmlToMarkdown(xml: string): string | null {
    const rendered = classDocXmlToMarkdownWithLines(xml);
    return rendered ? rendered.markdown : null;
}

/**
 * cite-class-reference-xml-by-its-own-lines: the same Markdown, plus, for each of
 * its lines (index 0 = Markdown line 1), the span of XML lines of the element that
 * line was rendered from. A chunk cut from the Markdown is cited by the union of
 * its lines' spans, so its range names the file on disk and not the rendering.
 */
export function classDocXmlToMarkdownWithLines(xml: string): { markdown: string; lineSpans: XmlLineSpan[] } | null {
    const open = CLASS_OPEN.exec(xml);
    if (!open) return null;
    const classAttrs = parseAttributes(open[1]);
    if (!classAttrs.name) return null;

    const newlines: number[] = [];
    for (let i = xml.indexOf('\n'); i >= 0; i = xml.indexOf('\n', i + 1)) newlines.push(i);
    const lineAt = (offset: number): number => {
        let lo = 0, hi = newlines.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (newlines[mid] < offset) lo = mid + 1; else hi = mid; }
        return lo + 1;
    };
    const spanOf = (m: RegExpExecArray | RegExpMatchArray, base = 0): XmlLineSpan =>
        [lineAt(base + m.index!), lineAt(base + m.index! + m[0].length - 1)];
    const elementSpan = (scope: string, tag: string, base = 0): XmlLineSpan | null => {
        const m = new RegExp(`<${tag}(?:\\s[^>]*)?>[\\s\\S]*?</${tag}>`).exec(scope);
        return m ? spanOf(m, base) : null;
    };
    const tagLine = (tag: string): XmlLineSpan => {
        const i = xml.search(new RegExp(`<${tag}>`));
        return i >= 0 ? [lineAt(i), lineAt(i)] : classLine;
    };

    // One entry per pushed string, which may hold several lines: they all share its span.
    const out: Array<[string, XmlLineSpan]> = [];
    const push = (span: XmlLineSpan, ...texts: string[]) => { for (const t of texts) out.push([t, span]); };

    const classLine: XmlLineSpan = [lineAt(open.index), lineAt(open.index)];
    push(classLine, `# ${classAttrs.name}`, '');
    if (classAttrs.inherits) push(classLine, `Inherits: \`${classAttrs.inherits}\``, '');

    // The class-level <description> is the one before the first entry section;
    // every later <description> belongs to a method/member/signal/constant.
    const bodyStart = xml.indexOf('</brief_description>');
    const firstSection = xml.search(/<(constructors|methods|operators|members|signals|constants|theme_items)>/);
    const headFrom = bodyStart >= 0 ? bodyStart : 0;
    const head = xml.slice(headFrom, firstSection >= 0 ? firstSection : undefined);

    for (const [text, span] of [
        [sectionText(xml, 'brief_description'), elementSpan(xml, 'brief_description')],
        [sectionText(head, 'description'), elementSpan(head, 'description', headFrom)],
    ] as const) {
        if (text) push(span ?? classLine, text, '');
    }

    const links = [...xml.matchAll(/<link(?:\s+title="([^"]*)")?\s*>([\s\S]*?)<\/link>/g)];
    if (links.length) {
        const tutorials = elementSpan(xml, 'tutorials') ?? classLine;
        push(tutorials, '## Tutorials', '');
        for (const l of links) push(spanOf(l), `- ${l[1] ? `[${decodeEntities(l[1])}](${l[2].trim()})` : l[2].trim()}`);
        push(tutorials, '');
    }

    for (const section of ENTRY_SECTIONS) {
        const singular = section.slice(0, -1);
        const entries = [...xml.matchAll(
            new RegExp(`<${singular}\\s+([^>]*?)>([\\s\\S]*?)</${singular}>`, 'g'))];
        if (!entries.length) continue;
        push(tagLine(section), `## ${section[0].toUpperCase()}${section.slice(1)}`, '');
        for (const entry of entries) {
            const [, rawAttrs, block] = entry;
            const attrs = parseAttributes(rawAttrs);
            const span = spanOf(entry);
            push(span, `### ${signature(attrs.name, block, attrs)}`, '');
            const desc = sectionText(block, 'description');
            if (desc) push(span, desc, '');
        }
    }

    const members = [...xml.matchAll(/<member\s+([^>]*?)>([\s\S]*?)<\/member>/g)];
    if (members.length) {
        push(tagLine('members'), '## Properties', '');
        for (const entry of members) {
            const [, rawAttrs, body] = entry;
            const a = parseAttributes(rawAttrs);
            const span = spanOf(entry);
            push(span, `### ${a.name}: ${a.type}${a.default ? ` = ${a.default}` : ''}`, '');
            const desc = bbcodeToMarkdown(body);
            if (desc) push(span, desc, '');
        }
    }

    const signals = [...xml.matchAll(/<signal\s+([^>]*?)>([\s\S]*?)<\/signal>/g)];
    if (signals.length) {
        push(tagLine('signals'), '## Signals', '');
        for (const entry of signals) {
            const [, rawAttrs, block] = entry;
            const a = parseAttributes(rawAttrs);
            const span = spanOf(entry);
            push(span, `### ${signature(a.name, block, a)}`, '');
            const desc = sectionText(block, 'description');
            if (desc) push(span, desc, '');
        }
    }

    const constants = [...xml.matchAll(/<constant\s+([^>]*?)>([\s\S]*?)<\/constant>/g)];
    if (constants.length) {
        push(tagLine('constants'), '## Constants', '');
        for (const entry of constants) {
            const [, rawAttrs, body] = entry;
            const a = parseAttributes(rawAttrs);
            const span = spanOf(entry);
            const enumTag = a.enum ? ` (enum \`${a.enum}\`)` : '';
            push(span, `### ${a.name} = ${a.value}${enumTag}`, '');
            const desc = bbcodeToMarkdown(body);
            if (desc) push(span, desc, '');
        }
    }

    const themeItems = [...xml.matchAll(/<theme_item\s+([^>]*?)>([\s\S]*?)<\/theme_item>/g)];
    if (themeItems.length) {
        push(tagLine('theme_items'), '## Theme properties', '');
        for (const entry of themeItems) {
            const [, rawAttrs, body] = entry;
            const a = parseAttributes(rawAttrs);
            const span = spanOf(entry);
            push(span, `### ${a.name}: ${a.type}${a.default ? ` = ${a.default}` : ''}`, '');
            const desc = bbcodeToMarkdown(body);
            if (desc) push(span, desc, '');
        }
    }

    // The text rule is the one the Markdown-only version applied to the joined string
    // (`\n{3,}` → `\n\n`, then trim): run on lines, a blank line directly after a blank
    // line is dropped, and leading and trailing blank lines go. Each kept line keeps its span.
    const lines: Array<[string, XmlLineSpan]> = [];
    for (const [text, span] of out) for (const line of text.split('\n')) {
        if (line === '' && (lines.length === 0 || lines[lines.length - 1][0] === '')) continue;
        lines.push([line, span]);
    }
    while (lines.length && lines[lines.length - 1][0] === '') lines.pop();
    return { markdown: lines.map(([line]) => line).join('\n') + '\n', lineSpans: lines.map(([, span]) => span) };
}
