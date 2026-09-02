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

/**
 * Convert one class-reference XML document to Markdown.
 * Returns null when the input is not such a document — the caller skips it.
 */
export function classDocXmlToMarkdown(xml: string): string | null {
    const open = CLASS_OPEN.exec(xml);
    if (!open) return null;
    const classAttrs = parseAttributes(open[1]);
    if (!classAttrs.name) return null;

    const out: string[] = [`# ${classAttrs.name}`, ''];
    if (classAttrs.inherits) out.push(`Inherits: \`${classAttrs.inherits}\``, '');

    // The class-level <description> is the one before the first entry section;
    // every later <description> belongs to a method/member/signal/constant.
    const bodyStart = xml.indexOf('</brief_description>');
    const firstSection = xml.search(/<(constructors|methods|operators|members|signals|constants|theme_items)>/);
    const head = xml.slice(bodyStart >= 0 ? bodyStart : 0, firstSection >= 0 ? firstSection : undefined);

    for (const [tag, text] of [
        ['brief_description', sectionText(xml, 'brief_description')],
        ['description', sectionText(head, 'description')],
    ] as const) {
        if (text) out.push(text, '');
        void tag;
    }

    const links = [...xml.matchAll(/<link(?:\s+title="([^"]*)")?\s*>([\s\S]*?)<\/link>/g)];
    if (links.length) {
        out.push('## Tutorials', '');
        for (const l of links) out.push(`- ${l[1] ? `[${decodeEntities(l[1])}](${l[2].trim()})` : l[2].trim()}`);
        out.push('');
    }

    for (const section of ENTRY_SECTIONS) {
        const singular = section.slice(0, -1);
        const entries = [...xml.matchAll(
            new RegExp(`<${singular}\\s+([^>]*?)>([\\s\\S]*?)</${singular}>`, 'g'))];
        if (!entries.length) continue;
        out.push(`## ${section[0].toUpperCase()}${section.slice(1)}`, '');
        for (const [, rawAttrs, block] of entries) {
            const attrs = parseAttributes(rawAttrs);
            out.push(`### ${signature(attrs.name, block, attrs)}`, '');
            const desc = sectionText(block, 'description');
            if (desc) out.push(desc, '');
        }
    }

    const members = [...xml.matchAll(/<member\s+([^>]*?)>([\s\S]*?)<\/member>/g)];
    if (members.length) {
        out.push('## Properties', '');
        for (const [, rawAttrs, body] of members) {
            const a = parseAttributes(rawAttrs);
            out.push(`### ${a.name}: ${a.type}${a.default ? ` = ${a.default}` : ''}`, '');
            const desc = bbcodeToMarkdown(body);
            if (desc) out.push(desc, '');
        }
    }

    const signals = [...xml.matchAll(/<signal\s+([^>]*?)>([\s\S]*?)<\/signal>/g)];
    if (signals.length) {
        out.push('## Signals', '');
        for (const [, rawAttrs, block] of signals) {
            const a = parseAttributes(rawAttrs);
            out.push(`### ${signature(a.name, block, a)}`, '');
            const desc = sectionText(block, 'description');
            if (desc) out.push(desc, '');
        }
    }

    const constants = [...xml.matchAll(/<constant\s+([^>]*?)>([\s\S]*?)<\/constant>/g)];
    if (constants.length) {
        out.push('## Constants', '');
        for (const [, rawAttrs, body] of constants) {
            const a = parseAttributes(rawAttrs);
            const enumTag = a.enum ? ` (enum \`${a.enum}\`)` : '';
            out.push(`### ${a.name} = ${a.value}${enumTag}`, '');
            const desc = bbcodeToMarkdown(body);
            if (desc) out.push(desc, '');
        }
    }

    const themeItems = [...xml.matchAll(/<theme_item\s+([^>]*?)>([\s\S]*?)<\/theme_item>/g)];
    if (themeItems.length) {
        out.push('## Theme properties', '');
        for (const [, rawAttrs, body] of themeItems) {
            const a = parseAttributes(rawAttrs);
            out.push(`### ${a.name}: ${a.type}${a.default ? ` = ${a.default}` : ''}`, '');
            const desc = bbcodeToMarkdown(body);
            if (desc) out.push(desc, '');
        }
    }

    return out.join('\n').replace(/\n{3,}/g, '\n\n').trim() + '\n';
}
