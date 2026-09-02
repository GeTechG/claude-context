// Class-reference XML → Markdown. The regression that matters most is the
// indentation one: Godot indents description text by element depth, and a
// continuation line that keeps its tabs is an indented code block to every
// Markdown parser — the prose would be indexed as a code listing.

import { classDocXmlToMarkdown } from './class-doc-xml';

const DOC = `<?xml version="1.0" encoding="UTF-8" ?>
<class name="Node" inherits="Object" xml:lang="en">
\t<brief_description>
\t\tBase class for all scene objects.
\t</brief_description>
\t<description>
\t\tNodes are Godot's building blocks.
\t\tA tree of nodes is called a [i]scene[/i]. See [method add_child] and [Object].
\t\t[b]Note:[/b] Use [code]get_child(0)[/code] first.
\t\t[codeblocks]
\t\t[gdscript]
\t\tvar child = get_child(0)
\t\tif child:
\t\t\tadd_child(child)
\t\t[/gdscript]
\t\t[/codeblocks]
\t</description>
\t<tutorials>
\t\t<link title="Nodes and scenes">https://docs.godotengine.org/nodes.html</link>
\t</tutorials>
\t<methods>
\t\t<method name="add_child">
\t\t\t<return type="void" />
\t\t\t<param index="0" name="node" type="Node" />
\t\t\t<param index="1" name="force_readable_name" type="bool" default="false" />
\t\t\t<description>
\t\t\t\tAdds a child [param node].
\t\t\t</description>
\t\t</method>
\t</methods>
\t<members>
\t\t<member name="name" type="StringName" setter="set_name" getter="get_name" default="&amp;&quot;&quot;">
\t\t\tThe name of the node.
\t\t</member>
\t</members>
\t<signals>
\t\t<signal name="child_entered_tree">
\t\t\t<param index="0" name="node" type="Node" />
\t\t\t<description>
\t\t\t\tEmitted when a child enters the tree.
\t\t\t</description>
\t\t</signal>
\t</signals>
\t<constants>
\t\t<constant name="NOTIFICATION_ENTER_TREE" value="10">
\t\t\tNotification received when the node enters the tree.
\t\t</constant>
\t</constants>
</class>`;

describe('classDocXmlToMarkdown', () => {
    const md = classDocXmlToMarkdown(DOC)!;

    it('returns null for XML that is not a class document', () => {
        expect(classDocXmlToMarkdown('<meta><uuid>abc</uuid></meta>')).toBeNull();
        expect(classDocXmlToMarkdown('not xml at all')).toBeNull();
    });

    it('emits the class as a heading with its base class', () => {
        expect(md.startsWith('# Node')).toBe(true);
        expect(md).toContain('Inherits: `Object`');
    });

    it('strips the XML element indentation from prose', () => {
        // A tab-indented continuation line would render as a code block.
        const outsideFences = md.split('```').filter((_, i) => i % 2 === 0).join('\n');
        expect(outsideFences).not.toMatch(/^[ \t]+\S/m);
        expect(md).toContain('A tree of nodes is called a *scene*.');
    });

    it('converts BBCode markup and cross-references', () => {
        expect(md).toContain('**Note:**');
        expect(md).toContain('`get_child(0)`');
        expect(md).toContain('`add_child`');   // [method add_child]
        expect(md).toContain('`Object`');      // [Object]
    });

    it('fences code examples with their language and dedents the body', () => {
        expect(md).toContain('```gdscript\nvar child = get_child(0)');
        // one level of real GDScript indentation survives, XML depth does not
        expect(md).toContain('\nif child:\n\tadd_child(child)\n```');
    });

    it('renders method signatures with parameter types, defaults and return', () => {
        expect(md).toContain('### add_child(node: Node, force_readable_name: bool = false) -> void');
        expect(md).toContain('Adds a child `node`.');
    });

    it('renders properties, signals and constants with decoded entities', () => {
        expect(md).toContain('### name: StringName = &""');
        expect(md).toContain('### child_entered_tree(node: Node)');
        expect(md).toContain('### NOTIFICATION_ENTER_TREE = 10');
    });

    // Regression: the wrapper-stripping step used to be written `codeblocks?`,
    // which also matched the SINGULAR `[codeblock]` and removed its markers
    // before the fencing step could see them. The example body then leaked out
    // as prose, and its GDScript `# comment` became a Markdown H1 that captured
    // the heading_path of every chunk in the file (116 of the 912 Godot class
    // docs contain a bare [codeblock]).
    it('fences a bare [codeblock] so its # comments cannot become headings', () => {
        const md = classDocXmlToMarkdown(`<class name="RenderingDevice">
\t<description>
\t\t[codeblock]
\t\t# Draw wire.
\t\tdraw_list_draw(list, false, 1)
\t\t[/codeblock]
\t</description>
</class>`)!;
        let inFence = false;
        const headings = md.split('\n').filter(line => {
            if (/^```/.test(line)) { inFence = !inFence; return false; }
            return !inFence && /^# /.test(line);
        });
        expect(headings).toEqual(['# RenderingDevice']);
        expect(md).toContain('```\n# Draw wire.\ndraw_list_draw(list, false, 1)\n```');
    });

    it('keeps tutorial links', () => {
        expect(md).toContain('[Nodes and scenes](https://docs.godotengine.org/nodes.html)');
    });
});
