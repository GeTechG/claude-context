// rag-graph-layer Phase 1.1: per-language extractors for structural code
// signals — imports / extends / implements. Pure tree-sitter walking, no
// network or LSP. Each extractor returns whatever it can find and `undefined`
// for fields it cannot extract; never throws on a malformed grammar.
//
// `extractStructural(rootNode, language)` is the file-level entry — it walks
// the top of the tree to gather imports. `extractClassStructural(node,
// language)` is the symbol-level entry, called per code-chunk node by the
// splitter to fill `extends` / `implements` for class-shaped symbols.

import Parser from 'tree-sitter';

export interface FileStructural {
    imports?: string[];
}

export interface ClassStructural {
    extends?: string;
    implements?: string[];
}

const STOPWORDS_IDENT = new Set(['true', 'false', 'null', 'none', 'undefined']);

function looksLikeIdentifier(text: string): boolean {
    if (!text) return false;
    if (STOPWORDS_IDENT.has(text.toLowerCase())) return false;
    return /^[A-Za-z_][A-Za-z0-9_.]*$/.test(text);
}

function dedup(values: string[]): string[] {
    if (values.length === 0) return values;
    const seen = new Set<string>();
    const out: string[] = [];
    for (const v of values) {
        if (!v || seen.has(v)) continue;
        seen.add(v);
        out.push(v);
    }
    return out;
}

/**
 * Walk a python `dotted_name` node into its dotted-string form (e.g. `os.path`).
 */
function dottedNameText(node: Parser.SyntaxNode): string {
    return node.text.replace(/\s+/g, '');
}

function findFirstChild(node: Parser.SyntaxNode, types: string[]): Parser.SyntaxNode | null {
    for (const child of node.children) {
        if (types.includes(child.type)) return child;
    }
    return null;
}

// ----------------------- imports (file-level) -----------------------

function extractImportsTypeScriptOrJavaScript(root: Parser.SyntaxNode): string[] {
    const out: string[] = [];
    for (const child of root.children) {
        if (child.type === 'import_statement') {
            // grammar: import_statement → import_clause? source:string
            const sourceNode = (child as any).childForFieldName?.('source')
                ?? findFirstChild(child, ['string']);
            if (sourceNode && typeof sourceNode.text === 'string') {
                const txt = sourceNode.text.replace(/^['"`]|['"`]$/g, '');
                if (txt) out.push(txt);
            }
        }
    }
    return dedup(out);
}

function extractImportsPython(root: Parser.SyntaxNode): string[] {
    const out: string[] = [];
    for (const child of root.children) {
        if (child.type === 'import_statement') {
            // import a, b.c → list of dotted_name
            for (const sub of child.children) {
                if (sub.type === 'dotted_name') {
                    out.push(dottedNameText(sub));
                } else if (sub.type === 'aliased_import') {
                    const dn = findFirstChild(sub, ['dotted_name']);
                    if (dn) out.push(dottedNameText(dn));
                }
            }
        } else if (child.type === 'import_from_statement') {
            // from a.b import c, d  →  treat the module as the import target;
            // names get added as `a.b.c`, `a.b.d` for higher-resolution graph.
            const moduleNode = (child as any).childForFieldName?.('module_name')
                ?? findFirstChild(child, ['dotted_name', 'relative_import']);
            const moduleName = moduleNode ? dottedNameText(moduleNode) : '';
            const namesAdded: string[] = [];
            for (const sub of child.children) {
                if (sub.type === 'dotted_name' && sub !== moduleNode) {
                    namesAdded.push(dottedNameText(sub));
                } else if (sub.type === 'aliased_import') {
                    const dn = findFirstChild(sub, ['dotted_name']);
                    if (dn) namesAdded.push(dottedNameText(dn));
                }
            }
            if (namesAdded.length > 0 && moduleName) {
                for (const n of namesAdded) out.push(`${moduleName}.${n}`);
            } else if (moduleName) {
                out.push(moduleName);
            } else {
                out.push(...namesAdded);
            }
        }
    }
    return dedup(out);
}

function extractImportsJava(root: Parser.SyntaxNode): string[] {
    const out: string[] = [];
    for (const child of root.children) {
        if (child.type === 'import_declaration') {
            // import_declaration → 'import' (static)? scoped_identifier (.* )? ;
            const scoped = findFirstChild(child, ['scoped_identifier', 'identifier']);
            if (scoped) out.push(scoped.text.replace(/\s+/g, ''));
        }
    }
    return dedup(out);
}

function extractImportsHaxe(root: Parser.SyntaxNode): string[] {
    // tree-sitter-haxe exposes module-level imports as `Import` or
    // `ImportDecl` (varies between grammar versions). Walk children and pick
    // up identifier-shaped children.
    const out: string[] = [];
    for (const child of root.children) {
        const t = child.type;
        if (t === 'Import' || t === 'ImportDecl' || t === 'import' || t === 'import_statement') {
            // Concatenate identifier-like subtokens to reconstruct
            // `pack.Module` or `pack.Module.Symbol`.
            const parts: string[] = [];
            for (const sub of child.children) {
                if (sub.type === 'IDENTIFIER' || sub.type === 'identifier' || sub.type === 'TypePath') {
                    parts.push(sub.text);
                } else if (sub.type === 'PackagePath' || sub.type === 'package_path') {
                    parts.push(sub.text);
                }
            }
            if (parts.length > 0) {
                const joined = parts.join('.').replace(/\s+/g, '');
                if (joined) out.push(joined);
            } else if (typeof child.text === 'string') {
                // Fallback: strip the leading `import` keyword and trailing `;`.
                const stripped = child.text
                    .replace(/^import\s+/, '')
                    .replace(/;\s*$/, '')
                    .trim();
                if (stripped) out.push(stripped);
            }
        }
    }
    return dedup(out);
}

export function extractStructural(root: Parser.SyntaxNode, language: string): FileStructural {
    if (!root) return {};
    const lang = language.toLowerCase();
    try {
        let imports: string[] = [];
        switch (lang) {
            case 'typescript':
            case 'ts':
            case 'javascript':
            case 'js':
                imports = extractImportsTypeScriptOrJavaScript(root);
                break;
            case 'python':
            case 'py':
                imports = extractImportsPython(root);
                break;
            case 'java':
                imports = extractImportsJava(root);
                break;
            case 'haxe':
            case 'hx':
                imports = extractImportsHaxe(root);
                break;
            default:
                return {};
        }
        const filtered = imports.filter(looksLikeIdentifierOrPath);
        return filtered.length > 0 ? { imports: filtered } : {};
    } catch {
        // Grammar mismatch or unknown node shape — degrade silently.
        return {};
    }
}

// imports may be path-shaped strings like `react/jsx-runtime` — we accept
// anything that's a non-empty token without whitespace.
function looksLikeIdentifierOrPath(text: string): boolean {
    if (!text) return false;
    if (/\s/.test(text)) return false;
    return text.length > 0;
}

// ----------------------- extends / implements (class-level) -----------------------

function extractClassTsJs(node: Parser.SyntaxNode): ClassStructural {
    // class_declaration → class_heritage → extends_clause? implements_clause?
    let extendsName: string | undefined;
    const implementsList: string[] = [];
    const heritage = findFirstChild(node, ['class_heritage']);
    if (heritage) {
        for (const part of heritage.children) {
            if (part.type === 'extends_clause' || part.type === 'extends_type_clause') {
                const ident = findFirstChild(part, ['identifier', 'type_identifier', 'member_expression']);
                if (ident) extendsName = ident.text.replace(/\s+/g, '');
            }
            if (part.type === 'implements_clause' || part.type === 'class_implements') {
                for (const sub of part.children) {
                    if (sub.type === 'type_identifier' || sub.type === 'identifier' || sub.type === 'generic_type') {
                        implementsList.push(sub.text.replace(/\s+/g, ''));
                    }
                }
            }
        }
    }
    return packClass(extendsName, implementsList);
}

function extractClassPython(node: Parser.SyntaxNode): ClassStructural {
    // class_definition → name superclasses? body. Python has no "implements".
    let extendsName: string | undefined;
    const argList = findFirstChild(node, ['argument_list']);
    if (argList) {
        for (const sub of argList.children) {
            if (sub.type === 'identifier' || sub.type === 'attribute' || sub.type === 'dotted_name') {
                if (!extendsName) extendsName = sub.text.replace(/\s+/g, '');
            }
        }
    }
    return packClass(extendsName, []);
}

function extractClassJava(node: Parser.SyntaxNode): ClassStructural {
    let extendsName: string | undefined;
    const implementsList: string[] = [];
    for (const child of node.children) {
        if (child.type === 'superclass') {
            const ident = findFirstChild(child, ['type_identifier', 'identifier', 'generic_type', 'scoped_type_identifier']);
            if (ident) extendsName = ident.text.replace(/\s+/g, '');
        } else if (child.type === 'super_interfaces') {
            const list = findFirstChild(child, ['interface_type_list', 'type_list']) || child;
            for (const sub of list.children) {
                if (sub.type === 'type_identifier' || sub.type === 'identifier' || sub.type === 'generic_type' || sub.type === 'scoped_type_identifier') {
                    implementsList.push(sub.text.replace(/\s+/g, ''));
                }
            }
        }
    }
    return packClass(extendsName, implementsList);
}

// rag-graph-supertype-extraction-fix: tree-sitter-haxe@0.4.6 emits `extends` /
// `implements` as bare keyword tokens (no wrapper nodes) directly under
// ClassType, with the type expression (`TypePath`) as the immediately-following
// sibling. The wrapper-node search the previous implementation relied on never
// matched the deployed grammar, silently dropping every Haxe heritage edge.
const HAXE_TYPEPATH_NODE_TYPES = new Set<string>([
    'TypePath', 'type_path',
    'IdentifierTypePath', 'identifier_type_path',
    'TypeName', 'type_name',
    'IDENTIFIER', 'identifier',
]);

// Normalize a Haxe type expression to its bare class identifier:
//   `haxe.ds.BalancedTree<K, V>` → `BalancedTree`
//   `haxe.Constraints.IMap<K, V>` → `IMap`
//   `Array<Map<Int, String>>` → `Array` (depth-balanced strip)
//   `K:EnumValue & Constructible` → null (not identifier-shaped)
//   `foo bar baz` → null (top-level whitespace separates tokens)
// Returns null when the result cannot key into the `by_symbol` index.
export function normalizeTypeName(raw: string): string | null {
    if (!raw) return null;
    const trimmed = raw.trim();
    if (!trimmed) return null;
    // Reject top-level whitespace (allow whitespace inside generic brackets).
    let depthWs = 0;
    for (let i = 0; i < trimmed.length; i++) {
        const c = trimmed[i];
        if (c === '<') depthWs++;
        else if (c === '>') depthWs--;
        else if (depthWs === 0 && /\s/.test(c)) return null;
    }
    // Strip generic suffix via depth-balanced parsing.
    let depth = 0;
    let cutAt = -1;
    for (let i = 0; i < trimmed.length; i++) {
        const c = trimmed[i];
        if (c === '<') {
            if (depth === 0 && cutAt < 0) cutAt = i;
            depth++;
        } else if (c === '>') {
            depth--;
        }
    }
    const noGenerics = depth === 0 && cutAt >= 0 ? trimmed.slice(0, cutAt) : trimmed;
    const lastDot = noGenerics.lastIndexOf('.');
    const out = lastDot >= 0 ? noGenerics.slice(lastDot + 1) : noGenerics;
    return /^[A-Za-z_][A-Za-z0-9_]*$/.test(out) ? out : null;
}

function extractHaxeHeritageFromAst(node: Parser.SyntaxNode): ClassStructural {
    let extendsName: string | undefined;
    const implementsList: string[] = [];
    let expectingExtends = false;
    let expectingImplements = false;
    for (const child of node.children) {
        if (!child) continue;
        const t = child.type;
        if (t === 'extends') {
            expectingExtends = true;
            expectingImplements = false;
            continue;
        }
        if (t === 'implements') {
            expectingImplements = true;
            expectingExtends = false;
            continue;
        }
        if (HAXE_TYPEPATH_NODE_TYPES.has(t)) {
            const norm = normalizeTypeName(child.text);
            if (expectingExtends) {
                if (norm) extendsName = norm;
                expectingExtends = false;
                continue;
            }
            if (expectingImplements) {
                if (norm) implementsList.push(norm);
                continue; // stay in implements mode for comma-separated list
            }
            continue;
        }
        if (t === ',' || t === 'comma') {
            // Comma between TypePaths keeps us in implements mode.
            continue;
        }
        // Anything else (e.g. `{`, another keyword) terminates the heritage list.
        expectingExtends = false;
        expectingImplements = false;
    }
    return packClass(extendsName, implementsList);
}

// Recover heritage from raw source for files the AST cannot dissect — typically
// the ~36% of Haxe stdlib that parses with `hasError=true` due to `#if` /
// `#elseif` conditional compilation, complex docstrings, or `~/regex/` literals.
// Anchored at start-of-line so `class Foo extends Bar` inside a block comment
// (which has leading `* ` or `// `) does not match.
function extractClassHaxeViaRegex(source: string, className: string | undefined): ClassStructural {
    if (!source) return {};
    const namePat = className
        ? className.replace(/[\\^$.|?*+(){}\[\]]/g, '\\$&')
        : '\\w+';
    const headerRe = new RegExp(
        `^\\s*(?:@:?\\w+(?:\\([^)]*\\))?\\s+)*(?:extern\\s+|abstract\\s+|final\\s+|private\\s+|public\\s+)*class\\s+${namePat}\\b([^{]{0,500})`,
        'm',
    );
    const headerMatch = source.match(headerRe);
    if (!headerMatch) return {};
    const tail = headerMatch[1] || '';
    let extendsName: string | undefined;
    const implementsList: string[] = [];
    // Walk left-to-right, alternating between `extends` / `implements` keyword
    // and a following type expression (with optional up-to-2-level generics).
    const heritageRe = /\b(extends|implements)\s+([\w.]+(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>)?)/g;
    let m: RegExpExecArray | null;
    while ((m = heritageRe.exec(tail)) !== null) {
        const keyword = m[1];
        const norm = normalizeTypeName(m[2]);
        if (!norm) continue;
        if (keyword === 'extends') {
            if (!extendsName) extendsName = norm;
        } else {
            implementsList.push(norm);
        }
    }
    return packClass(extendsName, implementsList);
}

function extractHaxeClassName(node: Parser.SyntaxNode): string | undefined {
    for (const child of node.children) {
        if (!child) continue;
        if (
            child.type === 'type_name' || child.type === 'TypeName' ||
            child.type === 'identifier' || child.type === 'IDENTIFIER'
        ) {
            const txt = (child.text || '').replace(/\s+/g, '');
            if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(txt)) return txt;
        }
    }
    return undefined;
}

export function extractClassHaxe(node: Parser.SyntaxNode): ClassStructural {
    const astResult = extractHaxeHeritageFromAst(node);
    if (astResult.extends || (astResult.implements && astResult.implements.length > 0)) {
        return astResult;
    }
    // Cheap pre-check before invoking the regex fallback.
    const src = node.text || '';
    if (!/class\s+\w+[\s\S]{0,500}?\b(extends|implements)\b/.test(src)) {
        return astResult;
    }
    const className = extractHaxeClassName(node);
    return extractClassHaxeViaRegex(src, className);
}

function packClass(extendsName: string | undefined, implementsList: string[]): ClassStructural {
    const out: ClassStructural = {};
    if (extendsName && looksLikeIdentifier(extendsName)) {
        out.extends = extendsName;
    }
    const cleanedImpl = dedup(implementsList).filter(looksLikeIdentifier);
    if (cleanedImpl.length > 0) {
        out.implements = cleanedImpl;
    }
    return out;
}

export function extractClassStructural(node: Parser.SyntaxNode, language: string): ClassStructural {
    if (!node) return {};
    const lang = language.toLowerCase();
    try {
        switch (lang) {
            case 'typescript':
            case 'ts':
            case 'javascript':
            case 'js':
                return extractClassTsJs(node);
            case 'python':
            case 'py':
                return extractClassPython(node);
            case 'java':
                return extractClassJava(node);
            case 'haxe':
            case 'hx':
                return extractClassHaxe(node);
            default:
                return {};
        }
    } catch {
        return {};
    }
}
