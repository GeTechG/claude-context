/**
 * show-the-reranker-the-file-path: does a question ask for a dependency
 * relation of a code entity ("what calls X", "what inherits from X", "what
 * would break if X changed")?
 *
 * Read from the question's wording only — no source name, no dataset id. The
 * reranker keeps the content-only passage for such a question under
 * `RERANKER_PASSAGE_PATH=unless_relational`, because on it the file that
 * DEFINES the subject carries the subject's name in its path while its callers
 * do not (`infra/cross-corpus-runs/rerank-path-text-2026-09-30/`). A false
 * positive only forgoes the path's gain; a false negative puts the path on a
 * relational question.
 */
const RELATIONAL_PATTERNS: RegExp[] = [
    // a wh-word followed, inside one clause, by a relation verb in its active form
    /\b(what|which|who)\b[^.?!;,]{0,60}?\b(calls|call|invokes?|uses|references|depends\s+on|inherits?|extends|implements|subclass(es)?|overrides?|imports|instantiates?|derives?\s+from)\b/i,
    // a plural or base-form subject: which classes implement X, what files use X, what does X depend on
    /\b(what|which)\b[^.?!;,]{0,40}?\b(classes|types|files|modules|functions|methods|scripts|nodes|components|objects|structs|interfaces|symbols|places|code)\s+(use|call|invoke|reference|extend|implement|import|override|instantiate|depend\s+on|inherit\s+from|derive\s+from)\b/i,
    /\b(what|which)\b[^.?!;,]{0,40}?\b(does|do)\b[^.?!;,]{0,60}?\bdepend\s+on\b/i,
    /\b(what|which)\b[^.?!;,]{0,40}?\b(is|are)\s+(derived|inherited)\s+from\b/i,
    // the relation stated as a noun: callers of X, call sites of X, usages of X, dependents of X
    /\b(callers?|call\s+sites?|usages?|dependents|subclasses|implementers|derived\s+(types|classes))\s+(of|for)\b/i,
    // the passive with an interrogative: called by what / referenced from where
    /\b(called|invoked|referenced|used|inherited|extended|implemented|overridden|imported)\s+(by|from)\s+(what|which|whom?|where)\b/i,
    /\bwhere\s+(is|are)\b[^.?!;,]{0,40}?\b(used|called|referenced|instantiated)\b/i,
    /\bwho\s+(calls|uses|references|inherits|extends|implements)\b/i,
    // impact: what would break / what would a change to X affect
    /\bwhat\s+(would|will|could|might|does)\s+break\b/i,
    /\b(what|which)\b[^.?!;]{0,80}?\b(would|will|could|might)\b[^.?!;]{0,60}?\b(affect|break|impact)\b/i,
    /\b(impact|blast\s+radius)\s+of\s+(changing|renaming|removing|deleting)\b/i,
    // Russian: кто вызывает / что использует / что сломается
    /(^|[\s«"])(кто|что|какие|какой|где)\s[^.?!;,]{0,60}?(вызывает|вызывают|использует|используют|наследу|реализу|ссылается|ссылаются|зависит|зависят)/i,
    /что\s+сломается/i,
];

export function isRelationalQuery(query: string | undefined | null): boolean {
    if (!query) return false;
    return RELATIONAL_PATTERNS.some((p) => p.test(query));
}
