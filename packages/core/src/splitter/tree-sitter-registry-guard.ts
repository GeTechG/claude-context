/**
 * Repairs `tree-sitter`'s prototype patches after a second execution of its
 * JS wrapper in one process. Issue #75.
 *
 * ## The defect (upstream, `tree-sitter@0.25.0/index.js`)
 *
 * The wrapper patches methods onto the *native* `Tree` / `TreeCursor` classes
 * by first capturing the native implementation off the prototype and then
 * replacing it:
 *
 *     const {rootNode, rootNodeWithOffset, edit} = Tree.prototype;   // line 13
 *     Object.defineProperty(Tree.prototype, 'rootNode', {
 *       get() { if (this instanceof Tree && rootNode) { ... } },     // line 15
 *       configurable: true,
 *     });
 *
 * That is not idempotent. On a **second** execution of the same file in the
 * same process, the destructuring on line 13 no longer reads the native
 * method — it *invokes the accessor installed by the first execution*, with
 * `this === Tree.prototype`. `Tree.prototype instanceof Tree` is false, so the
 * getter falls through and yields `undefined`. The second execution then
 * installs a getter whose captured `rootNode` is `undefined`, and its
 * `&& rootNode` guard makes it return `undefined` for **every** tree, forever.
 *
 * The same file does the same thing three more times — `TreeCursor.prototype`'s
 * `currentNode` / `startPosition` / `endPosition` (line 395) and, worse,
 * `Parser.prototype`'s `parse` and `setLanguage` (line 346). `parse` does not
 * fail loudly at all: the second execution's wrapper takes
 * `(input, oldTree, {bufferSize, includedRanges})`, and calls the FIRST
 * execution's wrapper with those three positionally, so the options object
 * arrives where the native `parse` expects nothing and `includedRanges` and
 * `bufferSize` are silently dropped. A parse restricted to one range then
 * returns the whole file. That is a **silent wrong answer**, which is worse in
 * kind than the undefined root, and it is invisible to any test that calls
 * `parser.parse(source)` with no options (local-rag #75, review).
 *
 * Nothing recovers on its own: the native method is only ever reachable through
 * the first execution's closure.
 *
 * ## Why jest triggers it
 *
 * Jest gives every *test file* a fresh module registry, so `tree-sitter/index.js`
 * is executed once per test file. The native `.node` addon is loaded through the
 * real `require` and stays cached for the process, so `Tree`, `TreeCursor` and
 * `Parser` are the *same objects* across all registries — the second test file's
 * execution clobbers the accessors the first one installed. In practice exactly
 * one test file per process ever sees a working `tree.rootNode`: the first one
 * to load `tree-sitter`. Which file that is depends on jest's sequencer order,
 * which is driven by cached per-file timings — so the failure moves around and
 * looks like a flake under CPU load. It is not a timeout and not a worker-pool
 * artefact (`--runInBand` has been the configured runner since 4062e7e).
 *
 * ## The repair
 *
 * The first execution's descriptors *are* correct. We stash them on the shared
 * native `Parser` object — the one thing that survives a module-registry reset —
 * the first time this guard runs, and reinstall them on every later run. Later
 * executions' `unmarshalNode` is never used, which is also why grammars stay
 * consistent: `Parser.prototype.setLanguage` initialises `language.nodeSubclasses`
 * only once per grammar object, so the node classes always match the stashed
 * first-execution closure.
 *
 * WHAT IS RESTORED IS DERIVED, NOT LISTED (local-rag #75, review). The first
 * version of this guard named the members it knew about — `Tree` and
 * `TreeCursor` — and left `Parser.prototype.parse` double-wrapped, which is how
 * the `includedRanges` defect above survived the fix for the one beside it. A
 * list of members is a list somebody has to keep in step with a dependency's
 * internals, and it will fall behind again on the next release. So the guard
 * captures EVERY own property descriptor of every prototype the wrapper can
 * reach — the exported `Parser` and every class hanging off it — and reinstalls
 * whichever ones a later execution changed. Restoring a descriptor that never
 * moved is a no-op, so being broad costs nothing and being narrow cost a silent
 * wrong answer.
 *
 * Wired in as `setupFiles` in jest.config.cjs, so it runs before every test
 * file's imports. Production loads the wrapper exactly once and is unaffected;
 * this is deliberately not imported by any production module.
 */

import Parser from 'tree-sitter';

/** Property key for the stash, on the process-shared native `Parser` object. */
const STASH_KEY = '__claudeContextPristineTreeSitterDescriptors';

type Descriptors = Array<[string | symbol, PropertyDescriptor]>;

/**
 * Every prototype the wrapper patches, derived rather than named: the exported
 * `Parser` itself plus every constructor hanging off it (`Tree`, `TreeCursor`,
 * `Query`, `LookaheadIterator`, …). A class the dependency adds tomorrow is
 * covered without editing this file.
 */
function prototypes(): object[] {
    const anchor = Parser as any;
    const out: object[] = [];
    const seen = new Set<object>();
    const add = (ctor: any) => {
        if (typeof ctor !== 'function' || !ctor.prototype || seen.has(ctor.prototype)) return;
        seen.add(ctor.prototype);
        out.push(ctor.prototype);
    };
    add(anchor);
    for (const key of Object.getOwnPropertyNames(anchor)) {
        if (key === 'prototype' || key === 'caller' || key === 'arguments') continue;
        let value: unknown;
        // A getter on the binding could throw; a prototype we cannot read is one we
        // cannot guard, and that is not a reason to fail the whole repair.
        try { value = anchor[key]; } catch { continue; }
        add(value);
    }
    return out;
}

function descriptorsOf(proto: object): Descriptors {
    const out: Descriptors = [];
    for (const key of Reflect.ownKeys(proto)) {
        if (key === 'constructor') continue;
        const descriptor = Object.getOwnPropertyDescriptor(proto, key);
        if (descriptor) out.push([key, descriptor]);
    }
    return out;
}

function sameDescriptor(a: PropertyDescriptor | undefined, b: PropertyDescriptor | undefined): boolean {
    if (!a || !b) return a === b;
    return a.get === b.get && a.set === b.set && a.value === b.value;
}

export type GuardResult = 'captured' | 'repaired' | 'intact' | 'unavailable';

/**
 * Capture the first execution's `tree-sitter` prototype patches, or reinstall
 * them if a later execution has overwritten them. Idempotent and cheap.
 */
export function guardTreeSitterModuleRegistry(): GuardResult {
    const anchor = Parser as any;
    const protos = prototypes();
    if (protos.length === 0) return 'unavailable';

    const stashed: Array<[object, Descriptors]> | undefined = anchor[STASH_KEY];
    if (!stashed) {
        Object.defineProperty(anchor, STASH_KEY, {
            value: protos.map((proto) => [proto, descriptorsOf(proto)] as [object, Descriptors]),
            configurable: true, enumerable: false, writable: true,
        });
        return 'captured';
    }

    let repaired = false;
    for (const [proto, descriptors] of stashed) {
        for (const [key, want] of descriptors) {
            if (sameDescriptor(Object.getOwnPropertyDescriptor(proto, key), want)) continue;
            // A non-configurable property cannot be put back, and refusing to repair the
            // rest because of one would be the worse outcome; it is reported by staying
            // different rather than by throwing here.
            try { Object.defineProperty(proto, key, want); repaired = true; } catch { /* not configurable */ }
        }
    }
    return repaired ? 'repaired' : 'intact';
}

// Importing the guard installs it — this module is a jest `setupFiles` entry.
guardTreeSitterModuleRegistry();
