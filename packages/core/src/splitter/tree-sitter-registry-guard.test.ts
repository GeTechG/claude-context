// Issue #75 — regression test for the tree-sitter module-registry defect.
//
// WHAT WAS REPRODUCED (2026-09-07): `npx jest --runInBand` in this package,
// under 32 busy loops on 24 cores, failed 8 tests in
// `src/splitter/ast-splitter.imports.test.ts` on run 3 of 4
// (`Tests: 8 failed, 349 passed, 357 total`) — `Received: undefined` from
// `extractStructural` and `Received: null` from `findFirst`, i.e. the parse
// came back empty. Adding two throwaway test files that merely parse made the
// same 8 failures deterministic with no load at all, which is what turned the
// flake into a mechanism: the failure follows *test-file ordering*, and load
// only perturbs the ordering (jest's sequencer sorts by cached per-file
// timings).
//
// THE MECHANISM: `tree-sitter@0.25.0/index.js` patches the native `Tree` /
// `TreeCursor` prototypes non-idempotently. Its second execution in one
// process captures `undefined` as the underlying native `rootNode` — because
// destructuring `Tree.prototype` now *invokes* the first execution's accessor
// with `this === Tree.prototype`, which fails that accessor's `instanceof`
// guard — and installs a getter that returns `undefined` for every tree.
// Jest resets the JS module registry per test file while the native `.node`
// addon stays cached for the process, so exactly one test file per process
// gets a working `tree.rootNode`: whichever loads `tree-sitter` first.
// See `tree-sitter-registry-guard.ts` for the annotated source lines.
//
// WHAT WAS RULED OUT: a timeout (the assertions fail on `undefined`, not on
// time; the suite takes the same 22-26 s in the failing and passing runs); the
// jest worker pool (`--runInBand` since 4062e7e — one worker); a collected
// `Tree`/`Parser` (the owner walked 200 dropped-reference roots after a forced
// `global.gc()` and they were fine); and `node.children` "transiently returning
// holes" (the defensive index walk in ast-splitter.imports.test.ts) — the tree
// is not holed, `tree.rootNode` is `undefined` before any walk starts.
//
// WHY THIS FIX: the defect is in a vendored dependency's module initialisation,
// and it is only *triggered* by jest's per-file registry reset — production
// loads the wrapper once. So the fix restores the first execution's prototype
// patches from a stash on the process-shared native `Parser` object, wired in
// as `setupFiles`. Raising a timeout would not have moved this at all.
//
// The test below re-creates the second execution deliberately, so it fails
// without the guard regardless of file ordering or machine load.

import Parser from 'tree-sitter';
import { guardTreeSitterModuleRegistry } from './tree-sitter-registry-guard';

const TypeScript = require('tree-sitter-typescript').typescript;

const SOURCE = "import { foo } from 'lodash';\nclass A extends B {}\n";

function parseRoot(): any {
    const parser = new Parser();
    parser.setLanguage(TypeScript);
    return (parser.parse(SOURCE) as any).rootNode;
}

// The first statement of SOURCE, as a byte range. `parse` with `includedRanges` must
// see ONLY that, so a correct parse has exactly one child.
const FIRST_STATEMENT_ONLY = {
    includedRanges: [{
        startIndex: 0,
        endIndex: SOURCE.indexOf('\n'),
        startPosition: { row: 0, column: 0 },
        endPosition: { row: 0, column: SOURCE.indexOf('\n') },
    }],
};

function parseFirstStatementOnly(): any {
    const parser = new Parser();
    parser.setLanguage(TypeScript);
    return (parser.parse(SOURCE, null as any, FIRST_STATEMENT_ONLY as any) as any).rootNode;
}

/** Re-execute `tree-sitter/index.js` the way a fresh jest test file does. */
function reExecuteTreeSitterWrapper(): void {
    jest.resetModules();
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('tree-sitter');
}

describe('tree-sitter module-registry guard (#75)', () => {
    afterEach(() => {
        // Leave the process in the repaired state for the rest of the file.
        guardTreeSitterModuleRegistry();
    });

    it('parses normally to begin with', () => {
        const root = parseRoot();
        expect(root).toBeDefined();
        expect(root.type).toBe('program');
        expect(root.childCount).toBeGreaterThan(0);
    });

    it('re-executing the tree-sitter wrapper breaks tree.rootNode (the upstream defect)', () => {
        reExecuteTreeSitterWrapper();
        // Documents the defect this guard exists for. If a future tree-sitter
        // release makes its prototype patching idempotent, this expectation
        // flips and the guard can be deleted — that is the intended signal.
        expect(parseRoot()).toBeUndefined();
    });

    it('the guard restores tree.rootNode after a re-execution', () => {
        reExecuteTreeSitterWrapper();
        expect(parseRoot()).toBeUndefined();

        expect(['repaired', 'captured', 'intact']).toContain(guardTreeSitterModuleRegistry());

        const root = parseRoot();
        expect(root).toBeDefined();
        expect(root.type).toBe('program');
        expect(root.childCount).toBe(2);
        expect(root.child(0).type).toBe('import_statement');
        // The whole walk, not just the root: this is what the imports tests do.
        expect(root.children.map((c: any) => c.type))
            .toEqual(['import_statement', 'class_declaration']);
    });

    it('the guard is idempotent when nothing has been clobbered', () => {
        guardTreeSitterModuleRegistry();
        expect(guardTreeSitterModuleRegistry()).toBe('intact');
        expect(parseRoot().type).toBe('program');
    });

    it('parse options survive a re-execution: includedRanges is still honoured', () => {
        // local-rag #75, REVIEW. The first version of this guard restored `Tree` and
        // `TreeCursor` and left `Parser.prototype.parse` double-wrapped. That one does
        // not fail loudly: the second execution's wrapper calls the first execution's
        // with `(input, oldTree, options)` positionally, so the native `parse` never
        // sees the options and `includedRanges` and `bufferSize` are silently dropped —
        // a parse restricted to one range quietly returns the whole file. Every test
        // above calls `parser.parse(SOURCE)` with no options, so none of them could see
        // it. This is the assertion that would have.
        expect(parseFirstStatementOnly().childCount).toBe(1);
        reExecuteTreeSitterWrapper();
        guardTreeSitterModuleRegistry();
        const root = parseFirstStatementOnly();
        expect(root).toBeDefined();
        expect(root.childCount).toBe(1);
        expect(root.child(0).type).toBe('import_statement');
    });

    it('the guard is WIRED IN, not merely present', () => {
        // The three tests above call the guard themselves, so they pass whether or not
        // anything runs it before each test file — and running it before each test file
        // is the entire fix. What breaks the suite is a `setupFiles` entry somebody
        // removes while refactoring the config; the guard module would still be here,
        // still unit-tested, still green, and `ast-splitter.imports.test.ts` would go
        // back to failing under load. So the wiring is asserted too.
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const config = require('../../jest.config.cjs');
        expect(config.setupFiles || []).toEqual(
            expect.arrayContaining([expect.stringContaining('tree-sitter-registry-guard')]),
        );
    });
});
