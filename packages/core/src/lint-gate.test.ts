// The lint gate (local-rag #129).
//
// #122 restored the ability to lint this package after it had been unlintable
// for the whole period its pin was moving; it reported 96 errors and fixed none
// of them, deliberately. #129 cleared them. This file is what stops them coming
// back: without it, "the lint runs" is a fact about a command nobody is obliged
// to type, which is the same defect as a lint that cannot start.
//
// What it asserts:
//
//   - zero findings at ERROR severity, over every package's `src`, under the
//     rule set the package ships. Not a rule set this file chooses: it runs
//     `eslint.config.js` from the submodule root, so weakening a rule to make
//     this test pass is visible as a change to that file.
//   - zero unused eslint-disable directives. A suppression that suppresses
//     nothing reads as a considered exception while granting none, and the
//     finding it was written for fires unattended beside it. Three of those had
//     accumulated by #129.
//   - that the file pattern still reaches every package. A gate whose glob has
//     stopped matching is green for the wrong reason, and that is the failure
//     mode this whole file exists to answer.
//
// It asserts NOTHING about the warning count. The shipped set puts
// `@typescript-eslint/no-explicit-any` at `warn` for several hundred deliberate
// occurrences; a test that held the warning count at zero would be answered by
// turning that rule off, which is the hollowing this gate is against.
//
// Findings that cannot be fixed without changing behaviour are not fixed: they
// carry a line-scoped `eslint-disable-next-line` with the reason beside it, and
// the change that added them lists them. That is how a site leaves this gate.
//
// eslint runs as a child process rather than through its Node API: the API's
// glob search uses a dynamic import, which jest's CommonJS VM refuses without
// --experimental-vm-modules. The child process is also the command a person
// runs, so this asserts the same thing `pnpm lint` reports.

import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// .../packages/core/src -> the submodule root
const ROOT = path.resolve(__dirname, '..', '..', '..');
const CONFIG = path.join(ROOT, 'eslint.config.js');
const PATTERN = 'packages/*/src/**/*.ts';

// eslint's own bin, beside the entry point node resolves for us — no assumption
// about where this pnpm layout puts the package.
const ESLINT_BIN = path.resolve(path.dirname(require.resolve('eslint')), '..', 'bin', 'eslint.js');

interface EslintMessage {
    ruleId: string | null;
    severity: number;
    message: string;
    line?: number;
    column?: number;
}

interface EslintResult {
    filePath: string;
    messages: EslintMessage[];
    errorCount: number;
    warningCount: number;
}

interface Finding {
    where: string;
    rule: string;
    message: string;
}

interface LintRun {
    errors: Finding[];
    unusedDirectives: Finding[];
    warningCount: number;
    packagesSeen: string[];
}

function packagesWithSources(): string[] {
    const packagesDir = path.join(ROOT, 'packages');
    return fs.readdirSync(packagesDir)
        .filter((name) => fs.existsSync(path.join(packagesDir, name, 'src')))
        .sort();
}

function runEslint(): EslintResult[] {
    let stdout: string;
    try {
        stdout = execFileSync(
            process.execPath,
            [ESLINT_BIN, '--config', CONFIG, PATTERN, '-f', 'json'],
            { cwd: ROOT, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'] },
        );
    } catch (err) {
        // eslint exits 1 when it has findings; that is a successful run with a
        // report on stdout. Anything with no parseable report is a broken run,
        // and a broken run must fail this gate rather than pass it empty.
        const failure = err as { stdout?: string; stderr?: string; message?: string };
        stdout = failure.stdout ?? '';
        if (!stdout.trim().startsWith('[')) {
            throw new Error(`eslint did not produce a report: ${failure.stderr || failure.message}`);
        }
    }
    const parsed = JSON.parse(stdout) as EslintResult[];
    if (parsed.length === 0) throw new Error(`eslint linted no files for pattern ${PATTERN}`);
    return parsed;
}

function collect(results: EslintResult[]): LintRun {
    const errors: Finding[] = [];
    const unusedDirectives: Finding[] = [];
    let warningCount = 0;
    const packagesSeen = new Set<string>();

    for (const result of results) {
        const relative = path.relative(ROOT, result.filePath);
        const owner = relative.split(path.sep)[1];
        if (owner) packagesSeen.add(owner);
        warningCount += result.warningCount;
        for (const message of result.messages) {
            const finding: Finding = {
                where: `${relative}:${message.line}:${message.column}`,
                rule: message.ruleId ?? '(unused disable directive)',
                message: message.message,
            };
            if (message.severity === 2) errors.push(finding);
            else if (message.ruleId === null) unusedDirectives.push(finding);
        }
    }

    return { errors, unusedDirectives, warningCount, packagesSeen: [...packagesSeen].sort() };
}

function render(findings: Finding[]): string {
    return findings.map((f) => `${f.where}  ${f.rule}  ${f.message}`).join('\n');
}

// The two classes of finding that decide the verdict. The warning count is not
// one of them, and the test below proves that rather than asserting it in prose.
function blocking(run: LintRun): Finding[] {
    return [...run.errors, ...run.unusedDirectives];
}

describe('the vendored package lints clean (#129)', () => {
    let run: LintRun;

    beforeAll(() => {
        run = collect(runEslint());
    }, 300000);

    it('reports zero errors under the rule set the package ships', () => {
        expect(render(run.errors)).toBe('');
        expect(run.errors).toHaveLength(0);
    });

    it('has no eslint-disable directive that suppresses nothing', () => {
        expect(render(run.unusedDirectives)).toBe('');
        expect(run.unusedDirectives).toHaveLength(0);
    });

    it('still reaches every package, so a green run is not an empty one', () => {
        expect(run.packagesSeen).toEqual(packagesWithSources());
    });

    it('does not gate on the warning count, at any value', () => {
        // The first version of this test asserted `warningCount > 0`, under a name
        // claiming the gate says nothing about warnings. It said the opposite:
        // clearing all 366 `no-explicit-any` warnings honestly — without touching
        // the rule — would have reddened the gate. A check that punishes the
        // improvement it is supposed to allow is worse than no check.
        const cleared: LintRun = { ...run, errors: [], unusedDirectives: [], warningCount: 0 };
        const noisier: LintRun = { ...run, errors: [], unusedDirectives: [], warningCount: 10000 };
        expect(blocking(cleared)).toEqual([]);
        expect(blocking(noisier)).toEqual([]);

        // …and an error still blocks, whatever the warning count is.
        const oneError: LintRun = {
            ...run,
            errors: [{ where: 'a.ts:1:1', rule: 'prefer-const', message: 'x' }],
            unusedDirectives: [],
            warningCount: 0,
        };
        expect(blocking(oneError)).toHaveLength(1);
    });
});
