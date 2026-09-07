// Flat config for eslint 9 (local-rag #122).
//
// This replaces `.eslintrc.js`, which was broken two independent ways and so
// had not been run since before the eslint 9 bump:
//
//   1. eslint >= 9 looks for `eslint.config.*` and refuses to start without one.
//   2. Under `ESLINT_USE_FLAT_CONFIG=false` it then failed on
//      `couldn't find the config "@typescript-eslint/recommended"` — in eslintrc
//      syntax that string names a shareable config *package*; the plugin config
//      is `plugin:@typescript-eslint/recommended`. So the extends was malformed
//      for eslint 8 too.
//
// The rule set below is what `.eslintrc.js` meant to express, unchanged: eslint's
// recommended rules, @typescript-eslint's recommended rules (with its
// `eslint-recommended` overrides that switch off the core rules TypeScript makes
// redundant), and the four explicit rule overrides that were already there.
//
// `.eslintrc.js` is deleted rather than repaired: a flat config takes precedence
// whenever it exists, so a corrected eslintrc beside it would be a file nothing
// loads — which is how the malformed extends survived unnoticed.

const { createRequire } = require('node:module');

// `@eslint/js` is not a direct dependency and pnpm's strict layout keeps it out
// of the root `node_modules`, so it is not resolvable from here. Reach it
// through eslint's own require instead of adding a devDependency: eslint depends
// on `@eslint/js` at its exact own version, so the two can never disagree.
const js = createRequire(require.resolve('eslint'))('@eslint/js');

const tseslint = require('@typescript-eslint/eslint-plugin');
const tsParser = require('@typescript-eslint/parser');

// `.eslintrc.js` used `env: { node: true, es6: true }`. Flat config has no `env`,
// so the globals are named. Jest globals are included because the test files
// linted here are jest suites.
const globals = {
    // node
    __dirname: 'readonly',
    __filename: 'readonly',
    Buffer: 'readonly',
    URL: 'readonly',
    URLSearchParams: 'readonly',
    TextDecoder: 'readonly',
    TextEncoder: 'readonly',
    AbortController: 'readonly',
    AbortSignal: 'readonly',
    fetch: 'readonly',
    console: 'readonly',
    exports: 'writable',
    global: 'readonly',
    module: 'writable',
    process: 'readonly',
    require: 'readonly',
    setTimeout: 'readonly',
    clearTimeout: 'readonly',
    setInterval: 'readonly',
    clearInterval: 'readonly',
    setImmediate: 'readonly',
    clearImmediate: 'readonly',
    queueMicrotask: 'readonly',
    structuredClone: 'readonly',
    // jest
    afterAll: 'readonly',
    afterEach: 'readonly',
    beforeAll: 'readonly',
    beforeEach: 'readonly',
    describe: 'readonly',
    expect: 'readonly',
    it: 'readonly',
    jest: 'readonly',
    test: 'readonly',
};

module.exports = [
    {
        // `.eslintrc.js`'s ignorePatterns. Its `'*.js'` entry is dropped in
        // favour of scoping the rules block below to TypeScript: in flat config a
        // bare `'*.js'` ignore would also swallow this config file.
        ignores: [
            '**/dist/**',
            '**/out/**',
            '**/build/**',
            '**/node_modules/**',
            '**/*.d.ts',
        ],
    },
    {
        files: ['**/*.ts', '**/*.tsx'],
        languageOptions: {
            parser: tsParser,
            ecmaVersion: 2020,
            sourceType: 'module',
            globals,
        },
        plugins: {
            '@typescript-eslint': tseslint,
        },
        linterOptions: {
            // A disable comment naming a rule that no longer exists suppresses
            // nothing, which is the same defect class as an unrunnable config.
            reportUnusedDisableDirectives: true,
        },
        rules: {
            ...js.configs.recommended.rules,
            ...tseslint.configs['eslint-recommended'].overrides[0].rules,
            ...tseslint.configs.recommended.rules,
            '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
            '@typescript-eslint/explicit-function-return-type': 'off',
            '@typescript-eslint/explicit-module-boundary-types': 'off',
            '@typescript-eslint/no-explicit-any': 'warn',
        },
    },
];
