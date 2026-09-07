module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'node',
    testMatch: ['<rootDir>/src/**/*.test.ts'],
    // Issue #75: `tree-sitter/index.js` is not idempotent, and jest re-executes
    // it in every test file's fresh module registry while the native addon stays
    // cached for the process — the second execution permanently installs a
    // `Tree.prototype.rootNode` getter that returns `undefined` for every tree.
    // This setup file captures the first execution's prototype patches and
    // reinstalls them before each test file runs. See the module for the full
    // mechanism.
    setupFiles: ['<rootDir>/src/splitter/tree-sitter-registry-guard.ts'],
};
