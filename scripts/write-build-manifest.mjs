#!/usr/bin/env node
// Writes <package>/dist/.build-manifest.json — the statement of what this
// build compiled (#132).
//
// The arms, the MCP server and anything consuming a package of this repo run
// its BUILT output; only the jest suite runs the source. dist is gitignored,
// so a stale build produces no diff and nothing could tell anyone — #132 fell
// through exactly that hole: source landed for #111/#108/#75 and was never
// compiled, and every consumer kept running the old artifact while the
// candidate stamp recorded the new pin beside the old bytes.
//
// This script runs as the last step of each package's `build` script, so every
// path that compiles also restates what it compiled: a sha256 per source file
// under <package>/src, beside the output it produced. The repo's freshness
// check (infra/lib/dist-freshness.js) recomputes the same map and refuses a
// run that is about to spawn arm drivers against an artifact that no longer
// matches.
//
// The manifest's content is a pure function of the source bytes: no
// timestamps, no compiler version, no absolute paths. It lives inside the
// directory the candidate identity hashes, so a rebuild of an unchanged tree
// must produce a byte-identical manifest — a no-op rebuild must not move the
// candidate identity.
//
// The build that writes no manifest is not fresh either: running bare
// `tsc --build --force` leaves a manifest that no longer matches the source,
// and the check refuses it. The documented rebuild command is the scripted
// one:
//
//   cd patches/claude-context && pnpm build:core && pnpm build:mcp

import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCHEMA = 'claude-context-build-manifest-v1';
const MANIFEST_NAME = '.build-manifest.json';

// The vendored root, resolved from THIS file so the script works from any cwd.
const VENDORED_ROOT = fileURLToPath(new URL('..', import.meta.url));

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function walkFiles(dir, visitor) {
  for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
    const child = join(dir, entry.name);
    if (entry.isDirectory()) walkFiles(child, visitor);
    else if (entry.isFile()) visitor(child);
  }
}

function relKeys(base, files) {
  const map = {};
  for (const file of files) {
    const key = relative(base, file).split(sep).join('/');
    map[key] = sha256(readFileSync(file));
  }
  return map;
}

// Every file under the package's src/, keyed package-relatively with forward
// slashes, so the map is identical across platforms and checkouts.
function sourceMap(pkgDir) {
  const files = [];
  walkFiles(join(pkgDir, 'src'), (file) => files.push(file));
  return relKeys(pkgDir, files);
}

// Every loadable output the build emitted, keyed dist-relatively. `.node`
// binaries are loadable too and travel with the artifact, so they are
// authenticated by bytes; they pair to no source. The manifest itself is the
// statement, not a compiled product, and is excluded — it cannot hash itself.
function outputMap(pkgDir) {
  const distDir = join(pkgDir, 'dist');
  const files = [];
  walkFiles(distDir, (file) => {
    const key = relative(distDir, file).split(sep).join('/');
    if (key === MANIFEST_NAME) return;
    if (/\.(js|json|node)$/.test(key)) files.push(file);
  });
  return relKeys(distDir, files);
}

function main() {
  const packageArg = process.argv[2];
  if (!packageArg) {
    console.error(`usage: write-build-manifest.mjs <package-dir-relative-to-vendored-root>  (e.g. packages/core)`);
    process.exit(1);
  }
  const pkgDir = join(VENDORED_ROOT, packageArg);
  if (!statSync(join(pkgDir, 'src'), { throwIfNoEntry: false })?.isDirectory()) {
    console.error(`write-build-manifest: ${packageArg}/src does not exist; nothing to state`);
    process.exit(1);
  }
  if (!statSync(join(pkgDir, 'dist'), { throwIfNoEntry: false })?.isDirectory()) {
    console.error(`write-build-manifest: ${packageArg}/dist does not exist; run the build before writing the manifest`);
    process.exit(1);
  }
  const manifest = { schema: SCHEMA, package: packageArg, sources: sourceMap(pkgDir), outputs: outputMap(pkgDir) };
  const target = join(pkgDir, 'dist', MANIFEST_NAME);
  writeFileSync(target, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`[build-manifest] wrote ${MANIFEST_NAME} (${Object.keys(manifest.sources).length} source files, ${Object.keys(manifest.outputs).length} outputs) for ${packageArg}`);
}

main();
