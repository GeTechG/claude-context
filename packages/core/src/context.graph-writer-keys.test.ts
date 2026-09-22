// serve-a-symbol-graph-of-the-served-index: the graph writer keeps every symbol name as
// data. Keyed in a plain `{}`, a symbol named `constructor` read the inherited
// Object.prototype.constructor, was dropped from the graph, and had its fields written
// onto the global `Object`. Drives the real writer and the real loader; no Milvus.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { GraphIndex } from './search/graph-expansion';
import { VectorDatabase } from './vectordb';

class TestEmbedding extends Embedding {
    protected maxTokens = 8192;
    async detectDimension(): Promise<number> { return 3; }
    async embed(_text: string): Promise<EmbeddingVector> { return { vector: [1, 0, 0], dimension: 3 }; }
    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> { return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 })); }
    getDimension(): number { return 3; }
    getProvider(): string { return 'test'; }
}

const INHERITED = ['constructor', 'toString', 'valueOf', 'hasOwnProperty', '__proto__', 'isPrototypeOf'];

describe('graph writer: symbol names are data', () => {
    let dir = '';
    beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'graph-writer-keys-')); });
    afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

    it('writes and loads back every inherited-member name, and leaves the runtime untouched', async () => {
        const ctx: any = new Context({ embedding: new TestEmbedding(), vectorDatabase: {} as VectorDatabase });
        const chunks = [
            ...INHERITED.map((name, i) => ({
                chunkId: `chunk_${String(i).padStart(16, '0')}`,
                relativePath: `lib/pkg/${name}.js`,
                contentType: 'code',
                symbolName: name,
                parentSymbol: '__proto__',
            })),
            { chunkId: 'chunk_ordinary000000', relativePath: 'lib/pkg/Ordinary.js', contentType: 'code', symbolName: 'Ordinary' },
        ];
        ctx.graphAccumulator = { snapshot: () => chunks };

        const objectKeysBefore = Object.getOwnPropertyNames(Object).sort();
        const protoKeysBefore = Object.getOwnPropertyNames(Object.prototype).sort();

        await ctx.persistGraphSideIndex(dir);

        expect(Object.getOwnPropertyNames(Object).sort()).toEqual(objectKeysBefore);
        expect(Object.getOwnPropertyNames(Object.prototype).sort()).toEqual(protoKeysBefore);
        expect((Object as any).canonical_chunk_ids).toBeUndefined();

        const file = path.join(dir, '.symbols-graph.json');
        const raw = JSON.parse(fs.readFileSync(file, 'utf-8'));
        for (const name of [...INHERITED, 'Ordinary']) {
            expect(Object.prototype.hasOwnProperty.call(raw.by_symbol, name)).toBe(true);
        }
        // A package derived from `parent_symbol` named `__proto__` is a bucket, not a prototype.
        expect(Object.prototype.hasOwnProperty.call(raw.by_package, '__proto__')).toBe(true);

        const idx = GraphIndex.load(file)!;
        expect(idx).not.toBeNull();
        INHERITED.forEach((name, i) => {
            expect(idx.lookupSymbol(name)?.canonical_chunk_ids).toEqual([`chunk_${String(i).padStart(16, '0')}`]);
        });
        expect(idx.symbolCount).toBe(INHERITED.length + 1);
    });
});
