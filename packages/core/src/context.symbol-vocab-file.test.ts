// name-declarations-and-refresh-the-vocabulary: `SYMBOL_VOCAB_FILE` points a build at a
// vocabulary derived from the corpus it is indexing.
//
// Without it a build filters its `mentioned_symbols` through whatever `.symbols-vocab.json`
// is beside the corpus, and a side build restores that file afterwards — so the served index
// was built through a vocabulary dated 2026-09-02, four index generations old, and nothing
// in its log said so. The override is the same shape as `CHUNK_CONTEXT_FILE`: a path under
// the corpus root, so the served file is untouched until a flip.

import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { VectorDatabase } from './vectordb';

class TestEmbedding extends Embedding {
    protected maxTokens = 8192;
    async detectDimension(): Promise<number> { return 3; }
    async embed(_text: string): Promise<EmbeddingVector> { return { vector: [1, 0, 0], dimension: 3 }; }
    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> { return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 })); }
    getDimension(): number { return 3; }
    getProvider(): string { return 'test'; }
}

const vectorDatabase = (): VectorDatabase => ({
    createCollection: jest.fn(), createHybridCollection: jest.fn(), dropCollection: jest.fn(),
    hasCollection: jest.fn(), listCollections: jest.fn(), insert: jest.fn(), insertHybrid: jest.fn(),
    search: jest.fn(), hybridSearch: jest.fn(), delete: jest.fn(), query: jest.fn(),
    getCollectionDescription: jest.fn(), checkCollectionLimit: jest.fn(), getCollectionRowCount: jest.fn(),
} as unknown as VectorDatabase);

describe('SYMBOL_VOCAB_FILE', () => {
    let root: string;
    let saved: string | undefined;

    beforeEach(async () => {
        root = await fs.mkdtemp(path.join(os.tmpdir(), 'vocab-file-'));
        saved = process.env.SYMBOL_VOCAB_FILE;
        delete process.env.SYMBOL_VOCAB_FILE;
        await fs.writeFile(path.join(root, '.symbols-vocab.json'), JSON.stringify({ symbols: ['ServedName'], generatedAt: '2026-09-02T00:00:00.000Z' }));
        await fs.writeFile(path.join(root, '.symbols-vocab.next.json'), JSON.stringify({ symbols: ['ServedName', 'FreshName'], generatedAt: '2026-09-20T00:00:00.000Z' }));
    });

    afterEach(async () => {
        if (saved === undefined) delete process.env.SYMBOL_VOCAB_FILE; else process.env.SYMBOL_VOCAB_FILE = saved;
        await fs.rm(root, { recursive: true, force: true });
    });

    const context = () => new Context({ embedding: new TestEmbedding(), vectorDatabase: vectorDatabase() });

    it('reads the served file when the override is unset', async () => {
        const vocab = await context().loadSymbolVocabulary(root);
        expect(vocab && [...vocab]).toEqual(['ServedName']);
    });

    it('reads the file the override names, leaving the served one alone', async () => {
        process.env.SYMBOL_VOCAB_FILE = '.symbols-vocab.next.json';
        const vocab = await context().loadSymbolVocabulary(root);
        expect(vocab && [...vocab].sort()).toEqual(['FreshName', 'ServedName']);
        // The point of the override: a build that uses it does not touch what is served.
        const served = JSON.parse(await fs.readFile(path.join(root, '.symbols-vocab.json'), 'utf-8'));
        expect(served.symbols).toEqual(['ServedName']);
    });

    it('refuses a path outside the corpus root instead of reading it', async () => {
        // A vocabulary from another corpus gates this corpus's edges, silently. Better to
        // fail the build than to write an index nobody can account for.
        for (const bad of ['/etc/passwd', '../elsewhere.json']) {
            process.env.SYMBOL_VOCAB_FILE = bad;
            await expect(context().loadSymbolVocabulary(root)).rejects.toThrow(/inside the codebase root/);
        }
    });
});
