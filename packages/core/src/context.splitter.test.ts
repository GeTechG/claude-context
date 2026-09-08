import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { Splitter, CodeChunk } from './splitter';
import { AstCodeSplitter } from './splitter/ast-splitter';
import { FileSynchronizer } from './sync/synchronizer';
import { VectorDatabase } from './vectordb';

class TestEmbedding extends Embedding {
    protected maxTokens = 8192;

    async detectDimension(): Promise<number> {
        return 3;
    }

    async embed(_text: string): Promise<EmbeddingVector> {
        return { vector: [1, 0, 0], dimension: 3 };
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 }));
    }

    getDimension(): number {
        return 3;
    }

    getProvider(): string {
        return 'test';
    }
}

class RecordingSplitter implements Splitter {
    public calls: Array<{ code: string; language: string; filePath?: string }> = [];

    constructor(private readonly label: string) { }

    async split(code: string, language: string, filePath?: string): Promise<CodeChunk[]> {
        this.calls.push({ code, language, filePath });
        return [{
            content: `${this.label}:${code}`,
            metadata: {
                startLine: 1,
                endLine: code.split('\n').length,
                language,
                filePath,
            },
        }];
    }

    setChunkSize(): void { }

    setChunkOverlap(): void { }
}

const createVectorDatabase = (): jest.Mocked<VectorDatabase> => ({
    createCollection: jest.fn().mockResolvedValue(undefined),
    createHybridCollection: jest.fn().mockResolvedValue(undefined),
    dropCollection: jest.fn().mockResolvedValue(undefined),
    hasCollection: jest.fn().mockResolvedValue(false),
    listCollections: jest.fn().mockResolvedValue([]),
    insert: jest.fn().mockResolvedValue(undefined),
    insertHybrid: jest.fn().mockResolvedValue(undefined),
    search: jest.fn().mockResolvedValue([]),
    hybridSearch: jest.fn().mockResolvedValue([]),
    delete: jest.fn().mockResolvedValue(undefined),
    query: jest.fn().mockResolvedValue([]),
    getCollectionDescription: jest.fn().mockResolvedValue(''),
    checkCollectionLimit: jest.fn().mockResolvedValue(true),
    getCollectionRowCount: jest.fn().mockResolvedValue(0),
});

describe('Context request-scoped splitters', () => {
    let tempRoot: string;
    let originalHome: string | undefined;
    let originalHybridMode: string | undefined;

    beforeEach(async () => {
        tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'claude-context-splitter-'));
        const homeDir = path.join(tempRoot, 'home');
        await fs.mkdir(homeDir, { recursive: true });
        originalHome = process.env.HOME;
        originalHybridMode = process.env.HYBRID_MODE;
        process.env.HOME = homeDir;
        process.env.HYBRID_MODE = 'false';
    });

    afterEach(async () => {
        if (originalHome === undefined) {
            delete process.env.HOME;
        } else {
            process.env.HOME = originalHome;
        }
        if (originalHybridMode === undefined) {
            delete process.env.HYBRID_MODE;
        } else {
            process.env.HYBRID_MODE = originalHybridMode;
        }
        await fs.rm(tempRoot, { recursive: true, force: true });
    });

    it('uses a request-scoped splitter for indexing without replacing the context splitter', async () => {
        const project = path.join(tempRoot, 'project');
        await fs.mkdir(project);
        await fs.writeFile(path.join(project, 'index.ts'), 'const value = 1;');

        const vectorDatabase = createVectorDatabase();
        const contextSplitter = new RecordingSplitter('context');
        const requestSplitter = new RecordingSplitter('request');
        const context = new Context({
            embedding: new TestEmbedding(),
            vectorDatabase,
            codeSplitter: contextSplitter,
        });

        await context.indexCodebase(project, undefined, false, [], [], requestSplitter);

        expect(contextSplitter.calls).toHaveLength(0);
        expect(requestSplitter.calls).toHaveLength(1);
        expect(context.getCodeSplitter()).toBe(contextSplitter);

        const insertedDocuments = vectorDatabase.insert.mock.calls
            .flatMap(([, documents]) => documents);
        expect(insertedDocuments).toHaveLength(1);
        expect(insertedDocuments[0].content).toBe('request:const value = 1;');
    });

    it('uses a request-scoped splitter for changed files during sync reindexing', async () => {
        const project = path.join(tempRoot, 'project');
        await fs.mkdir(project);
        const filePath = path.join(project, 'note.md');
        await fs.writeFile(filePath, 'first version');

        const vectorDatabase = createVectorDatabase();
        const contextSplitter = new RecordingSplitter('context');
        const requestSplitter = new RecordingSplitter('request');
        const context = new Context({
            embedding: new TestEmbedding(),
            vectorDatabase,
            codeSplitter: contextSplitter,
        });

        try {
            const synchronizer = new FileSynchronizer(
                project,
                await context.getEffectiveIgnorePatterns(project),
                context.getEffectiveSupportedExtensions()
            );
            await synchronizer.initialize();
            context.setSynchronizer(context.getCollectionName(project), synchronizer);

            await fs.writeFile(filePath, 'second version');
            await context.reindexByChange(project, undefined, [], [], requestSplitter);

            expect(contextSplitter.calls).toHaveLength(0);
            expect(requestSplitter.calls).toHaveLength(1);
            expect(context.getCodeSplitter()).toBe(contextSplitter);

            const insertedDocuments = vectorDatabase.insert.mock.calls
                .flatMap(([, documents]) => documents);
            expect(insertedDocuments).toHaveLength(1);
            expect(insertedDocuments[0].content).toBe('request:second version');
        } finally {
            await FileSynchronizer.deleteSnapshot(project);
        }
    });
});

// #130: `getSplitterInfo()` called a static that did not exist, through an
// untyped `require` that hid the TypeError from the compiler. These pin the
// repaired surface: the static exists, the introspection returns it through
// the typed binding, and the two readers of the language list read one list.
describe('splitter introspection reports what the splitter supports (#130)', () => {
    it('getSplitterInfo on the default (AST) splitter returns the actual language list', () => {
        const context = new Context({
            embedding: new TestEmbedding(),
            vectorDatabase: createVectorDatabase(),
        });
        const info = context.getSplitterInfo();
        expect(info.type).toBe('ast');
        expect(info.hasBuiltinFallback).toBe(true);
        expect(Array.isArray(info.supportedLanguages)).toBe(true);
        for (const language of ['typescript', 'ts', 'haxe', 'hx', 'hxml']) {
            expect(info.supportedLanguages).toContain(language);
        }
    });

    it('the language list agrees with the splitter support predicate, both ways', () => {
        const supported = AstCodeSplitter.getSupportedLanguages();
        for (const language of supported) {
            expect(AstCodeSplitter.isLanguageSupported(language)).toBe(true);
        }
        expect(AstCodeSplitter.isLanguageSupported('fortran')).toBe(false);
    });

    it('getSupportedLanguages returns a copy, so callers cannot move the splitter list', () => {
        const list = AstCodeSplitter.getSupportedLanguages();
        const length = list.length;
        list.push('fortran');
        expect(AstCodeSplitter.getSupportedLanguages()).not.toContain('fortran');
        expect(AstCodeSplitter.getSupportedLanguages()).toHaveLength(length);
    });
});
