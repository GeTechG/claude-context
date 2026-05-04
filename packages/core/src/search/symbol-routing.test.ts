import { parseQualifiedName } from './query-classifier';
import { buildSymbolFilter } from './symbol-routing';

describe('parseQualifiedName (Phase C)', () => {
    it('parses dot-separated qualified names', () => {
        expect(parseQualifiedName('Std.parseInt')).toEqual({
            className: 'Std', methodName: 'parseInt', fullyQualified: 'Std.parseInt',
        });
    });

    it('parses double-colon names', () => {
        expect(parseQualifiedName('Foo::bar')).toEqual({
            className: 'Foo', methodName: 'bar', fullyQualified: 'Foo::bar',
        });
    });

    it('parses slash path-style names', () => {
        expect(parseQualifiedName('Foo/Bar/baz')).toEqual({
            className: 'Bar', methodName: 'baz', fullyQualified: 'Foo/Bar/baz',
        });
    });

    it('takes the last component as methodName for 3+ level dotted names', () => {
        expect(parseQualifiedName('pkg.Class.method')).toEqual({
            className: 'Class', methodName: 'method', fullyQualified: 'pkg.Class.method',
        });
    });

    it('returns null for natural-language queries', () => {
        expect(parseQualifiedName('how to read a file')).toBeNull();
    });

    it('returns null for queries with NL tokens around a qualified name', () => {
        // Anchored: full string must be a qualified name. Embedded NL → null.
        expect(parseQualifiedName('Lambda.fold reduce list to single value')).toBeNull();
    });

    it('returns null for single identifiers without separator', () => {
        expect(parseQualifiedName('parseInt')).toBeNull();
    });

    it('returns null for malformed inputs', () => {
        expect(parseQualifiedName('a.')).toBeNull();
        expect(parseQualifiedName('.b')).toBeNull();
        expect(parseQualifiedName('a..b')).toBeNull();
    });

    it('returns null for empty / whitespace queries', () => {
        expect(parseQualifiedName('')).toBeNull();
        expect(parseQualifiedName('   ')).toBeNull();
    });

    it('does not allow mixed separators in one query', () => {
        expect(parseQualifiedName('Foo.bar::baz')).toBeNull();
        expect(parseQualifiedName('Foo/bar.baz')).toBeNull();
    });

    it('trims surrounding whitespace before parsing', () => {
        expect(parseQualifiedName('  Std.parseInt  ')).toEqual({
            className: 'Std', methodName: 'parseInt', fullyQualified: 'Std.parseInt',
        });
    });
});

describe('buildSymbolFilter (Phase C)', () => {
    it('returns null when vocab is provided but className is missing', () => {
        const parsed = { className: 'Unknown', methodName: 'foo', fullyQualified: 'Unknown.foo' };
        const out = buildSymbolFilter({ parsed, vocab: new Set(['Std', 'parseInt']) });
        expect(out).toBeNull();
    });

    it('emits exact symbol/parent_symbol filter when both are in vocab', () => {
        const parsed = { className: 'Std', methodName: 'parseInt', fullyQualified: 'Std.parseInt' };
        const out = buildSymbolFilter({ parsed, vocab: new Set(['Std', 'parseInt']) });
        expect(out).not.toBeNull();
        expect(out!).toContain('symbol_name == "parseInt" and parent_symbol == "Std"');
    });

    it('emits a basename fallback OR clause for parent_symbol-less indexes', () => {
        const parsed = { className: 'Std', methodName: 'parseInt', fullyQualified: 'Std.parseInt' };
        const out = buildSymbolFilter({ parsed, vocab: new Set(['Std', 'parseInt']) });
        expect(out).not.toBeNull();
        // Should contain at least one LIKE clause for the className.ext form.
        expect(out!).toMatch(/relativePath like "%Std\.\w+"/);
    });

    it('uses the language-extension override when provided', () => {
        const parsed = { className: 'Foo', methodName: 'bar', fullyQualified: 'Foo.bar' };
        const out = buildSymbolFilter({
            parsed,
            vocab: new Set(['Foo', 'bar']),
            languageExtensions: ['.hx'],
        });
        expect(out).not.toBeNull();
        expect(out!).toContain('relativePath like "%Foo.hx"');
        expect(out!).not.toMatch(/relativePath like "%Foo\.ts"/);
    });

    it('degrades to file-only LIKE filter when method is unknown but class is known', () => {
        const parsed = { className: 'Foo', methodName: 'unknownMethod', fullyQualified: 'Foo.unknownMethod' };
        const out = buildSymbolFilter({
            parsed,
            vocab: new Set(['Foo']),
            languageExtensions: ['.hx'],
        });
        expect(out).not.toBeNull();
        expect(out!).not.toContain('symbol_name ==');
        expect(out!).toContain('relativePath like "%Foo.hx"');
    });

    it('treats null vocab as "no gate" (returns the exact filter)', () => {
        const parsed = { className: 'AnyClass', methodName: 'anyMethod', fullyQualified: 'AnyClass.anyMethod' };
        const out = buildSymbolFilter({ parsed, vocab: null });
        expect(out).not.toBeNull();
        expect(out!).toContain('symbol_name == "anyMethod" and parent_symbol == "AnyClass"');
    });

    it('escapes double quotes in component names', () => {
        const parsed = { className: 'Foo"Bar', methodName: 'baz"', fullyQualified: 'Foo"Bar.baz"' };
        const out = buildSymbolFilter({ parsed, vocab: null });
        expect(out!).toContain('symbol_name == "baz\\""');
        expect(out!).toContain('parent_symbol == "Foo\\"Bar"');
    });
});
