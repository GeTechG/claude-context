// knowledge-router: classifyLexicalForm tests (task 2.3).
//
// The lexical-form axis is orthogonal to {codeSignal, docSignal}: it answers
// "which retrieval channel carries the query" (sparse/BM25 vs dense), not
// "which domain". 3+ examples per bucket, plus the boundary cases the
// task-2 calibration flagged.

import { classifyLexicalForm } from './query-classifier';

describe('classifyLexicalForm', () => {
    describe('identifier — code token, < 3 NL words', () => {
        it('classifies a qualified name "Std.parseInt"', () => {
            expect(classifyLexicalForm('Std.parseInt')).toBe('identifier');
        });

        it('classifies a qualified name "FileSystem.readDirectory"', () => {
            expect(classifyLexicalForm('FileSystem.readDirectory')).toBe('identifier');
        });

        it('classifies a qualified name "Reflect.callMethod"', () => {
            expect(classifyLexicalForm('Reflect.callMethod')).toBe('identifier');
        });

        it('classifies a bare camelCase token "parseInt"', () => {
            expect(classifyLexicalForm('parseInt')).toBe('identifier');
        });

        it('classifies a bare PascalCase token "BytesBuffer"', () => {
            expect(classifyLexicalForm('BytesBuffer')).toBe('identifier');
        });

        it('boundary: "Bytes vs BytesBuffer" is identifier-dominated (form, not shape)', () => {
            expect(classifyLexicalForm('Bytes vs BytesBuffer')).toBe('identifier');
        });

        // task-2.2 calibration finding: a dotted qualified name whose
        // components are lowercase (haxe, parse) must not have those
        // components counted as natural-language words.
        it('calibration: lowercase-component qualified name "haxe.Json.parse"', () => {
            expect(classifyLexicalForm('haxe.Json.parse')).toBe('identifier');
            expect(classifyLexicalForm('haxe.io.Path.join')).toBe('identifier');
        });

        it('calibration: two qualified names + connector stay identifier', () => {
            expect(classifyLexicalForm('haxe.io.Bytes vs haxe.io.BytesBuffer'))
                .toBe('identifier');
        });
    });

    describe('descriptive — >= 3 NL words, no dominating code token', () => {
        it('classifies "how to read a file line by line"', () => {
            expect(classifyLexicalForm('how to read a file line by line')).toBe('descriptive');
        });

        it('classifies "iterate and transform a collection map filter fold"', () => {
            expect(classifyLexicalForm('iterate and transform a collection map filter fold'))
                .toBe('descriptive');
        });

        it('classifies "parse the input string into tokens"', () => {
            expect(classifyLexicalForm('parse the input string into tokens')).toBe('descriptive');
        });
    });

    describe('mixed — both signals, or neither clearly present', () => {
        it('classifies "Lambda.fold reduce list to single value" (qname in NL prose)', () => {
            expect(classifyLexicalForm('Lambda.fold reduce list to single value')).toBe('mixed');
        });

        it('classifies "how does StringTools.replace handle empty input"', () => {
            expect(classifyLexicalForm('how does StringTools.replace handle empty input'))
                .toBe('mixed');
        });

        it('boundary: short pure-NL query "read file" falls to the neutral bucket', () => {
            expect(classifyLexicalForm('read file')).toBe('mixed');
        });

        it('boundary: empty query is neutral', () => {
            expect(classifyLexicalForm('')).toBe('mixed');
            expect(classifyLexicalForm('   ')).toBe('mixed');
        });
    });
});
