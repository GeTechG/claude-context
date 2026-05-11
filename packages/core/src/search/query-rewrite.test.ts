// rag-query-static-rewrite: unit tests for the three deterministic
// rewriters (splitComparison, expandCaseVariants, expandAbbreviations) and
// the applyRewriting composition entry point.

import {
    applyRewriting,
    expandAbbreviations,
    expandCaseVariants,
    splitComparison,
} from './query-rewrite';

describe('splitComparison', () => {
    describe('positive EN cases', () => {
        it('splits "Bytes vs BytesBuffer" on bare `vs`', () => {
            expect(splitComparison('Bytes vs BytesBuffer')).toEqual({
                left: 'Bytes',
                right: 'BytesBuffer',
            });
        });

        it('splits qualified-name comparison "haxe.io.Bytes vs haxe.io.BytesBuffer"', () => {
            expect(splitComparison('haxe.io.Bytes vs haxe.io.BytesBuffer')).toEqual({
                left: 'haxe.io.Bytes',
                right: 'haxe.io.BytesBuffer',
            });
        });

        it('splits "parseInt versus parseFloat"', () => {
            expect(splitComparison('parseInt versus parseFloat')).toEqual({
                left: 'parseInt',
                right: 'parseFloat',
            });
        });

        it('splits "difference between Promise and Future"', () => {
            expect(splitComparison('difference between Promise and Future')).toEqual({
                left: 'Promise',
                right: 'Future',
            });
        });

        it('splits "compare Map and StringMap"', () => {
            expect(splitComparison('compare Map and StringMap')).toEqual({
                left: 'Map',
                right: 'StringMap',
            });
        });

        it('preserves trailing context on bare `vs` (NL trailing OK if ≤6 tokens)', () => {
            expect(splitComparison('Map vs StringMap when to use which')).toEqual({
                left: 'Map',
                right: 'StringMap when to use which',
            });
        });
    });

    describe('positive RU cases', () => {
        it('splits `разница между` phrase', () => {
            expect(splitComparison('разница между Bytes и BytesBuffer')).toEqual({
                left: 'Bytes',
                right: 'BytesBuffer',
            });
        });

        it('splits `чем X отличается от Y` phrase', () => {
            expect(splitComparison('чем parseInt отличается от parseFloat')).toEqual({
                left: 'parseInt',
                right: 'parseFloat',
            });
        });

        it('splits bare `против`', () => {
            expect(splitComparison('Map против StringMap')).toEqual({
                left: 'Map',
                right: 'StringMap',
            });
        });

        it('splits `отличие X от Y`', () => {
            expect(splitComparison('отличие Bytes от BytesBuffer')).toEqual({
                left: 'Bytes',
                right: 'BytesBuffer',
            });
        });
    });

    describe('false-positive guards', () => {
        it('rejects "compared to last year" (left empty)', () => {
            expect(splitComparison('compared to last year')).toBeNull();
        });

        it('rejects "versus everyone" with left empty', () => {
            expect(splitComparison('versus everyone')).toBeNull();
        });

        it('rejects bare trigger without right side', () => {
            expect(splitComparison('Map vs')).toBeNull();
        });

        it('rejects subjects longer than 6 tokens', () => {
            const longRight =
                'a long natural language explanation of what comes next here';
            expect(splitComparison(`Map vs ${longRight}`)).toBeNull();
        });

        it('returns null on plain NL prose with no trigger', () => {
            expect(splitComparison('how to read a file line by line')).toBeNull();
        });

        it('returns null on empty / whitespace input', () => {
            expect(splitComparison('')).toBeNull();
            expect(splitComparison('   ')).toBeNull();
        });
    });
});

describe('expandCaseVariants', () => {
    it('camelCase → [snake_case, kebab-case, PascalCase]', () => {
        const out = expandCaseVariants('parseConfig');
        expect(out).toEqual(expect.arrayContaining(['parse_config', 'parse-config', 'ParseConfig']));
        expect(out).not.toContain('parseConfig');
    });

    it('snake_case → [camelCase, kebab-case, PascalCase]', () => {
        const out = expandCaseVariants('parse_config');
        expect(out).toEqual(expect.arrayContaining(['parseConfig', 'parse-config', 'ParseConfig']));
        expect(out).not.toContain('parse_config');
    });

    it('kebab-case → [camelCase, snake_case, PascalCase]', () => {
        const out = expandCaseVariants('parse-config');
        expect(out).toEqual(expect.arrayContaining(['parseConfig', 'parse_config', 'ParseConfig']));
        expect(out).not.toContain('parse-config');
    });

    it('PascalCase → [camelCase, snake_case, kebab-case]', () => {
        const out = expandCaseVariants('ParseConfig');
        expect(out).toEqual(expect.arrayContaining(['parseConfig', 'parse_config', 'parse-config']));
        expect(out).not.toContain('ParseConfig');
    });

    it('no-op for single-form all-lowercase token', () => {
        expect(expandCaseVariants('parse')).toEqual([]);
        expect(expandCaseVariants('config')).toEqual([]);
    });

    it('no-op for empty / whitespace input', () => {
        expect(expandCaseVariants('')).toEqual([]);
        expect(expandCaseVariants('   ')).toEqual([]);
    });

    it('no-op for all-uppercase acronym', () => {
        expect(expandCaseVariants('URL')).toEqual([]);
        expect(expandCaseVariants('HTTP')).toEqual([]);
    });

    it('no-op for token without case-boundary signal', () => {
        expect(expandCaseVariants('parseconfig')).toEqual([]);
    });
});

describe('expandAbbreviations', () => {
    it('expands `cfg` → [config, configuration]', () => {
        const out = expandAbbreviations('show cfg loader');
        expect(out).toEqual(['config', 'configuration']);
    });

    it('expands `auth` → [authentication]', () => {
        expect(expandAbbreviations('auth flow')).toEqual(['authentication']);
    });

    it('expands `db lib repo` together', () => {
        const out = expandAbbreviations('db lib repo intro');
        expect(out).toEqual(expect.arrayContaining(['database', 'library', 'repository']));
    });

    it('no-op for tokens outside whitelist', () => {
        expect(expandAbbreviations('how does foo work')).toEqual([]);
        expect(expandAbbreviations('parseConfig token')).toEqual([]);
    });

    it('idempotent: skips expansion if expanded form already present', () => {
        expect(expandAbbreviations('auth authentication helper')).toEqual([]);
        expect(expandAbbreviations('cfg config flag')).toEqual(['configuration']);
    });

    it('no-op on empty / whitespace input', () => {
        expect(expandAbbreviations('')).toEqual([]);
        expect(expandAbbreviations('   ')).toEqual([]);
    });

    it('respects word boundaries (does not match `cfgloader`)', () => {
        // `cfgloader` is a single token, not `cfg` + `loader`. Whitelist
        // expansion only fires on whole-token matches.
        expect(expandAbbreviations('cfgloader path')).toEqual([]);
    });
});

describe('applyRewriting (composition)', () => {
    it('all flags off → kind=single, no extras (no-op)', () => {
        const r = applyRewriting('Bytes vs BytesBuffer', { split: false, case: false, abbrev: false });
        expect(r.kind).toBe('single');
        expect(r.sparseExtra).toEqual([]);
        expect(r.debug.caseExpansions).toEqual([]);
        expect(r.debug.abbrevExpansions).toEqual([]);
    });

    it('split flag fires on comparison query', () => {
        const r = applyRewriting('Bytes vs BytesBuffer', { split: true, case: false, abbrev: false });
        expect(r.kind).toBe('split');
        expect(r.left).toBe('Bytes');
        expect(r.right).toBe('BytesBuffer');
        expect(r.debug.comparisonMatchedTrigger?.toLowerCase()).toBe('vs');
    });

    it('split flag with case-shape subjects pulls case variants for both sides', () => {
        const r = applyRewriting('parseConfig vs writeConfig', {
            split: true,
            case: true,
            abbrev: false,
        });
        expect(r.kind).toBe('split');
        expect(r.left).toBe('parseConfig');
        expect(r.right).toBe('writeConfig');
        expect(r.sparseExtra).toEqual(expect.arrayContaining(['parse_config', 'write_config']));
        expect(r.debug.caseExpansions.length).toBeGreaterThanOrEqual(2);
    });

    it('case flag on single query produces sparseExtra', () => {
        const r = applyRewriting('parseConfig handler', { split: false, case: true, abbrev: false });
        expect(r.kind).toBe('single');
        expect(r.sparseExtra).toEqual(expect.arrayContaining(['parse_config', 'parse-config', 'ParseConfig']));
    });

    it('abbrev flag on single query produces sparseExtra', () => {
        const r = applyRewriting('cfg loader', { split: false, case: false, abbrev: true });
        expect(r.kind).toBe('single');
        expect(r.sparseExtra).toEqual(['config', 'configuration']);
    });

    it('split=true with non-comparison query falls through to single path', () => {
        const r = applyRewriting('how to read a file', { split: true, case: false, abbrev: false });
        expect(r.kind).toBe('single');
        expect(r.left).toBeUndefined();
        expect(r.right).toBeUndefined();
    });

    it('split=true with comparison trigger but length-guard fail → debug rejection tag', () => {
        const r = applyRewriting('compared to last year', { split: true, case: false, abbrev: false });
        expect(r.kind).toBe('single');
        expect(r.debug.comparisonRejectedReason).toBe('guard_fail');
    });

    it('combined flags: split + case + abbrev all fire on a mixed query', () => {
        const r = applyRewriting('parseConfig vs cfg loader', {
            split: true,
            case: true,
            abbrev: true,
        });
        expect(r.kind).toBe('split');
        expect(r.left).toBe('parseConfig');
        expect(r.right).toBe('cfg loader');
        // case-expansion fires on `parseConfig`, abbrev on `cfg`.
        expect(r.sparseExtra).toEqual(expect.arrayContaining(['parse_config', 'config', 'configuration']));
    });

    it('dedups sparseExtra entries across case + abbrev', () => {
        // expandAbbreviations on `auth` produces `authentication`.
        // expandCaseVariants on `auth_token` produces `authToken`, `auth-token`, `AuthToken`.
        // Ensure no duplicates between the two channels.
        const r = applyRewriting('auth_token cfg', {
            split: false,
            case: true,
            abbrev: true,
        });
        const seen = new Set<string>();
        for (const e of r.sparseExtra) {
            expect(seen.has(e)).toBe(false);
            seen.add(e);
        }
    });
});
