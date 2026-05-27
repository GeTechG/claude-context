// knowledge-router: routeQuery + channelWeightsForLexicalForm tests
// (task 3.3).

import {
    routeQuery,
    channelWeightsForLexicalForm,
    DEFAULT_CHANNEL_WEIGHTS,
} from './query-router';

describe('channelWeightsForLexicalForm', () => {
    it('identifier leans sparse — sparse_bm25 above dense', () => {
        const w = channelWeightsForLexicalForm('identifier');
        expect(w.sparse_bm25).toBeGreaterThan(w.dense);
        expect(w.sparse_learned).toBeGreaterThan(w.dense);
    });

    it('descriptive leans dense — dense above both sparse channels', () => {
        const w = channelWeightsForLexicalForm('descriptive');
        expect(w.dense).toBeGreaterThan(w.sparse_learned);
        expect(w.dense).toBeGreaterThan(w.sparse_bm25);
    });

    it('mixed keeps the Phase-4 defaults 1.0 / 0.6 / 0.4', () => {
        expect(channelWeightsForLexicalForm('mixed')).toEqual(DEFAULT_CHANNEL_WEIGHTS);
    });

    it('returns a fresh object — callers cannot mutate the shared default', () => {
        const a = channelWeightsForLexicalForm('mixed');
        a.dense = 99;
        expect(channelWeightsForLexicalForm('mixed').dense).toBe(1.0);
        expect(DEFAULT_CHANNEL_WEIGHTS.dense).toBe(1.0);
    });
});

describe('routeQuery', () => {
    it('aggregates intent + lexical_form + shape for an identifier query', () => {
        const r = routeQuery('Std.parseInt');
        expect(r.lexical_form).toBe('identifier');
        expect(r.intent.codeSignal).toBe(true);
        expect(r.query_shape).toBe('single');
        expect(r.conceptQuota).toBe(false);
    });

    it('infers query_shape=comparison from isComparisonShape', () => {
        expect(routeQuery('Bytes vs BytesBuffer').query_shape).toBe('comparison');
    });

    it('defaults to query_shape=single for a non-comparison query', () => {
        expect(routeQuery('how to read a file line by line').query_shape).toBe('single');
    });

    it('honours an explicit shapeHint (gold-set label)', () => {
        expect(routeQuery('synthesise the iteration story', { shapeHint: 'concept' }).query_shape)
            .toBe('concept');
        expect(routeQuery('trace the call chain', { shapeHint: 'multi-hop' }).query_shape)
            .toBe('multi-hop');
    });

    describe('channelWeights gating', () => {
        it('flag off → static Phase-4 defaults regardless of lexical_form', () => {
            expect(routeQuery('Std.parseInt').channelWeights).toEqual(DEFAULT_CHANNEL_WEIGHTS);
            expect(routeQuery('how to read a file line by line').channelWeights)
                .toEqual(DEFAULT_CHANNEL_WEIGHTS);
        });

        it('flag on → per-lexical-form triplet', () => {
            const identifier = routeQuery('Std.parseInt', { channelWeightsEnabled: true });
            expect(identifier.channelWeights).toEqual(channelWeightsForLexicalForm('identifier'));

            const descriptive = routeQuery('how to read a file line by line', {
                channelWeightsEnabled: true,
            });
            expect(descriptive.channelWeights).toEqual(channelWeightsForLexicalForm('descriptive'));
        });
    });

    describe('conceptQuota gating', () => {
        it('flag on AND query_shape=concept → conceptQuota true', () => {
            const r = routeQuery('q', { shapeHint: 'concept', conceptQuotaEnabled: true });
            expect(r.conceptQuota).toBe(true);
        });

        it('flag on but shape != concept → conceptQuota false', () => {
            const r = routeQuery('Std.parseInt', { conceptQuotaEnabled: true });
            expect(r.conceptQuota).toBe(false);
        });

        it('flag off but shape = concept → conceptQuota false', () => {
            const r = routeQuery('q', { shapeHint: 'concept' });
            expect(r.conceptQuota).toBe(false);
        });
    });
});
