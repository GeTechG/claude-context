// knowledge-router: routeQuery — thin aggregator over the existing
// deterministic classifiers. A single place where every per-query routing
// decision is collected so it is visible in diagnostic logs and the eval
// dump. Existing call-sites that invoke classifyQuery / isComparisonShape
// directly are unaffected — this module *adds*, it does not replace.
//
// Two axes feed retrieval:
//   - intent ({codeSignal, docSignal}) → code-pool vs doc-pool domain weights
//     (handled by weightsForIntent, NOT touched here).
//   - lexical_form (identifier / descriptive / mixed) → dense vs sparse
//     channel weights (the bipolar-dense fix, handled here).

import {
    classifyQuery,
    isComparisonShape,
    classifyLexicalForm,
    QueryIntent,
    LexicalForm,
} from './query-classifier';
import { envManager } from '../utils/env-manager';

// Retrieval-channel weights for the inner per-channel Milvus ranker, in the
// canonical {dense, sparse_learned, sparse_bm25} order. The Phase-4 static
// defaults are 1.0 / 0.6 / 0.4.
export interface ChannelWeights {
    dense: number;
    sparse_learned: number;
    sparse_bm25: number;
}

export const DEFAULT_CHANNEL_WEIGHTS: ChannelWeights = {
    dense: 1.0,
    sparse_learned: 0.6,
    sparse_bm25: 0.4,
};

// knowledge-router task 3.2: lexical-form → channel-weight table.
// identifier leans on the literal-match channels (BM25 / learned-sparse);
// descriptive leans on the dense semantic channel; mixed keeps the Phase-4
// defaults.
//
// VERDICT (task 5.2 tuning grid, 2026-05-28): QUERY_ROUTER_CHANNEL_WEIGHTS is
// MEASURED-NEGATIVE and stays default `false`. The flag forces RRF→weighted
// channel fusion globally, and weighted fusion is strictly worse than RRF on
// this corpus: descriptive recall −0.087, mixed −0.111. identifier-form
// queries are weight-insensitive (qualified names hit RERANKER_BYPASS_FOR_
// QUALIFIED_NAME + symbol-routing, so the inner channel weights never reach
// their final ranking — even an extreme 0.1/1.0/1.0 triplet reproduces the
// RRF result bit-for-bit). No triplet lifts any lexical_form cell ≥+0.05 mrr.
// The triplets below are kept as documented bucket *direction* only
// (identifier → sparse-heavy, descriptive → dense-heavy); they are never used
// in production because the flag is off. The CHANNEL_WEIGHT_{ID,DESC}_* env
// grid (applyGridOverride) remains for any future re-bake. See
// infra/eval-results-knowledge-router-retrieval.json + infra/eval-summary.md
// «knowledge-router — bake-off».
export function channelWeightsForLexicalForm(form: LexicalForm): ChannelWeights {
    const base = ((): ChannelWeights => {
        switch (form) {
            case 'identifier':
                // sparse-heavy — identifier queries need literal token matches.
                return { dense: 0.6, sparse_learned: 0.8, sparse_bm25: 1.0 };
            case 'descriptive':
                // dense-heavy — descriptive queries have no literal matches; the
                // semantic channel carries them.
                return { dense: 1.0, sparse_learned: 0.5, sparse_bm25: 0.3 };
            case 'mixed':
            default:
                return { ...DEFAULT_CHANNEL_WEIGHTS };
        }
    })();
    return applyGridOverride(form, base);
}

// knowledge-router task 5.2: tuning-grid override. The grid sweeps candidate
// triplets without rebuilding dist per point, mirroring the repo's existing
// env-driven `CHANNEL_WEIGHT_*` sweeps. Env vars (per form):
//   CHANNEL_WEIGHT_ID_{DENSE,LEARNED,BM25}    — identifier triplet
//   CHANNEL_WEIGHT_DESC_{DENSE,LEARNED,BM25}  — descriptive triplet
// Each is optional; an unset channel keeps the hardcoded default for that
// channel. `mixed` is intentionally NOT overridable — it is pinned to the
// Phase-4 defaults by design. After the grid selects a point, the winning
// triplets are baked into the hardcoded table above and these env vars stay
// unset in production (byte-identical to the pinned table).
function applyGridOverride(form: LexicalForm, base: ChannelWeights): ChannelWeights {
    const prefix = form === 'identifier' ? 'CHANNEL_WEIGHT_ID_'
        : form === 'descriptive' ? 'CHANNEL_WEIGHT_DESC_'
        : null;
    if (!prefix) return base;
    const num = (suffix: string, fallback: number): number => {
        const raw = envManager.get(prefix + suffix);
        if (raw === undefined || raw.trim() === '') return fallback;
        const v = Number(raw);
        return Number.isFinite(v) ? v : fallback;
    };
    return {
        dense: num('DENSE', base.dense),
        sparse_learned: num('LEARNED', base.sparse_learned),
        sparse_bm25: num('BM25', base.sparse_bm25),
    };
}

// query_shape — synthesis axis. Only `comparison` is inferable from the
// deterministic classifiers (isComparisonShape); `concept` / `multi-hop`
// are gold-set labels supplied via shapeHint by the eval harness. Production
// callers leave shapeHint undefined, so query_shape collapses to
// comparison / single and the concept-span quota stays a no-op.
export type QueryShape = 'single' | 'comparison' | 'concept' | 'multi-hop';

export interface RouteDecision {
    intent: QueryIntent;
    query_shape: QueryShape;
    lexical_form: LexicalForm;
    channelWeights: ChannelWeights;
    conceptQuota: boolean;
}

export interface RouteOptions {
    // Ground-truth query_shape supplied by the caller (the eval harness
    // reads it from the gold-set). When undefined, routeQuery infers only
    // the shapes its deterministic classifiers can see.
    shapeHint?: QueryShape;
    // QUERY_ROUTER_CHANNEL_WEIGHTS — when false (default), channelWeights is
    // the static Phase-4 default for every query (pre-change behaviour).
    channelWeightsEnabled?: boolean;
    // CONCEPT_SPAN_QUOTA — when false (default), conceptQuota is always false.
    conceptQuotaEnabled?: boolean;
}

export function routeQuery(query: string, options: RouteOptions = {}): RouteDecision {
    const intent = classifyQuery(query);
    const lexical_form = classifyLexicalForm(query);

    const query_shape: QueryShape = options.shapeHint
        ?? (isComparisonShape(query) ? 'comparison' : 'single');

    const channelWeights = options.channelWeightsEnabled
        ? channelWeightsForLexicalForm(lexical_form)
        : { ...DEFAULT_CHANNEL_WEIGHTS };

    const conceptQuota = !!options.conceptQuotaEnabled && query_shape === 'concept';

    return { intent, query_shape, lexical_form, channelWeights, conceptQuota };
}
