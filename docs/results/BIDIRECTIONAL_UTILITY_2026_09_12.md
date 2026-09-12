# Bidirectional chronology: measurable utility, 2026-09-12

This additive extension preserves every original frozen Pythia and prior secant result. The native engine is `src/chronotrace/geometry/bidirectional.py`; its 124 new test items are in `tests/test_bidirectional.py`. Full experimental drivers, prior-method dependencies, raw oracle arrays, 1,073 available tests, and the complete report are delivered in the accompanying standalone research package. This PR does not claim to have integrated all experimental runners or executed full repository CI/style checks.

## Main result

The new decoder uses residual-controlled inverse suffixes and exact forward prefixes, joins complementary stage sets, then verifies joined candidates against the checkpoint's closed rounding cell. It does not use the old global second-order tail approximation. Every actual inverse gradient and residual check is charged; an exhausted inverse solver increases uncertainty, and untested candidates survive replay-budget exhaustion.

The primary protocol was locally source-locked before 12 fresh seed clusters. On one shared digits dataset, two 64-dimensional logistic task families use eight known stages, 32 full-batch updates per stage, effective durations 5/10/20, and one target per problem. The 72 distinct targets are each observed with FP32 and FP16 checkpoint export, producing 144 cases. Training and replay remain FP64. These are not 144 independent histories.

| Export | Duration | Cases | New unique verified | Equal-work ranked retrieval | Prior secant verified |
|---|---:|---:|---:|---:|---:|
| FP32 | 5 | 24 | 24 | 3 | 0 |
| FP32 | 10 | 24 | 24 | 3 | 0 |
| FP32 | 20 | 24 | 24 | 3 | 0 |
| FP16 | 5 | 24 | 24 | 3 | 0 |
| FP16 | 10 | 24 | 24 | 3 | 0 |
| FP16 | 20 | 24 | 0 | 8 | 0 |

All cases are retained. New method: 120/144 full histories, 3,360 inferred pairs, zero observed wrong full histories or pairs. Every oracle-compatible history is retained. Complete prefix-cached endpoint replay checks all 40,320 histories independently for every one of 72 problems, after decoder predictions are fixed. All oracle-compatible sets are singletons, including the 24 strong-FP16 failures: these are algorithm/budget failures, not a proof of intrinsic information loss.

FP32 paired gain over ranked retrieval is 87.5 percentage points. A descriptive paired seed-cluster bootstrap (20,000 replicates) gives a 95% interval of 76.4-97.2 points, on this fixed dataset. Zero observed errors is not a universal zero-error guarantee. Primary training-loss reductions range from 9.0% to 39.5%; held-out accuracy from 73.3% to 87.8%.

## Work versus runtime: separate comparisons

For N=8, the target-blind split rule selects k=5: 8,800 forward-prefix stage executions and 400 inverse-stage executions. Inverse gradients are counted individually. Complete prefix-cached replay needs 109,600 forward stages. Across FP32 cases, the median exhaustive/new gradient-work ratio is 8.738, range 8.213-9.135. Cold prefix acquisition and final verification are included; evaluator target/oracle generation is separately accounted.

Eight predeclared spent FP32 inputs were repeated serially on one CPU thread with alternating method order and prefix acquisition included. Median paired wall-clock speedup over complete replay: 7.327x. Median new time: 1.824s; median exhaustive time: 13.214s. All repeated outputs match their complete oracle. These are same-environment repeats, not new independent confirmation or external replication.

The ranked comparator ranks all second-order surrogate histories without reading labels and uses exact prefix caching until the first compatible endpoint. Its budget equals realized new-method gradient work. Costs are scored from the independent exhaustive endpoint oracle; this is not a timed standalone ranked implementation. No runtime speedup over ranked replay is claimed. Its first-match output does not by itself establish uniqueness.

## Learned-representation bridge

A separate local lock tests six new target heads, each with two exports. A 64-to-16 tanh encoder is trained on a disjoint encoder split and frozen; only a 170-parameter softmax head undergoes the eight audited stages. At duration 10: 12/12 observations uniquely verified, zero observed wrong outputs, 72.31-72.46% head-loss reduction, and 95.83-96.94% held-out accuracy. This is head-only adaptation, not full-network or LLM recovery. The targets share an encoder and dataset.

Only the first predefined seed has a full 40,320-endpoint oracle; both exports match it. The other five targets are not independently exhaustively checked. A stronger duration-20 FP16 pilot failure is retained, not promoted into confirmation. The fixed encoder hits its declared epoch limit; no optimizer-convergence claim is made.

## Assumptions and limits

Known base, immutable deterministic GD recipes, known gradients, global gradient Lipschitz constants, and eta*L<1 are required. Each stage occurs exactly once. The new method uses stronger regularity/access assumptions than executable-map ranked replay. Momentum/Adam state, unknown recipes, stochastic unknown batches, and an unknown base are not covered.

The proofs are exact-arithmetic and conditional on valid numerical allowances; fixed guards are not a formal interval proof. The current join explicitly checks N! middle-vector distances, and catalogue memory and worst-case verification remain combinatorial. There is no subfactorial total-computation or arbitrary-N claim.

Reversible optimization, bidirectional search, and set-membership sequence inference are established. The intended contribution is the specific conditional endpoint-audit construction and measured work/value boundary. See the accompanying theory note; do not claim inversion or temporal traces are newly discovered.

## Validation and provenance

124 new pytest items pass, including all 24 four-stage histories on 12 affine problems and three precisions (864 scalar-oracle comparisons), nonconverged inverses, true ambiguity, budget retention, quantization cells, malformed inputs, mutation isolation, and accounting. The standalone available suite passes 1,073 items in normal and optimized Python. This includes 949 prior available tests and is not the full repository regression suite. Full repository CI and style integration remain pending.

The artifact verifier checks all seven primary locked source files, all 144 records, all 72 raw problems, complete grids, raw hashes, work counts, and retained oracle sets. It reports zero set violations and zero wrong verified outputs. This is a stored-artifact verification, not an external rerun.

Primary protocol SHA-256: `931b5e2caa42a9d707ade143b9da9a22360ee0f10e9255dfe21a9c508ac1a0e5`.
Primary records SHA-256: `0da6ba2e6b06e65d1d640bfa79ea8cce187a2079cce3dbc9bbd26e336efb5181`.
Neural protocol SHA-256: `72c6f0e0c64cf80dc61719743df9fe0aff302227c4e6d05ce6cf481f6c56e7c1`.
Neural records SHA-256: `42d0d442bcbb1252fab3e6b4e2a895e880983699c1ceb7db2f9fcf7bc85da01e`.

Native engine test: `PYTHONPATH=src python -m pytest -q tests/test_bidirectional.py`.
Reproduction commands and all raw evidence are in the standalone package README.

Publication position: measurable research utility in a scoped known-recipe audit, ready for technical review. Not 100% submission-ready, externally replicated, or production-ready.
