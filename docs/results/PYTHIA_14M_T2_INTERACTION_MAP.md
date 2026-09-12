# Pythia-14M T2 Independent Interaction-Order Map

Date: 2026-08-29

Status: **independent support for structured partial chronology and state-conditioned tail failure; the pre-registered stage-length transition hypothesis is not supported.**

## Protocol

Workflow run: `33245167776`

Aggregate artifact: `pythia-14m-t2-aggregate`, artifact ID `9713044534`

Protocol version: `pythia-14m-state-conditioned-map-v1`

Protocol SHA256: `6eb95c404243cd64b74f4b761d99ae2db3c255c8ce68ce443c41f0c29426b7ab`

Model: `EleutherAI/pythia-14m-deduped`, revision `step143000`

Optimizer: portable deterministic full-batch plain SGD, no momentum, no weight decay

Learning rate: `1e-4`

Fresh codebook seeds were mechanically frozen before any T2 chronology output:

- `1011473075`
- `2229356454`
- `2700450505`
- `119806841`

Stage lengths were frozen to `{1,2,4,8,16,32}`. All four codebooks and all six lengths were required to be reported; there was no performance-based selection.

Each condition evaluated all six A/B/C permutations. Total ground-truth endpoints: `4 * 6 * 6 = 144`.

## Aggregate result

Across the full frozen map:

- exact full-order correct: **72/144 = 0.500**;
- first-stage correct: **144/144 = 1.000**;
- full-order errors: **72**;
- same-prefix tail-swap errors: **72/72 = 1.000**;
- pairwise precedence accuracy: **360/432 = 0.833333**;
- mean Kendall tau: **2/3**.

The dominant pattern is therefore not generic failure. The base-anchored finite-pair representation recovers the earliest stage perfectly, then frequently collapses the two possible orders of the remaining stages onto one candidate.

Most conditions recover exactly `3/6` full histories. Two deviations occur without changing the qualitative structure:

- seed `2700450505`, 2 updates/stage: `2/6` full order, all four errors still tail swaps;
- seed `119806841`, 32 updates/stage: `4/6` full order, both errors still tail swaps.

## Pre-registered checks

### Check 1 — first failure is explained by tail robustness

**PASS on all four seeds.**

Every seed's first observed full-order failure occurs where the pre-written tail-robustness quantity

`alignment - |midpoint bias|`

is non-positive.

The first full-order failure occurs at the smallest tested stage length, `1` update/stage, for all four seeds.

### Check 2 — errors are structurally tail-localized

**PASS.**

Frozen threshold: at least `75%` of errors should be same-prefix tail swaps if the state-conditioned-tail theory generalizes.

Observed: **72/72 = 100%**.

### Check 3 — partial chronology survives after full-order failure

**PASS.**

Frozen requirement: at least three of four seeds should retain a first-stage advantage when full-order recovery fails.

Observed: **4/4 seeds**, with first-stage recovery `6/6` in every one of the 24 conditions.

### Check 4 — interaction drift versus stage length

Pre-registered as descriptive only.

There is **no consistent monotonic stage-length relationship** across seeds. Pearson correlations between mean relative commutator drift and `log2(stage length)` have mixed signs (`-0.249`, `-0.326`, `-0.712`, `+0.512`). Mean base/conditioned cosine also remains low and noisy rather than displaying a clean duration-driven transition.

Therefore T2 does not support the specific prediction that increasing the number of stage updates causes a clean low-order-to-high-order transition over `{1,2,4,8,16,32}` at the frozen `1e-4` learning rate.

## One-step result is especially informative

The structured phenomenon already appears after **one optimizer update per stage** on every fresh codebook.

For example, seed `1011473075` at one update/stage has:

- full order `3/6`;
- first stage `6/6`;
- all `3/3` errors are tail swaps;
- minimum finite-pair signature separation `0.237516`;
- prefix tail base-commutator norms spanning `0.2375` to `1.2280`;
- conditioned tail-commutator norms only `0.0167` to `0.1867`;
- mean relative commutator drift approximately `1.005`;
- mean base/conditioned cosine approximately `0.0144`;
- minimum tail robustness `-0.9645`.

This rules out the simple story that many repeated SGD steps are required before the base pair geometry becomes stale. At `eta=1e-4`, a single full-batch update is already enough to move the relevant pair interaction field into a strongly state-conditioned regime for these synthetic tasks.

## Interpretation

T2 independently generalizes the **structure** seen in the motivating Pythia-14M instance:

1. low-order/base pair interactions retain strong coarse chronology information;
2. coarse information is specifically sufficient to identify the first stage;
3. the remaining ambiguity is highly localized to the ordering of later stages after a known prefix;
4. the conditioned late-stage interaction is dramatically different from the interaction measured only at the base checkpoint;
5. third-order/common-midpoint effects can dominate the tiny remaining aligned tail separation.

This is substantially stronger than the original one-instance observation because the same structural error occurs over four independently generated tokenizer-safe synthetic worlds.

However, T2 **falsifies or at least fails to support** the proposed stage-duration transition within the tested range. The smallest tested duration is already outside the static-pair regime.

## What T2 does not prove

T2 does not yet establish:

- that a sufficiently smaller learning rate restores the expected local pairwise regime;
- that degree-3 interactions reconstruct four-stage chronology without exhaustive replay;
- that the pattern holds at 31M/70M;
- that the effect survives stochastic minibatches or realistic optimizer state;
- that the same hierarchy appears for natural corpora;
- that chronology is recoverable with weaker access than replay-capable white-box access.

## Decision

**Keep 31M blocked.**

The next falsifier should vary the local step size rather than stage duration. Freeze a new independent one-update Pythia-14M map over substantially smaller learning rates before observing those endpoints.

The prediction is now sharper:

> For one update per stage, pair-order effects scale approximately as `O(eta^2)` while the omitted third-order contribution scales approximately as `O(eta^3)`. Therefore reducing `eta` should increase tail robustness and eventually restore exact finite-pair chronology recovery, provided the signal remains above the FP32 numerical floor.

If this asymptotic restoration is observed on new codebooks, it would connect the controlled commutator theorem directly to the model-scale state-conditioned hierarchy. If it is not observed before numerical resolution becomes limiting, the present local-to-finite theory must be revised.
