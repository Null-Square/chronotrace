# Pythia-14M T2 — Independent State-Conditioned Interaction Map

Status: **pre-registered before independent chronology outcomes**.

Lock: `configs/pythia_14m_t2.lock.json`

## Why T2 exists

The original portable Pythia-14M instance produced a reproducible `3/6` finite-pair full-order result. A subsequent diagnostic replay showed that the relevant tail-pair commutators changed dramatically after conditioning on the first stage. This suggested a state-conditioned interaction hierarchy, but the hypothesis was generated from the same instance.

T2 is the first independent test of that mechanism.

It is deliberately not a model-size scale-up and not a hyperparameter search.

## Frozen design

Model: `EleutherAI/pythia-14m-deduped`

Revision: `step143000`

Optimizer: deterministic full-batch plain SGD, no momentum, no weight decay

Learning rate: `1e-4`

Precision: FP32

Stages: A/B/C

Ground-truth histories: all six permutations

Tokenizer-safe worlds per codebook: 16

Stage lengths:

`{1, 2, 4, 8, 16, 32}` updates per stage.

All conditions are retained. There is no performance-based selection rule.

## Independent codebooks

Fresh codebook seeds were generated mechanically, not chosen by hand:

- take SHA256 of the literal string `chronotrace-t2-independent-codebooks-v1`;
- interpret the first four non-overlapping 4-byte chunks as big-endian unsigned 32-bit integers.

Frozen derivation hash:

`3c49dab384e147a6a0f59ac907241b7992399a0fbdbd91da5a3c9d8d4e1fc313`

Frozen seeds:

1. `1011473075`
2. `2229356454`
3. `2700450505`
4. `119806841`

The codebook builder must use the already-frozen Pythia tokenizer vocabulary and the exact context-boundary validation used by the previous scale gate.

## Numerical path

T2 inherits the portable CPU execution path that resolved the earlier host/backend contradiction:

- `ATEN_CPU_CAPABILITY=default`
- `MKL_CBWR=COMPATIBLE`
- MKL/OpenMP dynamic execution disabled
- one intra-op thread
- one inter-op thread
- MKLDNN disabled
- deterministic PyTorch algorithms
- pinned Python/PyTorch/Transformers stack in the workflow.

The frozen Pythia base-parameter hash must match:

`cba585ef12f0a770686bffb9d1c1d00e11400106b46d943c1cae04fa7e0df2ce`.

## Theory being tested

A base finite-pair reconstruction treats the interaction between stages i and j as if it were a static object measured at `theta_0`.

After a prefix p, the actual tail separation is instead

`C_ij(F_p(theta_0))`.

For the two histories `pij` and `pji`, define:

- `d0`: separation of their base finite-pair predicted endpoints;
- `dc`: separation of their actual endpoints;
- `b`: actual pair midpoint minus predicted pair midpoint.

The exact normalized decomposition is

`alignment = <dc,d0> / ||d0||^2`

`midpoint_bias = 2<b,d0> / ||d0||^2`.

The forward history wins against its tail swap iff

`alignment + midpoint_bias > 0`,

and the reverse history wins iff

`alignment - midpoint_bias > 0`.

Therefore both tail orders are simultaneously recoverable against each other iff

`alignment > |midpoint_bias|`.

Define

`tail_robustness = alignment - |midpoint_bias|`.

This is the main theory diagnostic for T2.

## Metrics reported for every seed × stage length

No metric is allowed to hide or remove a condition.

For each condition report:

- full-order recovery out of 6;
- first-stage recovery out of 6;
- pairwise precedence accuracy;
- mean Kendall tau;
- every true/decoded permutation;
- number and fraction of errors that are exact adjacent tail swaps;
- minimum finite-pair signature separation;
- maximum exact third-order residual norm;
- per-prefix base commutator norm;
- per-prefix conditioned commutator norm;
- relative commutator drift;
- base/conditioned cosine;
- tail alignment coefficient;
- tail midpoint-bias coefficient;
- tail robustness.

The exact directional contamination ratio may also be reported as a diagnostic, but its `chi=1` decision threshold is an algebraic restatement of nearest-signature competition and must not be presented as independent validation.

## Pre-registered checks

### Check 1 — tail mechanism at first failure

For each independent codebook, if full-order recovery first drops below 6/6 at one of the sampled stage lengths, at least one shared-prefix tail robustness value should be `<= 0` at that first failing length.

Failure of this check means the first failure is driven by a chronology direction other than the proposed tail mechanism and the theory must broaden.

### Check 2 — structure of errors

Across all T2 errors, at least 75% should preserve the first stage and differ from truth only by swapping positions 2 and 3.

This threshold was fixed before T2 outcomes. Failure means the J010 error structure does not generalize strongly enough to support the current coarse-to-fine chronology story.

### Check 3 — partial chronology survives longer

After full-order failures begin, first-stage recovery should remain higher than full-order recovery in at least 3 of the 4 independent codebooks.

Failure means the apparent first-stage robustness of J010 was likely instance-specific.

### Check 4 — interaction drift versus stage length

Report without selection whether increasing stage length tends to:

- reduce base/conditioned commutator cosine; and/or
- increase relative commutator drift.

This is descriptive rather than a hard pass criterion because nonlinear finite training operators need not vary monotonically with stage length.

## What T2 can establish

A successful T2 would support a general statement limited to this setting:

> Across multiple independent tokenizer-controlled Pythia-14M synthetic instances, base-anchored pair interactions can preserve coarse chronology while prefix-conditioned third-order effects control finer tail order as finite training stages become more nonlocal.

It would still not establish natural-data, cross-model, Adam, or black-box chronology reconstruction.

## What happens after T2

- If the state-conditioned pattern repeats, the next meaningful test is a four-stage experiment where an order-3 interaction basis can be evaluated against a factorial chronology space without simply replaying every full three-stage history.
- If the pattern does not repeat, revise the mechanism before spending compute on larger models.
- Pythia-31M remains blocked until this interpretation is complete.
