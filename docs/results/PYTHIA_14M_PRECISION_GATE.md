# Pythia-14M FP32/FP64 One-Step Precision Adjudication

Date: 2026-08-29

Status: **precision-boundary support confirmed across all four frozen T2b codebooks. The real-arithmetic second-order commutator law is recovered in FP64; the corresponding FP32 stage dynamics are not a reliable local-mechanism measurement.**

## Question

T2b found that the one-step Pythia-14M pair commutator scaled approximately linearly with learning rate in FP32 instead of following the smooth reset-SGD prediction

`theta_AB - theta_BA = eta^2 (H_B g_A - H_A g_B) + O(eta^3)`.

The frozen precision gate asked whether this failure came from finite-precision dynamics/cancellation or from a genuine failure to enter the smooth local regime.

## Frozen protocol

Workflow run: `33264106935`.

Protocol: `configs/pythia_14m_precision_gate.lock.json`.

Protocol version: `pythia-14m-one-step-precision-adjudication-v1`.

Canonical protocol SHA256: `610ac3ca6df7554668017ddfad6ff4de5e3ec388f7291aa1767feecbd7dc6848`.

Source T2b run: `33248443650`.

Model: `EleutherAI/pythia-14m-deduped`, revision `step143000`.

Optimizer: deterministic full-batch plain SGD, no momentum, no weight decay.

Updates per stage: `1`.

Precisions: FP32 and FP64 from the exact same parameter state, with the FP64 base required to equal the FP32 base lifted exactly to FP64.

Frozen codebook seeds:

- `1208340830`
- `2712532023`
- `798146982`
- `3670363774`

Frozen numerical learning-rate probe:

`{1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4}`.

No rate was selected scientifically from this sweep.

## Evidence artifacts

Aggregate artifact ID: `9718245253`.

Aggregate artifact ZIP SHA256: `529361d4fbd4d7c0bb088071c9fab52a05f9e645615273f9b167a2fbc5829373`.

Aggregate JSON SHA256: `8c9070d8f1e3b4ae026d234077e831f34705e0916ec79d26a07ec3f86ae00685`.

Per-seed JSON SHA256:

| Codebook seed | JSON SHA256 |
|---:|---|
| `1208340830` | `fe150205bbb2e5d88f8603bc5189f69f7fe269b9d28dbdd1c0726568f0e451b3` |
| `2712532023` | `0023f084d7da436bbe78ecfb13dc3a698ca5e07aa238b598a43117cf2c4836c3` |
| `798146982` | `4d897432b23bd2542857b4e9b02b12c36e320e0674048c9e9cae837ad3bb921b` |
| `3670363774` | `8c868a203d92b6a8558e543ad97a89970b8b15742670ff3c9f99ffb1d1dc9dfa` |

All four seed jobs and the frozen aggregate job completed successfully.

## Primary result — FP64 restores the second-order law

The most diagnostic quantities are the log-log slopes of mean pair-commutator norm against `eta` at the low-rate end.

| Seed | FP32 smallest 3 | FP32 smallest 4 | FP64 smallest 3 | FP64 smallest 4 |
|---:|---:|---:|---:|---:|
| `1208340830` | `0.90672` | `1.08906` | `1.99357` | `1.98648` |
| `2712532023` | `0.89870` | `0.93764` | `2.00454` | `2.00776` |
| `798146982` | `0.96547` | `0.94793` | `2.00357` | `2.00696` |
| `3670363774` | `1.00640` | `0.92170` | `1.99277` | `1.98175` |

All four FP64 slices independently approach the predicted slope `2` while all four FP32 slices remain near slope `1`.

This is a clean realization of the pre-registered **precision-boundary support** pattern.

## FP32/FP64 disagreement grows at the numerical boundary

At the smallest rates, FP32 and FP64 pair-commutator vectors cease to represent the same local interaction.

Across the four seeds, the low-rate mean FP32/FP64 commutator cosine is approximately zero:

- `0.01113`
- `0.00060`
- `0.02613`
- `-0.00499`

The corresponding mean relative vector error to FP64 is enormous:

- `3414.3`
- `3908.5`
- `2642.3`
- `2170.0`

At `eta=1e-8`, aggregated mean commutator norms are:

- FP32: `4.06224e-5`
- FP64: `4.05935e-9`

The FP32 quantity is therefore dominated by a precision-dependent floor rather than the true second-order chronology interaction.

## Additional numerical finding — the base gradient field is precision-sensitive

The gate also records singleton gradient norms at the identical base checkpoint. They are rate-independent because every singleton starts from the same base.

Across the 12 seed-by-stage comparisons, the FP32 gradient norm is between `2.85x` and `11.77x` the FP64 gradient norm, with mean ratio `5.54x`.

For example, seed `1208340830` gives:

| Stage | FP32 gradient norm | FP64 gradient norm | Ratio |
|---|---:|---:|---:|
| A | `1432.44` | `295.47` | `4.85x` |
| B | `2055.84` | `278.56` | `7.38x` |
| C | `864.35` | `282.18` | `3.06x` |

Thus the discrepancy is not only endpoint subtraction at tiny `eta`: the Pythia stage vector field itself is materially dtype-sensitive under this CPU execution stack.

This does **not** contradict the precision-gate conclusion. FP64 independently recovers the real-arithmetic second-order asymptotic law. It does mean that subsequent mechanism experiments should not treat the existing FP32 finite-pair geometry as a precision-neutral measurement of that law.

## Interpretation

The T2b asymptotic failure is classified as a **floating-point / dtype boundary**, not as evidence against the smooth reset-SGD commutator mechanism.

The precision gate therefore resolves the specific T2b ambiguity:

1. the exact one-step reset-SGD mechanism remains supported in a precision-calibrated representation;
2. FP32 at the tested Pythia checkpoint does not provide a trustworthy local asymptotic measurement as `eta` shrinks;
3. the earlier FP32 T2 partial-chronology phenomenon remains a valid fact about the implemented FP32 training operator, but its geometry must not be over-interpreted as precision-neutral real-arithmetic commutator structure;
4. the next higher-order mechanism experiment should use FP64 and choose its operating point by chronology-blind numerical/stability criteria before observing reconstruction outcomes.

## Decision

The numerical ambiguity that blocked the next mechanism experiment is resolved.

**Do not jump to 31M.** The higher-value next experiment is the Pythia-14M four-stage interaction-order gate:

- four stages A/B/C/D;
- fresh codebooks frozen before model outcomes;
- FP64 execution;
- chronology-blind operating-point selection using a previously used codebook or other non-confirmatory pilot evidence;
- compare the same decoder family at `K=2` and `K=3`;
- report complete order, prefix depth, and pairwise precedence separately;
- use all 24 four-stage permutations and explicitly report fixed-terminal-stage strata;
- only after the operating point is frozen run the four fresh confirmation codebooks.

The four fresh confirmation codebooks are frozen separately in `configs/pythia_14m_four_stage_codebooks.lock.json` and have not been used to observe any model chronology outcome.
