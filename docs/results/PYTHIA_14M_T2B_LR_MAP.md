# Pythia-14M T2b One-Step Learning-Rate Asymptotic Map

Date: 2026-08-29

Status: **pre-registered asymptotic restoration failed in FP32; numerical-precision adjudication required before revising the underlying real-arithmetic mechanism.**

## Question

T2 independently showed that the static finite-pair chronology decoder already loses later-stage ordering after one full-batch SGD update per stage at `eta=1e-4`. T2b asked whether reducing the one-step learning rate restores the local regime predicted by the reset-SGD commutator expansion:

`theta_AB - theta_BA = eta^2 (H_B g_A - H_A g_B) + O(eta^3)`.

If the finite-pair signal is degree 2 while the omitted three-stage term is degree 3, then reducing `eta` should make higher-order contamination smaller relative to pair-order separation and eventually restore exact pairwise full-order decoding, provided numerical resolution is adequate.

## Frozen protocol

Workflow run containing the four scientific slices: `33248443650`.

Protocol: `configs/pythia_14m_t2b_lr.lock.json`

Protocol version: `pythia-14m-one-step-lr-asymptotic-map-v1`

Protocol SHA256 recorded independently by all four slice artifacts:

`8561c471cac12ec2786f4f5cd9bf54f302acb00e6033dac36f0e75ee715fb5cc`

Model: `EleutherAI/pythia-14m-deduped`, revision `step143000`.

Optimizer: deterministic full-batch plain SGD, no momentum, no weight decay.

Precision: FP32 portable CPU execution.

Updates per stage: exactly `1`.

All six A/B/C histories were evaluated at every condition.

Fresh mechanically derived codebook seeds:

- `1208340830`
- `2712532023`
- `798146982`
- `3670363774`

Frozen learning-rate grid:

`{1e-6, 3e-6, 1e-5, 3e-5, 1e-4}`.

There was no learning-rate selection. The complete curve is the result.

## Exact source artifacts

All four scientific jobs completed successfully and uploaded their full JSON outputs before the workflow's aggregate-packaging step failed.

| Codebook seed | Artifact ID | Artifact ZIP SHA256 | JSON SHA256 |
|---:|---:|---|---|
| 1208340830 | 9713656451 | `007548763fdfd90624c2cb9dc75a84dad3c0ab167149c60f06af9fc885336dd0` | `ceb8e5bdd3455b22eb3455e54161c9264ec4095c49c489e34fc7e9d041f77d59` |
| 2712532023 | 9713656289 | `dadbea0858ea5c2948a4205e96867929348971124fa1ed37713279e0e21e6740` | `2b2645ead89441062cd581a621cd130ae344fa1967d58d771b464d065f056e3b` |
| 798146982 | 9713650272 | `3991af1c8b2be812e872400aeafca113a51402794d05ae4794843529284be159` | `d3fed8aa271b07069e71373659543b5bb4e54e66568b7482d253fcdd9064cce0` |
| 3670363774 | 9713652306 | `beca3b7a960438500cf3ef42875f988453ca508594964e90dd22bea6583bcf46` | `0ff030583a0214484a4328a75321f9200dab1b296ab0e8270093df3bbcee915d` |

The original aggregate job failed only because it did not install the local `chronotrace` package before importing `json_sha256`. It had already downloaded all four artifacts. This is an evidence-packaging failure, not a scientific-run failure.

The exact four archived JSONs were subsequently aggregated together using the pre-registered aggregation logic. The resulting aggregate JSON SHA256 is:

`b5908da57365abbd3e3b6c18b365e246fddb60011425efb785a3023faa0acd06`.

## Pre-registered checks

### Check 1 — asymptotic exact-order restoration

Frozen rule: at least `3/4` fresh codebooks must reach `6/6` exact full-order recovery at one or more rates below `1e-4`, while finite-pair signatures remain identifiable.

**FAIL: 0/4.**

No fresh codebook reaches `6/6` at any tested rate.

### Check 2 — tail robustness improves at the lowest rate

Frozen rule: at least `3/4` fresh codebooks must have greater minimum tail robustness at `1e-6` than at `1e-4`.

**PASS: 3/4.**

This is a partial directional trend, but it does not produce exact-order restoration.

### Check 3 — pair signal approaches `O(eta^2)`

This check was pre-registered as descriptive rather than hard pass/fail.

Observed per-seed log-log slopes of mean base-commutator norm versus `eta`:

| Seed | Slope |
|---:|---:|
| 1208340830 | `1.01040` |
| 2712532023 | `0.94527` |
| 798146982 | `0.97286` |
| 3670363774 | `0.84007` |

The predicted real-arithmetic local slope is approximately `2`. The observed FP32 slopes are instead close to `1`.

This is the most important T2b result because it means the tested sweep did **not** enter the expected asymptotic regime as represented by the current FP32 stage operator.

### Check 4 — relative degree-3 contamination shrinks

Frozen rule: at least `3/4` codebooks must have a smaller

`max degree-3 residual / minimum pair-signature separation`

at `1e-6` than at `1e-4`.

**FAIL: 1/4.**

The average ratio across seeds is:

- `1e-6`: `2.3951`
- `3e-6`: `2.0984`
- `1e-5`: `1.9670`
- `3e-5`: `1.9690`
- `1e-4`: `1.6946`

Thus the smallest step does not improve the relative truncation error; on average it is worse.

### Check 5 — error structure

Across the complete 4-seed x 5-rate map:

- total histories evaluated: `120`;
- exact-order errors: `72`;
- same-prefix tail-swap errors: `58/72 = 0.80556`.

The T2 tail-localized structure remains common but is no longer exact. At tiny FP32 steps, errors sometimes affect the first-stage identity as well.

## Aggregate accuracy by learning rate

| eta | Full order | First stage | Errors | Tail-swap errors |
|---:|---:|---:|---:|---:|
| `1e-6` | `10/24` | `22/24` | 14 | 12 |
| `3e-6` | `9/24` | `18/24` | 15 | 9 |
| `1e-5` | `11/24` | `22/24` | 13 | 11 |
| `3e-5` | `7/24` | `21/24` | 17 | 14 |
| `1e-4` | `11/24` | `23/24` | 13 | 12 |

There is no monotonic recovery as the learning rate shrinks.

## Signal scale

Mean singleton displacement norm across stages/seeds scales approximately linearly with `eta`, as expected for one SGD update:

| eta | Mean singleton displacement norm |
|---:|---:|
| `1e-6` | `0.0015980` |
| `3e-6` | `0.0047940` |
| `1e-5` | `0.0159799` |
| `3e-5` | `0.0479396` |
| `1e-4` | `0.1597983` |

But mean base-commutator norm is:

| eta | Mean base commutator norm |
|---:|---:|
| `1e-6` | `0.0038811` |
| `3e-6` | `0.0106114` |
| `1e-5` | `0.0276022` |
| `3e-5` | `0.0884823` |
| `1e-4` | `0.2827357` |

The measured FP32 commutator therefore remains on the same order as, and often larger than, the singleton displacement instead of shrinking quadratically relative to it.

## Interpretation

T2b **does not validate the intended local-to-finite asymptotic bridge in FP32**.

However, the result should not yet be interpreted as falsifying the exact reset-SGD real-arithmetic theorem. For one deterministic smooth SGD step per stage with fixed stage maps, the AB/BA difference is analytically second order in `eta`. The observed near-linear scaling therefore exposes a new adjudication problem:

1. the Pythia loss may have such large local curvature that even `eta=1e-6` remains outside the useful asymptotic regime; and/or
2. FP32 parameter-update and subtraction/endpoint-comparison effects may dominate the true second-order chronology signal as the step shrinks.

The nominal `>1e-12` finite-pair signature-identifiability threshold is not a sufficient floating-point error model. T2b shows that numerical identifiability must be calibrated empirically, not inferred only from a nonzero vector norm.

## Decision

**Do not move to 31M and do not proceed to the four-stage scientific reconstruction experiment yet.**

The next operation is a **numerical-only precision/locality adjudication** on these same frozen T2b instances. It must not introduce new codebooks or tune a scientific learning rate.

The adjudication should compare the identical one-step stage maps under FP32 and FP64 (and, where useful, the analytical local HVP/commutator prediction) and ask:

- does the pair-commutator slope move toward 2 in FP64?
- does relative degree-3 contamination decrease as `eta` shrinks in FP64?
- do FP32 and FP64 endpoint/order signatures diverge precisely where T2b loses the clean T2 partial chronology?

If FP64 restores the expected scaling, classify T2b as a floating-point resolution boundary and design subsequent geometry experiments in a precision-calibrated regime. If FP64 still fails to approach the second-order law over sufficiently small steps, then the assumed deterministic smooth stage-map model or implementation must be re-examined before any scale claim.
