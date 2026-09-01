# Pythia-14M Reverse-Peeling One-History Smoke

Date: 2026-08-29

Status: **FAIL under the frozen smoke rule. The target replayed exactly and the nonconverged inverse iterates ranked the true final stage `D` and true predecessor `ABC` first, but none of the four raw Picard inverse solves converged within the frozen 100-iteration budget. The all-24 exact-peeling run is therefore not authorized.**

## Role

This is a non-confirmatory methodology smoke on the already-spent four-stage pilot codebook seed `1011473075`. The held-out confirmation seeds remain prohibited and unobserved.

The goal was deliberately narrow: before evaluating any other history, test whether the one-step SGD operator could be inverted from the frozen `ABCD` endpoint by raw fixed-point iteration and whether the inferred predecessor geometry selected the true final stage.

## Frozen inputs

- Model: `EleutherAI/pythia-14m-deduped@step143000`
- Precision: FP64 portable CPU path
- Learning rate: `1e-4`
- Target history: `ABCD`
- Frozen target endpoint SHA256: `eaf246430f46da1550cbaa9b5c25695f66f370dd09a51c747214a84371ef3e8f`
- Candidate final stages: `A`, `B`, `C`, `D`
- Inverse iteration: `x_{k+1} = y + eta * grad L_j(x_k)`
- Initialization: final endpoint `y`
- Relative-update tolerance: `1e-12`
- Maximum iterations: `100`
- Predecessor score: minimum FP64 L2 distance to the six exact K3 three-stage predecessor references for the remaining stages
- Protocol: `configs/pythia_14m_reverse_peeling_smoke.lock.json`

## Evidence

- Workflow run: `33274469711`
- Artifact ID: `9721377079`
- Artifact digest: `sha256:c57967bd79de637b54231cc6f6928c0ab9c7e3942688be5dfac8c372e68e81c4`
- Raw result JSON SHA256: `40f70bc6aaaa69b70208a20c2e8b22cdfa3524534269db1538fd23b18aefdfe6`
- Canonical result SHA256: `c502def67266483b0e8787ad2c290ac4fdec616e151b7acc9e80aee9fe22efc4`
- Protocol SHA256: `4b7c192a73c41ce5aee145d9e9c07de853703b93a1a8e72c0b9b20cc08e9d09b`
- Numerical execution fingerprint: `deaad55af513e78d0c4c1d5636836bcb7d7325be64d8df0b196cd6a66b262d42`
- Stage executions: `44` (`40` K3 basis + `4` target)
- Inverse gradient evaluations: `404`
- Confirmation codebooks observed: `false`

## Frozen gate result

| Frozen check | Result |
|---|---|
| Target endpoint replay exact | PASS |
| All four candidate inverses converge | **FAIL: 0/4** |
| All forward-inverse errors finite | PASS |
| True final stage `D` selected | PASS |
| Selected `D` predecessor is `ABC` | PASS |
| Positive `D` vs runner-up margin | PASS |

Therefore `smoke_pass_all = false`.

## Candidate diagnostics at iteration 100

| Candidate last stage | Converged | Final relative update | Best predecessor | Predecessor residual | Forward-inverse relative error |
|---|---:|---:|---|---:|---:|
| A | no | `5.59244e-5` | `DCB` | `0.0802704` | `5.34172e-5` |
| B | no | `4.38140e-5` | `ADC` | `0.0831850` | `4.41541e-5` |
| C | no | `5.04066e-5` | `ABD` | `0.0602142` | `2.87768e-5` |
| D | no | `2.69402e-5` | **`ABC`** | **`0.0571779`** | `2.95099e-5` |

Residual ranking from the nonconverged iterates was

`D < C < A < B`,

with `D` vs `C` margin `0.0030363160`.

The minimum relative update seen anywhere in each 100-step trajectory was still only approximately `5.47e-6` to `8.54e-6`, far above the frozen `1e-12` tolerance. Update ratios repeatedly exceeded one, so the trajectories did not exhibit contraction toward a fixed point.

## Interpretation

The frozen experiment does **not** establish successful reverse inversion and cannot authorize all-history evaluation.

It does isolate the current failure more sharply than the earlier susceptibility experiment. The target and K3 geometry are internally consistent, and the true `D` / `ABC` hypothesis has the smallest predecessor residual even though the raw inverse iteration does not converge. Thus the immediate question is numerical/dynamical:

> Is the true one-step SGD map locally invertible, with Picard iteration failing only because `eta H` is non-contractive, or is explicit inverse recovery itself unstable/non-unique in this LLM regime?

The inverse equation is equivalently the stationarity condition of

`Psi_y(x) = 0.5 ||x-y||^2 - eta L_D(x)`.

A frozen D-only Armijo line-search diagnostic will test that distinction on the same spent `ABC -> D` transition. It may use the known `ABC` predecessor only for post-solver error measurement, never for initialization or search steering.

## Decision

- Do **not** launch the frozen all-24 reverse-peeling protocol.
- Do **not** touch confirmation codebooks.
- Do **not** scale model size.
- Run exactly one spent-history solver diagnostic using the frozen Armijo protocol.
- If that diagnostic fails, stop tuning explicit inverse solvers and pivot to forward reachable-set / Jacobian-conditioned chronology scoring.
