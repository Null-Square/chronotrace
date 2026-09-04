![ChronoTrace — Record the sequence. Rewind the cause.](assets/chronotrace-cover.jpg)

# ChronoTrace

[![CI](https://github.com/Null-Square/chronotrace/actions/workflows/ci.yml/badge.svg)](https://github.com/Null-Square/chronotrace/actions/workflows/ci.yml)

**Training-history forensics for language models.**

ChronoTrace studies an inverse problem in sequential learning:

> Given a trained language model, what information about the sequence of learning events that produced it remains recoverable from the final model?

The project is deliberately evidence-gated. It separates controlled mechanism results from large-model chronology claims, preserves negative results, and does not promote a detector result when ordinary capability already explains the signal.

## Reviewer status at a glance

| Workstream | Status | Current conclusion | Evidence |
| --- | --- | --- | --- |
| Phase-0 Pythia-70M AB/BA discovery | **Negative / confounded** | Endpoint order was perfectly classifiable, but capability-only features were also perfect and all eight matched seed pairs failed the capability-equivalence gate. This does **not** establish non-trivial path memory. | [`docs/results/PHASE0_V1.md`](docs/results/PHASE0_V1.md) |
| Phase-0b common-tail washout | **Design-only negative** | No tested common-tail duration satisfied the frozen capability gate. At `C=300`, the contextual order witness was at chance while the capability-only baseline remained above chance. | [`docs/results/PHASE0B_WASHOUT_PILOT.md`](docs/results/PHASE0B_WASHOUT_PILOT.md) |
| Inverse commutator mechanism | **Positive controlled result** | The predicted second-order chronology geometry appears in a smooth nonlinear system and a deterministic causal transformer; all `3! = 6` stage orders are recovered in the local theorem gate. | [`docs/results/commutator_macro_gate.md`](docs/results/commutator_macro_gate.md) |
| Finite macro-stage decoder | **Positive controlled result** | Treating complete training stages as finite operators retains `6/6` recovery through `64` updates/stage after the one-step HVP decoder has already failed. | [`docs/results/commutator_macro_gate.md`](docs/results/commutator_macro_gate.md) |
| Finite-pair interaction decoder | **Positive controlled result** | Exact singleton + ordered-pair probes recover all six A/B/C chronologies through the full fixed sweep to `256` updates/stage. This is the preferred white-box decoder for the first scale gate. | [`docs/results/finite_pair_gate.md`](docs/results/finite_pair_gate.md) |
| Realistic large-model chronology | **Pending** | Not yet established. The next gate isolates model scale while retaining known stages, deterministic execution, plain SGD, full weights, and the finite-pair decoder. | [`docs/RESULTS.md`](docs/RESULTS.md) |

For the consolidated evidence ledger, claim boundary, and exact reported values, start with **[`docs/RESULTS.md`](docs/RESULTS.md)**.

## Headline controlled result

The strongest completed result is the **Finite Pair Interaction Decoder**. For candidate stages `A, B, C`, it caches singleton stage effects and exact ordered-pair interactions

```text
I_{j<-i} = F_j(F_i(theta_0)) - theta_0 - Delta_i - Delta_j
```

and predicts complete histories without replaying every full chronology. The probe budget is `N^2` stage executions rather than factorial history replay.

On the fixed deterministic `1,032`-parameter causal transformer:

| Updates per stage | Micro HVP | Differential macro | Finite pair |
| ---: | ---: | ---: | ---: |
| 1 | 6/6 | 6/6 | 6/6 |
| 2 | 4/6 | 6/6 | 6/6 |
| 32 | 5/6 | 6/6 | 6/6 |
| 64 | 4/6 | 6/6 | 6/6 |
| 128 | 2/6 | 4/6 | 6/6 |
| 256 | 1/6 | 3/6 | 6/6 |

The simple sufficient nearest-signature certificate `2 ||r_high|| / delta_min < 1` holds through `32` updates/stage. It becomes inconclusive from `64` onward even though empirical decoding remains `6/6`. Those later points are successful controlled reconstructions, **not** formally certified by that worst-case norm bound.

## Evidence boundary

ChronoTrace currently supports these statements:

- sequential training order can produce an antisymmetric interaction signature in controlled nonlinear systems;
- the same inverse second-order geometry survives a genuine causal-transformer parameterization and language-model loss in a small deterministic model;
- finite training-stage operators extend chronology recovery beyond one-step local geometry;
- exact finite pair interactions extend recovery beyond the local finite-difference macro decoder in the fixed controlled stress test;
- the first Pythia AB/BA discovery result was confounded by ordinary capability differences and is recorded as a negative result, not evidence of path persistence.

ChronoTrace does **not** currently establish:

- reliable chronology reconstruction for realistic LLM training pipelines;
- robustness to stochastic batches, persistent Adam state, unknown stage recipes, distillation, merging, or black-box access;
- training provenance, ownership, or legal attribution from model endpoints;
- the original seed-held-out path-persistence hypothesis under capability-matched Pythia-scale endpoints.

## Research question

Let two models receive the same training-stage multiset in different orders:

```text
M_AB = Train(B, Train(A, M0))
M_BA = Train(A, Train(B, M0))
```

ChronoTrace asks whether an endpoint procedure `F` can infer the hidden chronology:

```text
F(M_AB) -> AB
F(M_BA) -> BA
```

For more than two stages, the project studies whether low-order interactions between finite stage maps are sufficient to identify a full or partial order without exhaustive replay.

## Reviewer quick start

A reviewer can inspect the project in this order:

1. **[`docs/RESULTS.md`](docs/RESULTS.md)** — consolidated claim/evidence ledger and limitations.
2. **[`docs/results/finite_pair_gate.md`](docs/results/finite_pair_gate.md)** — strongest current controlled reconstruction result.
3. **[`docs/results/commutator_macro_gate.md`](docs/results/commutator_macro_gate.md)** — smooth-system, tiny-transformer, and finite macro-stage mechanism gates.
4. **[`docs/results/PHASE0_V1.md`](docs/results/PHASE0_V1.md)** — confounded Pythia-70M discovery result and why confirmation stayed closed.
5. **[`docs/results/PHASE0B_WASHOUT_PILOT.md`](docs/results/PHASE0B_WASHOUT_PILOT.md)** — failed common-tail design pilot.
6. **[`docs/HYPOTHESES.md`](docs/HYPOTHESES.md)** — falsifiable hypotheses and null hypotheses.
7. **[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)** — reproducibility, split, metadata, and confirmation discipline.
8. **[`docs/DECISIONS.md`](docs/DECISIONS.md)** — research decisions, including the finite-pair scale-gate choice.

## Reproduce the controlled gates

The implementation targets Python `3.11+`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mvp,dev]"

ruff check .
pytest

python scripts/commutator_smoke.py
python scripts/transformer_commutator_smoke.py
python scripts/operator_commutator_smoke.py
python scripts/finite_pair_commutator_smoke.py
python scripts/smoke_mvp.py
```

The main CI workflow runs lint and unit tests on Python `3.11` and `3.12`, then runs the torch geometry regressions and all controlled smoke gates on CPU.

## Core terms

- **Training history** — an ordered sequence of learning stages.
- **Path persistence** — recoverable information about training history that remains in the final model.
- **Order witness** — a probe or endpoint feature that separates models with different histories.
- **PathBench** — the controlled benchmark for training-history reconstruction.
- **ChronoScore** — a signed chronology score used by the controlled inverse-geometry experiments.
- **Training-history interaction order** — proposed hierarchy describing the minimum singleton/pair/triple/... interaction order needed to identify chronology at a target error level. This is a research direction, not yet an established contribution.

## Research progression

### Completed evidence gates

- **Phase-0 discovery:** Pythia-scale AB/BA endpoint classification exposed a recency/capability confound.
- **Phase-0b washout pilot:** shuffled common-tail rehearsal did not achieve endpoint capability equivalence.
- **Commutator gate:** inverse second-order chronology geometry validated in a smooth nonlinear system and tiny causal transformer.
- **Macro-stage gate:** finite stage-map decoder retained perfect six-order recovery through `64` updates/stage.
- **Finite-pair gate:** epsilon-free singleton/pair decoder retained `6/6` recovery through `256` updates/stage.

### Next gate — isolate model scale

The next experiment should change **model scale only**: move the finite-pair white-box decoder to a Pythia checkpoint while retaining deterministic stage execution, known exact stages, plain SGD without momentum or weight decay, full endpoint weights, and a predeclared stage-length sweep. Chronology accuracy must not be used to tune the learning rate.

Only after this fixed scale gate is resolved should the project add stochastic data order, optimizer state, low-dimensional projection, unknown stage recipes, or black-box access.

## Repository map

```text
chronotrace/
├── configs/                 experiment configuration
├── docs/                    research specification, evidence, and decisions
│   └── results/             immutable per-gate result reports
├── paper/                   paper outline and bibliography
├── scripts/                 deterministic research and smoke utilities
├── src/chronotrace/         Python package
├── tests/                   fast deterministic and geometry regression tests
└── .github/workflows/       CI and controlled experiment workflows
```

## Reproducibility discipline

ChronoTrace uses explicit evidence gates:

1. hold out random seeds from detector fitting;
2. keep the data multiset matched when testing order;
3. measure ordinary capability and reject chronology claims explained by capability imbalance;
4. predefine stop conditions and keep confirmation closed after a failed gate;
5. preserve raw run metadata, model identifiers, optimizer settings, stage order, package versions, commit SHA, and artifact checksums;
6. separate discovery from confirmation;
7. report uncertainty and effect sizes rather than accuracy alone;
8. preserve negative results and method failure boundaries.

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) and [`docs/BASELINES.md`](docs/BASELINES.md).

## Safety and scope

ChronoTrace is intended for model auditing, research reproducibility, training-governance analysis, and authorized model forensics. Controlled and synthetic datasets are the default for validation. It estimates evidence about training history; it does not make ownership or legal provenance claims.

See [`docs/ETHICS_AND_SCOPE.md`](docs/ETHICS_AND_SCOPE.md).

## Paper

Working title:

> **ChronoTrace: The Inverse Problem of Sequential Learning in Language Models**

The paper is developed beside the implementation so that every main claim maps to an experiment and every experiment maps to a reproducible artifact. See [`paper/`](paper/) for the current outline and references.

## License

No open-source license has been selected yet. Until a license is added, normal copyright rules apply.
