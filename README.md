# ChronoTrace

**Training-history forensics for language models.**

ChronoTrace studies an inverse problem in sequential learning:

> Given a trained language model, can we recover information about the sequence of learning events that produced it?

The project tests whether optimization history leaves a measurable trace after ordinary model behavior has converged. The first target is deliberately narrow: distinguish models trained on the same two domains in opposite orders (`A → B` versus `B → A`) while controlling data, token count, optimizer, model architecture, and evaluation performance.

## Status

**Research stage:** hypothesis and benchmark setup.

The next implementation milestone is the **MVP validation slice**. It will test one binary claim before the project expands:

> A detector trained on some random seeds can predict `A → B` versus `B → A` on unseen seeds better than chance, while ordinary task metrics remain approximately matched.

If this claim does not survive seed-held-out evaluation, we stop or revise the core hypothesis.

## Core terms

- **Training history:** an ordered sequence of learning stages.
- **Path persistence:** recoverable information about training history that remains in the final model.
- **Order witness:** a probe that separates models with different histories.
- **PathBench:** the controlled benchmark for training-history reconstruction.
- **Training-history half-life:** decay of recoverable history signal under common subsequent training.

## Research question

Let two models receive the same training-stage multiset but in different orders:

```text
M_AB = Train(B, Train(A, M0))
M_BA = Train(A, Train(B, M0))
```

ChronoTrace asks whether a forensic procedure `F` can infer the hidden order from the final model:

```text
F(M_AB) -> AB
F(M_BA) -> BA
```

The strict evaluation uses random seeds that the detector did not see during detector training.

## First-principles motivation

Sequential gradient updates do not generally commute. For small updates, the difference between `A → B` and `B → A` appears in higher-order interaction terms. ChronoTrace tests whether those path-dependent effects remain measurable after training and whether they can be converted into useful forensic evidence.

The project does **not** assume that all training histories are identifiable. A major goal is to measure the conditions under which history is recoverable, unstable, or information-theoretically indistinguishable.

## Repository map

```text
chronotrace/
├── configs/                 experiment configuration
├── docs/                    research specification and decision records
├── paper/                   paper outline and bibliography
├── scripts/                 repository and experiment utilities
├── src/chronotrace/         Python package
├── tests/                   fast deterministic tests
└── .github/workflows/       continuous integration
```

Start with:

1. [`docs/RESEARCH_QUESTION.md`](docs/RESEARCH_QUESTION.md)
2. [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md)
3. [`docs/MVP.md`](docs/MVP.md)
4. [`docs/PATHBENCH.md`](docs/PATHBENCH.md)
5. [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)

## Non-negotiable research rules

1. **Hold out random seeds.** Never report a history detector only on training runs used to fit that detector.
2. **Keep the data multiset matched.** The primary AB/BA experiment changes order, not data membership.
3. **Measure ordinary capability.** A history classifier is not interesting if it only detects large performance differences between AB and BA models.
4. **Predefine stop conditions.** The MVP can falsify the project hypothesis.
5. **Keep raw run metadata.** Model, data, optimizer, seed, stage order, package versions, and commit SHA must be recorded.
6. **Separate discovery from confirmation.** Probe discovery and final confirmatory evaluation use different model seeds.
7. **Report uncertainty.** Use confidence intervals and effect sizes, not accuracy alone.
8. **Do not claim provenance proof from weak evidence.** ChronoTrace estimates evidence about training history; it does not make legal or ownership conclusions.

## Planned research progression

### Phase 0 — Binary order validation

Train controlled small language models under `AB` and `BA`. Test seed-held-out order classification.

### Phase 1 — Order witnesses

Search for behavioral and white-box features that are selectively sensitive to cross-stage interactions.

### Phase 2 — Forensic half-life

Apply identical subsequent stage `C` and measure how fast the AB/BA signal decays.

### Phase 3 — Multi-stage reconstruction

Recover partial or total orders for histories containing more than two stages.

### Phase 4 — Acquisition mechanism

Test whether the model reveals whether a capability arose from direct memorization, distributed reconstruction, rule learning, distillation, or later adaptation.

### Phase 5 — Black-box transfer

Test whether order witnesses discovered on shadow models transfer to models available only through generation APIs.

## Installation

The implementation scaffold targets Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run repository checks:

```bash
python scripts/doctor.py
pytest
ruff check .
```

## Reproducibility

Experiment outputs must not be committed to Git. Each run will write a machine-readable manifest that includes the repository commit, configuration, seeds, model identifier, training stages, environment, and artifact checksums.

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Safety and scope

ChronoTrace is intended for model auditing, research reproducibility, training-governance analysis, and authorized model forensics. Controlled and synthetic datasets are the default for validation. See [`docs/ETHICS_AND_SCOPE.md`](docs/ETHICS_AND_SCOPE.md).

## Paper

Working title:

> **ChronoTrace: The Inverse Problem of Sequential Learning in Language Models**

The paper is developed beside the implementation so that every main claim maps to an experiment and every experiment maps to a reproducible artifact.

## License

No open-source license has been selected yet. Until a license is added, normal copyright rules apply.
