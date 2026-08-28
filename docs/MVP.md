# MVP Validation Slice

## Goal

Validate or reject one claim:

> The final model contains a seed-generalizing signal that distinguishes `A -> B` from `B -> A` when the training-stage multiset is held constant.

The MVP is not a product demo. It is a falsification experiment.

## Frozen Phase-0 design

The first executable design uses:

- base model: `EleutherAI/pythia-70m-deduped`, revision `step143000`;
- stage A: learn synthetic `alias -> entity` mappings;
- stage B: learn synthetic `entity -> signal` mappings;
- 96 deterministic synthetic worlds;
- the same A and B examples in both histories;
- optimizer reset between macro stages;
- stage-specific shuffle/dropout seeds that do not depend on macro order;
- discovery model-training seeds separate from confirmation seeds.

The central forensic feature is **directional contextual binding**. For a fixed queried relation and fixed correct answer, compare the target margin under a congruent cue from the other stage with the margin under an incongruent cue. Measure this in both directions.

## Minimum experiment

Train multiple small causal language models under two histories:

```text
AB = base -> A -> B
BA = base -> B -> A
```

Use identical stage datasets and matched training budgets.

Split model runs by random seed into two groups:

- **discovery seeds:** feature and detector development;
- **confirmation seeds:** one-time confirmatory evaluation.

The detector must never fit on confirmation-seed artifacts.

## Stage design requirements

A and B satisfy the following design requirements:

1. Each stage teaches measurable structure.
2. The stages interact through a shared `alias -> entity -> signal` chain.
3. Both histories can in principle retain similar individual A and B task performance.
4. The exact nonce facts are generated for this benchmark.
5. The probe bank contains A-only, B-only, and cross-stage contextual-binding probes.

If this construction produces only trivial recency/forgetting differences, it does not validate the ChronoTrace hypothesis. The capability-only baseline and capability-matching checks are required to detect that failure mode.

## First feature ladder

Test features from simplest to more mechanistic. Do not jump directly to a complex detector.

1. scalar A-only and B-only capability metrics;
2. token-level log probabilities on fixed held-out probes;
3. congruent-versus-incongruent contextual-binding margins;
4. behavioral response vectors over a fixed probe bank;
5. hidden-state summaries;
6. layer-wise activation differences;
7. targeted interaction or commutator-motivated features.

The current implementation stops at level 3. More complex features are justified only if the behavioral slice shows a reproducible signal or provides a clear falsification target.

## Primary metric

**Seed-held-out balanced accuracy** on the confirmation models.

Balanced accuracy is primary because the final confirmation set contains matched AB and BA histories but the metric remains stable if a run is excluded by a predeclared quality gate.

## Secondary metrics

- AUROC;
- paired-seed bootstrap confidence interval;
- cross-seed decision stability;
- effect size for each feature family;
- AB/BA capability gap on A-only and B-only tasks;
- performance of a baseline detector that sees only ordinary capability features.

## Quality gate

A run is eligible for the primary analysis only if:

- training completes without numerical failure;
- generated stage and probe artifacts match their recorded SHA-256 hashes;
- both stage datasets were consumed under the frozen configuration;
- no discovery/confirmation model-seed overlap exists;
- an existing run directory was not silently overwritten;
- the checkpoint and run manifest pass integrity checks.

A capability-matching threshold is recorded in `configs/mvp.yaml` before confirmation runs begin.

## Run sequence

Install the executable MVP stack:

```bash
pip install -e ".[dev,mvp]"
```

Generate the immutable synthetic artifacts:

```bash
chronotrace --config configs/mvp.yaml generate
```

Inspect the frozen endpoint matrix:

```bash
chronotrace --config configs/mvp.yaml matrix --split discovery
chronotrace --config configs/mvp.yaml matrix --split confirmation
```

Train one endpoint:

```bash
chronotrace --config configs/mvp.yaml train --history AB --seed 11
```

Extract its fixed behavioral features:

```bash
chronotrace --config configs/mvp.yaml features --run-dir runs/phase0-ab-seed11
```

After all discovery and confirmation endpoints have features, generate the report:

```bash
chronotrace --config configs/mvp.yaml detect
```

Do not inspect confirmation predictions while modifying probes or detector features. If confirmation is opened, that evaluation is spent and a new untouched confirmation seed set is required for further confirmatory claims.

## Success bands

These bands guide the next decision; they are not a substitute for statistical analysis.

- **<= 0.55 balanced accuracy:** no useful MVP signal; inspect controls before investing further.
- **0.55–0.65:** weak signal; repeat with stronger interaction design and more seeds.
- **0.65–0.80:** meaningful path signal; proceed to mechanistic localization.
- **> 0.80:** strong validation; immediately test confounds, continuation decay, and task transfer.

The confirmatory confidence interval and controls matter more than the point estimate.

## Stop conditions

Stop or redesign the core experiment if:

1. detection fails on unseen seeds after two independently designed A/B stage constructions;
2. detection disappears after matching ordinary task metrics;
3. the detector depends on accidental run metadata or checkpoint serialization artifacts;
4. the signal cannot be reproduced from a clean rerun of the full pipeline.

## Deliverables

The MVP is complete when the repo can produce:

- a set of AB and BA checkpoints;
- one manifest per run;
- a fixed probe bank;
- extracted forensic features;
- a seed-held-out classifier result;
- capability-control results;
- a single reproducible report/figure set.
