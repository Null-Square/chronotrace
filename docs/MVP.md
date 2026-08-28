# MVP Validation Slice

## Goal

Validate or reject one claim:

> The final model contains a seed-generalizing signal that distinguishes `A -> B` from `B -> A` when the training-stage multiset is held constant.

The MVP is not a product demo. It is a falsification experiment.

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

A and B should satisfy all of the following:

1. Each stage teaches measurable structure.
2. The stages interact, so order can plausibly matter at higher order.
3. Both histories can reach similar individual A and B task performance.
4. The data are synthetic or controlled, so the base model cannot already contain the exact facts.
5. We can generate held-out A-only, B-only, and A×B probes.

The first implementation decision is the exact stage construction. It is intentionally not frozen in this scaffold.

## First feature ladder

Test features from simplest to more mechanistic. Do not jump directly to a complex detector.

1. scalar task metrics;
2. token-level log probabilities on held-out probes;
3. behavioral response vectors over a fixed probe bank;
4. hidden-state summaries;
5. layer-wise activation differences;
6. targeted interaction or commutator-motivated features.

Each level should be compared with the previous one.

## Primary metric

**Seed-held-out balanced accuracy** on the confirmation models.

Balanced accuracy is primary because the final confirmation set must contain equal or near-equal AB and BA histories but the metric remains stable if a run is excluded by a predeclared quality gate.

## Secondary metrics

- AUROC;
- bootstrap confidence interval;
- cross-seed decision stability;
- effect size for each feature family;
- AB/BA capability gap on A-only and B-only tasks;
- performance of a baseline detector that sees only ordinary evaluation metrics.

## Quality gate

A run is eligible for the primary analysis only if:

- training completes without numerical failure;
- both stage datasets were consumed according to the manifest;
- no data leak crosses discovery and confirmation probe sets;
- the checkpoint and run manifest pass integrity checks.

A capability-matching threshold will be chosen before confirmation runs begin.

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
