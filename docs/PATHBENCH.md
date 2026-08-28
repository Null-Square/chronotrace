# PathBench Specification

PathBench is the controlled benchmark for training-history reconstruction.

## Unit of evaluation

The unit is a **model endpoint with a hidden training history**.

A benchmark case contains:

- base model identifier and revision;
- ordered stage identifiers;
- stage data hashes;
- training hyperparameters;
- random seed;
- final checkpoint hash;
- capability evaluation;
- forensic probe outputs or approved white-box features.

The label is the training history. The detector must not receive fields that trivially reveal that label.

## Phase-0 task

Binary history classification:

```text
AB vs BA
```

The same A and B datasets appear in both classes.

## Required controls

### Membership control

AB and BA must contain the same stage-example multiset. A detector cannot win because one class contains a unique example.

### Budget control

Token count, optimization steps, batch construction policy, and main hyperparameters must match unless an experiment explicitly varies one of them.

### Capability control

Report held-out A-only and B-only performance. A strong history result is not sufficient if AB and BA have obviously different end capabilities.

### Seed control

Detector training and final evaluation use disjoint model-training seeds.

### Serialization control

When testing white-box features, eliminate file-name, checkpoint-path, timestamp, trainer-state, and serialization metadata as possible shortcuts.

### Probe isolation

Probe text must not be used as training text.

## Dataset generations

### Generation 0 — synthetic interaction worlds

Small controlled domains designed to maximize interpretability and cheap falsification.

### Generation 1 — naturalistic synthetic text

The same latent structures expressed through varied natural-language documents and tasks.

### Generation 2 — real public domains

Public or licensed corpora with controlled stage boundaries and strong matched-history controls.

### Generation 3 — heterogeneous training objectives

Different learning objectives rather than only different data domains.

## Extension tasks

### History decay

```text
AB -> C_t
BA -> C_t
```

Estimate recoverability as a function of common continuation training.

### Permutation recovery

For stages `A..K`, infer a hidden permutation. Use Kendall's tau and exact-order accuracy.

### Partial-order recovery

When stages can overlap or interleave, infer directed precedence edges and evaluate graph precision, recall, and calibration.

### Acquisition-path classification

Hold endpoint capability as constant as practical and vary whether the capability entered through direct examples, distributed fragments, rules, distillation, or later adaptation.

## Benchmark integrity

Every benchmark release should include enough metadata to reproduce the endpoint but should expose history labels separately from detector-facing features. Confirmation labels should remain hidden until the analysis code is frozen when practical.
