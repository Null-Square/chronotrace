# Paper Outline

## Title

**ChronoTrace: The Inverse Problem of Sequential Learning in Language Models**

## Abstract — target structure

1. Training pipelines are sequential, but provenance methods mainly identify data or ancestry.
2. Define training-history reconstruction from a final model.
3. Introduce PathBench with matched data multisets and different stage orders.
4. Introduce Order Witnesses / ChronoTrace features.
5. Report seed-held-out history recovery and capability controls.
6. Measure persistence under common continuation training.
7. State limits and implications for model governance.

## 1. Introduction

- Modern LLMs pass through many learning stages.
- The same endpoint capability can arise from different histories.
- Sequential optimization is path dependent because update operators do not generally commute.
- Existing provenance asks *what* influenced a model or *which model* it descended from.
- We ask *in what order did learning events occur?*

### Contributions — provisional

1. Formalize **training-history reconstruction** as an inverse sequential-learning problem.
2. Introduce **PathBench**, which changes stage order while keeping the stage-data multiset fixed.
3. Establish whether history is recoverable across unseen training seeds.
4. Characterize where the signal lives and how long it survives common continuation training.

## 2. Related Work

- curriculum and sequential learning;
- continual learning and forgetting;
- membership inference;
- data attribution and counterfactual memorization;
- model lineage/provenance;
- unlearning fingerprints.

## 3. Problem Formulation

Define base state `theta_0`, stage update operators `U_A`, `U_B`, and histories:

```text
theta_AB = U_B(U_A(theta_0))
theta_BA = U_A(U_B(theta_0))
```

Define observer access regimes:

- behavior only;
- logits;
- activations;
- weights.

Define history estimator `F(theta or observations)` and identifiability.

## 4. PathBench

- synthetic interaction worlds;
- matched example multiset;
- matched optimization budget;
- held-out model seeds;
- A-only, B-only, and A×B evaluations;
- continuation stage C.

## 5. ChronoTrace Method

Begin with simple feature families, then introduce Order Witness selection if needed.

Potential principle: find observations with large divergence between shadow AB and BA endpoint distributions while remaining stable within each history across seeds.

## 6. Phase-0 Results

- ordinary capability comparison;
- history classification;
- seed-held-out confidence intervals;
- feature ablations;
- serialization and metadata controls.

## 7. Mechanistic Analysis

- layer localization;
- interaction probes;
- relationship to gradient/Hessian or Lie-bracket predictions.

## 8. Forensic Half-Life

- matched continuation `C_t`;
- history score versus additional updates;
- scale and layer dependence.

## 9. Limitations and Identifiability

- endpoint-equivalent histories;
- finite shadow-model coverage;
- distribution shift;
- proprietary model confounders;
- legal interpretation.

## 10. Discussion

Potential use in model governance, unlearning verification, checkpoint audit, capability-origin studies, and safety pipeline analysis.
