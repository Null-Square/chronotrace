# Hypotheses

The project uses explicit, falsifiable hypotheses.

## H1 — Path persistence

A detector can distinguish `AB` from `BA` on model seeds that were not used to fit the detector.

**MVP evidence:** seed-held-out balanced accuracy above chance with a confidence interval that excludes 0.5 under the predefined confirmatory split.

## H2 — Interaction localization

Features derived from interactions between stage-A and stage-B knowledge carry more history information than features derived from either stage alone.

**Prediction:** `A × B` probes outperform matched `A-only` and `B-only` controls.

## H3 — Benchmark invisibility

History remains detectable when standard task performance for AB and BA is approximately matched.

**Prediction:** a history detector remains useful after excluding runs with material capability imbalance and after controlling for scalar task metrics.

## H4 — Forensic half-life

Identical subsequent training weakens the history signal gradually rather than erasing it immediately.

For common continuation stage `C_t`, define a history score `S(t)`. We will estimate the decay of `S(t)` with continued optimization. The characteristic decay interval is the **training-history half-life**.

## H5 — Stage-type signature

Different learning mechanisms can leave distinguishable historical traces even when they teach overlapping capabilities.

Candidate mechanisms include:

- continued pretraining;
- supervised fine-tuning;
- preference optimization;
- distillation;
- targeted unlearning.

This is outside the MVP.

## H6 — Partial-order recovery

For more than two stages, pairwise or structured forensic evidence can recover a non-trivial part of the hidden training order.

Evaluation can use Kendall rank correlation for total orders and edge precision/recall for partial-order graphs.

## H7 — Acquisition mechanism

A model's endpoint can contain evidence that separates direct memorization, distributed reconstruction, rule learning, distillation, and later adaptation as different acquisition paths.

This is a long-term hypothesis and must not be claimed from binary AB/BA results.

## H8 — Black-box transfer

Order witnesses discovered on controlled shadow models can retain predictive power when only model outputs are available for an unseen target.

This is intentionally the hardest extension and is not required for Phase 0.

## Null hypotheses

The project must take these seriously:

- seed-specific variance dominates the order signal;
- any separability is explained by ordinary performance differences;
- history information exists only in weights and does not appear in meaningful behavior;
- common continuation training erases the signal too quickly for practical use;
- history is recoverable only in synthetic tasks with unrealistically strong stage separation.
