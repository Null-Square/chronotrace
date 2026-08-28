# ChronoTrace Baseline Ladder

ChronoTrace is not allowed to treat any above-chance history classification as evidence for a nontrivial path trace. A claimed Order Witness must be interpreted against progressively stronger alternative explanations.

This document defines the baseline ladder to freeze **before fresh Phase-0b discovery**. The current washout design pilot may inform only the choice of the common terminal-stage length; it must not be used to tune these baselines against held-out confirmation seeds.

## B0 — Chance

For the balanced binary history task (`ABC` vs `BAC`), chance balanced accuracy is 0.5.

This is necessary but scientifically weak.

## B1 — Capability-only baseline

Use only endpoint control features measuring direct A and B capability, with no contextual-binding features.

This baseline is already implemented. Phase-0 v1 demonstrated why it is mandatory: both the forensic detector and capability-only detector achieved perfect order classification because ordinary recency/forgetting exposed the last stage.

A valid Phase-0b protocol therefore requires paired endpoint capability matching before confirmation can be unlocked.

## B2 — Marginal stage-likelihood / recency baseline

Measure the final model's likelihood on the A-stage and B-stage training examples separately, without cross-stage contextual probes.

Candidate features include:

- mean and variance of A-example NLL;
- mean and variance of B-example NLL;
- A-minus-B mean NLL;
- quantiles of the per-example NLL distributions;
- paired differences for corresponding synthetic worlds.

This captures a stronger form of ordinary recency than the current capability prompts. If history remains decodable from these marginal likelihoods, the result may still be useful as temporal provenance, but it is not evidence that cross-stage interaction structure is necessary.

## B3 — Palimpsestic candidate-transcript baseline

Kuditipudi et al. (NeurIPS 2025), *Blackbox Model Provenance via Palimpsestic Membership Inference*, show that final model likelihood can correlate with known randomized example-level training order even after later training.

For ChronoTrace, construct the strongest fair adaptation available in the controlled benchmark:

1. Treat `ABC` and `BAC` as two candidate transcripts over the same A/B/C example multiset.
2. Assign each example its candidate training positions, including its later appearance in the common C stage.
3. Evaluate the endpoint model's per-example log-likelihoods.
4. Compute a transcript-order correlation statistic for each candidate history.
5. Predict the history whose candidate transcript is more compatible with the endpoint likelihood pattern.

This baseline is intentionally advantaged: it is given the candidate source examples and their candidate positions. It tests whether a conventional memorization-recency explanation already solves the inverse macro-order problem.

If B3 succeeds after capability matching, ChronoTrace should report that result directly. The method claim must then focus on whether Order Witnesses add signal beyond known-transcript likelihood correlation, require fewer queries, or work when exact source examples/order positions are unavailable.

## B4 — Single-direction contextual baselines

Evaluate the two contextual directions independently:

- A-context -> B query;
- B-context -> A query.

This checks whether the full directional-asymmetry feature is genuinely using an interaction between stages rather than one unusually sensitive prompt family.

## B5 — Full Order-Witness detector

Use the frozen cross-stage contextual-binding features and directional asymmetry.

A strong positive result is not merely `B5 > 0.5`. The strongest target pattern is:

1. paired A/B endpoint capabilities satisfy the frozen equivalence gate;
2. B1 is near chance;
3. B2 is near chance or materially weaker than B5;
4. B3 is near chance or materially weaker/more query-expensive than B5;
5. B5 generalizes across unseen training seeds;
6. the effect replicates on a fresh task family and at least one additional model scale.

## Attribution ladder for paper claims

### Level 0 — Confounded

Order is decodable, but B1 already decodes it.

Interpretation: ordinary endpoint capability/forgetting exposes chronology. This is the Phase-0 v1 result.

### Level 1 — Memorization-recency provenance

Capabilities are matched, but B2 or B3 decodes history comparably to the full method.

Interpretation: training chronology is identifiable, but a conventional likelihood/recency trace is sufficient.

This can still be a useful inverse-provenance result, but the mechanism claim must remain narrow.

### Level 2 — Interaction-specific path trace

Capabilities are matched, marginal/Palimpsestic baselines are weak, and cross-stage Order Witnesses recover history on unseen seeds.

Interpretation: endpoint interaction structure contains chronology information beyond simple marginal recency.

This is the main target for ChronoTrace.

### Level 3 — Transcript-free temporal forensics

Level 2 holds when the detector is not supplied exact original training examples or per-example training positions, for example using only semantic descriptions of candidate stages or generated probes.

This is the strongest long-term direction because it separates ChronoTrace most clearly from provenance tests that start with the original ordered transcript.

## Freeze rule

The exact B2/B3 feature definitions, query budgets, classifier family, and comparison metrics must be frozen before fresh Phase-0b discovery. Confirmation seeds must never be used to select among baseline variants.
