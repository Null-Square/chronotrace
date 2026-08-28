# Research Decision Log

Use this file for decisions that can affect interpretation of results. Add entries; do not rewrite history.

## 2026-08-28 — D001 — Define the core problem as training-history reconstruction

**Decision:** Focus ChronoTrace on the inverse problem of sequential learning rather than ordinary document membership inference or source ranking.

**Reason:** Existing research already covers strong versions of membership inference, lineage verification, and data attribution. Training order has a known optimization mechanism but post-hoc history reconstruction appears less explored.

**Consequence:** The first benchmark holds the training-stage multiset constant and changes only macro stage order.

## 2026-08-28 — D002 — Make seed-held-out evaluation mandatory

**Decision:** A detector must generalize to model-training seeds not used during detector development.

**Reason:** Otherwise the detector can exploit run-specific noise rather than training history.

## 2026-08-28 — D003 — Do not freeze the MVP model or A/B stage design during repository initialization

**Decision:** Keep `configs/mvp.yaml` unresolved for model and stage choices.

**Reason:** The next implementation slice should choose the cheapest design that produces meaningful stage interaction. Freezing a stack before that design review would create unnecessary technical debt.

## 2026-08-28 — D004 — Freeze the first executable Phase-0 design

**Decision:** Use `EleutherAI/pythia-70m-deduped` at revision `step143000` for the first shadow-model experiment. Stage A teaches `alias -> entity`. Stage B teaches `entity -> signal`. The first forensic feature family is directional contextual binding measured by congruent-versus-incongruent cross-stage cues.

**Reason:** The model is small enough for repeated controlled runs. The two-stage synthetic chain creates a real cross-stage interaction without using contradictory labels. The binding probes hold the queried relation and answer fixed, so they can test whether context from the other learned relation changes access to the target relation.

**Consequence:** The first experiment resets optimizer state at each macro stage and uses stage-specific random seeds that do not depend on macro order. This isolates path dependence in the model weights from a simpler Adam-moment or data-shuffle explanation. Confirmation seeds remain frozen and separate from discovery seeds.

## 2026-08-28 — D005 — Reject Phase-0 v1 as capability-confounded

**Decision:** Do not treat the perfect AB/BA discovery classification as evidence for a nontrivial training-path trace.

**Reason:** The forensic LOSO detector and the capability-only LOSO baseline both reached 1.00 balanced accuracy / 1.00 AUROC, and every matched discovery-seed pair violated the frozen capability-equivalence threshold. Ordinary recency/forgetting is sufficient to expose the order.

**Consequence:** Confirmation remains locked. Phase-0b introduces an identical balanced terminal stage C and tests `ABC` versus `BAC` instead.

## 2026-08-28 — D006 — Fix Phase-0b washout selection before reading forensic performance

**Decision:** On already-consumed design seeds `13, 23, 29`, evaluate only C-step candidates `50, 150, 300`. Select the smallest candidate for which every paired A-control and B-control mean-margin gap is at most `1.0`.

**Reason:** The design pilot exists to remove the v1 capability confound, not to optimize history classification. Allowing forensic accuracy to select C would leak the target signal into protocol design.

**Consequence:** Forensic metrics are reported for diagnosis but cannot determine the chosen washout length. If no candidate passes, Phase-0b is not frozen.

## 2026-08-28 — D007 — Reserve fresh Phase-0b discovery seeds before the design result

**Decision:** If a Phase-0b C length qualifies under D006, the next fresh discovery split will use training seeds `41, 43, 47, 53, 59, 61, 67, 71`. The untouched confirmation seeds remain `101, 103, 107, 109`.

**Reason:** These eight seeds are disjoint from the v1 discovery set, the Phase-0b design seeds, and confirmation. Fixing them before reading the valid washout-pilot result prevents favorable-seed selection after observing the design outcome.

**Consequence:** The only design-pilot quantity permitted to enter the fresh protocol is the C-step length selected by the capability-only rule in D006. The fresh discovery seeds cannot be replaced based on pilot or discovery performance.
