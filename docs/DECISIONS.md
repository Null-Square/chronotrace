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
