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

## 2026-08-28 — D008 — Reject shuffled-union Phase-0b as an endpoint-equivalence mechanism

**Decision:** Do not freeze any of the tested shuffled-union terminal-stage lengths `C in {50, 150, 300}`.

**Reason:** None satisfied the predeclared `<= 1.0` paired capability-gap gate. The maximum observed gap was `12.8486` at C=50, `5.4859` at C=150, and `9.4641` at C=300. The capability-only LOSO balanced accuracy remained `1.000`, `0.833`, and `0.833`, respectively. The Order-Witness detector fell from `1.000` at C=50 to `0.833` at C=150 and `0.500` at C=300. Thus longer shuffled common rehearsal can erase the interaction witness before it reliably equalizes ordinary endpoint capability, and capability matching is not monotonic under this operator.

**Consequence:** Fresh discovery seeds `41, 43, 47, 53, 59, 61, 67, 71` and confirmation seeds `101, 103, 107, 109` remain untouched. Phase-0b is a completed design-only negative result, not a positive chronology result.

## 2026-08-28 — D009 — Test Balanced Joint Washout before increasing terminal duration

**Decision:** Phase-0c will replace shuffled-union C sampling with **Balanced Joint Washout (BJW)**. Every C optimizer step must contain an equal number of matched A and B examples, paired by synthetic world and template, in the same minibatch. The terminal corpus remains exactly the A+B multiset and the batch schedule is identical for matched `ABC` and `BAC` histories.

**Reason:** The Phase-0b operator still introduced stochastic local imbalance inside C even though its aggregate corpus was balanced. Increasing C alone would not directly remove that mechanism. BJW makes each terminal gradient estimate approximately symmetric, `g_C ~= 0.5 g_A + 0.5 g_B`, and therefore provides a stronger test of path identifiability under endpoint equivalence.

**Consequence:** The Phase-0c design pilot will reuse only consumed design seeds `13, 23, 29`, keep C-step candidates `50, 150, 300` for direct operator comparison, and use the same capability-only selection rule as D006. Forensic performance cannot select the BJW duration. Fresh discovery and confirmation remain locked until a BJW candidate passes the capability gate.

## 2026-08-28 — D010 — Supersede BJW as the next compute experiment with inverse commutator decoding

**Decision:** Archive the tested BJW implementation before launching its Pythia matrix. The next research gate is a local **Commutator Decoder**: infer a finished model's candidate stage order directly from the antisymmetric second-order endpoint residual predicted by the noncommutative gradient geometry.

**Reason:** Phase-0b revealed a structural weakness in post-hoc equalization: extra common training can erase the current Order-Witness before it reliably removes ordinary capability differences. A cleaner experiment should make the confound vanish at the mathematical source instead of training it away afterward. For one gradient step on A and one on B,

`theta_AB = theta_0 - eta(g_A + g_B) + eta^2 H_B g_A + O(eta^3)`

`theta_BA = theta_0 - eta(g_A + g_B) + eta^2 H_A g_B + O(eta^3)`.

Both histories therefore share the entire first-order learning displacement. Their chronology is encoded in the second-order bracket `b_AB = H_B g_A - H_A g_B`. The order-independent midpoint gives a direct endpoint score whose asymptotic targets are `+1` for AB and `-1` for BA. For N stages, the second-order residual is a signed sum of pairwise bracket vectors, so a candidate permutation can be decoded without inserting a washout stage.

Forward work on sequential-learning geometry uses the same Lie-bracket mechanism to predict which curriculum will perform better. ChronoTrace targets the inverse problem: infer which candidate chronology produced an observed endpoint. Targeted literature search also found example-order provenance, model-lineage fingerprints, and training-order effects, but not a direct method centered on reconstructing an unknown semantic macro-stage permutation from endpoint commutator residuals. This novelty statement remains provisional and must be re-audited before publication.

**Consequence:** PR #8 remains unmerged as a tested alternative equalization operator and future ablation. No BJW Pythia matrix is run. Before any further Pythia experiment, CI must verify on a tiny smooth nonlinear system that (1) commutator prediction error is `O(eta^3)`, (2) ordinary AB/BA held-out behavior divergence is `O(eta^2)` while shared learning displacement is `O(eta)`, (3) the pairwise ChronoScore approaches `+/-1`, and (4) the bracket-basis decoder recovers all `3! = 6` three-stage permutations. Fresh discovery and confirmation seeds remain untouched.

## 2026-08-28 — D011 — Lift chronology decoding from gradient steps to finite training operators

**Decision:** Do not move directly from the successful one-update transformer theorem to Pythia. First treat a complete multi-update training stage as a near-identity operator `F_D(theta) = theta + Delta_D(theta)` and test a **macro-operator commutator decoder** against the one-step HVP decoder as stage duration increases.

**Reason:** A frontier-model training stage is not one gradient evaluation. Replacing a long stage by one effective gradient step can leave the local Taylor regime almost immediately. For finite stage maps,

`F_B(F_A(theta_0)) - F_A(F_B(theta_0)) ~= J Delta_B Delta_A - J Delta_A Delta_B`.

The directional derivatives can be estimated with centered finite differences of complete stage runs, avoiding Hessian materialization and second-order autograd. This also bypasses the double-backward limitation observed with fused SDPA/Flash attention. The method needs only pairwise stage-map probes, `O(N^2)` stage executions, rather than replaying all `N!` candidate chronologies.

**Consequence:** The fixed tiny-transformer stress test uses plain SGD with per-update learning rate `0.01`, stage lengths `{1,2,4,8,16,32,64}`, and finite-difference epsilon `1e-4`. Both decoders must recover the one-update control. The experiment only earns larger-model compute if the macro-operator decoder remains correct after the local HVP decoder loses perfect three-stage permutation recovery. Fresh discovery and confirmation seeds remain untouched.

## 2026-08-28 — D012 — Accept the macro-operator gate and isolate model scale next

**Decision:** Treat the controlled commutator/macro-operator work as the first positive ChronoTrace mechanism milestone. The next experiment may scale the same white-box operator decoder to Pythia-70M, but must keep optimizer and stage determinism simple rather than adding Adam-state persistence or stochasticity at the same time.

**Reason:** On the fixed 1,032-parameter causal transformer, the local HVP decoder recovered 6/6 permutations at one update but lost perfect recovery at two updates and fell as low as 2/6 in the predeclared sweep. The finite macro-operator decoder recovered 6/6 permutations at every stage length from 1 through 64 updates. At 64 updates the maximum individual stage displacement norm was `0.6117`, while macro AB/BA scores retained the correct signs (`+1.0450` and `-0.9083`). This demonstrates that the finite stage-map formulation extends the chronology-identifiable regime beyond the one-step approximation on the same model and data.

The result is documented in `docs/results/commutator_macro_gate.md`. It remains a controlled mechanism result: the base checkpoint, candidate stage procedures, optimizer, data, and full endpoint weights are all known.

**Consequence:** The next large-model gate should isolate **scale** first. Use a fixed Pythia checkpoint, deterministic synthetic stage data, full-weight endpoints, plain SGD without momentum, and a small predeclared stage-length sweep. Do not reuse the old classifier-based confirmation protocol as evidence for this new mechanism. Adam/AdamW state, stochastic data order, approximate candidate stages, and black-box inference remain later independent stressors.
