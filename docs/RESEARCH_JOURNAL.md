# ChronoTrace Research Journal

This is the append-only experimental and theoretical journal for ChronoTrace.

**Rule:** add new entries; do not rewrite old entries to make the research path look cleaner in hindsight. If an interpretation changes, append a correction entry that points back to the earlier entry.

Every substantive experiment should record:

1. question / hypothesis;
2. what was frozen before results;
3. implementation commit / PR / workflow run;
4. raw result summary;
5. what the result does and does not imply;
6. decision taken because of it;
7. the next falsifier.

Detailed result tables remain in `docs/results/`. Scientific protocol decisions remain in `docs/DECISIONS.md`. This journal is the chronological narrative connecting them.

---

## 2026-08-28 — J001 — Phase-0 v1 detects order for a trivial reason

**Question.** Can a finished Pythia-70M endpoint distinguish `AB` from `BA` using directional cross-stage binding features?

**Frozen controls.** Same A/B artifacts, same architecture/checkpoint, optimizer reset between stages, stage randomness independent of macro-order, seed-held-out detector evaluation, capability-only baseline, capability-equivalence gate, confirmation seeds untouched.

**Evidence.** Workflow run `33201301575`; detailed result in `docs/results/PHASE0_V1.md`.

**Result.** 16/16 FP32 discovery endpoints completed. Forensic detector: balanced accuracy `1.000`, AUROC `1.000`. Capability-only baseline: balanced accuracy `1.000`, AUROC `1.000`. Every matched seed violated the capability-equivalence gate; worst observed A/B mean-margin gap was `13.9531` against the frozen `<=1.0` threshold.

**Interpretation.** This is not evidence for non-trivial path memory. Last-stage recency / forgetting already reveals the order perfectly.

**Decision.** Reject v1 as capability-confounded. Do not touch confirmation.

**Next falsifier.** Equalize terminal A/B capability without selecting on forensic accuracy.

---

## 2026-08-28 — J002 — Common shuffled washout erases the witness before equalizing capability

**Question.** Does a common balanced terminal stage C make `ABC` and `BAC` capability-equivalent while preserving history information?

**Frozen controls.** Design seeds `13,23,29`; C lengths `{50,150,300}`; C is the same deterministic shuffled A+B union for both histories; candidate selected only if every matched A- and B-control gap is `<=1.0`; forensic metrics forbidden from choosing C.

**Evidence.** Workflow run `33207229999`; `docs/results/PHASE0B_WASHOUT_PILOT.md`.

**Result.** No C length qualified. At C=300, capability-only balanced accuracy remained `0.833` while the Order-Witness detector fell to `0.500` balanced accuracy / `0.444` AUROC. Capability gaps were non-monotonic with longer C.

**Interpretation.** Post-hoc washout is not a principled route to the desired endpoint-equivalent regime. It can destroy the current chronology witness before removing ordinary capability evidence.

**Decision.** Do not extend C blindly and do not freeze Phase-0b.

**Next falsifier.** Initially considered Balanced Joint Washout, then superseded before expensive compute by a mechanism-first inverse formulation.

---

## 2026-08-28 — J003 — Pivot from washout to inverse noncommutative geometry

**Question.** Can chronology be decoded directly from the antisymmetric endpoint residual created by noncommuting stage updates, rather than manufacturing endpoint equivalence after training?

**Theory.** For one small update on A and B,

`theta_AB - theta_BA = eta^2 (H_B g_A - H_A g_B) + O(eta^3)`.

Both histories share the full first-order displacement. Order appears at second order through the commutator.

**Decision.** Archive the BJW Pythia design unspent. Make the commutator scaling law the next gate. See D010 in `docs/DECISIONS.md`.

**Novelty boundary.** Lie-bracket order effects are not new; the candidate contribution is inverse endpoint reconstruction rather than forward curriculum selection.

---

## 2026-08-28 — J004 — Local theorem and causal-transformer mechanism gate pass

**Question.** Does the inverse commutator formula have the predicted asymptotics, and does it survive a real transformer parameterization / LM loss?

**Evidence.** PR #9, workflow run `33213263777`; `docs/results/commutator_macro_gate.md`.

**Result.** Smooth system slopes: commutator remainder `2.99346`, behavior gap `2.06531`, shared displacement `0.95967`. Tiny 1,032-parameter causal transformer slopes: `3.00290`, `1.98968`, `0.99879`. All six A/B/C permutations recovered across the fixed learning-rate sweep; pair ChronoScores approached `+1/-1`.

**Interpretation.** This is a positive controlled mechanism result. It validates the inverse second-order geometry, not realistic LLM chronology reconstruction.

**Next falsifier.** Replace one-step gradients by multi-update training-stage operators.

---

## 2026-08-28 — J005 — Macro training-operator decoder outlives the one-step approximation

**Question.** If each stage contains many SGD updates, does treating the complete stage as an operator extend the recoverable regime?

**Frozen protocol.** Same tiny causal transformer; SGD lr `0.01`; stage lengths `{1,2,4,8,16,32,64}`; centered finite-difference macro-JVP epsilon `1e-4`.

**Evidence.** Same PR #9 / run `33213263777`; `docs/results/commutator_macro_gate.md`.

**Result.** Local HVP decoder first loses 6/6 recovery at 2 updates/stage. Macro-operator decoder stays 6/6 through all 64 updates/stage. At 64 updates, max singleton displacement norm is `0.61171`; AB/BA macro scores remain correctly signed (`+1.04502`, `-0.90832`).

**Interpretation.** A complete finite training stage is a better forensic primitive than a single effective gradient once the trajectory is nonlocal.

**Next falsifier.** Remove finite-difference epsilon and measure exact finite directed pair interactions.

---

## 2026-08-29 — J006 — Finite pair interaction decoder passes the controlled stress test

**Question.** Are exact singleton + directed-pair stage interactions enough to reconstruct full three-stage histories after local and differential approximations fail?

**Evidence.** PR #10, workflow run `33214025787`; `docs/results/finite_pair_gate.md`.

**Result.** Finite-pair decoder stays `6/6` from 1 through 256 updates/stage. Micro HVP first fails at 2. Differential macro first fails at 128 (`4/6`) and reaches `3/6` at 256. At 256 updates, max singleton displacement is `2.08180`.

The sufficient bound `2||r_high||/delta_min < 1` holds only through 32 updates. It is `1.4289` at 64, `2.0025` at 128, and `2.0173` at 256 even though empirical recovery remains perfect.

**Interpretation.** Pair interactions can work well outside the conservative worst-case certificate on the controlled model. The unmodeled object is an explicit triple-and-higher interaction residual.

**Decision.** Promote finite pairs to the first scale-gate decoder, while treating interaction order as a first-class research object.

---

## 2026-08-29 — J007 — Chronology-blind Pythia scale learning-rate gate freezes `1e-4`

**Question.** What common plain-SGD rate is numerically stable across Pythia 14M/31M/70M without observing any chronology endpoint?

**Evidence.** Workflow `33216943852`; `docs/results/pythia_scale_lr_gate.md`.

**Frozen candidates.** `{1e-4,3e-4,1e-3,3e-3,1e-2}`; stage A only; 8 updates; exact tokenizer-controlled data; no chronology labels or scores available to the selector.

**Result.** `1e-4` is the only common passing rate because 14M rejects every larger candidate. The tokenizer/codebook/data hashes are identical across all 15 singleton probes.

**Decision.** Freeze `1e-4`, the exact codebook, dataset hashes, Pythia revision, and 16-update chronology bridge before training any full A/B/C order.

---

## 2026-08-29 — J008 — First Pythia-14M bridge is contradictory and therefore rejected

**Question.** Does the finite-pair basis recover all six Pythia-14M A/B/C histories at the frozen 16-update protocol?

**Observed issue.** Nominally identical executions produced contradictory `3/6` and `6/6` outcomes. The first evidence writer also raised before preserving a failed result; that observability bug was fixed without changing scientific settings.

**Interpretation at the time.** No scale claim was accepted. Reproducibility became a hard gate before 31M.

**Decision.** Fingerprint base weights, concrete token batches, singleton/pair basis, candidate signatures, all six endpoints, runtime stack, and numerical settings. Require independent exact-hash replicas.

---

## 2026-08-29 — J009 — Initial reproducibility gate shows host/backend sensitivity

**Evidence.** Workflow `33218688360`; `docs/results/pythia_14m_reproducibility_gate.md` records the first adjudication.

**Result.** With pinned package versions, deterministic algorithms, and single-thread execution, two replicas produced `3/6` and one produced `6/6`; learned singleton/pair/endpoint hashes differed despite identical base-model and batch hashes. Two AMD replicas also differed from each other.

**Interpretation.** The contradiction was numerical execution/backend contamination, not a trustworthy model-scale chronology result.

**Decision.** Keep 31M blocked. Force a portable CPU path without changing scientific hyperparameters.

---

## 2026-08-29 — J010 — Portable Pythia-14M gate is exactly reproducible and fails full-order recovery

**Question.** Under a portable frozen CPU numerical path, what does the original 14M scientific protocol actually do?

**Numerical-only intervention.** `ATEN_CPU_CAPABILITY=default`, `MKL_CBWR=COMPATIBLE`, MKL/OpenMP dynamic execution disabled, MKLDNN disabled, one intra-op/inter-op thread, pinned software versions. No change to model, data, LR, optimizer, stage duration, basis, decoder, or success criterion.

**Evidence.** Workflow `33219286064`.

**Result.** All three independent replicas are bit-for-bit identical in the finite-pair basis and all six endpoint hashes. All three recover exactly `3/6` histories. Therefore the reproducible result is a scientific negative for **full-order pairwise decoding**, not a runner failure.

Portable decoded histories:

- `ABC -> ACB` (wrong)
- `ACB -> ACB` (correct)
- `BAC -> BAC` (correct)
- `BCA -> BAC` (wrong)
- `CAB -> CAB` (correct)
- `CBA -> CAB` (wrong)

Numerics:

- min finite-pair signature separation: `0.2056145`
- min decode margin: `0.0150283`
- max triple+ remainder norm: `0.2054421`
- max `2||r_high||/delta_min`: `1.99832`
- singleton displacement norms: A `0.2062331`, B `0.1697814`, C `0.1847850`

**Structured partial-order observation.** The three errors are all a swap of positions 2 and 3 while preserving the first stage. Across the six histories, the decoded permutations contain `15/18 = 83.3%` correct pairwise precedence relations, first-stage accuracy `6/6`, and mean Kendall tau `2/3`. These are post-hoc descriptive diagnostics from one frozen synthetic instance, not yet a generalization claim.

**Interpretation.** The base-anchored finite-pair truncation is insufficient for complete Pythia-14M chronology at this protocol. However, the error structure strongly suggests a missing **prefix-conditioned higher-order interaction**, not arbitrary loss of all order information.

**Next falsifier.** Test the theory that the relevant pair commutator for late stages must be evaluated at the state produced by the earlier prefix, and that the triple residual measures the drift of that commutator away from its base-checkpoint value. Freeze this theory and independent-seed evaluation before running new chronology experiments.

---

## 2026-08-29 — J011 — T1 directly supports prefix-conditioned third-order geometry on the frozen 14M instance

**Question.** Do the exact third-order residual and prefix-conditioned commutator explain the structured `3/6` failure already observed in J010?

**Frozen before the run.** T1 was diagnostic-only. It was required to reproduce the exact portable finite-pair basis and all six exact endpoint hashes from run `33219286064`. No decoder, threshold, model, data, optimizer, learning rate, stage length, or success rule could change. Before seeing T1, the theory predicted that a failed history should cross its tail-swap decision boundary when the directional contamination ratio `chi` exceeds `1`, and that the tail-pair commutator should drift materially after conditioning on the first stage.

**Evidence.** Workflow `33243747235`; artifact ID `9712269414`; `docs/results/pythia_14m_theory_diagnostic.md`.

**Identity check.** The finite-pair basis and all six endpoints exactly match J010. Therefore this is mechanistic analysis of the same frozen instance, not independent generalization evidence.

**Directional result.** The three wrong histories are exactly the three tail-swap cases with `chi > 1`:

- `ABC -> ACB`: `chi = 1.118318`
- `BCA -> BAC`: `chi = 1.172054`
- `CBA -> CAB`: `chi = 1.231791`

Every correctly decoded history remains below the exact boundary against all competitors; its maximum `chi` is at most `0.843755`.

**State-conditioning result.** Conditioning on the first stage nearly replaces the relevant tail commutator rather than weakly perturbing it. For prefixes A/B/C, relative commutator drift is approximately `0.998`, `1.010`, and `1.020`; base/conditioned cosine is only `0.103`, `0.031`, and `0.052`. The conditioned commutator norms shrink from base norms `0.206–0.260` to `0.038–0.063`.

**Interpretation.** T1 strongly supports the specific state-conditioned interaction mechanism on the motivating instance. The pairwise decoder fails not merely because an omitted term is large, but because the exact third-order residual is aligned strongly enough with a specific tail-swap chronology direction to cross the nearest-signature boundary.

**What this does not imply.** The hypothesis was generated from J010, so T1 cannot establish that the same mechanism repeats across independent worlds/codebooks, stage lengths, model sizes, optimizers, or natural corpora.

**Decision.** Keep the state-conditioned interaction theory. Keep 31M blocked. Move to an independent Pythia-14M interaction-order map whose codebook seeds and stage-length sweep are frozen before chronology results are observed.

**Next falsifier.** T2 must test whether the relationship between stage length, prefix-conditioned commutator drift, directional contamination, full-order recovery, first-stage recovery, pairwise precedence, and Kendall tau repeats across fresh tokenizer-safe synthetic instances. No T2 parameter may be selected using chronology performance.

---

## Journal checkpoint — current state

Accepted controlled positives:

- inverse one-step commutator scaling law;
- causal-transformer realization;
- multi-update macro-stage recovery through 64 updates on the controlled transformer;
- exact finite-pair recovery through 256 updates on the controlled transformer;
- on the frozen Pythia-14M J010 instance, exact directional contamination and prefix-conditioned commutator drift explain the structured pairwise-decoder failures.

Accepted negatives / constraints:

- v1 order classification is capability-confounded;
- shuffled common-tail washout does not create robust endpoint equivalence;
- Pythia-14M base-anchored finite-pair truncation does **not** recover full 3-stage order at the frozen 16-update scale protocol;
- portable numerical execution is required before interpreting small endpoint-distance differences;
- T1 is explanatory evidence on the same instance, not generalization.

Current live research question:

> Across independent model/data instances, is training chronology encoded hierarchically through **state-conditioned interaction order**, where lower-order interactions recover coarse precedence and prefix-conditioned / higher-order interactions are required for deeper chronology?

---

## 2026-08-29 — J012 — Correction: chi is a decision identity; midpoint bias is the sharper T1 mechanism

**Why this entry exists.** J011 treated the exact directional contamination threshold `chi = 1` as if its agreement with correct/incorrect cases were substantive support for the mechanism. A later derivation showed that this was too strong.

**Correction.** For a fixed true pairwise signature and competitor, `chi < 1` is algebraically equivalent to the nearest-signature decision preferring the true candidate. Therefore the observation “failed histories have `chi > 1`” is a **diagnostic restatement of the decision boundary**, not independent empirical validation.

The useful information from chi is which competing chronology direction receives the strongest contamination. On the frozen 14M instance, that direction is the observed same-prefix tail swap for all three failures.

**Pre-result refinement.** Before reading the new replay output, `docs/PREFIX_CONDITIONED_DECOMPOSITION.md` separated each shared-prefix pair into:

- conditioned tail-separation alignment with the base pair direction;
- a common third-order midpoint bias.

Both tail orders are simultaneously recoverable against each other iff

`alignment > |midpoint_bias|`.

**Evidence.** Midpoint-decomposition replay workflow `33245010517`; artifact `9712666884`. The replay again exactly matched the portable finite-pair basis and all six endpoint hashes.

**Result.** For prefixes A/B/C respectively:

- alignment: `0.019095`, `0.005551`, `0.013423`;
- midpoint bias: `-0.137284`, `+0.177508`, `+0.245139`;
- forward/reverse boundary scores:
  - A/BC: `-0.118189`, `+0.156379`;
  - B/AC: `+0.183059`, `-0.171957`;
  - C/AB: `+0.258562`, `-0.231716`.

The conditioned tail separation retains only a tiny positive projection along the static pair direction, while the common midpoint bias is about `7.2x`, `32.0x`, and `18.3x` larger in magnitude. Its sign selects exactly the surviving tail order in each prefix pair.

**Revised interpretation.** On this instance, the static pair decoder fails primarily because useful conditioned tail separation nearly collapses while a much larger third-order midpoint drift pushes both endpoints toward one static-pair candidate. It is not best described as a simple commutator reversal.

**Epistemic limit.** The midpoint/alignment formulas are exact decompositions of the already-observed endpoints. They explain this instance; they do not by themselves demonstrate generalization or prediction.

**Decision.** Keep T2 unchanged. Treat the independent structure of errors and partial chronology across fresh codebooks as the real falsifier. Keep 31M blocked.

---

## 2026-08-29 — J013 — Literature update narrows the novelty boundary around training memory and tomography

**Question.** Does the newer state-dependent / optimizer-memory framing collide with existing 2026 work?

**Search outcome.** Yes, materially.

Newly reviewed neighboring work includes:

- Sevetlidis & Pavlidis, **“Process-Tensor Tomography of SGD: Measuring Non-Markovian Memory via Back-Flow of Distinguishability”** (AISTATS 2026; arXiv:2601.16563). It models training as a multi-time process from controlled interventions to observables, measures training memory/non-Markovianity, and uses optimizer-state reset as a causal break.
- Sevetlidis & Pavlidis, **“Training Memory in Deep Neural Networks: Mechanisms, Evidence, and Measurement Gaps”** (arXiv:2601.21624). It already organizes optimizer state, data order, nonconvex path, and auxiliary state as training-memory mechanisms.
- Xu, **“Stored in Optimizer State, Valued by Later Training”** (arXiv:2608.20442). It treats parameters and optimizer moments as a full trainer state, identifies first-moment transport, and shows later routes assign different behavioral value to stored ancestry.
- Guo, **“Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions”** (arXiv:2608.24593). It directly studies optimizer-state transport interacting with future minibatch paths.

**Consequences.** The following are no longer defensible as ChronoTrace novelty claims:

- training has memory;
- training can be described as a multi-time process;
- “tomography” applied to training memory;
- optimizer moments carry history;
- future continuation changes the effect/value of earlier perturbations;
- a taxonomy of optimizer/data/path state as memory channels.

**Branding correction.** “Training-History Tomography” should be deemphasized as an umbrella phrase because “Process-Tensor Tomography of SGD” already occupies closely adjacent terminology.

**Remaining candidate gap.** The strongest still-defensible target is narrower:

> post-hoc reconstruction or partial-order identification of an **unknown semantic macro-stage chronology** from a finished model, together with the interaction degree and observation/access regime required for that inverse problem.

The process-tensor paper is highly adjacent but positions itself as a measurement/non-Markovianity witness under known interventions. Searches of its available full text for “recover”, “infer”, “unknown”, and “chronology” did not reveal a centered unknown-permutation reconstruction task. This distinction is promising but is not proof of firstness.

**Decision.** Update `docs/LITERATURE_MAP.md`, the repository README, and theory language. Future Adam experiments can study how optimizer memory changes **inverse chronology identifiability**, but cannot claim optimizer memory itself as new.

**Next novelty falsifier.** Continue searching specifically for endpoint-only or simulator-assisted reconstruction of unknown task/stage order, partial chronology, interaction-order inversion, or path-signature inversion of training trajectories.

---

## Journal checkpoint — after T1 refinement and literature update

What is currently supported:

- controlled reset-SGD chronology begins through noncommutative interaction geometry;
- finite stage operators require progressively richer interaction models as locality is lost;
- the portable Pythia-14M 16-update instance retains strong coarse chronology but defeats the static pair model on tail order;
- on that instance, prefix conditioning nearly collapses useful tail alignment while a larger third-order midpoint drift selects one tail candidate per prefix.

What remains unconfirmed:

- whether this coarse-to-fine / midpoint-dominance structure repeats on independent codebooks;
- whether a fixed low interaction order can reconstruct larger chronology spaces efficiently;
- whether the same information is observable behaviorally rather than through weights;
- how realistic optimizer/schedule channels change inverse identifiability.

Current independent falsifier:

> **T2 — four fresh mechanically derived Pythia-14M codebooks × stage lengths `{1,2,4,8,16,32}`, with all conditions reported under a frozen no-selection rule.**

31M remains blocked until T2 is interpreted.

---

## 2026-08-29 — J014 — T2 independently replicates coarse chronology but falsifies the expected stage-length transition at eta=1e-4

**Question.** Across fresh tokenizer-safe codebooks, does the static finite-pair decoder move from a low-order regime into a higher-order/prefix-conditioned regime as stage length increases from 1 to 32 updates?

**Frozen before results.** Four codebook seeds were derived mechanically from a SHA-256 label; stage lengths were exactly `{1,2,4,8,16,32}`; model was `EleutherAI/pythia-14m-deduped@step143000`; optimizer was deterministic plain SGD; learning rate stayed at the chronology-blind frozen `1e-4`; all six A/B/C histories and every condition had to be reported; no condition could be selected by chronology accuracy. Protocol SHA256: `6eb95c404243cd64b74f4b761d99ae2db3c255c8ce68ce443c41f0c29426b7ab`.

**Evidence.** Workflow `33245167776`; aggregate artifact `pythia-14m-t2-aggregate`, artifact ID `9713044534`.

**Pre-registered checks.** All three hard structural checks passed:

1. every seed's first observed full-order failure had non-positive tail robustness;
2. `72/72` observed errors across the complete map were same-prefix tail swaps;
3. all `4/4` fresh seeds showed a first-stage reconstruction advantage.

**Raw reconstruction pattern.** At nearly every condition, static finite-pair decoding recovered `3/6` complete histories, `6/6` first stages, `15/18 = 0.8333` pairwise precedence relations, and mean Kendall tau `2/3`. There were two small deviations: seed `2700450505` at 2 updates recovered `2/6`, and seed `119806841` at 32 updates recovered `4/6`. Across all four seeds and six lengths there were 72 errors total, and every one was a same-prefix swap of stages 2 and 3.

**State-conditioning pattern.** Mean base/conditioned tail-commutator cosine remained near zero across most conditions, while mean relative commutator drift stayed near 1.0. Thus the useful late-pair interaction measured at the base checkpoint is already badly misaligned with the interaction after the first stage.

**Important negative result.** The pre-T2 narrative expected a stage-length transition: pairwise recovery should work at short stages and degrade as prefixes become more nonlocal. T2 does **not** show that transition at `eta=1e-4`. Every fresh seed already fails full-order recovery at **one update per stage**. Therefore the hypothesis “stage length is the variable that moves Pythia-14M out of the pairwise regime under the frozen `1e-4` rate” is rejected.

**Implementation audit.** The T2 runner reloads the supplied flat weight vector before every stage map, creates a fresh zero-momentum SGD optimizer for every invocation, and constructs each stage-length condition from the same frozen `theta0`. No obvious weight or optimizer-state carry-over across pair probes or stage-length conditions was found. This does not substitute for an independent implementation, but the first audit found no state-reuse explanation for the one-step result.

**Interpretation.** The robust result is **partial chronology**, not full reconstruction: on these controlled Pythia-14M tasks, a static base-anchored pair basis identifies the first stage extremely reliably while failing on later order through a highly structured tail-swap mode. The surprising one-step failure says that `eta=1e-4` is already outside the useful static-pair asymptotic regime for these task gradients, or that prefix conditioning enters strongly enough that stage-count alone is the wrong locality variable.

**Decision.** Keep 31M blocked. Do not tune stage length. Test the actual asymptotic control variable: one-step learning rate.

**Next falsifier.** T2b is already frozen on four new mechanically derived codebooks, one update per stage, and rates `{1e-6,3e-6,1e-5,3e-5,1e-4}`. It asks whether sufficiently small `eta` restores the predicted pairwise `O(eta^2)` chronology regime while signatures remain numerically identifiable. T2b must remain manual and should not be launched merely because code changed.
