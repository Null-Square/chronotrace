# Hypotheses

The project uses explicit, falsifiable hypotheses. Status labels are provisional research-state summaries, not publication claims.

## H1 — Path persistence

A detector can distinguish `AB` from `BA` on model seeds that were not used to fit the detector.

**Original MVP evidence target:** seed-held-out balanced accuracy above chance with a confidence interval that excludes 0.5 under the predefined confirmatory split.

**Current status:** **not established in the non-trivial regime.** Phase-0 v1 separated AB/BA perfectly, but capability-only controls also separated them perfectly. The result was rejected as recency/capability-confounded and confirmation was never touched.

## H2 — Interaction localization

Features derived from interactions between stage-A and stage-B knowledge carry more history information than features derived from either stage alone.

**Prediction:** `A × B` probes outperform matched `A-only` and `B-only` controls.

**Current status:** **mixed / superseded as the main mechanism.** The original behavioral Order-Witness can be erased by common continuation before capability is equalized. Later white-box operator experiments show strong interaction geometry, but they are not the same behavioral feature claim.

## H3 — Benchmark invisibility

History remains detectable when standard task performance for different histories is approximately matched.

**Prediction:** a history detector remains useful after excluding runs with material capability imbalance and after controlling for scalar task metrics.

**Current status:** **not established.** Phase-0b failed to create robust capability equivalence with shuffled common rehearsal. The research program moved to mechanism-first endpoint geometry instead of continuing to tune washout.

## H4 — Forensic half-life

Identical subsequent training weakens the history signal gradually rather than erasing it immediately.

For common continuation stage `C_t`, define a history score `S(t)`. We will estimate the decay of `S(t)` with continued optimization. The characteristic decay interval is the **training-history half-life**.

**Current status:** **unsupported by Phase-0b as originally formulated.** Shuffled common rehearsal produced non-monotonic capability gaps and erased the current Order-Witness before robust endpoint equivalence. A future half-life study needs a better-defined trace observable.

## H5 — Stage-type signature

Different learning mechanisms can leave distinguishable historical traces even when they teach overlapping capabilities.

Candidate mechanisms include:

- continued pretraining;
- supervised fine-tuning;
- preference optimization;
- distillation;
- targeted unlearning.

**Current status:** outside the current mechanism gate.

## H6 — Partial-order recovery

For more than two stages, pairwise or structured forensic evidence can recover a non-trivial part of the hidden training order.

Evaluation can use Kendall rank correlation for total orders and edge precision/recall for partial-order graphs.

**Current status:** **hypothesis strengthened, but not confirmed.** In the final portable Pythia-14M scale gate, full finite-pair recovery was only `3/6`. However, all three errors preserved the true first stage and swapped only positions 2 and 3. Descriptively, the single locked instance had first-stage accuracy `6/6`, pairwise precedence accuracy `15/18 = 83.3%`, and mean Kendall tau `2/3`. These diagnostics were noticed after the result and require independent-seed confirmation.

## H7 — Acquisition mechanism

A model's endpoint can contain evidence that separates direct memorization, distributed reconstruction, rule learning, distillation, and later adaptation as different acquisition paths.

**Current status:** long-term; must not be claimed from current synthetic chronology experiments.

## H8 — Black-box transfer

Order witnesses discovered on controlled shadow models can retain predictive power when only model outputs are available for an unseen target.

**Current status:** intentionally deferred until white-box identifiability is understood.

## H9 — State-conditioned interaction hierarchy

The interaction needed to identify chronology depends on where in parameter/optimizer state space later stages are executed.

For finite stage maps define

`C_BC(theta) = F_C(F_B(theta)) - F_B(F_C(theta))`.

Then histories sharing prefix A satisfy the exact identity

`E_ABC - E_ACB = C_BC(F_A(theta_0))`.

The current base finite-pair decoder uses `C_BC(theta_0)`. Define the prefix-conditioned drift

`T_{A;BC} = C_BC(F_A(theta_0)) - C_BC(theta_0)`.

**Prediction H9a:** the three Pythia-14M tail-swap failures are associated with large directional contamination from `T_{prefix;tail-pair}` relative to the base pair-signature margin.

**Prediction H9b:** as stage duration / displacement is reduced, base-state pair interactions should become sufficient more often because conditioned commutator drift shrinks faster than the leading pair-order signal.

**Prediction H9c:** the minimum interaction order required for chronology recovery increases as stage operators move farther from the base state or their commutator fields vary more strongly over the reachable path.

**Current status:** **new primary theory, untested directly.** It was derived from the exact finite-stage identity after the reproducible 14M failure pattern was observed.

## H10 — Training-history interaction order

A complete chronology endpoint admits an exact ordered interaction decomposition by Möbius inversion over stage subsets. Truncating after order K gives a polynomial-probe approximation whose residual contains interactions of order `K+1` and above.

**Prediction:** there exist regimes where K=2 fails but K=3 restores chronology identification on independently generated histories, and the transition is predictable from interaction/separation diagnostics rather than chosen post hoc.

**Complexity target:** for fixed K, ordered interaction construction scales as `O(N^K)` stage-map extensions versus factorial full-history replay.

**Current status:** the K=2 finite-pair construction is implemented. On the controlled tiny transformer it remains 6/6 through 256 updates/stage. On the frozen Pythia-14M 16-update protocol it reproducibly achieves only 3/6 full-order recovery. K=3 has not yet been tested as a replay-efficient decoder on `N>=4` stages.

## H11 — Optimizer-memory trace

When optimizer state persists across stage boundaries, history is encoded not only in the weight geometry but also in hidden optimizer state such as Adam moments.

**Prediction:** persistent Adam should produce chronology effects that differ measurably from optimizer-reset Adam at matched weight checkpoints, and weights-only reconstruction may lose information available in the full `(theta,m,v,step)` state.

**Current status:** theory-only. Current mechanism experiments intentionally use reset / stateless plain SGD to isolate the weight-geometry channel.

## Null hypotheses

The project must take these seriously:

- seed/world-specific variance dominates the apparent partial-order signal;
- the 6/6 first-stage observation on Pythia-14M is specific to one synthetic codebook/world instance;
- any separability is explained by ordinary performance differences;
- history information exists only in weights but is too unstable / high-order for practical reconstruction;
- common continuation training erases useful traces too quickly;
- chronology is recoverable only in synthetic tasks with unrealistically strong stage separation;
- interaction order grows too quickly with stage duration for polynomial-order reconstruction to beat replay;
- realistic optimizer state and stochasticity alias many histories after projection to weights;
- current novelty disappears under a more complete literature audit.
