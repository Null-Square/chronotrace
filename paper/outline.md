# Paper Outline

## Working title

**ChronoTrace: The Inverse Problem of Sequential Learning in Language Models**

Alternative framing if the interaction-order result becomes the central contribution:

**ChronoTrace: Reconstructing Training Order from Noncommutative Learning Interactions**

No title should use “training-history tomography” as a novelty claim because related 2026 work already uses process-tensor tomography for SGD memory.

## Abstract — target structure

1. A final neural network is the endpoint of an ordered training process, but most provenance methods ask about membership, influence, or ancestry rather than the unknown order of semantic learning stages.
2. Define **training-history reconstruction**: infer a hidden macro-stage chronology or partial order from a finished model under an explicit observer-access regime.
3. Show why simple endpoint differences are insufficient: exact affine controls separate ordinary recency/translation effects from genuinely noncommuting weight geometry.
4. Derive a graded interaction hierarchy: local order enters through second-order commutators; finite stages admit exact singleton, pair, triple, and higher interactions; later pair effects are conditioned on the earlier prefix.
5. Validate the mechanism first on smooth systems and a tiny causal transformer, then on a numerically reproducible Pythia-14M bridge.
6. Report the independent T2 finding: over 144 three-stage endpoints from four fresh synthetic codebooks, a static pair decoder recovers the first stage 144/144 while exact full order is 72/144; every one of 72 errors is a same-prefix tail swap.
7. Report the T2b local-step asymptotic result only after the frozen experiment completes. If it validates restoration, connect the controlled commutator theorem to model-scale interaction order; if it fails, present the resulting identifiability boundary instead.
8. State limitations clearly: current scale experiments are replay-capable white-box mechanism experiments, not yet black-box provenance.

## 1. Introduction

- Modern language models are produced by ordered mixtures of pretraining, continued training, domain stages, alignment stages, synthetic-data stages, and other interventions.
- A model artifact usually exposes the endpoint but not the complete process that produced it.
- Existing provenance work primarily asks:
  - was example/document X used?;
  - did model B descend from model A?;
  - which data influenced a behavior?;
  - does a model retain memory of a disclosed training transcript?
- ChronoTrace asks a different inverse question:

> Given candidate learning stages and a finished model, what can we infer about the **unknown order** in which those stages occurred?

- The answer need not be a complete permutation. A scientifically meaningful output can be a robust prefix or partial order.

### Contributions — current candidate set

1. **Inverse formulation.** Formalize unknown semantic macro-stage chronology reconstruction under explicit observer-access assumptions.
2. **Exact null theory.** Give solvable affine/quadratic controls that separate trivial recency/translation chronology from noncommuting geometric chronology and characterize exact common-continuation decay.
3. **Interaction hierarchy.** Develop exact finite stage interactions and the notion of training-history interaction order; show that later pair interactions are prefix-conditioned.
4. **Mechanism validation.** Verify the local commutator scaling law and finite interaction decoders in controlled neural systems.
5. **Structured Pythia result.** Establish under portable numerical execution that Pythia-14M retains robust coarse chronology while a base-anchored pair truncation loses later ordering in a highly structured same-prefix manner across independent synthetic instances.
6. **Partial chronology.** Introduce recoverable prefix depth / partial-order reporting rather than forcing an unsupported total-order claim.
7. **Asymptotic bridge.** T2b conditionally tests whether decreasing one-step learning rate restores the local pairwise regime predicted by the theory. Include this contribution only if the frozen result supports it.

## 2. Related Work and Claim Boundary

### 2.1 Forward task/data ordering

- curriculum learning;
- continual-learning task-order optimization;
- data ordering and scheduling;
- geometric/Lie-bracket analyses of beneficial stage order.

Distinction: these works choose, optimize, or analyze order in the forward direction. ChronoTrace treats the order as latent and attempts to invert it from an endpoint.

### 2.2 Membership and model provenance

- membership inference;
- palimpsestic membership / known-transcript order correlations;
- model ancestry and lineage signatures;
- training-data attribution.

Distinction: inclusion/ancestry/known-transcript dependence is not the same as recovering an unknown semantic stage permutation.

### 2.3 Training memory and optimizer state

- process-tensor approaches to SGD memory;
- optimizer-state transport/memory;
- historical-learning methods.

Claim boundary: “training has memory,” “optimizer state carries history,” and “tomography of SGD memory” are not ChronoTrace novelty claims.

### 2.4 Ordered-path mathematics

- BCH/Magnus expansions;
- Lie brackets;
- Chen/path signatures and iterated interactions.

Claim boundary: the mathematics of noncommuting ordered paths is classical. The candidate contribution is applying a graded empirical interaction representation to the **inverse chronology problem for trained language models**.

## 3. Problem Formulation

### 3.1 Extended training state

Let

`z_t = (theta_t, optimizer_t, scheduler_t, rng_t, stream_t, ...)`.

A stage is an operator `F_i` over this extended state. The released artifact often exposes only a projection, typically weights:

`theta_final = P(F_{pi_N} ... F_{pi_1}(z_0))`.

Define the hidden chronology `pi` and the observation/access regime available to an auditor.

### 3.2 Access regimes

Separate at least:

- black-box generation;
- logits/probabilities;
- activations;
- weights only;
- checkpoint + candidate stage descriptions/data;
- replay-capable white-box access to the original/base state and stage procedures.

The current Pythia mechanism bridge lives in the strongest replay-capable regime.

### 3.3 Outputs

Do not require only an exact permutation. Define:

- exact total-order recovery;
- pairwise precedence accuracy;
- recoverable prefix depth;
- robust partial order / edge set;
- uncertainty or ambiguity among candidate histories.

## 4. Exact Null Theory: Affine Training Stages

For quadratic full-batch losses, finite repeated-GD stages are exact affine maps

`F_i(theta) = A_i theta + c_i`.

Use this solvable model to establish:

1. different stage optima can produce chronology in one dimension even when linear maps commute — a formal recency confound;
2. shared optimum + commuting maps gives exact order unidentifiability;
3. noncommuting matrices create geometric order dependence;
4. under a common continuation W, history difference obeys exactly
   `Delta_k = A_W^k Delta_0`,
   giving mode-dependent spectral provenance half-lives.

This section defines negative controls for all later empirical claims.

## 5. From Local Commutators to Finite Interaction Order

### 5.1 Local one-step result

For reset SGD:

`theta_AB - theta_BA = eta^2 (H_B g_A - H_A g_B) + O(eta^3)`.

The shared first-order displacement is chronology-blind; pure geometric chronology first enters at second order.

### 5.2 Exact finite interactions

For complete finite stage operators, use Möbius/interaction decomposition over ordered subsets:

- degree 1: singleton stage effects;
- degree 2: directed finite pair interactions;
- degree 3: triple / prefix-conditioned interactions;
- higher degree: deeper path dependence.

Define **training-history interaction order** as the smallest retained degree sufficient for a specified chronology claim and robustness level.

### 5.3 Prefix conditioning

For shared prefix A:

`E_ABC - E_ACB = C_BC(F_A(theta_0))`.

A static degree-2 decoder measures `C_BC(theta_0)`. Their difference is an exact degree-3 effect.

Decompose the failure into:

- conditioned tail separation/alignment;
- common higher-order midpoint bias.

For the motivating Pythia-14M instance, the aligned conditioned tail signal nearly collapses while the midpoint bias dominates it.

### 5.4 Probe complexity versus inference complexity

Measuring all interactions through fixed degree K can require `O(N^K)` stage-map extensions, but scoring all N! total orders is still factorial without additional structure.

Keep probe complexity and inference complexity as separate claims.

## 6. Controlled Mechanism Experiments

### 6.1 Smooth system scaling law

Verify predicted orders of the shared displacement, commutator signal, and truncation remainder.

### 6.2 Tiny causal transformer

Show the same scaling law in a real LM loss/transformer parameterization.

### 6.3 Multi-update macro stages

Compare:

- local HVP approximation;
- differential macro operator;
- exact finite directed-pair interactions.

Result already established: exact finite pairs retain controlled full-order recovery far beyond the local approximation regime.

## 7. Pythia-14M Reproducible Scale Bridge

### 7.1 Chronology-blind stability gate

Freeze tokenizer/codebook/data and select a stable common SGD rate without observing chronology outcomes.

### 7.2 Numerical reproducibility gate

Document the initial contradictory results across hosted CPU backends and the portable numerical execution intervention.

This section is scientifically important: small endpoint-geometry differences cannot be interpreted until the training operator is reproducible enough for exact comparison.

### 7.3 Frozen 16-update result

Portable result:

- exact full order: `3/6`;
- first stage: `6/6`;
- pairwise precedence: `15/18`;
- all mistakes are swaps of positions 2 and 3.

Treat this as a negative for static full-order pair decoding and a hypothesis-generating partial-order observation.

## 8. T1: Exact State-Conditioned Failure Mechanism

Replay the exact frozen tensors and decompose each shared-prefix pair.

Report:

- base vs conditioned commutator norm/angle;
- exact degree-3 residual;
- midpoint bias;
- aligned conditioned tail signal;
- tail recoverability inequality.

Main motivating-instance result: conditioned tail separation is strongly attenuated/rotated and the common third-order midpoint bias is substantially larger than the remaining aligned signal.

Explicitly state that directional contamination `chi` is an exact decision identity, not an independent predictive result.

## 9. T2: Independent State-Conditioned Interaction Map

Frozen independent protocol:

- four mechanically derived fresh tokenizer-safe codebooks;
- stage lengths `{1,2,4,8,16,32}`;
- fixed `eta=1e-4`;
- all six A/B/C histories.

Aggregate:

- `72/144` exact full order;
- `144/144` first-stage recovery;
- `72/72` errors are same-prefix tail swaps;
- `360/432` pairwise precedence;
- mean Kendall tau `2/3`.

Interpretation:

- independent support for structured coarse chronology and state-conditioned late-stage ambiguity;
- no support for the proposed stage-duration transition, because the same failure is already present at one update/stage.

## 10. T2b: One-Step Learning-Rate Asymptotic Map

This section is currently frozen but **result pending**.

Fresh codebooks, one update/stage, rates

`{1e-6, 3e-6, 1e-5, 3e-5, 1e-4}`.

Pre-registered prediction:

- degree-2 pair separation approaches `O(eta^2)`;
- omitted degree-3 effects approach `O(eta^3)`;
- therefore relative contamination should shrink with eta and exact pairwise full-order recovery should reappear before numerical resolution is lost.

Possible paper branches:

### If T2b restores full order

Use it as the bridge connecting the controlled theorem to Pythia-scale finite training geometry. Then proceed to the four-stage interaction-order experiment.

### If T2b fails while signatures remain identifiable

Do not force the theorem narrative. Treat the result as evidence that the practical Pythia operator is outside the assumed smooth/local scaling regime or that an omitted mechanism remains. Diagnose before scaling.

### If low rates are numerically unresolved

Use a numerical-only precision adjudication on the identical frozen protocol and report the identifiability floor.

## 11. Hierarchical / Partial Chronology Decoding

Define prefix groups `G(p)` and group error

`E(p) = min_{pi in G(p)} ||theta* - theta_hat_pi||`.

Define prefix margin and **recoverable prefix depth**. Also define pairwise precedence edge margins.

Motivation: T2 shows that coarse chronology can be perfectly preserved even while exact total order is not.

For N>3, consider prefix-adaptive beam decoding with conditioned interaction probes allocated only to ambiguous branches.

Do not claim polynomial-time inference until hypothesis-expansion complexity is measured directly.

## 12. Conditional T3: Four-Stage Interaction-Order Reconstruction

Only execute if T2b justifies the local-to-finite bridge.

Use at least four stages so degree-3 interactions are not equivalent to replaying every complete three-stage history.

Report separately:

- degree-2 versus degree-3 total-order recovery;
- prefix depth 1/2/3;
- pairwise partial order;
- probe executions;
- chronology hypotheses expanded;
- interaction residual/separation geometry.

The central question becomes:

> Does increasing interaction degree recover progressively deeper chronology without exhaustive full-history replay?

## 13. Continuation / Provenance Survivability

Return to common continuation only after the inverse mechanism is understood.

Use the affine spectral theory as a null/reference and measure how different chronology modes decay under shared subsequent training.

Avoid claiming one universal “history half-life”; decay can be direction/mode dependent.

## 14. Realistic Training-State Channels

Add one channel at a time:

1. reset-SGD weight geometry;
2. changing learning-rate/global-step schedule;
3. persistent momentum/Adam state;
4. stochastic minibatch/data-order state;
5. mixed precision/distributed numerics.

Keep first-order clock/optimizer memory distinct from second-order reset-SGD geometric chronology.

## 15. Weaker Access and Semantic Generalization

Only after the replay-capable mechanism is established:

- replace exact synthetic examples with semantic stage descriptions;
- test approximate/reconstructed stage data;
- move from weights to activations/logits;
- test shadow-model transfer;
- compare directly to capability, recency, and palimpsestic provenance baselines.

The strongest eventual result would be transcript-free temporal forensics under weaker access, but the present paper should not assume that result in advance.

## 16. Limitations and Identifiability

- access-regime dependence;
- non-identifiable histories after projection;
- numerical sensitivity of white-box endpoint geometry;
- finite candidate-stage set;
- synthetic-to-natural distribution shift;
- possible optimizer/schedule confounders;
- computation required to replay candidate stages;
- legal/ownership conclusions are outside the method’s evidentiary scope.

## 17. Discussion

Frame the broader question around **how much of an optimization path survives in a trained artifact and at what interaction depth it becomes identifiable**, rather than the already-known claim that training order can matter.
