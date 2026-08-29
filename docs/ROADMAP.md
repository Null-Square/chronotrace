# Roadmap

## R0 — Research scaffold

Status: **complete enough for active experiments**

- inverse sequential-learning problem defined;
- deterministic protocol locks implemented;
- append-only research journal added;
- result reports preserve positive and negative gates;
- exact endpoint / data / runtime fingerprinting available.

## R1 — Behavioral path-persistence validation

Status: **completed as a negative/confounded first route**

- Phase-0 v1 separated AB/BA perfectly but capability-only controls also separated them perfectly;
- shuffled common-tail Phase-0b did not produce robust capability equivalence;
- original confirmation split was never touched.

**Conclusion:** do not spend more effort tuning behavioral washout until the underlying training-history geometry is understood.

## R2 — Noncommutative mechanism

Status: **positive controlled milestone**

- one-step Lie-bracket inverse scaling verified;
- tiny causal-transformer theorem gate verified;
- finite multi-update stage operator decoder outlived the one-step approximation;
- exact finite-pair interaction decoder recovered 6/6 through 256 updates/stage on the controlled tiny transformer.

## R3 — Scale bridge

Status: **Pythia-14M full-order pair decoder failed reproducibly**

Completed:

- chronology-blind LR selection across Pythia 14M/31M/70M;
- tokenizer-controlled, hash-locked synthetic stage data;
- portable numerical execution gate;
- exact three-replica Pythia-14M result.

Final portable 14M result:

- full-order finite-pair recovery: `3/6`;
- first-stage identity, post-hoc diagnostic: `6/6`;
- pairwise precedence, post-hoc diagnostic: `15/18`;
- mean Kendall tau, post-hoc diagnostic: `2/3`;
- triple+ residual comparable to pair-signature separation.

**Decision:** 31M chronology remains blocked. Do not scale a decoder whose interaction model is already known to be insufficient.

## R4 — State-conditioned interaction theory

Status: **current priority**

Theory work:

- formalize exact ordered Möbius interaction decomposition;
- separate singleton, pair, triple, and higher interaction orders;
- model pair commutators as state-dependent fields `C_ij(theta)`;
- quantify prefix-conditioned commutator drift;
- replace conservative residual-norm diagnostics with exact directional-contamination diagnostics;
- distinguish weight-geometry traces from optimizer-memory / scheduler / stochastic traces.

Immediate diagnostic experiment:

- rerun the exact portable 14M frozen instance only to retain vectors needed for theory diagnostics;
- compute triple interactions, conditioned commutator drift, angles, decision-boundary contamination, and per-edge margins;
- do not treat this rerun as independent evidence.

## R5 — Independent 14M falsification map

Before execution, freeze independent world/codebook seeds and a stage-length grid designed to map the interaction-order transition rather than maximize accuracy.

Primary reported endpoints:

- full permutation recovery;
- first-stage recovery;
- pairwise precedence accuracy;
- Kendall tau;
- pair-signature separation;
- prefix-conditioned commutator drift;
- exact directional contamination;
- required interaction order.

**Exit condition:** continue only if the structured partial-order / conditioned-commutator mechanism generalizes beyond the original synthetic instance.

## R6 — Polynomial interaction-order reconstruction

Use at least four stages so an order-3 interaction basis is meaningfully cheaper than exhaustive replay.

Goals:

- compare K=1, K=2, K=3 truncations;
- measure the minimum interaction order required for chronology identification;
- test whether fixed-K construction grows polynomially while the candidate chronology space grows factorially;
- develop an adaptive prefix-conditioned decoder rather than enumerating all full histories.

## R7 — Realistic training state

Only after the deterministic weight-geometry hierarchy generalizes:

- persistent Adam / AdamW moments;
- optimizer reset versus persistence;
- learning-rate schedules and global-step state;
- stochastic minibatch order;
- mixed precision / distributed execution;
- weights-only versus full optimizer-state observability.

The objective is to identify which channel actually carries chronology in realistic training.

## R8 — Model-scale progression

Conditional on R5/R6 success:

- Pythia 31M;
- Pythia 70M;
- different architecture family;
- larger stage corpora / more semantic stage types.

Scale one difficulty at a time. Do not bundle optimizer realism, stochasticity, architecture transfer, and model size into one experiment.

## R9 — Training-history half-life

Revisit only after a mechanistically stable trace quantity exists.

- common continuation;
- trace decay by interaction order;
- layer / parameter-subspace persistence;
- distinguish capability decay from chronology-information decay.

## R10 — Black-box / behavioral transfer

- project white-box chronology signatures into logits / representations;
- discover query-efficient witnesses on shadow models;
- evaluate transfer across seeds and architectures;
- characterize information loss from weights to behavior.

## R11 — Frontier-lab utility studies

Potential authorized applications:

- post-training pipeline audit;
- safety-stage ordering analysis;
- unlearning / relearning chronology verification;
- capability-origin analysis;
- checkpoint governance;
- forensic comparison of internal experimental branches.
