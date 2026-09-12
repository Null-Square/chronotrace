# ChronoTrace Theory — Access Regimes, Observability, and Trace Channels

Date: 2026-08-29

Status: theory note written while the pre-registered Pythia-14M T2 map was already frozen and running. It does not change T2.

## 1. Why the access model must be explicit

“Recover the training history from a finished model” is not one inverse problem. It is a family of inverse problems whose difficulty changes radically with what the auditor knows and can observe.

Let a candidate chronology be `pi` and let realistic training act on an extended state

`z = (theta, optimizer_state, global_step, rng_state, data_stream_state, scaler_state, ...)`.

A history produces

`z_pi = F_pi(z_0)`.

An auditor observes only

`o_pi = H(z_pi)`,

where `H` is an observation map.

Chronology is identifiable only relative to a specified candidate family, initial state, training procedure, and observation map.

This should be stated explicitly in every ChronoTrace experiment.

## 2. Access regimes

### R0 — unconstrained endpoint-only inverse problem

The auditor receives one final model but does not know the candidate stage identities, their update operators, or the initial state.

In general this problem is not identifiable. Arbitrarily many operator sequences can be constructed to produce the same endpoint.

ChronoTrace should never imply otherwise.

### R1 — white-box simulator provenance

Known:

- initial checkpoint `theta_0`;
- candidate stage identities/data;
- training algorithm and hyperparameters;
- ability to replay candidate stage operators;
- final weights.

Unknown:

- chronology.

This is the regime of the current commutator, finite-pair, and Pythia mechanism experiments.

The scientific question is whether the map

`pi -> F_pi(theta_0)`

is sufficiently separated that chronology can be reconstructed using fewer / lower-order probes than exhaustive replay of every full history.

This is a legitimate inverse-identifiability problem, but it is not yet a practical black-box forensic audit.

### R2 — weights-only provenance with incomplete simulator knowledge

The auditor has final weights and perhaps the base checkpoint, but only approximate candidate stage models or partial information about their data/training procedure.

Now there are two error sources:

- interaction truncation error;
- simulator/model-mismatch error.

A future robust ChronoTrace method would need to remain identifiable when both are present.

### R3 — black-box behavioral provenance

The auditor cannot inspect weights and observes only model responses to a chosen query set `Q`.

Define

`H_Q(theta) = (f_q(theta))_(q in Q)`,

where `f_q` may be a logit, log probability, margin, or response statistic.

Chronology must then be separated in **behavior space**, not weight space.

This is the eventual Order-Witness regime.

### R4 — trajectory-assisted provenance

The auditor has intermediate checkpoints, optimizer states, logs, or training metadata.

This is easier and highly relevant to compliance, but it is a different problem from final-checkpoint forensics. It should be treated as a separate benchmark/access setting rather than mixed into endpoint-only claims.

## 3. Identifiability is separation after observation

For a finite candidate set `Pi`, define

`delta_H = min_(pi != sigma) d(H(z_pi), H(z_sigma))`.

Exact candidate-history identifiability requires

`delta_H > 0`.

If the observation is perturbed by worst-case error at most `epsilon`, nearest-candidate decoding is robust whenever

`delta_H > 2 epsilon`.

This simple statement clarifies several layers of ChronoTrace:

- full-state histories may be distinct while weight projections collide;
- weight endpoints may be distinct while ordinary benchmark behavior collides;
- a carefully designed Order Witness can recover a direction that a generic benchmark projects into its null space.

For deterministic transformations,

`history -> full state -> weights -> queried behavior`

is a chain of projections. Information about history cannot increase under those projections. In information-theoretic language,

`I(history; behavior) <= I(history; weights) <= I(history; full state)`.

The practical problem is therefore not only whether a chronology trace exists, but whether it is **observable through the permitted interface**.

## 4. Weight-space trace versus behavioral observability

Suppose two histories differ by a small endpoint vector

`Delta_theta = theta_pi - theta_sigma`.

For a scalar query statistic `f_q(theta)`, locally

`f_q(theta_pi) - f_q(theta_sigma) ~= grad f_q(theta)^T Delta_theta`.

For a two-stage reset-SGD chronology,

`Delta_theta ~= eta^2 [g_A, g_B]`,

up to sign convention and higher-order terms.

Therefore a black-box witness is useful when its behavioral gradient has a large projection onto the order-sensitive weight direction:

`|grad f_q(theta)^T [g_A,g_B]|`.

With query noise or model stochasticity, a natural signal-to-noise objective is schematically

`WitnessScore(q) = |grad f_q^T Delta_order| / sigma_q`.

This gives a mechanistic interpretation of the Order Witness:

> an Order Witness is an observation whose sensitivity vector aligns with an otherwise hidden chronology direction.

This also explains Benchmark Invisibility: ordinary tasks can be unchanged because their observation gradients are nearly orthogonal to the chronology subspace even when weight-space chronology is identifiable.

Higher-order chronology should analogously require observations aligned with higher-order / prefix-conditioned interaction directions.

## 5. Real training contains multiple chronology channels

The reset-SGD experiments deliberately isolate one channel. Normal LLM training mixes several.

### Channel G — geometric noncommutativity

Assume:

- constant learning rate `eta`;
- optimizer state reset between macro stages;
- no exogenous stage-position dependence.

For one small gradient step per stage,

`F_A(theta)=theta-eta g_A(theta)`.

Then the first-order displacement is independent of A/B order, while

`theta_AB - theta_BA = eta^2(H_B g_A - H_A g_B) + O(eta^3)`.

Chronology begins at **second order**.

This is the pure weight-geometry channel studied by the current ChronoTrace mechanism program.

### Channel T — exogenous time / schedule asymmetry

Now suppose the first and second positions use different learning rates `eta_1` and `eta_2`, even if gradients are evaluated at the same base state to first order.

Then

`theta_AB = theta_0 - eta_1 g_A - eta_2 g_B + O(eta^2)`

and

`theta_BA = theta_0 - eta_1 g_B - eta_2 g_A + O(eta^2)`.

Therefore

`theta_AB - theta_BA = (eta_2-eta_1)(g_A-g_B) + O(eta^2)`.

Unless `eta_1=eta_2`, chronology is visible already at **first order**.

A cosine schedule, warmup, stage-dependent regularization, curriculum position, or any other global-step-dependent rule can therefore create a direct time-stamp-like provenance channel.

This signal may be useful for auditing, but it is conceptually different from noncommutative path memory.

### Channel O — optimizer-memory asymmetry

Consider simple momentum with persistent velocity

`v_t = mu v_(t-1) + g_t`

`theta_t = theta_(t-1) - eta v_t`,

starting from `v_0=0`.

Ignoring gradient-state drift beyond first order,

`theta_AB = theta_0 - eta[(1+mu)g_A + g_B] + O(eta^2)`

while

`theta_BA = theta_0 - eta[(1+mu)g_B + g_A] + O(eta^2)`.

Thus

`theta_AB - theta_BA = -eta mu (g_A-g_B) + O(eta^2)`.

Persistent optimizer memory can therefore inject chronology into the **weights themselves at first order**, even if the optimizer state is discarded before forensic inspection.

Adam is more nonlinear because its first and second moments, bias corrections, and coordinate-wise normalization all carry history, but the same qualitative point holds: the second stage update depends on statistics of gradients seen earlier.

This means a positive chronology result under realistic persistent Adam cannot automatically be attributed to the geometric Lie-bracket channel.

### Channel S — stochastic-stream chronology

If minibatch sampling, dropout, data shuffling, distributed reduction order, or mixed-precision state is carried continuously through training, the realized stage operator depends on hidden stochastic state.

There are two scientifically different cases:

1. randomness is independently reset / matched across candidate histories, so it acts mainly as observation noise;
2. random/data-stream state is carried across stage boundaries, so chronology changes which random states are paired with which data and becomes part of the provenance signal.

These should not be conflated.

## 6. A hierarchy of provenance mechanisms

The asymptotic picture suggests a useful taxonomy:

| Mechanism | Example | Leading chronology order in simplified expansion |
|---|---|---|
| exogenous time asymmetry | different LR/global step by stage position | `O(eta)` |
| optimizer memory | persistent momentum / Adam moments | `O(eta)` in simple momentum example |
| geometric noncommutativity | constant-step reset SGD | `O(eta^2)` |
| prefix-conditioned interaction | one stage changes a later pair commutator | third interaction order |
| deeper path interaction | nested state-conditioned effects | fourth order and above |

This table is about mechanism order, not guaranteed empirical magnitude.

It suggests that realistic training-history tomography should eventually **factor the trace** rather than report one undifferentiated accuracy number.

## 7. Why current Pythia mechanism experiments still matter

The official Pythia suite was trained on roughly 300B tokens, and the 14M configuration uses Adam with `beta1=0.9`, `beta2=0.95`, cosine learning-rate decay, 1% warmup, weight decay, clipping, and FP16. The public configuration specifies 143,000 training iterations and a roughly 2M-token batch.

The current ChronoTrace bridge intentionally does not reproduce that optimizer dynamics. It uses constant-rate deterministic plain SGD with optimizer reset semantics so that the measured chronology cannot be explained by hidden Adam moments or changing learning rates.

That makes it a **mechanism-isolation experiment**, not a realistic pretraining simulation.

Representative source references:

- EleutherAI Pythia repository: https://github.com/EleutherAI/pythia
- Pythia-14M configuration: https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml

## 8. Theory-driven experimental sequence

The mechanism taxonomy implies a cleaner progression than simply increasing parameter count.

### Step A — finish geometric identifiability

Current T2/T3 program:

- reset optimizer state;
- constant learning rate;
- deterministic batches;
- measure interaction order and partial chronology.

Question: how far can weights alone encode path order through geometry?

### Step B — behavioral projection

Keep the same controlled training histories but replace white-box weight distance with query-space observations.

Question: which chronology directions survive projection into logits/behavior, and can an Order Witness be constructed without directly reading parameters?

### Step C — add optimizer memory as a factorial intervention

Compare otherwise identical histories under:

- optimizer reset at stage boundaries;
- persistent momentum;
- persistent Adam moments.

Question: how much extra provenance is introduced by optimizer memory, and does it survive after optimizer state itself is discarded?

### Step D — add schedule/time asymmetry

Compare constant-rate training with realistic warmup/decay while controlling stage token counts.

Question: how much chronology becomes recoverable because the training clock assigns different effective weights to earlier versus later data?

### Step E — stochastic/naturalistic training

Only after the individual channels are understood should they be combined in a realistic mini-pretraining pipeline.

## 9. Stronger eventual claim

The strongest paper is unlikely to be simply “training order can be predicted.”

A deeper target is:

> **Training chronology is encoded through multiple ordered dynamical channels whose information survives different projections to different degrees. ChronoTrace characterizes when history is identifiable, what interaction order is required, and which part of the signal comes from geometry, optimizer memory, the training clock, or stochastic state.**

That framing naturally connects mechanism, inverse identifiability, black-box observability, and practical provenance auditing without pretending that one controlled decoder already solves all four.
