# ChronoTrace

**Inverse temporal provenance for language-model training.**

ChronoTrace studies a specific inverse problem in sequential learning:

> Given a finished model, candidate learning stages, and a defined observation/access regime, what information about the unknown order of those stages is identifiable?

The project is no longer testing the broad claim that “training order matters.” That is well established. The current target is narrower: reconstruct or constrain an **unknown macro-stage chronology** from the endpoint and characterize the **interaction order** required for that inversion.

## Current scientific status — 2026-08-29

ChronoTrace has both negative and positive results. The repository keeps them all in an append-only research journal rather than rewriting the story around successful experiments.

### 1. Original behavioral AB/BA experiment — confounded negative

The first Pythia-70M discovery run classified `A -> B` versus `B -> A` perfectly, but a capability-only baseline also classified the order perfectly. Every matched seed violated the pre-registered capability-equivalence gate.

Conclusion: the original detector was reading ordinary recency / forgetting, not a nontrivial chronology trace. Confirmation was not run.

### 2. Common terminal washout — negative

A shared balanced terminal stage C did not reliably equalize ordinary A/B capability before the original forensic witness disappeared.

Conclusion: do not manufacture endpoint equivalence by blindly extending a washout stage.

### 3. Noncommutative mechanism program — controlled positive

The project pivoted to the geometry of training operators.

For small reset-SGD updates,

```text
theta_AB - theta_BA
  = eta^2 (H_B g_A - H_A g_B) + O(eta^3).
```

Controlled tests established:

- the predicted Lie-bracket scaling law;
- the mechanism in a small causal transformer;
- multi-update macro-stage reconstruction after the one-step approximation fails;
- exact finite-pair reconstruction through long stages on the controlled transformer.

These are mechanism results, not yet realistic LLM provenance.

### 4. Portable Pythia-14M scale bridge — reproducible 3/6 full-order result

A chronology-blind stability gate froze a plain-SGD learning rate before any Pythia chronology result was observed.

Early Pythia-14M executions were host/backend sensitive, so no result was accepted until a portable CPU numerical path produced exact tensor-hash agreement across independent runners.

Under that portable path, all replicas recovered exactly **3/6** three-stage histories.

The error structure is not random:

```text
ABC -> ACB
ACB -> ACB
BAC -> BAC
BCA -> BAC
CAB -> CAB
CBA -> CAB
```

Descriptively on this frozen instance:

- full permutation: `3/6`;
- first stage: `6/6`;
- pairwise precedence: `15/18 = 83.3%`;
- mean Kendall tau: `2/3`.

Every error preserves the first stage and swaps only the final two stages.

### 5. State-conditioned interaction diagnostic — mechanistic support, not generalization

For histories sharing prefix A,

```text
E_ABC - E_ACB = C_BC(F_A(theta_0)).
```

The static finite-pair decoder instead uses the B/C interaction measured at `theta_0`.

On the already-observed Pythia-14M instance, conditioning on the first stage dramatically rotates and shrinks the relevant tail commutator. Base/conditioned cosines are approximately `0.03–0.10`, and commutator-drift norms are roughly as large as the base commutators themselves.

This supports a **state-conditioned interaction** explanation for the structured tail errors, but it was generated from the same instance and therefore is not independent evidence.

### 6. T2 independent interaction map — current falsifier

The next experiment was frozen before its chronology outcomes:

- Pythia-14M at the same checkpoint;
- four independently generated tokenizer-safe codebooks;
- stage lengths `{1,2,4,8,16,32}`;
- all six A/B/C permutations;
- no condition selected or discarded by chronology performance;
- portable deterministic numerical path;
- pre-registered partial-order and prefix-conditioned interaction checks.

Pythia-31M remains blocked until this map is interpreted.

See `configs/pythia_14m_t2.lock.json` and `docs/experiments/PYTHIA_14M_T2_PROTOCOL.md`.

## Current theory

A training stage is a nonlinear operator. For a chronology

```text
pi = (pi_1, ..., pi_N),
```

the endpoint is an ordered composition

```text
E_pi = F_pi_N ... F_pi_1(theta_0).
```

For finite stages, ChronoTrace uses an exact interaction decomposition. The endpoint can be organized by interaction degree:

```text
degree 1: singleton stage effects
degree 2: directed pair interactions
degree 3: prefix-conditioned / three-stage interactions
...
```

This motivates **Training-History Interaction Order**:

> the minimum interaction degree required to distinguish or constrain a candidate training chronology under a specified observation/error tolerance.

The current hypothesis is that low-order interactions may preserve **coarse chronology** even when they are insufficient for the exact total order. Higher-order, prefix-conditioned interactions may then resolve later branches.

For a shared prefix and two possible tail orders, the project now uses an exact decomposition into:

- rotation/shrinkage of the tail-order separation;
- a common higher-order midpoint bias.

Both tail orders are simultaneously recoverable against each other exactly when

```text
alignment > |midpoint_bias|.
```

This quantity is being tested on independent Pythia-14M instances rather than tuned on the motivating result.

## Real training is an extended-state path

The current mechanism bridge deliberately uses reset/plain SGD to isolate **weight geometry**.

Real LLM training is better represented by a state such as

```text
z = (weights,
     optimizer moments,
     global step / scheduler,
     RNG and sampler state,
     mixed-precision state,
     ...).
```

That creates distinct chronology channels:

1. **geometric noncommutativity** — order enters at second order under constant-step reset SGD;
2. **optimizer memory** — momentum/Adam can carry earlier gradients into later updates;
3. **training-clock / schedule asymmetry** — the same data at different global steps can receive different effective weight;
4. **stochastic-stream state** — shuffle, minibatch, dropout, distributed and numerical state can couple to temporal position.

These mechanisms must be introduced separately. Optimizer memory and training non-Markovianity are established neighboring research areas; they are not ChronoTrace novelty claims.

See `docs/THEORY_ACCESS_AND_TRACE_CHANNELS.md`.

## Access regimes

ChronoTrace claims are meaningful only with an explicit access model.

The **current mechanism experiments** use a white-box simulator regime:

- known base checkpoint;
- known candidate stages;
- known training rule;
- ability to replay candidate stage operators;
- access to final weights;
- unknown chronology.

This does **not** yet constitute black-box forensic provenance.

Later stages of the program will ask whether chronology directions observable in weights survive projection into logits or model responses. An Order Witness can be interpreted as a query whose behavioral sensitivity aligns with a chronology-sensitive parameter direction.

See `docs/THEORY_ACCESS_AND_TRACE_CHANNELS.md`.

## Novelty boundary

ChronoTrace must not claim as novel that:

- training order matters;
- training order can leave persistent traces;
- optimization is path dependent;
- training has memory / can be modeled as a multi-time process;
- optimizer moments carry past gradients;
- later training can change the value of an earlier perturbation.

Close work includes palimpsestic black-box provenance, data-order/curriculum studies, process-tensor tomography of SGD, training-memory surveys, optimizer-state transport, unlearning fingerprints, and forward Lie-bracket analyses of sequential learning.

The current candidate contribution is:

> **Post-hoc identification or partial-order reconstruction of an unknown semantic macro-stage chronology from a finished model, together with a graded interaction hierarchy that characterizes when that inverse problem is identifiable.**

That remains a candidate novelty boundary, not a claim of first discovery. The literature map is continuously updated.

See `docs/LITERATURE_MAP.md`.

## Probe complexity versus inference complexity

A fixed degree-K interaction basis can require only polynomially many stage-map probes,

```text
sum_(r=1..K) P(N,r) = O(N^K)
```

for fixed K.

That does **not** make exact chronology decoding automatically polynomial: there are still `N!` candidate total orders, and consistency among pair orientations is combinatorial.

ChronoTrace therefore distinguishes:

- **probe complexity** — cost to construct the interaction representation;
- **inference complexity** — cost to recover a consistent chronology from it.

A future scalable approach may use prefix beams, pruning, partial orders, or constrained ranking rather than exhaustive permutation enumeration.

See `docs/PATH_SIGNATURE_FRAMING.md`.

## Research records

Start here:

1. [`docs/RESEARCH_JOURNAL.md`](docs/RESEARCH_JOURNAL.md) — append-only chronology of hypotheses, experiments, failures, and decisions.
2. [`docs/TRAINING_HISTORY_THEORY.md`](docs/TRAINING_HISTORY_THEORY.md) — current mathematical theory.
3. [`docs/THEORY_ACCESS_AND_TRACE_CHANNELS.md`](docs/THEORY_ACCESS_AND_TRACE_CHANNELS.md) — identifiability/access model and real-training chronology channels.
4. [`docs/PATH_SIGNATURE_FRAMING.md`](docs/PATH_SIGNATURE_FRAMING.md) — relation to graded noncommutative path representations.
5. [`docs/PREFIX_CONDITIONED_DECOMPOSITION.md`](docs/PREFIX_CONDITIONED_DECOMPOSITION.md) — exact shared-prefix tail decomposition.
6. [`docs/LITERATURE_MAP.md`](docs/LITERATURE_MAP.md) — novelty boundary and neighboring work.
7. [`docs/experiments/PYTHIA_14M_T2_PROTOCOL.md`](docs/experiments/PYTHIA_14M_T2_PROTOCOL.md) — current independent falsifier.

## Non-negotiable research rules

1. **Write the mechanism before the expensive test.** New compute must answer a pre-written question or falsifier.
2. **Keep the research journal append-only.** Corrections are new entries, not silent rewrites of history.
3. **Freeze selection rules before outcomes.** Do not tune on chronology performance and then call the result confirmation.
4. **Preserve exact artifacts and hashes.** Model, data, optimizer, codebook, endpoint and numerical execution identity matter.
5. **Separate numerical reproducibility from scientific success.** A perfectly reproducible negative result is scientifically valid.
6. **Separate discovery from independent evidence.** A diagnostic on the instance that generated a hypothesis is explanatory, not confirmation.
7. **Report partial chronology.** Exact permutation accuracy alone can hide meaningful structure.
8. **State the access regime.** White-box simulator evidence must not be described as black-box provenance.
9. **Distinguish chronology channels.** Geometry, optimizer memory, schedule/time and stochastic state require separate interventions.
10. **Do not claim novelty from weak wording.** The novelty claim must survive direct comparison with current training-memory and provenance literature.

## Repository map

```text
chronotrace/
├── configs/                 frozen and design experiment configurations
├── docs/                    theory, literature, protocols, decisions and results
├── paper/                   paper outline and bibliography
├── scripts/                 experiment and aggregation utilities
├── src/chronotrace/         Python package
├── tests/                   deterministic scientific/drift tests
└── .github/workflows/       CI and reproducible experiment runners
```

## Installation

The implementation scaffold targets Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
