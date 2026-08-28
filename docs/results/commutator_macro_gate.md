# Commutator and Macro-Stage Gate Results

Date: 2026-08-28

Status: **positive controlled mechanism result; not yet a large-model chronology result**.

This file records the fixed results that justified moving beyond the local one-update experiment. The values come from CI run `33213263777` on PR #9.

## 1. Smooth nonlinear theorem gate

The float64 nonlinear system satisfied the predicted asymptotic orders:

- commutator-prediction remainder slope: `2.99346` (`O(eta^3)` target);
- held-out AB/BA behavior-gap slope: `2.06531` (`O(eta^2)` target);
- shared endpoint-displacement slope: `0.95967` (`O(eta)` target);
- all `3! = 6` stage permutations recovered at every tested step size;
- pairwise ChronoScore approached `+1` for AB and `-1` for BA.

This establishes the algebraic implementation before testing a neural network.

## 2. Tiny causal-transformer theorem gate

Model:

- 1 causal self-attention block;
- explicit eager attention;
- LayerNorm, GELU, and cross-entropy LM loss;
- `1,032` trainable parameters;
- full-weight forensic endpoint;
- plain SGD without momentum.

Observed scaling:

- commutator-prediction remainder slope: `3.00290`;
- held-out loss-gap slope: `1.98968`;
- shared endpoint-displacement slope: `0.99879`.

At the smallest tested step size (`eta=0.0025`):

- AB ChronoScore: `0.99909`;
- BA ChronoScore: `-0.99513`;
- relative inferred-step-size error: `0.0003595`.

Three-stage result:

- all 6 permutations decoded correctly at `eta in {0.02, 0.01, 0.005, 0.0025}`.

Identifiability diagnostics:

- pairwise bracket count: `3`;
- bracket rank: `3`;
- bracket condition number: `3.07410`;
- minimum unscaled permutation-signature separation: `0.490046`;
- normalized higher-order/separation guarantee ratio stayed below `0.253` across the sweep.

This establishes that the inverse second-order chronology geometry survives a genuine causal transformer parameterization and language-model loss.

## 3. Multi-update macro-stage operator gate

Fixed protocol:

- same `1,032`-parameter causal transformer;
- optimizer: plain SGD, no momentum;
- per-update learning rate: `0.01`;
- candidate stages: A, B, C;
- updates per stage: `{1, 2, 4, 8, 16, 32, 64}`;
- centered finite-difference epsilon: `1e-4`;
- all six A/B/C permutations tested.

The **micro decoder** reused only the base one-step gradients/HVPs and substituted an effective step size `k * lr`.

The **macro decoder** treated each complete `k`-update stage as a finite map `F_D(theta)=theta+Delta_D(theta)` and estimated pairwise `J Delta` interactions by centered finite differences of stage runs.

| Updates/stage | Micro correct | Macro correct | Macro minimum margin | Max stage displacement norm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6/6 | 6/6 | `4.50e-05` | `0.01085` |
| 2 | 4/6 | 6/6 | `1.65e-04` | `0.02158` |
| 4 | 2/6 | 6/6 | `5.73e-04` | `0.04271` |
| 8 | 2/6 | 6/6 | `1.87e-03` | `0.08384` |
| 16 | 3/6 | 6/6 | `6.28e-03` | `0.16262` |
| 32 | 5/6 | 6/6 | `2.36e-02` | `0.31768` |
| 64 | 4/6 | 6/6 | `4.36e-02` | `0.61171` |

The local HVP decoder first lost perfect recovery at **2 updates per stage**. The finite macro-operator decoder retained **6/6 recovery through the full predeclared 64-update sweep**.

Pairwise macro ChronoScore signs also remained correct throughout. At 64 updates:

- AB score: `+1.04502`;
- BA score: `-0.90832`.

## Interpretation

This result supports a more useful statement than the original one-update theorem:

> Training chronology can remain recoverable from the antisymmetric interaction of **finite training-stage operators** after the one-step gradient/Hessian approximation has already left its reliable permutation-decoding regime.

It also validates the finite-difference route as a practical alternative to full Hessian-vector products for longer stages.

## What this does not establish

The result does **not** yet show that ChronoTrace works on realistic LLM training pipelines. The experiment still assumes:

- a known base checkpoint;
- exact candidate stage procedures;
- deterministic stage execution;
- plain SGD;
- full access to endpoint weights;
- only three candidate stages;
- and a small model.

No old Phase-0 confirmation seeds were used by this gate.

## Next gate

Before paying for Pythia-scale operator experiments, remove the remaining perturbation-scale choice from the finite-difference method. The next controlled decoder uses exact singleton and ordered-pair stage executions to construct **finite pair interactions** and finite pair commutators. This keeps stage-probe complexity quadratic while moving approximation error into explicit triple-and-higher stage interactions rather than a local epsilon derivative. Only after that fixed comparison succeeds should the preferred operator decoder be scaled to Pythia.
