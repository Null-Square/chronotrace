# Macro-Operator Chronology Decoder

This is the bridge between the one-update commutator theorem and realistic multi-update training stages.

## Motivation

The local Commutator Decoder is exact to second order for one small gradient update per stage. That is a mechanism check, but a real training stage can contain thousands of optimizer updates. Treating such a stage as one effective gradient step is generally wrong once the trajectory moves far enough from the base checkpoint.

ChronoTrace therefore models a complete deterministic stage `D` as an operator

`F_D(theta) = theta + Delta_D(theta)`.

`Delta_D` includes the entire effect of the stage: repeated updates, data traversal, schedules, and any deterministic optimizer state internal to the stage implementation.

For two near-identity stage maps,

`F_B(F_A(theta_0)) = theta_0 + Delta_A + Delta_B + J Delta_B Delta_A + higher order`,

`F_A(F_B(theta_0)) = theta_0 + Delta_A + Delta_B + J Delta_A Delta_B + higher order`.

Thus

`F_B(F_A(theta_0)) - F_A(F_B(theta_0))`

is locally governed by the macro-stage commutator

`B_AB = J Delta_B Delta_A - J Delta_A Delta_B`.

The first-order stage displacements are identical across AB and BA. Chronology again lives in an antisymmetric interaction term, but the object is now the **complete stage map**, not a single loss Hessian.

## Finite-difference cross-stage JVPs

ChronoTrace estimates

`J Delta_B(theta_0) @ Delta_A(theta_0)`

by centered finite difference. Let

`u_A = Delta_A / ||Delta_A||`.

Then

`J Delta_B @ Delta_A`

is approximated by

`||Delta_A|| * [Delta_B(theta_0 + eps u_A) - Delta_B(theta_0 - eps u_A)] / (2 eps)`.

This requires ordinary forward stage executions only. It does not require second derivatives through the training implementation.

That matters because the tiny-transformer HVP experiment exposed a real engineering limitation: PyTorch's fused SDPA/Flash attention path did not provide the double-backward operation required for exact Hessian-vector products. The finite operator formulation is compatible with training stacks that are differentiable only at the first-order update level or are treated as black-box deterministic maps over weights.

## N-stage signature

For candidate stages `1..N`, compute each stage displacement and every directed cross-JVP. For an unordered pair `{i,j}`, define the macro bracket

`B_ij = J Delta_j Delta_i - J Delta_i Delta_j`.

As in the local HVP decoder, a candidate chronology is represented by a signed combination of pairwise bracket vectors after subtracting a symmetric reference.

The current implementation exhaustively scores permutations only for small controlled experiments. The forensic geometry itself requires pairwise stage interactions rather than complete replay of every permutation.

## Computational distinction from exhaustive replay

For `N` candidate stages:

- stage displacements require `N` stage executions;
- centered cross-JVPs require `2 N (N-1)` stage executions;
- geometry construction is therefore `O(N^2)` stage runs;
- exhaustive endpoint replay requires `N!` complete histories, each containing `N` stages, or `N * N!` stage executions.

For three stages this distinction is small. It becomes the point of the method as `N` grows.

A scalable decoder must eventually avoid explicit enumeration of all `N!` signatures as well, for example by estimating pairwise orientations and projecting them onto a transitive ranking. The current repository does **not** claim that step is solved.

## Fixed multi-update gate

Before Pythia, use the same deterministic 1,032-parameter causal transformer from the one-step theorem gate.

Fixed settings:

- optimizer: plain SGD, no momentum;
- per-update learning rate: `0.01`;
- candidate stages: A, B, C;
- updates per stage: `{1, 2, 4, 8, 16, 32, 64}`;
- finite-difference epsilon: `1e-4`;
- full model weights are the forensic endpoint;
- all six A/B/C permutations are evaluated.

Two decoders are compared:

1. **Micro decoder.** Reuses the one-step base gradients/HVPs and substitutes an effective step size `k * lr` for a `k`-update stage.
2. **Macro decoder.** Recomputes the complete `k`-update stage displacement and pairwise finite-difference cross-JVPs.

The gate passes only if:

- both methods recover 6/6 permutations at one update;
- the micro decoder loses perfect recovery somewhere in the fixed sweep;
- the macro decoder still recovers 6/6 through 64 updates;
- AB and BA macro pair scores retain the correct sign through the sweep.

The purpose is not to maximize a benchmark. It is to show that the operator formulation extends the chronology-identifiable regime beyond the one-step Taylor approximation.

## Limits

Passing this gate still does not establish realistic LLM training-history reconstruction. Important unsolved issues include:

- stochastic data order and dropout;
- optimizer state that persists across macro stages;
- unknown or approximate stage data;
- unknown stage duration and hyperparameters;
- parameter-efficient tuning and model merging;
- quantization and checkpoint transformation;
- whether a low-dimensional parameter projection preserves chronology;
- finite-difference cost on frontier-scale checkpoints;
- and partial identifiability when bracket signatures are ill-conditioned.

The next large-model experiment should isolate one of these difficulties at a time rather than bundle all of them into a single Pythia run.
