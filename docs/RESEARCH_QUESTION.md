# Research Question

## Primary question

Can a finished language model retain recoverable information about the **order of learning stages** that produced it?

ChronoTrace studies the inverse problem:

> Given a finished model endpoint and candidate training-stage procedures, what aspects of the hidden training chronology are identifiable, at what interaction order, and under what conditions?

This is different from the better-studied forward problem:

> How does training order affect the final model, or which order should we choose?

## Why order can be encoded

Sequential training stages are generally noncommuting operators:

`F_B(F_A(theta)) != F_A(F_B(theta))`.

In the local regime the first order-dependent term is a Lie-bracket / commutator effect. For complete finite training stages, the endpoint admits a hierarchy of singleton, pairwise, triple, and higher ordered interactions.

The research question is therefore not simply whether two histories produce different weights. They almost always can. The hard inverse question is whether those differences have enough reusable structure to recover chronology **without exhaustively replaying every candidate history**.

## Current refinement after the first scale gate

The project initially targeted complete AB/BA or multi-stage order recovery from a fixed pairwise signature. Controlled tiny-transformer experiments supported this strongly, but the first portable Pythia-14M three-stage gate reproducibly recovered only `3/6` complete orders.

The error structure was not random: all three wrong predictions preserved the true first stage and swapped only the final two stages. On that single locked instance the base pairwise decoder descriptively recovered `15/18` precedence relations and the first stage `6/6`.

This motivates the current sharper question:

> Is training chronology encoded **hierarchically and state-conditionally**, such that base-state pair interactions recover coarse precedence while prefix-conditioned / higher-order interactions are required for deeper order?

For example, the exact difference between histories `ABC` and `ACB` is

`C_BC(F_A(theta_0))`,

where

`C_BC(theta) = F_C(F_B(theta)) - F_B(F_C(theta))`.

The current finite-pair basis uses `C_BC(theta_0)`. Their difference is a third-order prefix-conditioned interaction.

## Scope of the next paper candidate

A credible first paper no longer needs to force one binary detector story. It should establish, if the data support it:

1. **Inverse mechanism:** order information appears as noncommutative training-stage interactions.
2. **Interaction hierarchy:** singleton/pair/triple truncations have measurable regimes of sufficiency and failure.
3. **Partial identifiability:** coarse chronology can remain recoverable even when complete order is not.
4. **Predictive boundary:** interaction/separation diagnostics predict when a lower-order decoder will fail.
5. **Replay advantage:** a fixed interaction order can reconstruct or constrain a factorial chronology space using polynomially many stage probes.
6. **Scale evidence:** the mechanism generalizes beyond the tiny controlled transformer to independently generated Pythia-scale experiments.

If these do not generalize, the project should stop rather than reinterpret one synthetic instance as evidence.

## What this is not

ChronoTrace is not primarily:

- training-data membership inference;
- model lineage verification;
- document provenance ranking;
- contamination detection;
- curriculum optimization;
- a watermark inserted during training.

Those areas are relevant baselines and neighboring work. ChronoTrace targets latent chronology created by ordinary optimization without requiring a purpose-built provenance marker.

## Identifiability boundary

Not every pair of histories is distinguishable. If two histories induce the same distribution over the observable checkpoint, no forensic method can recover the difference.

Real training also has hidden state. For Adam-like optimization the true state includes weights, optimizer moments, scheduler step, RNG/data position, and mixed-precision state, while a published checkpoint often exposes only weights. Projecting away that hidden state can make otherwise distinct histories observationally equivalent.

Therefore the primary scientific object is the boundary:

> Which stage interactions, model states, optimizer memories, and observation channels make training chronology identifiable?

Negative results at this boundary are first-class results, not failed engineering.
