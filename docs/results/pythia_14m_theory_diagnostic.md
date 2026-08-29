# Pythia-14M T1 State-Conditioned Interaction Diagnostic

Date: 2026-08-29

Status: **mechanistic support on the already-observed frozen instance; not independent generalization evidence**.

## Question

The portable Pythia-14M finite-pair bridge reproducibly recovered only 3/6 full A/B/C histories, with every error preserving the first stage and swapping the final two stages. T1 asked whether this structured failure is explained by omitted third-order / prefix-conditioned interaction geometry rather than arbitrary loss of chronology information.

The pre-written mechanism was:

`E_ABC - E_ACB = C_BC(F_A(theta_0))`,

where the base finite-pair decoder uses `C_BC(theta_0)` instead. Define the prefix-conditioned drift

`T_{A;BC} = C_BC(F_A(theta_0)) - C_BC(theta_0)`.

The second pre-written diagnostic was the exact directional contamination ratio

`chi_(pi,sigma) = -2 <r_pi, d_(pi,sigma)> / ||d_(pi,sigma)||^2`,

for which nearest-signature decoding prefers the true chronology `pi` over competitor `sigma` iff `chi < 1`.

## Frozen evidence identity

Workflow run: `33243747235`

Artifact: `pythia-14m-theory-diagnostic`, artifact ID `9712269414`

Source portable scientific gate: workflow `33219286064`

Model: `EleutherAI/pythia-14m-deduped`, revision `step143000`

Learning rate: `1e-4`

Optimizer: deterministic plain SGD, no momentum, no weight decay

Updates per stage: `16`

The diagnostic refused to proceed unless it reproduced the already-recorded portable tensors. It did:

- finite-pair basis matches portable gate: **true**
- all six history endpoints match portable gate: **true**
- finite-pair basis SHA256: `1afeaa53d3b98c32473fcdfc50c297e6f2db14226d9a5f868ef3aa5366f882c2`
- combined history endpoint SHA256: `ca3e24c4139d87ec4004e78c26c229aa0cf00d86956c5ffd8b4163469622a9c7`
- numerical execution fingerprint: `deaad55af513e78d0c4c1d5636836bcb7d7325be64d8df0b196cd6a66b262d42`

Therefore T1 is a diagnostic replay of the same scientific instance, not a new dataset/model draw.

## Result 1 — exact directional contamination separates every correct and incorrect history

| True history | Decoded | Correct? | Tail-swap competitor | Tail-swap chi | Crosses boundary? |
|---|---|---:|---|---:|---:|
| ABC | ACB | no | ACB | 1.118318 | yes |
| ACB | ACB | yes | ABC | 0.843755 | no |
| BAC | BAC | yes | BCA | 0.817074 | no |
| BCA | BAC | no | BAC | 1.172054 | yes |
| CAB | CAB | yes | CBA | 0.741561 | no |
| CBA | CAB | no | CAB | 1.231791 | yes |

This is exact for the measured finite-pair decoder geometry: every failed history has `chi > 1` against its observed tail-swap competitor, and every correctly decoded history has `chi < 1` against all competitors.

The maximum directional contamination competitor is the observed decoded alternative for all three failures.

Aggregate descriptive reconstruction remains:

- full order: `3/6`
- first stage: `6/6`
- pairwise precedence: `15/18 = 0.833333`
- mean Kendall tau: `2/3`

## Result 2 — the tail commutator changes dramatically after the prefix

| Prefix | Tail pair | Base commutator norm | Conditioned norm | Base/conditioned cosine | Drift norm | Relative drift |
|---|---|---:|---:|---:|---:|---:|
| A | BC | 0.205614 | 0.037957 | 0.103456 | 0.205194 | 0.997954 |
| B | AC | 0.259722 | 0.046565 | 0.030969 | 0.262441 | 1.010468 |
| C | AB | 0.244604 | 0.062985 | 0.052139 | 0.249385 | 1.019547 |

Conditioning on the first stage therefore does not merely perturb the base tail interaction. On this instance it nearly replaces it:

- the conditioned commutators retain only roughly 15–26% of the base norm;
- their cosine with the base commutator is only roughly 0.03–0.10;
- the drift norm is approximately the full base-commutator norm.

The exact triple-residual-difference identity is numerically satisfied to about `2.8e-5` absolute error for all three prefix choices.

## Result 3 — the omitted term is exactly third-order for three stages

The measured pairwise prediction errors equal the exact three-stage interaction residual norms:

- ABC: `0.1737822`
- ACB: `0.1563738`
- BAC: `0.1747150`
- BCA: `0.2054421`
- CAB: `0.1612215`
- CBA: `0.2036168`

Because there are exactly three stages, this residual is the exact third-order Mobius interaction, not an unspecified mixture of third and higher interaction orders.

## Interpretation

T1 strongly supports the specific mechanism proposed after the portable 3/6 result:

1. base-checkpoint pair interactions preserve substantial coarse chronology;
2. the first stage moves the model into a state where the interaction between the two remaining stages is dramatically different;
3. the omitted exact third-order residual is directionally aligned with the tail-swap decision boundary for the three failures;
4. full pairwise chronology reconstruction fails exactly when that directional contamination crosses the theoretical threshold `chi = 1`.

This makes **state-conditioned interaction order** a substantially better theory of the current result than a generic statement that the residual norm became large.

## What this does not establish

T1 does **not** show that the same mechanism generalizes across:

- independent synthetic codebooks/worlds;
- other stage lengths;
- other Pythia checkpoints or model sizes;
- stochastic minibatch training;
- persistent Adam state;
- natural corpora;
- black-box behavioral access.

The hypothesis was motivated by this same frozen instance, so independent instances are mandatory before treating it as a general phenomenon.

## Decision

**Keep the prefix-conditioned interaction theory. Do not move to 31M yet.**

The next test is T2: freeze independent tokenizer-safe codebooks and a stage-length sweep *before* observing their chronology outcomes. T2 will map whether the transition from low-order to higher-order history reconstruction repeats across independent instances and whether `chi`, commutator drift, and partial-order metrics predict that transition.
