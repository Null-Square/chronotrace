# Pythia-14M T1 State-Conditioned Interaction Diagnostic

Date: 2026-08-29

Status: **mechanistic diagnosis on the already-observed frozen instance; not independent generalization evidence**.

## Question

The portable Pythia-14M finite-pair bridge reproducibly recovered only 3/6 full A/B/C histories, with every error preserving the first stage and swapping the final two stages. T1 asked whether this structured failure is explained by omitted exact third-order / prefix-conditioned interaction geometry rather than arbitrary loss of chronology information.

The pre-written mechanism was:

`E_ABC - E_ACB = C_BC(F_A(theta_0))`,

where the base finite-pair decoder uses `C_BC(theta_0)` instead. Define the prefix-conditioned drift

`T_{A;BC} = C_BC(F_A(theta_0)) - C_BC(theta_0)`.

A later, still-before-independent-T2 decomposition separated two possible causes of tail-order failure:

1. change in the actual tail separation relative to the base finite-pair direction;
2. common higher-order midpoint bias moving both actual endpoints toward one of the static-pair candidates.

For a shared prefix p and two tail orders `pij` and `pji`, define

`d0 = P_pij - P_pji`

for the static finite-pair prediction separation,

`dc = E_pij - E_pji`

for the actual conditioned separation, and

`b = (E_pij+E_pji)/2 - (P_pij+P_pji)/2`

for the actual-minus-predicted midpoint shift.

Then

`alignment = <dc,d0> / ||d0||^2`

`midpoint_bias = 2<b,d0> / ||d0||^2`.

The forward tail wins against its swap iff

`alignment + midpoint_bias > 0`,

and the reverse tail wins iff

`alignment - midpoint_bias > 0`.

Both tail orders are simultaneously recoverable against one another iff

`alignment > |midpoint_bias|`.

## Frozen evidence identity

Original T1 workflow run: `33243747235`

Original artifact ID: `9712269414`

Midpoint-decomposition replay workflow run: `33245010517`

Midpoint-decomposition artifact ID: `9712666884`

Source portable scientific gate: workflow `33219286064`

Model: `EleutherAI/pythia-14m-deduped`, revision `step143000`

Learning rate: `1e-4`

Optimizer: deterministic plain SGD, no momentum, no weight decay

Updates per stage: `16`

Both diagnostic runs refused to proceed unless they reproduced the already-recorded portable tensors. The midpoint replay again matched exactly:

- finite-pair basis matches portable gate: **true**
- all six history endpoints match portable gate: **true**
- finite-pair basis SHA256: `1afeaa53d3b98c32473fcdfc50c297e6f2db14226d9a5f868ef3aa5366f882c2`
- combined history endpoint SHA256: `ca3e24c4139d87ec4004e78c26c229aa0cf00d86956c5ffd8b4163469622a9c7`
- numerical execution fingerprint: `deaad55af513e78d0c4c1d5636836bcb7d7325be64d8df0b196cd6a66b262d42`

Therefore T1 and its midpoint replay are diagnostic analyses of the same scientific instance, not new dataset/model draws.

## Result 1 — descriptive reconstruction structure

The portable outcome remains:

| True history | Decoded | Correct? |
|---|---|---:|
| ABC | ACB | no |
| ACB | ACB | yes |
| BAC | BAC | yes |
| BCA | BAC | no |
| CAB | CAB | yes |
| CBA | CAB | no |

Aggregate:

- full order: `3/6`
- first stage: `6/6`
- pairwise precedence: `15/18 = 0.833333`
- mean Kendall tau: `2/3`

All three errors preserve the first stage and swap only the final two stages.

## Result 2 — correction on directional contamination

The directional contamination ratio was

`chi_(pi,sigma) = -2 <r_pi, d_(pi,sigma)> / ||d_(pi,sigma)||^2`.

Observed tail-swap values were:

| True history | Decoded | Tail-swap chi |
|---|---|---:|
| ABC | ACB | 1.118318 |
| ACB | ACB | 0.843755 |
| BAC | BAC | 0.817074 |
| BCA | BAC | 1.172054 |
| CAB | CAB | 0.741561 |
| CBA | CAB | 1.231791 |

Every failure has `chi > 1` against its decoded tail swap and every correct tail order remains below 1.

**Important interpretation correction:** `chi < 1` is algebraically equivalent to the two-candidate nearest-signature decision inequality. Therefore this perfect separation is a diagnostic restatement of the observed decision boundary, not independent empirical validation of the proposed mechanism.

Its useful role is directional accounting: on all three failures, the largest contamination direction is the observed same-prefix tail-swap alternative.

## Result 3 — the tail commutator changes dramatically after the prefix

| Prefix | Tail pair | Base commutator norm | Conditioned norm | Base/conditioned cosine | Drift norm | Relative drift |
|---|---|---:|---:|---:|---:|---:|
| A | BC | 0.205614 | 0.037957 | 0.103456 | 0.205194 | 0.997954 |
| B | AC | 0.259722 | 0.046565 | 0.030969 | 0.262441 | 1.010468 |
| C | AB | 0.244604 | 0.062985 | 0.052139 | 0.249385 | 1.019547 |

Conditioning on the first stage therefore does not merely perturb the base tail interaction on this instance:

- the conditioned commutators retain only roughly 15–26% of the base norm;
- their cosine with the base commutator is only roughly 0.03–0.10;
- the drift norm is approximately the full base-commutator norm.

The exact triple-residual-difference identity is numerically satisfied to about `2.8e-5` absolute error for all three prefix choices.

## Result 4 — midpoint bias dominates the surviving conditioned tail signal

The stronger midpoint replay gives:

| Prefix | Tail pair | Alignment | Midpoint bias | Forward score | Reverse score | Both tails recoverable? |
|---|---|---:|---:|---:|---:|---:|
| A | BC | 0.019095 | -0.137284 | -0.118189 | 0.156379 | no |
| B | AC | 0.005551 | +0.177508 | 0.183059 | -0.171957 | no |
| C | AB | 0.013423 | +0.245139 | 0.258562 | -0.231716 | no |

This sharpens the mechanism considerably.

The actual prefix-conditioned tail separation is not strongly reversed relative to the static pair direction. Its normalized projection remains **small but positive** for all three prefixes:

- A: `0.019095`
- B: `0.005551`
- C: `0.013423`.

However, the common higher-order midpoint bias is much larger in magnitude:

- A: `|-0.137284|`, about `7.2x` the alignment;
- B: `|+0.177508|`, about `32.0x` the alignment;
- C: `|+0.245139|`, about `18.3x` the alignment.

Thus `alignment - |midpoint_bias| < 0` for every prefix.

The sign of the midpoint bias selects exactly the surviving static-pair tail candidate:

- prefix A / BC: negative bias makes `ABC` lose and `ACB` win;
- prefix B / AC: positive bias makes `BAC` win and `BCA` lose;
- prefix C / AB: positive bias makes `CAB` win and `CBA` lose.

This matches the observed asymmetric collapse within every shared-prefix pair.

## Result 5 — the omitted term is exactly third-order for three stages

The measured pairwise prediction errors equal the exact three-stage interaction residual norms:

- ABC: `0.1737822`
- ACB: `0.1563738`
- BAC: `0.1747150`
- BCA: `0.2054421`
- CAB: `0.1612215`
- CBA: `0.2036168`

Because there are exactly three stages, this residual is the exact third-order Möbius interaction, not an unspecified mixture of third and higher interaction orders.

## Revised mechanistic interpretation

The best current description of the frozen Pythia-14M result is:

1. base-checkpoint pair interactions preserve substantial coarse chronology;
2. the first stage moves the model to a state where the remaining pair's order-sensitive separation becomes much smaller and nearly orthogonal to the base pair direction;
3. an exact third-order midpoint displacement is much larger, along the static tail-decision direction, than the small remaining aligned tail signal;
4. this midpoint displacement causes both actual endpoints sharing a first-stage prefix to favor the same static-pair tail candidate;
5. consequently the first stage remains recoverable while one of the two tail orders in each prefix pair is lost.

So the failure is better described as **collapse of useful conditioned tail separation plus dominating third-order midpoint bias**, not simply “the commutator reverses” and not merely “the higher-order residual norm becomes large.”

## Epistemic status

The midpoint/alignment equations are exact algebraic decompositions of measured endpoints. They explain the geometry of this particular result but, by themselves, cannot establish that the same mechanism is typical.

The genuinely independent question is whether fresh codebooks and stage lengths repeatedly show:

- structured same-prefix tail errors rather than arbitrary permutation errors;
- partial chronology surviving after exact-order recovery degrades;
- shrinking/rotating prefix-conditioned tail separation and/or growing midpoint dominance near the transition.

Those are being tested in T2, which was frozen before its outcomes.

## What this does not establish

T1 does **not** show that the same mechanism generalizes across:

- independent synthetic codebooks/worlds;
- other stage lengths;
- other Pythia checkpoints or model sizes;
- stochastic minibatch training;
- persistent Adam state;
- natural corpora;
- black-box behavioral access.

## Decision

**Retain the state-conditioned interaction theory, but narrow it to the midpoint/separation mechanism and keep the claim diagnostic until T2 reports. Do not move to 31M yet.**

T2 is the independent test: four fresh mechanically derived tokenizer-safe codebooks × stage lengths `{1,2,4,8,16,32}`, all reported under a frozen no-selection rule.
