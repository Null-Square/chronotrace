# Pythia-14M T2 — Pre-Result Interpretation Decision Tree

Date frozen: 2026-08-29

Status: written while workflow `33245167776` is still running and before any T2 seed outcome has been observed.

This document does not alter the T2 lock, codebooks, stage lengths, metrics, or pre-registered checks. Its purpose is to prevent every possible T2 result from being rationalized as support after the fact.

## Core independent question

T2 asks whether the structured Pythia-14M result that motivated the state-conditioned hierarchy repeats across fresh synthetic instances:

> As finite training stages become more nonlocal, does exact static-pair chronology degrade in a structured coarse-to-fine way, with early/partial chronology surviving more strongly than the final tail order?

The strongest independent evidence is the **error structure across fresh codebooks**, not an algebraic diagnostic computed from the same endpoints.

## Outcome A — structured coarse-to-fine failures repeat

Operational signs:

- full-order recovery drops below 6/6 for multiple independent codebooks at some sampled stage length;
- at least 75% of all observed errors are the pre-registered same-prefix adjacent tail swaps;
- first-stage recovery exceeds full-order recovery at the first failure in at least 3/4 codebooks;
- state-conditioned tail geometry near failure is consistent with reduced robustness, without requiring monotonicity in every scalar diagnostic.

Interpretation:

**Independent support for the current coarse-to-fine state-conditioned hierarchy in this controlled Pythia-14M setting.**

Allowed next step:

Design T3 with at least four stages, where degree-3 information can be tested against a factorial chronology space and probe complexity can be separated from exhaustive history replay.

Not yet allowed:

- claim natural-data provenance;
- claim black-box provenance;
- claim cross-model scaling;
- claim optimizer-memory novelty;
- move directly to 31M solely because T2 is positive.

## Outcome B — static pair recovery remains 6/6 for all four codebooks through 32 updates

Interpretation:

T2 does **not** independently reproduce the 16-update failure transition from the motivating codebook. It shows that the static pair model is more robust on the fresh sampled instances than expected.

This is not evidence against chronology identifiability; it is evidence that the proposed transition is instance-dependent or lies outside the sampled regime.

Allowed next step:

Pre-register a new stage-length extension using the same already-consumed T2 codebooks, with lengths chosen by a simple geometric rule rather than observed chronology accuracy, for example `{64,128}` or another fixed doubling extension justified by numerical stability.

Required caution:

The extension is a **design/calibration study on consumed codebooks**, not independent confirmation. Fresh codebooks would still be needed after the transition range is fixed.

Not allowed:

Cherry-pick a codebook or stage length that happens to fail and call it confirmation.

## Outcome C — failures occur but are not predominantly same-prefix tail swaps

Examples:

- first-stage errors are common;
- errors jump across multiple precedence edges;
- pre-registered tail-swap fraction is below 75%;
- first-stage recovery does not systematically exceed full-order recovery.

Interpretation:

**The specific coarse-to-fine / tail-collapse mechanism inferred from the motivating 14M instance is not supported as a general description across fresh codebooks.**

The broad fact that higher-order interactions matter may remain true, but the current structured hierarchy must be revised.

Allowed next step:

Analyze which chronology edges fail, interaction subspaces involved, and whether a more general degree-3 representation explains the fresh errors. Any new mechanism inferred from T2 is discovery and must receive a fresh independent test before being promoted.

Not allowed:

Proceed to T3 under the existing “first stage survives, tail collapses” narrative without revising the theory.

## Outcome D — exact order degrades but partial-order metrics remain high without a dominant tail-swap pattern

Example pattern:

- exact 6-way accuracy degrades;
- pairwise precedence/Kendall remain substantially above chance;
- errors are spread over different one-edge or short-distance permutation mistakes rather than specifically the last two positions.

Interpretation:

This would weaken the **specific tail mechanism** but support a broader and potentially more important hypothesis:

> low-order interactions recover a useful partial order even when they cannot recover the exact permutation.

Allowed next step:

Reframe T3 around partial-order reconstruction and edge confidence rather than a fixed prefix-tail recursion. Pre-register which partial-order metrics constitute success before four-stage outcomes.

## Outcome E — chronology collapses globally near the first failure

Example pattern:

- full-order accuracy drops sharply;
- first-stage accuracy also approaches chance;
- pairwise precedence and Kendall tau collapse together;
- no stable partial chronology survives.

Interpretation:

Evidence against a useful low-order hierarchy in this regime. The motivating J010 partial-order pattern was likely instance-specific.

Allowed next step:

Study the identifiability boundary itself: signature separation, conditioning, and endpoint collisions as stage duration grows. A negative paper direction may still be scientifically useful if it characterizes when chronology becomes unrecoverable.

Not allowed:

Scale the same decoder to larger models expecting parameter count alone to fix the problem.

## Outcome F — numerical or implementation failure

Examples:

- non-finite parameters;
- tokenizer/codebook contract failure;
- base checkpoint hash mismatch;
- portable numerical fingerprint drift;
- incomplete condition set;
- artifact/aggregation failure.

Interpretation:

No scientific conclusion from affected conditions.

Allowed next step:

Fix only the infrastructure/numerical defect, document it in the journal, and rerun the exact frozen scientific protocol. If a fix changes model/data/optimizer/stage-length semantics, it becomes a new protocol and must be frozen separately.

## How to treat the exact midpoint/alignment diagnostic

For a shared-prefix tail pair, the condition

`alignment > |midpoint_bias|`

is an exact decomposition of whether both actual endpoints remain on their respective sides of the static-pair tail boundary.

Therefore its agreement with a realized same-prefix tail failure is primarily **mechanistic classification of that failure**, not independent prediction by itself.

The independent content is whether fresh instances repeatedly enter this failure geometry in the pre-specified stage-length map, and whether the resulting chronology errors have the pre-registered structure.

## Scale decision

For all T2 outcomes above:

**Pythia-31M remains blocked until the result is documented, journaled, and a new theory-driven protocol is frozen.**

A positive T2 does not automatically trigger 31M. The next theoretically meaningful experiment is likely four-stage T3 because the central question is interaction order, not parameter scaling.
