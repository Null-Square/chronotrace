# ChronoTrace Research Operating Protocol

Purpose: preserve a defensible scientific audit trail from exploratory idea through paper-ready evidence.

This protocol is intentionally stricter than ordinary experiment logging because ChronoTrace studies small path-dependent effects that are unusually vulnerable to confounds, numerical artifacts, and post-hoc story selection.

## 1. Journal is append-only

Canonical chronological narrative: `docs/RESEARCH_JOURNAL.md`.

Do not rewrite an old entry because a later interpretation is cleaner. Append a correction / superseding entry and link it to the earlier journal ID.

Every substantive experiment or theory pivot gets a stable journal ID `J###`.

## 2. Before compute

For any experiment capable of changing a scientific claim, write down first:

- research question;
- hypothesis and null hypothesis;
- exact model / revision;
- dataset / stage construction and hashes;
- optimizer / schedule / numerical state;
- discovery versus confirmation status;
- candidate hyperparameters, if any;
- selection rule that is allowed to choose among candidates;
- metrics that are forbidden from influencing selection;
- success / failure criterion;
- expected falsifying outcome;
- whether the run is exploratory, diagnostic, discovery, or confirmatory.

If a hypothesis was noticed after observing a result, mark it **post-hoc / hypothesis-generating** and require independent frozen data before using confirmatory language.

## 3. During compute

Preserve:

- Git commit / PR head;
- workflow run ID and job IDs;
- exact protocol fingerprint;
- data/tokenizer hashes;
- base checkpoint hash when white-box weights are used;
- numerical execution fingerprint when endpoint differences are small;
- all failed-run evidence, not only passing artifacts.

A scientific gate must write its result artifact before raising on failure.

## 4. After compute

Add a journal entry containing:

1. **Question**
2. **Frozen protocol**
3. **Evidence** — run/commit/artifact IDs
4. **Raw result**
5. **Interpretation**
6. **What it does not imply**
7. **Decision**
8. **Next falsifier**

Detailed numeric tables belong under `docs/results/` and should be linked from the journal.

## 5. Theory-first rule

Do not launch a new expensive experiment merely because the previous accuracy was low.

Before changing the mechanism, answer:

- What exact term in the current model was omitted or mis-modeled?
- What new observable quantity distinguishes the competing explanations?
- What result would falsify the proposed explanation?
- Can that explanation be tested on an already-frozen instance before generating new data?

Prefer diagnostic replay over new tuning when the old endpoint already contains enough information to test the theory.

## 6. One difficulty at a time

Scale dimensions separately whenever possible:

- model size;
- stage duration;
- number of stages;
- optimizer-state persistence;
- stochasticity;
- mixed precision;
- architecture family;
- white-box versus black-box observability.

Do not increase several at once and then attribute failure to one of them.

## 7. Evidence classes

### Diagnostic

Reuses an already-observed instance to understand mechanism. Cannot establish generalization.

### Exploratory

Can generate hypotheses. Metrics chosen after seeing results must be labeled post-hoc.

### Discovery

Protocol and metrics frozen before fresh data/seeds. Can establish a candidate effect and estimate its behavior.

### Confirmation

Uses untouched seeds/models/data under a sealed protocol. No mechanism or threshold changes after opening confirmation.

## 8. Negative results are first-class

Record:

- confounded positives;
- numerical failures;
- reproducibility failures;
- failed theoretical approximations;
- failed scale gates;
- abandoned designs that were superseded before execution.

A later paper should be able to reconstruct why each direction was accepted or rejected without relying on memory.

## 9. Claim ladder

ChronoTrace claims should advance only through this ladder:

1. **mathematical identity / controlled mechanism**
2. **controlled neural-model realization**
3. **fresh-seed generalization at one model scale**
4. **model-scale / architecture replication**
5. **realistic optimizer / stochastic training**
6. **behavioral / black-box transfer**
7. **practical training-pipeline audit claim**

Evidence at one rung must not be described as evidence for later rungs.

## 10. Current special guardrails

- Pythia-31M chronology is blocked until the state-conditioned / interaction-order theory survives independent 14M instances.
- The observed Pythia-14M `6/6` first-stage identity is post-hoc and not a confirmed claim.
- The portable Pythia-14M pairwise full-order result is a reproducible negative: `3/6`.
- New decoders must not be tuned on the original locked 14M instance and then evaluated on the same instance as if they were confirmation.
- Confirmation seeds from the original Phase-0 program remain untouched unless a future protocol explicitly redefines and seals their role before use.
