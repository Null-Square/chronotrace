# ChronoTrace Paper Workspace

Current manuscript:

> **ChronoTrace: Certified Reconstruction of Training Order from Noncommutative Learning Interactions**

The research method is frozen. This directory is now a publication workspace, not an experiment-planning scratchpad.

## Start here

- `main.tex` — complete journal-neutral manuscript draft.
- `CLAIMS_AND_EVIDENCE.md` — claim-to-evidence guardrail; use this when editing strong statements.
- `FIGURE_PLAN.md` — purpose and source data for each main figure/table.
- `figures/` — reproducible TikZ sources for all main conceptual/result figures.
- `references.bib` — working verified bibliography.
- `SUBMISSION_CHECKLIST.md` — remaining author/venue/export actions.
- `Q1_READINESS.md` — internal reviewer-risk assessment.

Repository-level audit files:

- `../docs/REVIEWER_GUIDE.md`
- `../docs/RESULTS_FREEZE.md`
- `../configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json`

## Frozen headline result

Fresh label-blind Pythia-14M confirmation:

```text
27/32 complete histories certified
182/192 pairwise precedences certified
5 full-history abstentions
10 ambiguous pair decisions
0 contradictory certified pairs
0 double-exclusions
preregistered tier: STRONG
```

The manuscript must preserve the distinction among:

- spent/development results;
- the preregistered single-witness K4 scientific negative;
- post-hoc multi-witness methodology development;
- fresh v3 confirmation.

## Build

From this directory, with a standard TeX distribution:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Fallback:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript is intentionally journal-neutral. Convert the preamble/front matter to the selected journal template only after the scientific text and figure set have passed reviewer audit.

## Claim discipline

Do not claim that ChronoTrace:

- discovers that training order matters;
- introduces Möbius inversion, Lie brackets, Sherali–Adams/RLT, convex duality, or L1 witness combination as standalone mathematics;
- is currently black-box provenance;
- reconstructs arbitrary-N histories exactly in polynomial time;
- has an exact fixed-K guarantee for `N>K` without omitted-tail control;
- confirmed the post-hoc spent result directly.

The defensible novelty is the inverse chronology problem plus the integration of ordered interactions, frozen witness banks, proof-safe property certificates, explicit abstention, and the fresh terminal confirmation.

## Author metadata

`main.tex` uses `Anonymous Authors` for review. Final names, affiliations, acknowledgements, conflicts, funding, and venue-specific declarations must be supplied by the authors; they are not inferred from repository metadata.
