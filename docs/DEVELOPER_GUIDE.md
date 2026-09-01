# ChronoTrace Developer Guide

ChronoTrace is now in a **paper freeze** for the current result. This guide is for people who want to reproduce the package, understand the active implementation, or continue the research without contaminating the frozen v3 claim.

## 1. First commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make audit
make reviewer
```

`make audit` is intentionally lightweight. It checks the frozen selection arithmetic, fresh-seed ledger, canonical protocol hashes, outcome tier, numerical-validity flags, and paper-facing result copy without downloading Pythia weights.

`make reviewer` additionally checks generated assets, Ruff, and the full test suite.

With a TeX distribution installed:

```bash
make reviewer-full
```

This compiles the journal-neutral manuscript after the code/release checks.

## 2. Frozen versus active research

The current paper result is frozen at:

```text
configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json
```

Do not edit that selection, its method thresholds, its fresh target set, or its scientific interpretation to improve the current paper. Presentation-only changes are allowed if they are mechanically derived from the frozen evidence.

Any new scientific work should use a **new protocol version** and should not overwrite v3. Examples:

```text
..._v4.lock.json
..._v4.launch
..._v4.selection.json
```

The current v3 result remains the immutable baseline for comparison.

## 3. Active implementation path

Read the certificate implementation in this order:

1. `src/chronotrace/geometry/interactions.py`
   - exact ordered Möbius interactions;
   - streaming exact-prefix measurement.
2. `src/chronotrace/geometry/projected_interactions.py`
   - linear projection of endpoint/interactions.
3. `src/chronotrace/geometry/local_order_hierarchy.py`
   - K-local order marginal coordinates and constraints.
4. `src/chronotrace/geometry/local_order_lp.py`
   - proof-safe corrected dual lower bounds.
5. `src/chronotrace/geometry/multi_witness.py`
   - L1-safe combinations of frozen unit witnesses.
6. `src/chronotrace/geometry/multi_witness_local_order.py`
   - multi-witness infinity-norm certificate over the local hierarchy.
7. `src/chronotrace/geometry/pairwise_certificate.py`
   - label-blind two-sided precedence decisions.

The final scientific engine is:

```text
scripts/pythia_14m_pairwise_multi_witness_confirmation_v2.py
```

The v3 runner is a provenance/fresh-seed wrapper:

```text
scripts/pythia_14m_pairwise_multi_witness_confirmation_v3.py
```

See `docs/ARCHIVE_MAP.md` before interpreting similarly named historical runners.

## 4. Numerical safety contract

A ChronoTrace certificate must remain conservative under floating-point solver error.

Do not replace the corrected dual bound with a raw LP solver objective. The implementation corrects reduced-cost violations blockwise before reporting a lower bound. Terminal tests independently compare local-order LP primals with complete permutation convex-hull solves at `K=N=4`.

The frozen confirmation also requires:

- exact target active-lift replay;
- witness unit-norm checks;
- projected Möbius reconstruction checks;
- lower-bound soundness in witness geometry;
- lower-bound soundness against exact Euclidean class vertices;
- invalidation if both orientations of a pair are excluded.

A new method should preserve or strengthen these checks.

## 5. Research provenance contract

For a new confirmation experiment:

1. Develop on explicitly spent/development data.
2. Freeze the scientific method and decision thresholds before generating confirmation outputs.
3. Freeze the seed/target derivation rule before launch.
4. Launch the complete confirmation suite without adaptation between seeds.
5. Treat any seed touched by a prior scientific runner as spent.
6. Record failed attempts and implementation amendments rather than deleting them.
7. Keep post-hoc methodology development labeled post-hoc.
8. Never relabel a preregistered negative after method development.

The v1-to-v3 provenance correction is an example of how to handle a contaminated held-out set transparently.

## 6. Extending beyond N=4

The fixed-depth hierarchy contains

```text
sum_(r=1..K) P(N,r)
```

coordinates and is polynomial in `N` for fixed `K`. That does **not** make fixed `K<N` exact automatically.

For `N>K`, omitted interactions above degree `K` can change the true endpoint. `docs/K_LOCAL_INFORMATION_BARRIER.md` shows that arbitrary unseen `K+1` behavior cannot be universally bounded from finite degree-`<=K` observations for arbitrary smooth one-step SGD.

Promising follow-up directions therefore need at least one of:

- explicit regularity assumptions that imply a usable tail bound;
- selective higher-order probes acquired adaptively;
- a model/optimizer family with a provable truncation structure;
- probabilistic rather than universal guarantees;
- richer observed optimizer state in addition to final weights.

Do not describe a fixed-K experiment as exact unless its omitted-tail assumption or additional measurement channel is stated and tested.

## 7. Adding a new model-scale experiment

A new experiment should include:

- a JSON protocol lock;
- exact model/tokenizer revisions and hashes where available;
- deterministic execution settings;
- explicit stage-call accounting;
- a launch marker or equivalent one-shot gate;
- tests that freeze the method/seed semantics;
- a selection JSON containing final metrics and provenance;
- artifact IDs/digests for external run evidence;
- a short addition to `docs/RESEARCH_JOURNAL.md` and `docs/DECISIONS.md`.

Avoid putting one-off scientific logic only in a workflow YAML file. The workflow should call a versioned script whose invariants can be unit-tested.

## 8. Generated publication assets

Browser-visible and paper-facing release assets are generated from the frozen v3 selection:

```bash
python scripts/generate_release_assets.py --write
python scripts/generate_release_assets.py --check
```

Generated files include:

```text
assets/chronotrace-results.svg
assets/chronotrace-pipeline.svg
paper/generated/results_macros.tex
paper/figures/confirmation_matrix.tex
```

Do not edit those by hand. Change the generator only for presentation, never to change scientific values.

## 9. Paper workflow

The generic manuscript is under `paper/`. The dedicated GitHub Actions `Paper` workflow compiles `paper/main.tex`, fails on unresolved citations/references, and uploads the PDF artifact.

When a journal is selected, create a venue-specific subdirectory rather than destroying the neutral source, for example:

```text
paper/venues/<journal-slug>/
```

Keep the journal class/style/template files there and port the content from `paper/main.tex`. This makes future resubmission or venue changes reproducible.

## 10. Historical files

The repository is intentionally larger than a minimal software package because invalid attempts and negative experiments are part of the audit trail. Do not mass-delete historical `configs/`, `scripts/`, or workflows just to make the tree look smaller.

The cleanup rule is:

- current path is obvious;
- historical path is documented;
- generated/build outputs are ignored;
- raw model checkpoints/datasets are not committed;
- scientific history remains inspectable.

Use `docs/ARCHIVE_MAP.md` for the historical taxonomy.

## 11. Recommended follow-up projects

The cleanest next-paper extensions are:

1. **Nonterminal certification:** add assumptions or probes that make `K<N` tails certifiable.
2. **More realistic optimizer state:** momentum/Adam and stochastic minibatch chronology.
3. **Natural stage domains:** move from tokenizer-safe codebooks to semantic post-training domains while preserving deterministic auditability.
4. **Scale:** test whether certificate margins survive larger models and longer histories.
5. **Reduced replay access:** determine which controlled counterfactual queries are actually necessary.

Each should be a new protocol family, not a silent extension of the current paper freeze.
