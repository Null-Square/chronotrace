# ChronoTrace Historical Archive Map

ChronoTrace intentionally preserves failed protocols, superseded runners, negative selections, and provenance records. They are part of the scientific audit trail. Reviewers should not interpret every file under `configs/` or `scripts/` as part of the final execution path.

## Current paper-facing path

Use these first:

```text
README.md
docs/REVIEWER_GUIDE.md
docs/RESULTS_FREEZE.md
paper/main.tex
paper/CLAIMS_AND_EVIDENCE.md
configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json
```

## Current scientific engine

The final fresh v3 confirmation uses:

```text
configs/chronotrace_pairwise_multi_witness_methodology_v3.lock.json
configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json
scripts/pythia_14m_pairwise_multi_witness_confirmation_v2.py
scripts/pythia_14m_pairwise_multi_witness_confirmation_v3.py
scripts/analyze_pairwise_multi_witness_confirmation_suite_v3.py
src/chronotrace/geometry/interactions.py
src/chronotrace/geometry/projected_interactions.py
src/chronotrace/geometry/local_order_hierarchy.py
src/chronotrace/geometry/local_order_lp.py
src/chronotrace/geometry/multi_witness.py
src/chronotrace/geometry/multi_witness_local_order.py
src/chronotrace/geometry/pairwise_certificate.py
```

The v3 runner is a provenance/fresh-seed wrapper over the already-tested v2 scientific engine; the label-blind certificate logic itself was not changed after the method freeze.

## Historical evidence classes

### 1. Early behavioral probes

Files containing names such as:

```text
ab_ba
behavioral
capability
recency
```

belong to the original behavioral-discovery stage. They established that a naive order detector could be confounded by ordinary capability/recency effects. They are not part of the final claim.

### 2. Finite-pair / K2 / K3 pilots

Files containing:

```text
finite_pair
k23
k3
three_stage
```

record the transition from local pair signatures to structured finite-stage failures and later K3 convex certificates. These include useful negative results and partial chronology recovery.

### 3. Exact reachable N=4 mechanism

The forward-reachable all-24 selection establishes separability of the controlled spent system:

```text
configs/pythia_14m_forward_reachable_all24.selection.json
```

This is an identifiability/mechanism result and deliberately not the scalable method claim because it enumerates complete histories.

### 4. Projected K4 single-witness diagnostic

Files containing:

```text
projected_k4_survivor
exact_prefix_repair
implementation_amendment
invalid_attempts
```

record the preregistered single-witness K4 experiment and its implementation repair. The valid scientific result is negative. Earlier failed attempts are retained because they document why exact measured prefixes were required for nonlinear active lifts.

### 5. Multi-witness methodology development

Files containing:

```text
multi_witness
pairwise_certificate
```

record the post-hoc development step on the spent ABCD instance and the subsequent label-blind property-certificate implementation. The spent multi-witness success is methodology development, not confirmation.

### 6. Confirmation v1/v2/v3 provenance

Version labels matter:

- **v1:** historical confirmation infrastructure that consumed the originally reserved seed set; those seeds are spent.
- **v2:** label-blind scientific engine/interface developed before final fresh confirmation.
- **v3:** final provenance-corrected fresh seed freeze and confirmatory result.

The key provenance record is:

```text
configs/chronotrace_pairwise_multi_witness_confirmation_v1_spent.provenance.json
```

Do not use v1 numerical outputs as confirmation evidence for the final paper.

## Invalid-attempt convention

A file named `attempt`, `invalid`, `amendment`, or `launch` may document a failed or preflight execution. Such files are not silently deleted because doing so would make the scientific history look cleaner than it actually was.

The final result is always identified by a `selection.json` file and the corresponding run/artifact hashes in `docs/RESULTS_FREEZE.md`.

## Research journal

`docs/RESEARCH_JOURNAL.md` is append-only and intentionally verbose. It is useful for reconstructing the complete research sequence but should not be used as the first reviewer entry point.

## Why historical files remain in the branch

Deleting superseded experiments would make the repository smaller but scientifically weaker. The publication cleanup therefore follows a different rule:

- remove stale claims from the default landing pages;
- create an explicit reviewer path;
- keep negative/invalid/superseded evidence auditable;
- clearly label which files are current versus archival.

This is the same distinction the manuscript makes between spent development, preregistered negatives, post-hoc methodology development, and fresh confirmation.