# ChronoTrace Submission Checklist

This checklist separates completed scientific work from author/venue-specific actions that cannot be inferred automatically.

## Scientific freeze — complete

- [x] Final method frozen before fresh v3 confirmation.
- [x] Fresh v3 seed set mechanically derived after provenance audit.
- [x] Four seed jobs launched together with no intermediate adaptation.
- [x] 4/4 scientific seed jobs completed successfully.
- [x] Final result frozen: 27/32 complete histories, 182/192 pair relations.
- [x] Zero contradictory inferred pairs and zero double-exclusions.
- [x] Terminal exactness/soundness checks pass.
- [x] Preregistered single-witness K4 negative preserved as negative.
- [x] Post-hoc multi-witness development labeled as post-hoc.
- [x] v1 consumed seeds recorded as spent and excluded from final confirmation claim.
- [x] Post-science aggregation ordering bug documented and regression-tested.
- [x] No new Pythia confirmation reruns after observing v3 outcomes.

## Repository reviewer path — complete

- [x] Root README reflects final frozen result rather than historical T2/T2b status.
- [x] Browser-visible frozen result graphic and method pipeline are included.
- [x] `docs/REVIEWER_GUIDE.md` provides shortest audit path.
- [x] `docs/RESULTS_FREEZE.md` records immutable result/provenance ledger.
- [x] `docs/ARCHIVE_MAP.md` distinguishes current and historical machinery.
- [x] `docs/DEVELOPER_GUIDE.md` documents safe continuation after the paper freeze.
- [x] `paper/CLAIMS_AND_EVIDENCE.md` constrains manuscript claims.
- [x] `scripts/audit_release.py` verifies the frozen result/protocol/package ledger.
- [x] Frozen-data-derived publication assets have a deterministic generator and sync test.
- [x] PR #11 body updated to current science; PR remains open/unmerged.
- [x] Issue #12 contains frozen scientific/provenance summary.
- [x] Historical research journal/protocols preserved rather than silently rewritten.

## Manuscript — complete draft

- [x] Abstract centered on fresh confirmation.
- [x] Introduction and contributions.
- [x] Formal problem/access regime.
- [x] Exact ordered interaction theory.
- [x] Multi-witness certificate theorem.
- [x] Local-order/proof-safe dual method.
- [x] Information barrier.
- [x] Development/falsification narrative.
- [x] Fresh v3 methods/results.
- [x] Related work and novelty boundary.
- [x] Discussion.
- [x] Limitations.
- [x] Conclusion.
- [x] Reproducibility/provenance appendix.

## Figures/tables — sources complete

- [x] Pipeline/access-regime figure.
- [x] Scientific development/falsification ladder.
- [x] Fresh confirmation coverage figure.
- [x] Multi-witness geometry figure.
- [x] Fixed-K / terminal exactness boundary figure.
- [x] Case-level certification/abstention matrix source.
- [x] Related-work comparison table.
- [x] Fresh per-seed result table.
- [x] Numerical validity table.
- [x] Frozen result macros for journal-template conversion.

## Bibliography — core complete, final venue audit required

- [x] Pythia.
- [x] curriculum learning/data ordering.
- [x] 2025/2026 forward Lie-bracket ordering work.
- [x] membership inference.
- [x] influence/data attribution.
- [x] model lineage/provenance.
- [x] palimpsestic provenance.
- [x] process-tensor training memory.
- [x] permutation Fourier inference.
- [x] Sherali--Adams hierarchy.
- [x] classical Möbius reference.
- [ ] Re-run DOI/title/author metadata audit immediately before submission.
- [ ] Add any venue/reviewer-mandated references discovered during final formatting.

## Author/release actions still required

- [ ] Choose and approve a software license before inviting external reuse or creating an archival release. No license is inferred automatically.
- [ ] Replace `Anonymous Authors` after double-blind review requirements permit.
- [ ] Supply exact author names and affiliations.
- [ ] Add ORCID IDs if desired/required.
- [ ] Add acknowledgements/funding.
- [ ] Add conflict-of-interest statement if venue requires it.
- [ ] Add author-contribution statement if venue requires it.
- [ ] Confirm whether repository must be anonymized for review.
- [ ] Create archival release/DOI after acceptance or at submission if permitted.

## Venue selection/formatting still required

- [ ] Choose target Q1 journal/venue based on scope and page policy.
- [ ] Replace journal-neutral `article` preamble with official template.
- [ ] Apply venue page/word/figure limits.
- [ ] Confirm supplementary-material policy.
- [ ] Confirm code/data anonymization policy.
- [ ] Confirm whether arXiv preprint is allowed before/during review.
- [ ] Write venue-specific cover letter.
- [ ] Export final PDF and supplement.

## Final scientific language audit

Before submission, search the manuscript for and reject unsupported formulations such as:

```text
first ever
universal provenance
black-box chronology
polynomial-time exact reconstruction
arbitrary N
guaranteed full history at fixed K
100% accuracy
```

Preferred headline wording:

> ChronoTrace certifies 27/32 complete four-stage histories and 182/192 pairwise precedences on a frozen fresh Pythia-14M terminal confirmation, with five abstentions and zero contradictory certified pair relations.

## Go/no-go

**Scientific go:** yes.  
**Repository go:** yes once the final CI/release-audit/package gates are green.  
**Manuscript content go:** yes after the current LaTeX compile remains green.  
**Venue submission go:** requires author metadata, license decision for external reuse, template conversion, final literature audit, and venue-specific declarations—not additional method development.
