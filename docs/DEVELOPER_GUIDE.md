# Developing the current publication artifact

Read `README.md`, `docs/PUBLICATION_EVIDENCE.md`, and `paper/bidirectional/main.tex` first. The current decoder uses residual-controlled bidirectional joins; the old terminal witness hierarchy is historical, not the current headline.

## Gates

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -O -m pytest -q
python scripts/audit_publication.py
python scripts/audit_release.py
python scripts/generate_release_assets.py --check
make paper
```

The first audit checks the current ledger and archived manuscript identity. The next two preserve the historical Pythia release. A test pass does not imply full-repository lint is clean; archived exploratory code and frozen sources must not be silently reformatted merely to change that status.

## Scientific changes

Use a new branch and protocol version. Preserve the current primary lock, original source snapshot, raw arrays and all failed cells. Count every gradient and inverse residual check. Unconverged inverses enlarge uncertainty; replay-cap exhaustion retains untested candidates. Do not feed evaluator-only exhaustive distances or labels into decoder decisions.

Use Supplement S1's separate pinned workspace for independent numerical reproduction. Native development dependencies are intentionally broader; changing them is not a rerun of the frozen experiment. Record current hardware explicitly in any new timing study—the original serial timing artifact omitted its CPU model.

## Publication changes

The named author is Omar Al-Tawil. Do not invent affiliation, email, funding, conflicts, contributor roles, a license, DOI, acceptance, or author approval. The current author-action and journal checks are in `paper/SUBMISSION_CHECKLIST.md`. Archive old paper-facing documents before replacing their narrative; keep scientific records append-only.

The previous developer guide is preserved in `docs/history/DEVELOPER_GUIDE_PYTHIA.md`.
