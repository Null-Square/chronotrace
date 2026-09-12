# Current and historical research map

## Current publication

- `paper/bidirectional/`: canonical Elsevier-class source, section files, references, reproducible figure and highlights.
- `paper/main.tex`: current build entry point; `make paper` compiles the bidirectional article.
- `src/chronotrace/geometry/bidirectional.py`: frozen scientific engine.
- `tests/test_bidirectional.py`: native regression tests for the engine.
- `docs/PUBLICATION_EVIDENCE.md` and `research/bidirectional_v1/`: current claims, result ledger, protocol and Supplement S1 reproduction instructions.
- `scripts/audit_publication.py`: current narrative/count checks and historical manuscript SHA-256 preservation.

## Historical, not pooled with current evidence

- `paper/legacy_pythia/`: complete original paper tree, including its old planning/checklist files.
- `docs/RESULTS_FREEZE.md`: immutable Pythia 27/32 terminal confirmation ledger.
- `configs/chronotrace_pairwise_multi_witness_confirmation_v3.*`: original protocol and selection.
- `docs/history/MAIN_README_2026_09_05.md` and `RESULTS_2026_09_05.md`: preserved main-branch presentation.
- `docs/history/ARCHIVE_MAP_PYTHIA.md` and `DEVELOPER_GUIDE_PYTHIA.md`: earlier research navigation.
- Historical scripts, root paper macros and figures remain for reproducibility; `make audit` and `make assets-check` still audit the Pythia freeze.

The full experimental workspace for the current paper is supplied as Supplement S1 rather than duplicated into the native root scripts. See `research/bidirectional_v1/REPRODUCE.md`. No original scientific lock, source hash, or numerical result is changed by this publication alignment.
