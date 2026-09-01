# Contributing

ChronoTrace is an experiment-first research repository. The current v3 paper result is scientifically frozen; new research should extend it under a new protocol/version rather than rewriting the frozen evidence.

## Before changing code

1. Read `docs/REVIEWER_GUIDE.md`, `docs/RESULTS_FREEZE.md`, and `docs/DEVELOPER_GUIDE.md`.
2. State whether the change is **presentation/reproducibility**, **bug fixing**, or **new scientific research**.
3. Do not change the frozen v3 method, thresholds, target set, or selection to improve the current paper result.
4. For new science, create a new versioned protocol and record the change in `docs/DECISIONS.md` before examining confirmatory outputs.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make reviewer
```

Useful targets:

```text
make audit          verify the frozen result/protocol/package ledger
make assets-check   verify generated release assets are in sync
make check          run Ruff + tests
make paper          compile the journal-neutral manuscript
make reviewer-full  run reviewer checks and compile the paper
```

## Generated assets

The browser-visible result figures and paper result macros are generated from the frozen selection JSON:

```bash
python scripts/generate_release_assets.py --write
python scripts/generate_release_assets.py --check
```

Do not edit generated assets by hand. Presentation changes belong in the generator; scientific values belong only in a new frozen selection produced by a new protocol.

## Pull requests

A pull request should include:

- the research or engineering purpose;
- tests or an explanation of why tests do not apply;
- reproducibility impact;
- whether frozen paper-facing evidence changes;
- any change to experiment definitions, metrics, or stop conditions;
- any new external dependency.

Do not commit checkpoints, generated datasets, secrets, API keys, or experiment output directories.

## Research changes

Changes to hypotheses, primary metrics, seed splits, witness construction, interaction depth, certificate thresholds, or stop conditions are protocol changes. They must use a new protocol/version if they occur after the v3 paper freeze.

For a new confirmatory experiment:

1. develop only on explicitly spent/development data;
2. freeze the method and seed/target derivation before confirmation output exists;
3. launch the complete suite without adaptation between seeds;
4. mark any accidentally touched held-out seed as spent;
5. preserve negative and invalid attempts in the audit trail;
6. write a new `selection.json` rather than modifying the v3 selection.

See `docs/DEVELOPER_GUIDE.md` for the full continuation contract.
