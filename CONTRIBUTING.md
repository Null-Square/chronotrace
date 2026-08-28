# Contributing

ChronoTrace is an experiment-first research repository.

## Before changing code

1. Read `docs/RESEARCH_QUESTION.md` and `docs/EXPERIMENT_PROTOCOL.md`.
2. State which hypothesis or infrastructure requirement the change serves.
3. Do not change a confirmatory experiment after looking at confirmatory results without recording the change in `docs/DECISIONS.md`.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Pull requests

A pull request should include:

- the research or engineering purpose;
- tests or an explanation of why tests do not apply;
- reproducibility impact;
- any change to experiment definitions, metrics, or stop conditions;
- any new external dependency.

Do not commit checkpoints, generated datasets, secrets, API keys, or experiment output directories.

## Research changes

Changes to hypotheses, primary metrics, seed splits, or MVP stop conditions are protocol changes. Record them in `docs/DECISIONS.md` before running the affected confirmatory experiment.
