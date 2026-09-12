# GitHub Actions Compute Policy

Date established: 2026-08-29

Purpose: preserve fast correctness checks during development without spending model-training / scientific-compute minutes on routine commits.

## Automatic checks

Pull requests run only one cheap Python 3.11 lane:

- package install with `[dev]` only;
- Ruff;
- lightweight unit and protocol-drift tests.

Python 3.12 compatibility is checked after code lands on `main` or by explicit manual dispatch, not on every PR commit.

A PR whose changed-file set is entirely under `docs/**`, `paper/**`, or `README.md` is ignored by CI. Note that GitHub path filters are evaluated against the PR's overall changed-file set, so a documentation commit inside an already mixed code+docs PR can still start the cheap PR lane.

Concurrency cancels superseded CI runs on the same ref.

## Heavy mechanism smoke

The CPU PyTorch theorem / transformer / macro-operator / finite-pair / end-to-end smoke is **not run on routine PR commits**.

It runs only:

1. by explicit `workflow_dispatch`, when we intentionally want the full mechanism gate; or
2. after code lands on `main` through the `push` event.

This keeps strong integration coverage without paying for it on every intermediate commit.

## Scientific and model-training workflows

All Pythia and historical model-training workflows are **manual-dispatch only**. This includes:

- Pythia scale LR selection;
- Pythia-14M reproducibility bridge;
- Pythia-14M T2 interaction map;
- Pythia-14M T2b LR map;
- archived T2b aggregation;
- Pythia theory diagnostic;
- FP32 Pythia validation;
- legacy Phase-0b washout pilot.

These workflows must not gain `push` or `pull_request` triggers without an explicit research/compute decision.

Every heavy workflow should retain a concurrency group with `cancel-in-progress: true` where reruns can supersede one another.

## Research rule

A scientific workflow is dispatched only after its hypothesis, frozen protocol, selection rule, and stopping/interpretation rule are recorded. Routine code or documentation development must not implicitly launch scientific compute.

If a scientific run is already complete, prefer aggregating its archived artifacts over rerunning model training.
