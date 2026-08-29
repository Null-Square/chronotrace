# GitHub Actions Compute Policy

Date established: 2026-08-29

Purpose: preserve fast correctness checks during development without spending model-training / scientific-compute minutes on routine commits.

## Automatic checks

Pull requests run only the cheap `CI` test matrix:

- Python 3.11 and 3.12;
- package install with `[dev]` only;
- Ruff;
- lightweight unit and protocol-drift tests.

Documentation-only changes under `docs/**`, `paper/**`, or `README.md` do not start CI.

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
