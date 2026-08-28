"""Orchestration for the frozen Phase-0 discovery and confirmation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronotrace.config import ExperimentConfig
from chronotrace.data import generate_dataset
from chronotrace.protocol import file_set_fingerprint, verify_protocol_lock


def ensure_frozen_dataset(
    config: ExperimentConfig,
    *,
    lock_path: str | Path,
) -> dict[str, Any]:
    """Generate the dataset if absent, then verify every byte against the protocol lock."""

    data_root = Path(config.data["root"])
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        generate_dataset(
            data_root,
            seed=int(config.data["seed"]),
            worlds=int(config.data["worlds"]),
            decoys_per_probe=int(config.data.get("decoys_per_probe", 3)),
        )
    lock = verify_protocol_lock(config, lock_path=lock_path, data_root=data_root)
    return lock


def _run_id(history: str, seed: int) -> str:
    return f"phase0-{history.lower()}-seed{seed}"


def _ensure_endpoint_features(
    config: ExperimentConfig,
    *,
    history: str,
    seed: int,
    runs_root: Path,
) -> Path:
    """Train a missing endpoint or resume feature extraction from a completed endpoint."""

    from chronotrace.features import extract_run_features
    from chronotrace.training import train_endpoint

    run_dir = runs_root / _run_id(history, seed)
    features_path = run_dir / "features.json"
    if features_path.exists():
        return features_path

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        train_endpoint(config, history=history, training_seed=seed, output_root=runs_root)
    elif not (run_dir / "endpoint").exists():
        raise RuntimeError(f"Incomplete run cannot be resumed safely: {run_dir}")

    return extract_run_features(config, run_dir=run_dir)


def run_discovery(
    config: ExperimentConfig,
    *,
    lock_path: str | Path,
    runs_root: str | Path | None = None,
) -> Path:
    """Execute only discovery seeds and write the leave-one-seed-out discovery report."""

    from chronotrace.detector import discovery_only_report

    lock = ensure_frozen_dataset(config, lock_path=lock_path)
    root = Path(runs_root or config.artifacts["root"])
    root.mkdir(parents=True, exist_ok=True)

    for seed in config.seeds["discovery"]:
        for history in config.histories:
            _ensure_endpoint_features(
                config,
                history=history,
                seed=int(seed),
                runs_root=root,
            )

    report = discovery_only_report(config, runs_root=root)
    report["protocol_fingerprint"] = lock["fingerprint"]
    destination = root / "discovery_report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def freeze_confirmation(
    config: ExperimentConfig,
    *,
    lock_path: str | Path,
    runs_root: str | Path | None = None,
) -> Path:
    """Seal discovery artifacts before any confirmation endpoint is allowed to run."""

    lock = ensure_frozen_dataset(config, lock_path=lock_path)
    root = Path(runs_root or config.artifacts["root"])
    report_path = root / "discovery_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Run the discovery split before sealing confirmation")

    feature_paths: list[Path] = []
    for seed in config.seeds["discovery"]:
        for history in config.histories:
            path = root / _run_id(history, int(seed)) / "features.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing discovery feature artifact: {path}")
            feature_paths.append(path)

    from chronotrace.data import file_sha256

    seal = {
        "schema_version": 1,
        "protocol_fingerprint": lock["fingerprint"],
        "discovery_report_sha256": file_sha256(report_path),
        "discovery_feature_set_fingerprint": file_set_fingerprint(feature_paths),
        "discovery_seeds": list(config.seeds["discovery"]),
        "confirmation_seeds": list(config.seeds["confirmation"]),
    }
    destination = root / "confirmation_seal.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != seal:
            raise ValueError(
                "Existing confirmation seal does not match current discovery artifacts"
            )
        return destination
    destination.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def verify_confirmation_seal(
    config: ExperimentConfig,
    *,
    lock_path: str | Path,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify that nothing relevant changed after discovery was frozen."""

    lock = ensure_frozen_dataset(config, lock_path=lock_path)
    root = Path(runs_root or config.artifacts["root"])
    seal_path = root / "confirmation_seal.json"
    if not seal_path.exists():
        raise PermissionError(
            "Confirmation is locked. Review discovery results, then run `chronotrace freeze`."
        )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["protocol_fingerprint"] != lock["fingerprint"]:
        raise ValueError("Protocol changed after the confirmation seal was created")

    from chronotrace.data import file_sha256

    report_path = root / "discovery_report.json"
    if file_sha256(report_path) != seal["discovery_report_sha256"]:
        raise ValueError("Discovery report changed after confirmation was sealed")

    feature_paths = [
        root / _run_id(history, int(seed)) / "features.json"
        for seed in config.seeds["discovery"]
        for history in config.histories
    ]
    if file_set_fingerprint(feature_paths) != seal["discovery_feature_set_fingerprint"]:
        raise ValueError("Discovery feature artifacts changed after confirmation was sealed")
    return seal


def run_confirmation(
    config: ExperimentConfig,
    *,
    lock_path: str | Path,
    runs_root: str | Path | None = None,
) -> Path:
    """Execute confirmation exactly after a valid discovery seal exists."""

    from chronotrace.detector import fit_and_evaluate

    seal = verify_confirmation_seal(config, lock_path=lock_path, runs_root=runs_root)
    root = Path(runs_root or config.artifacts["root"])
    for seed in config.seeds["confirmation"]:
        for history in config.histories:
            _ensure_endpoint_features(
                config,
                history=history,
                seed=int(seed),
                runs_root=root,
            )

    report = fit_and_evaluate(config, runs_root=root)
    report["protocol_fingerprint"] = seal["protocol_fingerprint"]
    report["confirmation_seal"] = seal
    destination = root / "phase0_report.json"
    if destination.exists():
        raise FileExistsError(
            "Final confirmation report already exists. Confirmation is intentionally one-shot."
        )
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
