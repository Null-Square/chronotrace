"""Protocol locking and drift detection for confirmatory ChronoTrace experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chronotrace.config import ExperimentConfig
from chronotrace.data import file_sha256


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def config_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    """Return only fields that can change the scientific interpretation of Phase 0."""

    experiment = {
        "name": config.experiment.get("name"),
        "protocol_version": config.experiment.get("protocol_version"),
    }
    return _normalize(
        {
            "experiment": experiment,
            "model": config.model,
            "data": config.data,
            "training": config.training,
            "histories": config.histories,
            "seeds": config.seeds,
            "controls": config.controls,
            "forensics": config.forensics,
            "metrics": config.metrics,
        }
    )


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_protocol_lock(
    config: ExperimentConfig,
    generated_sha256: dict[str, str],
) -> dict[str, Any]:
    """Build an explicit lock object for a frozen experiment protocol."""

    body = {
        "lock_version": 1,
        "config": config_snapshot(config),
        "expected_generated_sha256": _normalize(generated_sha256),
    }
    return {**body, "fingerprint": _fingerprint_payload(body)}


def write_protocol_lock(
    config: ExperimentConfig,
    generated_sha256: dict[str, str],
    path: str | Path,
) -> Path:
    """Write a protocol lock. Calling this is an explicit protocol-change action."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = build_protocol_lock(config, generated_sha256)
    destination.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_protocol_lock(path: str | Path) -> dict[str, Any]:
    """Load and internally validate a protocol lock."""

    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    fingerprint = lock.pop("fingerprint", None)
    actual = _fingerprint_payload(lock)
    if fingerprint != actual:
        raise ValueError(
            "Protocol lock fingerprint mismatch. The lock was edited without being regenerated."
        )
    lock["fingerprint"] = fingerprint
    return lock


def verify_protocol_lock(
    config: ExperimentConfig,
    *,
    lock_path: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed if config or generated artifacts drift from the frozen lock."""

    lock = load_protocol_lock(lock_path)
    actual_config = config_snapshot(config)
    if actual_config != lock["config"]:
        raise ValueError(
            "Experiment protocol drift detected: current config no longer matches the lock. "
            "Do not update the lock unless this is an intentional new protocol version."
        )

    if data_root is not None:
        root = Path(data_root)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing generated metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = lock["expected_generated_sha256"]
        if metadata.get("sha256") != expected:
            raise ValueError("Generated-data metadata drift detected against the protocol lock")

        known_artifacts = {
            "worlds": "worlds.jsonl",
            "stage_a": "stage_a.jsonl",
            "stage_b": "stage_b.jsonl",
            "stage_c": "stage_c.jsonl",
            "probes": "probes.jsonl",
        }
        unknown = set(expected) - set(known_artifacts)
        if unknown:
            raise ValueError(f"Protocol lock contains unknown generated artifacts: {sorted(unknown)}")
        for name in sorted(expected):
            path = root / known_artifacts[name]
            if not path.exists():
                raise FileNotFoundError(f"Missing generated artifact: {path}")
            actual = file_sha256(path)
            if actual != expected[name]:
                raise ValueError(f"Generated artifact drift detected for {path}")
    return lock


def file_set_fingerprint(paths: list[str | Path]) -> str:
    """Fingerprint an ordered set of files by path label and content hash."""

    rows = []
    for raw in sorted(str(Path(path)) for path in paths):
        rows.append({"path": raw, "sha256": file_sha256(raw)})
    return _fingerprint_payload({"files": rows})
