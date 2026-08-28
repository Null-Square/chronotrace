from __future__ import annotations

import pytest

from chronotrace.config import load_config
from chronotrace.detector import FeatureSample, capability_matching_report
from chronotrace.phase0 import _require_discovery_capability_gate


def _sample(seed: int, history: str, a: float, b: float) -> FeatureSample:
    return FeatureSample(
        run_id=f"phase0-{history.lower()}-seed{seed}",
        history=history,
        seed=seed,
        features={"a_control.mean": a, "b_control.mean": b},
        feature_groups={"capability": ["a_control.mean", "b_control.mean"], "forensic": []},
    )


def test_capability_matching_uses_frozen_pairwise_margin_threshold() -> None:
    config = load_config("configs/mvp.yaml")
    samples = [
        _sample(11, "AB", 3.0, 4.0),
        _sample(11, "BA", 3.5, 3.2),
        _sample(13, "AB", 2.0, 5.0),
        _sample(13, "BA", 2.8, 4.1),
    ]

    report = capability_matching_report(config, samples)

    assert report["required"] is True
    assert report["threshold_max_mean_margin_gap"] == 1.0
    assert report["passed"] is True
    assert report["per_seed"]["11"]["max_mean_margin_gap"] == pytest.approx(0.8)
    assert report["per_seed"]["13"]["max_mean_margin_gap"] == pytest.approx(0.9)


def test_capability_matching_fails_when_one_matched_pair_exceeds_threshold() -> None:
    config = load_config("configs/mvp.yaml")
    samples = [
        _sample(11, "AB", 3.0, 4.0),
        _sample(11, "BA", 4.01, 4.0),
    ]

    report = capability_matching_report(config, samples)

    assert report["passed"] is False
    assert report["max_observed_mean_margin_gap"] == pytest.approx(1.01)


def test_confirmation_gate_refuses_failed_capability_matching() -> None:
    config = load_config("configs/mvp.yaml")
    fingerprint = "locked-protocol"
    report = {
        "protocol_fingerprint": fingerprint,
        "capability_matching": {
            "required": True,
            "passed": False,
            "max_observed_mean_margin_gap": 2.5,
            "threshold_max_mean_margin_gap": 1.0,
        },
    }

    with pytest.raises(PermissionError, match="Confirmation remains locked"):
        _require_discovery_capability_gate(
            config,
            report,
            protocol_fingerprint=fingerprint,
        )


def test_confirmation_gate_accepts_matching_discovery_report() -> None:
    config = load_config("configs/mvp.yaml")
    fingerprint = "locked-protocol"
    report = {
        "protocol_fingerprint": fingerprint,
        "capability_matching": {
            "required": True,
            "passed": True,
            "max_observed_mean_margin_gap": 0.75,
            "threshold_max_mean_margin_gap": 1.0,
        },
    }

    _require_discovery_capability_gate(
        config,
        report,
        protocol_fingerprint=fingerprint,
    )
