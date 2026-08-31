import json
import subprocess
import sys
from pathlib import Path

from chronotrace.reproducibility import json_sha256


def test_confirmation_aggregator_reproduces_frozen_excellent_threshold(tmp_path: Path) -> None:
    lock_path = Path("configs/chronotrace_pairwise_multi_witness_confirmation.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha = json_sha256(lock)
    targets = tuple(lock["target_histories_per_seed"])
    inputs: list[str] = []

    for seed in lock["heldout_seeds"]:
        cases = {}
        for target in targets:
            pairs = {
                label: {
                    "true_relation_sanity_passed": True,
                    "wrong_lower_bound_sound": True,
                }
                for label in ("AB", "AC", "AD", "BC", "BD", "CD")
            }
            cases[target] = {
                "active_lift_target_hash_match": True,
                "pairwise": pairs,
            }
        result = {
            "seed": seed,
            "confirmation_lock_sha256": lock_sha,
            "cases": cases,
            "stage_executions": 96,
            "witness_freeze_stage_executions": 72,
            "retained_degree4_endpoint_projection_scalars": 768,
            "retained_degree4_interaction_projection_scalars": 768,
            "full_k4_model_space_tensors_retained": False,
            "full_history_certificate_coverage": 7,
            "pairwise_wrong_orientation_certificate_coverage": 42,
            "minimum_wrong_orientation_margin_over_guard": 0.001,
            "invalid_seed_job": False,
            "codebook_sha256": f"codebook-{seed}",
            "dataset_sha256": {"worlds": f"dataset-{seed}"},
        }
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        inputs.append(str(path))

    output = tmp_path / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pairwise_multi_witness_confirmation_suite.py",
            "--lock",
            str(lock_path),
            "--inputs",
            *inputs,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["full_history_certificate_coverage"] == 28
    assert aggregate["pairwise_wrong_orientation_certificate_coverage"] == 168
    assert aggregate["full_history_abstention_count"] == 4
    assert aggregate["outcome_classification"] == "excellent"
    assert aggregate["invalid_suite"] is False


def test_confirmation_aggregator_source_freezes_all_coverage_thresholds() -> None:
    source = Path(
        "scripts/analyze_pairwise_multi_witness_confirmation_suite.py"
    ).read_text(encoding="utf-8")
    assert "full_coverage >= 28" in source
    assert "full_coverage >= 24" in source
    assert "full_coverage >= 16" in source
    assert 'return "scientific_negative"' in source
