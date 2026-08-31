import json
import subprocess
import sys
from pathlib import Path

from chronotrace.reproducibility import json_sha256

PAIR_LABELS = ("AB", "AC", "AD", "BC", "BD", "CD")


def _synthetic_seed_result(lock: dict, seed: int, *, full_coverage: int = 7) -> dict:
    cases = {}
    for target in lock["target_histories_per_seed"]:
        pairs = {
            label: {
                "terminal_exactness_passed": True,
                "lower_bound_sound_in_witness_geometry": True,
                "lower_bound_sound_against_euclidean_vertices": True,
            }
            for label in PAIR_LABELS
        }
        cases[target] = {
            "active_lift_target_hash_match": True,
            "pairwise": pairs,
        }
    return {
        "result_version": "chronotrace-pairwise-multi-witness-confirmation-seed-v2",
        "seed": seed,
        "confirmation_lock_sha256": json_sha256(lock),
        "cases": cases,
        "stage_executions": 96,
        "witness_freeze_stage_executions": 72,
        "retained_degree4_endpoint_projection_scalars": 768,
        "retained_degree4_interaction_projection_scalars": 768,
        "full_k4_model_space_tensors_retained": False,
        "full_history_certificate_coverage": full_coverage,
        "label_blind_pairwise_orientation_certificate_coverage": 42,
        "ambiguous_pair_count": 6,
        "contradictory_pair_count": 0,
        "both_orientations_excluded_count": 0,
        "minimum_excluded_orientation_margin_over_guard": 0.001,
        "projected_reconstruction_residual_max": 1e-12,
        "maximum_terminal_primal_exactness_error": 1e-12,
        "invalid_seed_job": False,
        "codebook_sha256": f"fresh-codebook-{seed}",
        "dataset_sha256": {"worlds": f"fresh-dataset-{seed}"},
    }


def test_v3_aggregator_reproduces_frozen_excellent_threshold(tmp_path: Path) -> None:
    lock_path = Path("configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inputs = []
    for seed in lock["fresh_heldout_seeds"]:
        result = _synthetic_seed_result(lock, int(seed))
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        inputs.append(str(path))

    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pairwise_multi_witness_confirmation_suite_v3.py",
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
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["full_history_certificate_coverage"] == 28
    assert selection["label_blind_pairwise_orientation_certificate_coverage"] == 168
    assert selection["full_history_abstention_count"] == 4
    assert selection["outcome_classification"] == "excellent"
    assert selection["invalid_suite"] is False
    assert selection["v1_seed_results_used_in_v3_selection"] is False


def test_v3_aggregator_invalidates_contradictions(tmp_path: Path) -> None:
    lock_path = Path("configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inputs = []
    for index, seed in enumerate(lock["fresh_heldout_seeds"]):
        result = _synthetic_seed_result(lock, int(seed), full_coverage=8)
        if index == 0:
            result["contradictory_pair_count"] = 1
            result["label_blind_pairwise_orientation_certificate_coverage"] = 47
            result["ambiguous_pair_count"] = 0
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        inputs.append(str(path))

    output = tmp_path / "selection-invalid.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pairwise_multi_witness_confirmation_suite_v3.py",
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
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["invalid_suite"] is True
    assert selection["outcome_classification"] == "invalid"


def test_v3_aggregator_freezes_all_coverage_thresholds() -> None:
    source = Path(
        "scripts/analyze_pairwise_multi_witness_confirmation_suite_v3.py"
    ).read_text(encoding="utf-8")
    assert "full_coverage >= 28" in source
    assert "full_coverage >= 24" in source
    assert "full_coverage >= 16" in source
    assert 'return "scientific_negative"' in source
