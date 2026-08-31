import json
import subprocess
import sys
from pathlib import Path

from chronotrace.reproducibility import json_sha256


PAIR_LABELS = ("AB", "AC", "AD", "BC", "BD", "CD")


def _seed_result(lock: dict, seed: int) -> dict:
    cases = {}
    targets = tuple(lock["target_histories_per_seed"])
    for index, target in enumerate(targets):
        full = index < 7
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
            "full_history_certified": full,
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
        "projected_reconstruction_residual_max": 1e-16,
        "maximum_terminal_primal_exactness_error": 1e-12,
        "full_history_certificate_coverage": 7,
        "label_blind_pairwise_orientation_certificate_coverage": 42,
        "ambiguous_pair_count": 6,
        "contradictory_pair_count": 0,
        "both_orientations_excluded_count": 0,
        "minimum_excluded_orientation_margin_over_guard": 0.001,
        "invalid_seed_job": False,
        "codebook_sha256": f"codebook-{seed}",
        "dataset_sha256": {"worlds": f"dataset-{seed}"},
    }


def test_v2_aggregator_reproduces_frozen_excellent_threshold(tmp_path: Path) -> None:
    lock_path = Path("configs/chronotrace_pairwise_multi_witness_confirmation.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inputs: list[str] = []
    for seed in lock["heldout_seeds"]:
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(
            json.dumps(_seed_result(lock, int(seed))),
            encoding="utf-8",
        )
        inputs.append(str(path))

    output = tmp_path / "aggregate.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pairwise_multi_witness_confirmation_suite_v2.py",
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
    assert aggregate["label_blind_pairwise_orientation_certificate_coverage"] == 168
    assert aggregate["ambiguous_pair_count"] == 24
    assert aggregate["contradictory_pair_count"] == 0
    assert aggregate["both_orientations_excluded_count"] == 0
    assert aggregate["full_history_abstention_count"] == 4
    assert aggregate["outcome_classification"] == "excellent"
    assert aggregate["invalid_suite"] is False


def test_v2_aggregator_invalidates_one_contradictory_pair(tmp_path: Path) -> None:
    lock_path = Path("configs/chronotrace_pairwise_multi_witness_confirmation.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inputs: list[str] = []
    for index, seed in enumerate(lock["heldout_seeds"]):
        result = _seed_result(lock, int(seed))
        if index == 0:
            result["contradictory_pair_count"] = 1
            result["invalid_seed_job"] = True
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        inputs.append(str(path))

    output = tmp_path / "aggregate-invalid.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_pairwise_multi_witness_confirmation_suite_v2.py",
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
    assert aggregate["invalid_suite"] is True
    assert aggregate["outcome_classification"] == "invalid"


def test_v2_aggregator_source_freezes_all_coverage_thresholds() -> None:
    source = Path(
        "scripts/analyze_pairwise_multi_witness_confirmation_suite_v2.py"
    ).read_text(encoding="utf-8")
    assert "full_coverage >= 28" in source
    assert "full_coverage >= 24" in source
    assert "full_coverage >= 16" in source
    assert 'return "scientific_negative"' in source
    assert "contradictory_pairs != 0 or both_excluded_pairs != 0" in source
