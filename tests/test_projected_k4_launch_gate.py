import hashlib
import json
from pathlib import Path


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _launch_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_projected_k4_launch_remains_spent_only_and_confirmation_blocked() -> None:
    protocol_path = Path("configs/pythia_14m_projected_k4_survivor_diagnostic.lock.json")
    source_protocol_path = Path("configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json")
    source_selection_path = Path(
        "configs/pythia_14m_k3_convex_last_stage_diagnostic.selection.json"
    )
    k23_path = Path("configs/pythia_14m_four_stage_k23_pilot.lock.json")
    attempt1_path = Path("configs/pythia_14m_projected_k4_survivor_diagnostic.attempt1.json")
    runner_path = Path("scripts/pythia_14m_projected_k4_survivor_diagnostic.py")
    entry_path = Path("scripts/pythia_14m_projected_k4_survivor_entry.py")
    workflow_path = Path(
        ".github/workflows/pythia-14m-projected-k4-survivor-diagnostic.yml"
    )
    launch_path = Path("configs/pythia_14m_projected_k4_survivor_diagnostic.launch")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
    k23 = json.loads(k23_path.read_text(encoding="utf-8"))
    attempt1 = json.loads(attempt1_path.read_text(encoding="utf-8"))
    runner = runner_path.read_text(encoding="utf-8")
    entry = entry_path.read_text(encoding="utf-8")
    workflow = workflow_path.read_text(encoding="utf-8")
    launch = _launch_values(launch_path)
    compile(runner, str(runner_path), "exec")
    compile(entry, str(entry_path), "exec")

    assert protocol["freeze_status"] == "frozen_before_any_projected_k4_pythia_output"
    assert (
        protocol["experiment_role"]
        == "non_confirmatory_spent_survivor_active_lift_diagnostic"
    )
    assert protocol["confirmation_codebooks_observed"] is False
    assert protocol["heldout_confirmation_launch_authorized"] is False
    assert protocol["pilot_codebook_seed"] not in protocol["confirmation_codebooks_prohibited"]
    assert protocol["source_k3_eliminated_last_stages"] == ["A", "B"]
    assert protocol["source_k3_surviving_last_stages"] == ["C", "D"]
    assert protocol["degree4_word_count"] == 24
    assert protocol["retained_degree4_projection_scalars"] == 96
    assert protocol["total_expected_stage_executions"] == 68
    assert protocol["hierarchy_max_degree"] == 4
    assert protocol["hierarchy_coordinate_count"] == 64
    assert _json_sha256(source_protocol) == protocol["source_k3_convex_protocol_sha256"]
    assert _json_sha256(source_selection) == protocol["source_k3_convex_selection_sha256"]

    assert source_selection["eliminated_last_stages"] == ["A", "B"]
    assert source_selection["surviving_last_stages"] == ["C", "D"]
    assert source_selection["all_diagnostic_checks_passed"] is True
    assert source_selection["heldout_confirmation_next_step_authorized"] is False

    assert "portable_kernel_gate" in k23
    assert k23["confirmation_codebooks_observed"] is False
    assert _json_sha256(k23) == source_protocol["source_k23_protocol_sha256"]

    assert attempt1["workflow_run_id"] == 33323546560
    assert attempt1["workflow_job_id"] == 99289590738
    assert attempt1["failure_class"] == "pre_model_configuration_plumbing_failure"
    assert attempt1["stage_executions"] == 0
    assert attempt1["model_weights_loaded"] is False
    assert attempt1["scientific_outputs_observed"] is False
    assert attempt1["output_artifact_created"] is False
    assert attempt1["confirmation_codebooks_observed"] is False
    assert attempt1["scientific_settings_changed_by_correction"] is False

    assert "projected_interaction_from_endpoint_delta" in runner
    assert "solve_local_order_lp" in runner
    assert "witness_freeze_stage_calls" in runner
    assert "active_lift_target_hash_match" in runner
    assert "heldout_confirmation_launch_authorized" in runner

    assert "pythia_14m_four_stage_k23_pilot.lock.json" in entry
    assert "source_k23_protocol_sha256" in entry
    assert "portable_kernel_gate" in entry
    assert "_configure_portable_numerics" in entry
    assert "pythia_14m_projected_k4_survivor_diagnostic" in entry

    assert "pythia_14m_projected_k4_survivor_diagnostic.launch" in workflow
    assert "pythia_14m_projected_k4_survivor_entry.py" in workflow
    assert "pythia_14m_k3_convex_last_stage_diagnostic.lock.json" in workflow
    assert "pythia_14m_k3_convex_last_stage_diagnostic.selection.json" in workflow

    if launch:
        assert launch["protocol"] == protocol["protocol_version"]
        assert int(launch["spent_codebook_seed"]) == protocol["pilot_codebook_seed"]
        assert launch["target_history"] == protocol["target_history"]
        assert launch["survivor_last_stages"] == "C,D"
        assert launch["confirmation_codebooks_observed"] == "false"
        assert launch["heldout_confirmation_launch_authorized"] == "false"
        assert launch["launch_role"] == protocol["experiment_role"]
