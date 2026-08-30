import json
from pathlib import Path


def test_projected_k4_exact_prefix_repair_is_protocol_conforming() -> None:
    protocol = json.loads(
        Path("configs/pythia_14m_projected_k4_survivor_diagnostic.lock.json").read_text(
            encoding="utf-8"
        )
    )
    amendment = json.loads(
        Path(
            "configs/pythia_14m_projected_k4_survivor_diagnostic.implementation_amendment.json"
        ).read_text(encoding="utf-8")
    )
    invalid = json.loads(
        Path(
            "configs/pythia_14m_projected_k4_survivor_diagnostic.invalid_attempts.json"
        ).read_text(encoding="utf-8")
    )
    runner_path = Path("scripts/pythia_14m_projected_k4_survivor_diagnostic.py")
    source = runner_path.read_text(encoding="utf-8")
    compile(source, str(runner_path), "exec")

    assert amendment["status"] == "frozen_after_invalid_unreported_attempt_before_any_retry_output"
    assert amendment["source_protocol_version"] == protocol["protocol_version"]
    assert amendment["scientific_decision_rule_changed"] is False
    assert amendment["survivor_set_changed"] is False
    assert amendment["witness_definition_changed"] is False
    assert amendment["thresholds_changed"] is False
    assert amendment["heldout_policy_changed"] is False
    assert amendment["stage_execution_budget_changed"] is False
    assert amendment["total_expected_stage_executions"] == 68
    assert protocol["total_expected_stage_executions"] == 68
    assert amendment["confirmation_codebooks_observed"] is False
    assert amendment["heldout_confirmation_launch_authorized"] is False

    attempts = {item["run_id"]: item for item in invalid["attempts"]}
    second = attempts[33323819378]
    assert second["job_id"] == 99290329962
    assert second["planned_stage_executions_completed_before_failure"] == 68
    assert second["scientific_metrics_computed_in_process"] is True
    assert second["scientific_metrics_emitted"] is False
    assert second["artifact_emitted"] is False
    assert second["scientific_metrics_observed_by_researcher"] is False
    assert second["c_or_d_scores_logged"] is False
    assert second["decision_rule_changed_after_failure"] is False

    assert "measure_ordered_interaction_basis_streaming_exact" in source
    assert "measure_ordered_interaction_basis_compact" not in source
    assert "TemporaryDirectory" in source
    assert "torch.save(endpoint, path)" in source
    assert "torch.load(prefix_path" in source
    assert "prefix_path.unlink()" in source
    assert "exact K3 prefix cache coverage drift" in source
    assert "K3 witness {stage} lower-bound reproduction drift" in source
    assert "ABCD K3-prefix active lift did not reproduce the target endpoint" in source
    assert "implementation_amendment_sha256" in source
