import json
from pathlib import Path


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


def test_k3_convex_launch_remains_spent_only_and_confirmation_blocked() -> None:
    protocol_path = Path("configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json")
    runner_path = Path("scripts/pythia_14m_k3_convex_last_stage_diagnostic.py")
    workflow_path = Path(".github/workflows/pythia-14m-k3-convex-last-stage-diagnostic.yml")
    launch_path = Path("configs/pythia_14m_k3_convex_last_stage_diagnostic.launch")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = runner_path.read_text(encoding="utf-8")
    workflow = workflow_path.read_text(encoding="utf-8")
    launch = _launch_values(launch_path)
    compile(source, str(runner_path), "exec")

    assert protocol["freeze_status"] == "frozen_before_any_k3_convex_last_stage_pythia_output"
    assert protocol["experiment_role"] == "non_confirmatory_spent_scalability_diagnostic"
    assert protocol["confirmation_codebooks_observed"] is False
    assert protocol["heldout_confirmation_launch_authorized"] is False
    assert protocol["pilot_codebook_seed"] not in protocol["confirmation_codebooks_prohibited"]
    assert len(protocol["launch_blockers"]) == 3

    assert "project_quadratic_simplex" in source
    assert "true_chronology_certified_impossible" in source
    assert "heldout_confirmation_launch_authorized" in source
    assert "pythia_14m_k3_convex_last_stage_diagnostic.launch" in workflow
    assert "pythia_14m_k3_convex_last_stage_diagnostic.py" in workflow
    assert "pythia_14m_four_stage_k23_pilot.lock.json" in workflow
    assert "pythia_14m_forward_reachable_all24.selection.json" in workflow
    assert "pythia_14m_k3_affine_last_stage_diagnostic.selection.json" in workflow

    if launch:
        assert launch["protocol"] == protocol["protocol_version"]
        assert int(launch["spent_codebook_seed"]) == protocol["pilot_codebook_seed"]
        assert launch["target_history"] == protocol["target_history"]
        assert launch["confirmation_codebooks_observed"] == "false"
        assert launch["heldout_confirmation_launch_authorized"] == "false"
        assert launch["launch_role"] == protocol["experiment_role"]
