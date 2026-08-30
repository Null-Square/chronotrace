import json
from pathlib import Path


def test_k3_convex_runner_remains_unlaunched_and_protocol_blocked() -> None:
    protocol_path = Path("configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json")
    runner_path = Path("scripts/pythia_14m_k3_convex_last_stage_diagnostic.py")
    workflow_path = Path(".github/workflows/pythia-14m-k3-convex-last-stage-diagnostic.yml")
    launch_path = Path("configs/pythia_14m_k3_convex_last_stage_diagnostic.launch")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = runner_path.read_text(encoding="utf-8")
    compile(source, str(runner_path), "exec")

    assert protocol["freeze_status"] == "frozen_before_any_k3_convex_last_stage_pythia_output"
    assert protocol["confirmation_codebooks_observed"] is False
    assert protocol["heldout_confirmation_launch_authorized"] is False
    assert len(protocol["launch_blockers"]) == 3
    assert "project_quadratic_simplex" in source
    assert "true_chronology_certified_impossible" in source
    assert not workflow_path.exists()
    assert not launch_path.exists()
