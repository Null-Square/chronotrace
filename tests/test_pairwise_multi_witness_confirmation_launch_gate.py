import json
from pathlib import Path


def _marker_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_confirmation_workflow_is_marker_only_and_frozen() -> None:
    lock = json.loads(
        Path(
            "configs/chronotrace_pairwise_multi_witness_confirmation.lock.json"
        ).read_text(encoding="utf-8")
    )
    workflow = Path(
        ".github/workflows/pythia-14m-pairwise-multi-witness-confirmation.yml"
    ).read_text(encoding="utf-8")

    assert lock["heldout_confirmation_launch_authorized"] is True
    assert lock["confirmation_codebooks_observed_before_freeze"] is False
    assert lock["no_intermediate_adaptation"] is True
    assert "workflow_dispatch" not in workflow
    assert "configs/chronotrace_pairwise_multi_witness_confirmation.launch" in workflow
    assert "fail-fast: false" in workflow
    assert "max-parallel: 4" in workflow
    assert "4294917749" in workflow
    assert "3885207466" in workflow
    assert "402469483" in workflow
    assert "2000073798" in workflow
    assert "needs: confirm-seed" in workflow
    assert "analyze_pairwise_multi_witness_confirmation_suite_v2.py" in workflow


def test_confirmation_launch_marker_is_valid_if_present() -> None:
    marker = Path("configs/chronotrace_pairwise_multi_witness_confirmation.launch")
    if not marker.exists():
        return
    values = _marker_values(marker)
    assert values["protocol"] == "chronotrace-pairwise-multi-witness-confirmation-32-v2"
    assert values["launch_gate_ci_conclusion"] == "success"
    assert values["confirmation_case_count"] == "32"
    assert values["pairwise_decision_count"] == "192"
    assert values["seed_job_count"] == "4"
    assert values["targets_per_seed"] == "8"
    assert values["total_expected_stage_executions"] == "384"
    assert values["no_intermediate_adaptation"] == "true"
    assert values["confirmation_codebooks_observed_before_launch"] == "false"
    assert values["heldout_confirmation_launch_authorized"] == "true"
