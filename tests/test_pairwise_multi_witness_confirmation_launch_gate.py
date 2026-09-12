import json
from pathlib import Path


def _marker_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_confirmation_workflow_is_fresh_v3_marker_only() -> None:
    lock = json.loads(
        Path(
            "configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json"
        ).read_text(encoding="utf-8")
    )
    workflow = Path(
        ".github/workflows/pythia-14m-pairwise-multi-witness-confirmation.yml"
    ).read_text(encoding="utf-8")

    assert lock["fresh_confirmation_launch_authorized"] is True
    assert lock["fresh_confirmation_outputs_observed_before_freeze"] is False
    assert lock["no_intermediate_adaptation"] is True
    assert "workflow_dispatch" not in workflow
    assert "configs/chronotrace_pairwise_multi_witness_confirmation_v3.launch" in workflow
    assert "configs/chronotrace_pairwise_multi_witness_confirmation.launch" not in workflow
    assert "fail-fast: false" in workflow
    assert "max-parallel: 4" in workflow
    for seed in (2186192236, 1368008047, 92712904, 1944430236):
        assert str(seed) in workflow
    for spent in (4294917749, 3885207466, 402469483, 2000073798):
        assert str(spent) not in workflow
    assert "needs: confirm-seed" in workflow
    assert "analyze_pairwise_multi_witness_confirmation_suite_v3.py" in workflow


def test_fresh_v3_launch_marker_is_valid_if_present() -> None:
    marker = Path("configs/chronotrace_pairwise_multi_witness_confirmation_v3.launch")
    if not marker.exists():
        return
    values = _marker_values(marker)
    assert values["protocol"] == "chronotrace-pairwise-multi-witness-confirmation-32-v3"
    assert values["launch_gate_ci_conclusion"] == "success"
    assert values["confirmation_case_count"] == "32"
    assert values["pairwise_decision_count"] == "192"
    assert values["orientation_class_lp_solve_count"] == "384"
    assert values["seed_job_count"] == "4"
    assert values["targets_per_seed"] == "8"
    assert values["total_expected_stage_executions"] == "384"
    assert values["no_intermediate_adaptation"] == "true"
    assert values["fresh_confirmation_outputs_observed_before_launch"] == "false"
    assert values["fresh_confirmation_launch_authorized"] == "true"
