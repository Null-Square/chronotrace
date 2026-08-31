import json
from collections import Counter
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_confirmation_design_is_frozen_balanced_and_previously_heldout() -> None:
    confirmation = _load(
        "configs/chronotrace_pairwise_multi_witness_confirmation.lock.json"
    )
    methodology = _load(
        "configs/chronotrace_pairwise_multi_witness_methodology.lock.json"
    )
    source_k3 = _load(
        "configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json"
    )

    assert confirmation["freeze_status"] == "frozen_before_any_heldout_confirmation_output"
    assert confirmation["experiment_role"] == "confirmatory_heldout_validation"
    assert confirmation["confirmation_codebooks_observed_before_freeze"] is False
    assert confirmation["heldout_confirmation_launch_authorized"] is True
    assert confirmation["no_intermediate_adaptation"] is True
    assert confirmation["methodology_version"] == methodology["methodology_version"]
    assert confirmation["method_head_commit"] == methodology["method_head_commit"]
    assert confirmation["method_ci_run"] == methodology["method_ci_run"] == 33328870033
    assert methodology["method_ci_conclusion"] == "success"

    seeds = tuple(int(value) for value in confirmation["heldout_seeds"])
    assert seeds == (4294917749, 3885207466, 402469483, 2000073798)
    assert set(seeds) == {
        int(value) for value in source_k3["confirmation_codebooks_prohibited"]
    }

    targets = tuple(confirmation["target_histories_per_seed"])
    assert targets == (
        "ABCD",
        "BCDA",
        "CDAB",
        "DABC",
        "DCBA",
        "ADCB",
        "BADC",
        "CBAD",
    )
    assert len(targets) * len(seeds) == confirmation["confirmation_case_count"] == 32
    assert confirmation["pairwise_decision_count"] == 192
    for position in range(4):
        counts = Counter(target[position] for target in targets)
        assert counts == {"A": 2, "B": 2, "C": 2, "D": 2}

    sharing = confirmation["execution_sharing"]
    assert sharing["k3_basis_stage_executions_per_seed"] == 40
    assert sharing["target_replay_stage_executions_per_seed"] == 32
    assert sharing["witness_freeze_stage_executions_per_seed"] == 72
    assert sharing["k4_active_lift_stage_executions_per_seed"] == 24
    assert sharing["total_stage_executions_per_seed"] == 96
    assert sharing["total_stage_executions_suite"] == 384
    assert sharing["retained_degree4_endpoint_projection_scalars_per_seed"] == 768
    assert sharing["retained_degree4_interaction_projection_scalars_per_seed"] == 768

    certificate = confirmation["certificate"]
    assert certificate["hierarchy_degree"] == 4
    assert certificate["hierarchy_dimension"] == 64
    assert certificate["certificate_guard"] == 1e-10
    assert certificate["elimination_guard"] == 1e-6
    assert certificate["property_queries_per_target"] == 6


def test_confirmation_runner_preserves_freeze_before_k4_and_scalar_only_output() -> None:
    path = Path("scripts/pythia_14m_pairwise_multi_witness_confirmation.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    freeze_check = source.index("confirmation witness freeze boundary drift")
    basis_release = source.index("del basis", freeze_check)
    k4_loop = source.index("for word in degree4_words:", basis_release)
    assert freeze_check < basis_release < k4_loop
    assert "expected_freeze_calls != 72" in source
    assert "expected_stage_calls != 96" in source
    assert "endpoint_scalar_count != 768" in source
    assert "interaction_scalar_count != 768" in source
    assert '"full_k4_model_space_tensors_retained": False' in source
    assert "active_hash_match" in source
    assert "wrong_lower_bound_sound" in source
    assert "true_relation_sanity_passed" in source
