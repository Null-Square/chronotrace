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

    assert confirmation["confirmation_version"] == (
        "chronotrace-pairwise-multi-witness-confirmation-32-v2"
    )
    assert confirmation["freeze_status"] == (
        "frozen_before_any_heldout_confirmation_output_after_label_blind_audit"
    )
    assert confirmation["experiment_role"] == "confirmatory_heldout_validation"
    assert confirmation["confirmation_codebooks_observed_before_freeze"] is False
    assert confirmation["heldout_confirmation_launch_authorized"] is True
    assert confirmation["no_intermediate_adaptation"] is True
    assert methodology["methodology_version"] == confirmation["methodology_version"]
    assert methodology["freeze_status"] == (
        "frozen_before_any_heldout_confirmation_output_after_label_blind_audit"
    )
    assert confirmation["method_head_commit"] == methodology["method_head_commit"]
    assert confirmation["method_head_commit"] == (
        "d30b712e14fcb8ff90663791d8aebc976bb99fa5"
    )
    assert confirmation["pre_confirmation_label_blind_audit"][
        "completed_before_any_heldout_codebook_generation"
    ] is True
    assert confirmation["pre_confirmation_label_blind_audit"][
        "heldout_output_observed"
    ] is False

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
    assert confirmation["orientation_class_lp_solve_count"] == 384
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
    assert certificate["pair_decisions_per_target"] == 6
    assert certificate["orientation_class_lp_solves_per_pair"] == 2


def test_v2_confirmation_runner_is_label_blind_and_freezes_before_k4() -> None:
    path = Path("scripts/pythia_14m_pairwise_multi_witness_confirmation_v2.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    freeze_check = source.index("confirmation witness freeze boundary drift")
    basis_release = source.index("del basis", freeze_check)
    k4_loop = source.index("for word in degree4_words:", basis_release)
    decision_call = source.index("certify_pairwise_orientation(", k4_loop)
    evaluation = source.index("expected_relation =", decision_call)
    assert freeze_check < basis_release < k4_loop < decision_call < evaluation
    assert "expected_freeze_calls != 72" in source
    assert "expected_stage_calls != 96" in source
    assert "endpoint_scalar_count != 768" in source
    assert "interaction_scalar_count != 768" in source
    assert '"full_k4_model_space_tensors_retained": False' in source
    assert "_exact_terminal_hull_distance" in source
    assert "terminal_exactness_passed" in source
    assert "lower_bound_sound_in_witness_geometry" in source
    assert "lower_bound_sound_against_euclidean_vertices" in source
    assert "evaluation_contradictory" in source
    assert "both_orientations_excluded_count" in source
    assert "wrong_relation = (true_relation[1]" not in source


def test_workflow_launches_all_seeds_from_one_immutable_marker() -> None:
    source = Path(
        ".github/workflows/pythia-14m-pairwise-multi-witness-confirmation.yml"
    ).read_text(encoding="utf-8")
    assert "configs/chronotrace_pairwise_multi_witness_confirmation.launch" in source
    assert "fail-fast: false" in source
    assert "max-parallel: 4" in source
    for seed in (4294917749, 3885207466, 402469483, 2000073798):
        assert str(seed) in source
    assert "pythia_14m_pairwise_multi_witness_confirmation_v2.py" in source
    assert "analyze_pairwise_multi_witness_confirmation_suite_v2.py" in source
