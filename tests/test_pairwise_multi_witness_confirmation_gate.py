from pathlib import Path


def test_v2_scientific_engine_is_label_blind_and_freezes_before_k4() -> None:
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


def test_active_workflow_launches_fresh_v3_suite_from_one_marker() -> None:
    source = Path(
        ".github/workflows/pythia-14m-pairwise-multi-witness-confirmation.yml"
    ).read_text(encoding="utf-8")
    assert "configs/chronotrace_pairwise_multi_witness_confirmation_v3.launch" in source
    assert "configs/chronotrace_pairwise_multi_witness_confirmation.launch" not in source
    assert "fail-fast: false" in source
    assert "max-parallel: 4" in source
    for seed in (2186192236, 1368008047, 92712904, 1944430236):
        assert str(seed) in source
    for spent in (4294917749, 3885207466, 402469483, 2000073798):
        assert str(spent) not in source
    assert "pythia_14m_pairwise_multi_witness_confirmation_v3.py" in source
    assert "analyze_pairwise_multi_witness_confirmation_suite_v3.py" in source
