import hashlib
import json
from collections import Counter
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_v3_fresh_seed_derivation_is_deterministic_and_disjoint() -> None:
    lock = _load("configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json")
    methodology = _load("configs/chronotrace_pairwise_multi_witness_methodology_v3.lock.json")
    provenance = _load(
        "configs/chronotrace_pairwise_multi_witness_confirmation_v1_spent.provenance.json"
    )

    assert lock["confirmation_version"] == (
        "chronotrace-pairwise-multi-witness-confirmation-32-v3"
    )
    assert lock["fresh_confirmation_outputs_observed_before_freeze"] is False
    assert lock["fresh_confirmation_launch_authorized"] is True
    assert lock["no_intermediate_adaptation"] is True
    assert methodology["v1_numeric_outputs_used_for_v3_method_tuning"] is False
    assert provenance["status"] == (
        "v1_confirmation_seeds_spent_not_eligible_for_final_confirmation"
    )
    assert provenance["workflow_run_id"] == 33361355778

    spent = {int(value) for value in lock["spent_seeds_excluded"]}
    expected = []
    for record in lock["fresh_seed_derivation"]["records"]:
        index = int(record["i"])
        label = f"chronotrace-v2-confirmation-seed:{index}"
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
        seed = int.from_bytes(bytes.fromhex(digest)[:4], "big")
        assert record["label"] == label
        assert record["sha256"] == digest
        assert int(record["seed"]) == seed
        assert seed not in spent
        expected.append(seed)
    assert tuple(expected) == tuple(lock["fresh_heldout_seeds"])
    assert tuple(expected) == (2186192236, 1368008047, 92712904, 1944430236)


def test_v3_design_remains_balanced_and_budget_frozen() -> None:
    lock = _load("configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json")
    targets = tuple(lock["target_histories_per_seed"])
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
    for position in range(4):
        assert Counter(target[position] for target in targets) == {
            "A": 2,
            "B": 2,
            "C": 2,
            "D": 2,
        }
    assert lock["confirmation_case_count"] == 32
    assert lock["pairwise_decision_count"] == 192
    assert lock["orientation_class_lp_solve_count"] == 384
    sharing = lock["execution_sharing"]
    assert sharing["witness_freeze_stage_executions_per_seed"] == 72
    assert sharing["total_stage_executions_per_seed"] == 96
    assert sharing["total_stage_executions_suite"] == 384
    assert sharing["retained_degree4_endpoint_projection_scalars_per_seed"] == 768
    assert sharing["retained_degree4_interaction_projection_scalars_per_seed"] == 768


def test_v3_workflow_uses_new_marker_and_fresh_seeds_only() -> None:
    workflow = Path(
        ".github/workflows/pythia-14m-pairwise-multi-witness-confirmation.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch" not in workflow
    assert "configs/chronotrace_pairwise_multi_witness_confirmation_v3.launch" in workflow
    assert "configs/chronotrace_pairwise_multi_witness_confirmation.launch" not in workflow
    for seed in (2186192236, 1368008047, 92712904, 1944430236):
        assert str(seed) in workflow
    for spent in (4294917749, 3885207466, 402469483, 2000073798):
        assert str(spent) not in workflow
    assert "pythia_14m_pairwise_multi_witness_confirmation_v3.py" in workflow
    assert "analyze_pairwise_multi_witness_confirmation_suite_v3.py" in workflow
    assert "fail-fast: false" in workflow
    assert "max-parallel: 4" in workflow


def test_v3_entrypoint_only_replaces_provenance_validation() -> None:
    source = Path(
        "scripts/pythia_14m_pairwise_multi_witness_confirmation_v3.py"
    ).read_text(encoding="utf-8")
    compile(source, "confirmation-v3", "exec")
    assert "import pythia_14m_pairwise_multi_witness_confirmation_v2 as v2" in source
    assert "v2._validate_locks = _validate_locks_v3" in source
    assert "v2.main()" in source
    assert "certify_pairwise_orientation" not in source
    assert "solve_local_order_multi_witness_lp" not in source
