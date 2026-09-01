#!/usr/bin/env python3
"""Run the frozen label-blind confirmation on one freshly derived v3 seed."""

from __future__ import annotations

import hashlib
from typing import Any

import pythia_14m_pairwise_multi_witness_confirmation_v2 as v2


def _derived_fresh_seeds(confirmation: dict[str, Any]) -> tuple[int, ...]:
    spent = {int(value) for value in confirmation["spent_seeds_excluded"]}
    chosen: list[int] = []
    records = confirmation["fresh_seed_derivation"]["records"]
    for expected_index, record in enumerate(records):
        index = int(record["i"])
        if index != expected_index:
            raise RuntimeError("fresh seed derivation index drift")
        label = f"chronotrace-v2-confirmation-seed:{index}"
        if str(record["label"]) != label:
            raise RuntimeError("fresh seed derivation label drift")
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
        if str(record["sha256"]) != digest:
            raise RuntimeError("fresh seed derivation digest drift")
        seed = int.from_bytes(bytes.fromhex(digest)[:4], "big")
        if int(record["seed"]) != seed:
            raise RuntimeError("fresh seed derivation integer drift")
        if seed in spent or seed in chosen:
            raise RuntimeError("fresh seed collides with spent/selected seed")
        chosen.append(seed)
    return tuple(chosen)


def _validate_locks_v3(
    confirmation: dict[str, Any],
    methodology: dict[str, Any],
    source_k3: dict[str, Any],
    source_k23: dict[str, Any],
    seed: int,
) -> None:
    if confirmation.get("confirmation_version") != (
        "chronotrace-pairwise-multi-witness-confirmation-32-v3"
    ):
        raise RuntimeError("fresh confirmation version drift")
    if confirmation.get("freeze_status") != (
        "frozen_before_any_fresh_v3_confirmation_codebook_generation_or_output"
    ):
        raise RuntimeError("fresh confirmation freeze-status drift")
    if confirmation.get("experiment_role") != (
        "fresh_confirmatory_validation_of_frozen_label_blind_method"
    ):
        raise RuntimeError("fresh confirmation role drift")
    if confirmation.get("fresh_confirmation_outputs_observed_before_freeze") is not False:
        raise RuntimeError("fresh confirmation outputs were observed before freeze")
    if confirmation.get("fresh_confirmation_launch_authorized") is not True:
        raise RuntimeError("fresh confirmation launch is not authorized")
    if confirmation.get("no_intermediate_adaptation") is not True:
        raise RuntimeError("fresh confirmation must prohibit intermediate adaptation")
    if methodology.get("methodology_version") != confirmation["methodology_version"]:
        raise RuntimeError("fresh methodology version drift")
    if methodology.get("freeze_status") != (
        "frozen_for_fresh_confirmation_after_v1_provenance_audit"
    ):
        raise RuntimeError("fresh methodology freeze-status drift")
    if methodology.get("method_head_commit") != confirmation["method_head_commit"]:
        raise RuntimeError("fresh method provenance drift")
    if methodology.get("v1_numeric_outputs_used_for_v3_method_tuning") is not False:
        raise RuntimeError("v1 outputs were used to tune the v3 method")
    if methodology.get("fresh_v3_confirmation_outputs_observed_before_freeze") is not False:
        raise RuntimeError("v3 methodology observed fresh confirmation output")
    if source_k3.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K3 development lock provenance drift")
    if source_k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K23 development lock provenance drift")
    if v2.json_sha256(source_k23) != str(source_k3["source_k23_protocol_sha256"]):
        raise RuntimeError("source K23 protocol hash drift")

    derived = _derived_fresh_seeds(confirmation)
    frozen = tuple(int(value) for value in confirmation["fresh_heldout_seeds"])
    if derived != frozen or len(frozen) != 4:
        raise RuntimeError("fresh seed set does not match deterministic derivation")
    if seed not in frozen:
        raise RuntimeError("seed is not one of the frozen fresh v3 seeds")
    spent = {int(value) for value in confirmation["spent_seeds_excluded"]}
    if seed in spent:
        raise RuntimeError("fresh v3 seed collides with a spent seed")
    if tuple(confirmation["stages"]) != ("A", "B", "C", "D"):
        raise RuntimeError("fresh confirmation stage set drift")
    if int(confirmation["confirmation_case_count"]) != 32:
        raise RuntimeError("fresh confirmation case count drift")
    if int(confirmation["pairwise_decision_count"]) != 192:
        raise RuntimeError("fresh confirmation pairwise decision count drift")
    if int(confirmation["orientation_class_lp_solve_count"]) != 384:
        raise RuntimeError("fresh confirmation orientation-class LP count drift")


if __name__ == "__main__":
    v2._validate_locks = _validate_locks_v3
    v2.main()
