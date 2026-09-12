#!/usr/bin/env python3
"""Verify the frozen ChronoTrace paper/reviewer package without model downloads."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from chronotrace.reproducibility import json_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / (
    "configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json"
)


class ReleaseAuditError(RuntimeError):
    """Raised when a frozen release invariant is violated."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseAuditError(message)


def _classification(full_coverage: int, invalid: bool) -> str:
    if invalid:
        return "invalid"
    if full_coverage >= 28:
        return "excellent"
    if full_coverage >= 24:
        return "strong"
    if full_coverage >= 16:
        return "partial"
    return "scientific_negative"


def _verify_hash(selection: dict[str, Any], field: str, path: str) -> None:
    value = _load(ROOT / path)
    actual = json_sha256(value)
    _require(actual == selection[field], f"canonical hash drift for {path}: {actual}")


def _verify_balanced_targets(lock: dict[str, Any]) -> None:
    targets = tuple(str(value) for value in lock["target_histories_per_seed"])
    _require(len(targets) == 8, "fresh confirmation must contain eight targets per seed")
    _require(len(set(targets)) == 8, "fresh confirmation target histories must be unique")
    for position in range(4):
        counts = Counter(target[position] for target in targets)
        _require(
            counts == {"A": 2, "B": 2, "C": 2, "D": 2},
            "target position balance drift",
        )


def _verify_public_copy() -> None:
    required = {
        "README.md": ("27 / 32", "182 / 192", "STRONG"),
        "docs/RESULTS_FREEZE.md": ("27 / 32", "182 / 192", "strong"),
        "paper/main.tex": ("27/32", "182/192", "five conservative"),
        "docs/REVIEWER_GUIDE.md": ("27/32", "182/192", "zero contradictory"),
    }
    for relative, snippets in required.items():
        path = ROOT / relative
        _require(path.exists(), f"paper-facing file missing: {relative}")
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            _require(
                snippet in text,
                f"paper-facing result drift in {relative}: {snippet!r}",
            )


def audit(selection_path: Path) -> dict[str, Any]:
    selection = _load(selection_path)
    _require(
        selection.get("selection_version")
        == "chronotrace-pairwise-multi-witness-confirmation-selection-v3-frozen",
        "unexpected frozen selection version",
    )
    _require(
        selection.get("scientific_seed_jobs_all_success") is True,
        "seed job failure recorded",
    )
    _require(
        selection.get("seed_rerun_after_scientific_execution") is False,
        "seed rerun recorded",
    )
    _require(
        selection.get("method_changed_after_fresh_confirmation_started") is False,
        "method drift",
    )
    _require(
        selection.get("v1_seed_results_used_in_v3_selection") is False,
        "spent v1 evidence reused",
    )

    _verify_hash(
        selection,
        "confirmation_lock_sha256",
        "configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json",
    )
    _verify_hash(
        selection,
        "methodology_lock_sha256",
        "configs/chronotrace_pairwise_multi_witness_methodology_v3.lock.json",
    )
    _verify_hash(
        selection,
        "source_k3_protocol_sha256",
        "configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json",
    )
    _verify_hash(
        selection,
        "source_k23_protocol_sha256",
        "configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )

    confirmation_lock = _load(
        ROOT / "configs/chronotrace_pairwise_multi_witness_confirmation_v3.lock.json"
    )
    _verify_balanced_targets(confirmation_lock)
    expected_seeds = {
        str(int(seed)) for seed in confirmation_lock["fresh_heldout_seeds"]
    }
    per_seed = selection["per_seed"]
    _require(
        set(per_seed) == expected_seeds,
        "fresh seed set drift between lock and selection",
    )

    full = sum(
        int(row["full_history_certificate_coverage"]) for row in per_seed.values()
    )
    pairs = sum(
        int(row["label_blind_pairwise_orientation_certificate_coverage"])
        for row in per_seed.values()
    )
    ambiguous = sum(int(row["ambiguous_pair_count"]) for row in per_seed.values())
    contradictions = sum(
        int(row["contradictory_pair_count"]) for row in per_seed.values()
    )
    both = sum(
        int(row["both_orientations_excluded_count"]) for row in per_seed.values()
    )

    _require(
        full == int(selection["full_history_certificate_coverage"]),
        "history aggregate drift",
    )
    _require(
        pairs
        == int(selection["label_blind_pairwise_orientation_certificate_coverage"]),
        "pair aggregate drift",
    )
    _require(
        ambiguous == int(selection["ambiguous_pair_count"]),
        "ambiguity aggregate drift",
    )
    _require(
        contradictions == int(selection["contradictory_pair_count"]),
        "contradiction drift",
    )
    _require(
        both == int(selection["both_orientations_excluded_count"]),
        "double-exclusion drift",
    )
    _require(
        pairs + ambiguous == int(selection["pairwise_decision_count"]),
        "pair accounting drift",
    )
    _require(
        int(selection["full_history_abstention_count"]) == 32 - full,
        "abstention drift",
    )

    abstention_cases = 0
    abstention_pairs = 0
    for by_target in selection["abstentions_by_seed_target_and_pair"].values():
        for labels in by_target.values():
            abstention_cases += 1
            abstention_pairs += len(labels)
    _require(
        abstention_cases == int(selection["full_history_abstention_count"]),
        "case abstention drift",
    )
    _require(abstention_pairs == ambiguous, "ambiguous pair ledger drift")

    invalid = bool(selection["invalid_suite"])
    expected_class = _classification(full, invalid)
    _require(
        selection["outcome_classification"] == expected_class,
        "outcome tier drift",
    )
    _require(
        expected_class == "strong",
        "frozen paper result is no longer the strong tier",
    )
    _require(
        contradictions == 0 and both == 0 and not invalid,
        "frozen suite validity failure",
    )

    digest_re = re.compile(r"^sha256:[0-9a-f]{64}$")
    for seed, row in per_seed.items():
        _require(
            int(row["stage_executions"]) == 96,
            f"stage execution drift for seed {seed}",
        )
        _require(
            int(row["witness_freeze_stage_executions"]) == 72,
            f"witness freeze boundary drift for seed {seed}",
        )
        _require(
            row["invalid"] is False,
            f"invalid fresh seed job recorded for {seed}",
        )
        _require(
            bool(digest_re.match(row["artifact_digest"])),
            f"artifact digest malformed for {seed}",
        )
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", row["result_raw_sha256"])),
            f"result digest malformed for {seed}",
        )

    _require(
        selection["all_terminal_witness_hull_exactness_passed"] is True,
        "terminal exactness failed",
    )
    _require(
        selection["all_corrected_bounds_sound_in_witness_geometry"] is True,
        "witness soundness failed",
    )
    _require(
        selection["all_corrected_bounds_sound_against_euclidean_vertices"] is True,
        "Euclidean soundness failed",
    )
    _require(
        selection["all_target_active_lifts_replayed_exactly"] is True,
        "active-lift replay failed",
    )

    _verify_public_copy()

    return {
        "status": "ok",
        "selection": str(selection_path.relative_to(ROOT)),
        "scientific_run_id": int(selection["scientific_run_id"]),
        "outcome_classification": selection["outcome_classification"],
        "complete_histories": f"{full}/32",
        "pairwise_precedences": f"{pairs}/192",
        "full_history_abstentions": int(selection["full_history_abstention_count"]),
        "contradictions": contradictions,
        "double_exclusions": both,
        "fresh_seed_count": len(per_seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    args = parser.parse_args()
    try:
        summary = audit(args.selection.resolve())
    except (ReleaseAuditError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ChronoTrace release audit: FAIL: {exc}")
        return 1
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print("ChronoTrace release audit: PASS")
        for key, value in summary.items():
            if key != "status":
                print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
