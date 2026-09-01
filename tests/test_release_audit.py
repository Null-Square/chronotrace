from __future__ import annotations

import json
import subprocess
import sys


def test_frozen_release_audit_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/audit_release.py", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "ok"
    assert summary["outcome_classification"] == "strong"
    assert summary["complete_histories"] == "27/32"
    assert summary["pairwise_precedences"] == "182/192"
    assert summary["full_history_abstentions"] == 5
    assert summary["contradictions"] == 0
    assert summary["double_exclusions"] == 0
    assert summary["fresh_seed_count"] == 4
