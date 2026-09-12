#!/usr/bin/env python3
"""Audit the current manuscript-facing ledger, not regenerate model evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / "research/bidirectional_v1"
    s = json.loads((p / "publication_summary.json").read_text())
    u = json.loads((p / "utility_summary.json").read_text())
    expected = [("fp32", 5, 24, 3), ("fp32", 10, 24, 3),
                ("fp32", 20, 24, 3), ("fp16", 5, 24, 3),
                ("fp16", 10, 24, 3), ("fp16", 20, 0, 8)]
    actual = [(c["precision"], c["duration"], c["verified"], c["ranked"])
              for c in s["cells"]]
    if actual != expected or any(c["cases"] != 24 for c in s["cells"]):
        raise ValueError("paper cell ledger drift")
    if sum(c["verified"] for c in s["cells"]) != u["primary"]["total_verified"]:
        raise ValueError("aggregate drift")
    if (s["observations"], s["problems"], s["seed_clusters"],
            s["distinct_sampled_orders"], s["inferred_pairs"],
            s["wrong_full"], s["wrong_pairs"]) != (144, 72, 12, 12, 3360, 0, 0):
        raise ValueError("evidence unit or error count drift")
    paper_root = ROOT / "paper/bidirectional"
    paper = (paper_root / "main.tex").read_text()
    paper += "".join(p.read_text() for p in sorted((paper_root / "sections").glob("*.tex")))
    required = ["Omar Al-Tawil", "All 72 FP32 observations", "0/24", "8.74", "7.33",
                "12 distinct sampled orders", "not a subfactorial", "ChatGPT"]
    if any(item not in paper for item in required):
        raise ValueError("paper narrative/identity drift")
    for rel in ["src/chronotrace/geometry/bidirectional.py",
                "tests/test_bidirectional.py", "paper/legacy_pythia/main.tex"]:
        if not (ROOT / rel).is_file():
            raise ValueError(f"missing current or historical file: {rel}")
    historical = ROOT / "paper/legacy_pythia/main.tex"
    original_sha256 = "ba0550f2844c73e02312396611fe70d6ede01b1e8015f4d9225d2e724546f1c2"
    if hashlib.sha256(historical.read_bytes()).hexdigest() != original_sha256:
        raise ValueError("immutable historical manuscript changed")
    print(json.dumps({"status": "passed", "cells": len(expected),
        "scientific_reexecution": False,
        "manuscript_sha256": hashlib.sha256(paper.encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
