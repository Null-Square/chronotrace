#!/usr/bin/env python3
# ruff: noqa: I001
"""Pre-model entrypoint for the frozen projected-K4 survivor diagnostic.

The scientific runner intentionally remains unchanged after launch. This adapter repairs
only the first-attempt configuration plumbing bug: portable numerical controls live in the
already-frozen K23 source lock, not in the K3-convex protocol. The adapter verifies the
K23 canonical JSON hash against the K3-convex source lock before substituting that lock
only for ``_configure_portable_numerics``.
"""

from __future__ import annotations

from pathlib import Path

from pythia_finite_pair_bridge import _load_json
from pythia_14m_projected_k4_survivor_diagnostic import main as diagnostic_main
import pythia_14m_projected_k4_survivor_diagnostic as diagnostic

from chronotrace.reproducibility import json_sha256


K23_PROTOCOL = Path("configs/pythia_14m_four_stage_k23_pilot.lock.json")
K3_CONVEX_PROTOCOL = Path("configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json")


def main() -> None:
    k23 = _load_json(K23_PROTOCOL)
    source_k3 = _load_json(K3_CONVEX_PROTOCOL)
    expected_k23_hash = str(source_k3["source_k23_protocol_sha256"])
    observed_k23_hash = json_sha256(k23)
    if observed_k23_hash != expected_k23_hash:
        raise RuntimeError("source K23 protocol hash drift before projected K4 execution")
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K23 protocol touched confirmation codebooks")
    if int(k23["pilot_codebook_seed"]) != 1011473075:
        raise RuntimeError("source K23 spent codebook drift")
    if "portable_kernel_gate" not in k23:
        raise RuntimeError("source K23 protocol is missing portable numerical controls")

    original_configure = diagnostic._configure_portable_numerics

    def configure_from_k23(torch: object, _source_k3_lock: object, threads: int) -> object:
        return original_configure(torch, k23, threads)

    diagnostic._configure_portable_numerics = configure_from_k23
    diagnostic_main()


if __name__ == "__main__":
    main()
