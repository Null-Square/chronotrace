#!/usr/bin/env python3
"""Run fast repository setup checks without GPU dependencies."""

from __future__ import annotations

from pathlib import Path
import sys

from chronotrace import __version__, load_config


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "configs" / "mvp.yaml",
        root / "docs" / "MVP.md",
        root / "docs" / "EXPERIMENT_PROTOCOL.md",
        root / "paper" / "outline.md",
    ]

    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        print("ChronoTrace doctor: missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    config = load_config(root / "configs" / "mvp.yaml")
    print(f"ChronoTrace {__version__}")
    print(f"Protocol: {config.experiment.get('protocol_version')}")
    print(f"Histories: {', '.join(config.histories)}")
    print("Repository scaffold: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
