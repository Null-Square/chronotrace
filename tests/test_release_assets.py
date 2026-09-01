from __future__ import annotations

import subprocess
import sys


def test_generated_release_assets_are_in_sync() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_release_assets.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
