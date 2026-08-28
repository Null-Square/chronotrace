"""Reproducibility manifest contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ALLOWED_HISTORIES = frozenset({"AB", "BA", "ABC", "BAC"})


@dataclass
class RunManifest:
    """Machine-readable record for one independently trained endpoint."""

    run_id: str
    history: str
    training_seed: int
    git_commit: str
    status: str = "created"
    base_model: str | None = None
    base_revision: str | None = None
    stage_artifacts: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.history not in _ALLOWED_HISTORIES:
            allowed = ", ".join(sorted(_ALLOWED_HISTORIES))
            raise ValueError(
                f"Unsupported ChronoTrace history {self.history!r}; allowed: {allowed}"
            )
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.git_commit:
            raise ValueError("git_commit must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write the manifest using stable, human-readable JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
