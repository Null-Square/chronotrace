"""Matched sequential fine-tuning for the Phase-0 histories."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Iterator

from chronotrace.config import ExperimentConfig
from chronotrace.data import file_sha256, read_jsonl
from chronotrace.manifest import RunManifest


def _require_mvp_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch, AdamW, (DataLoader, Dataset), (AutoModelForCausalLM, AutoTokenizer)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def _stable_stage_seed(training_seed: int, stage: str) -> int:
    """Make stage randomness independent of whether the stage came first or second."""

    return training_seed * 1009 + (17 if stage == "A" else 43)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _read_stage_rows(data_root: Path, stage: str) -> list[dict[str, Any]]:
    filename = "stage_a.jsonl" if stage == "A" else "stage_b.jsonl"
    return read_jsonl(data_root / filename)


def _batch_stream(loader: Any) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def train_endpoint(
    config: ExperimentConfig,
    *,
    history: str,
    training_seed: int,
    output_root: str | Path | None = None,
) -> Path:
    """Train one AB or BA endpoint from the same base checkpoint.

    The optimizer is reset at each macro stage by default. This avoids making the first
    experiment depend on Adam moment carry-over. A/B data shuffling also uses a
    stage-specific seed that does not depend on whether the stage came first or second.
    """

    if history not in {"AB", "BA"}:
        raise ValueError("history must be AB or BA")

    torch, AdamW, loader_types, hf_types = _require_mvp_stack()
    DataLoader, Dataset = loader_types
    AutoModelForCausalLM, AutoTokenizer = hf_types

    model_cfg = config.model
    training_cfg = config.training
    data_cfg = config.data
    base_model = str(model_cfg["checkpoint"])
    revision = model_cfg.get("revision")
    data_root = Path(data_cfg["root"])
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))

    run_root = Path(output_root or config.artifacts["root"])
    run_id = f"phase0-{history.lower()}-seed{training_seed}"
    run_dir = run_root / run_id
    endpoint_dir = run_dir / "endpoint"
    endpoint_dir.mkdir(parents=True, exist_ok=True)

    device = _device(torch, str(training_cfg.get("device", "auto")))
    _seed_everything(torch, training_seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision)
    model.to(device)

    max_length = int(training_cfg["max_length"])

    class CompletionDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]
            prompt_ids = tokenizer(
                row["prompt"], add_special_tokens=False, truncation=False
            )["input_ids"]
            completion_ids = tokenizer(
                row["completion"], add_special_tokens=False, truncation=False
            )["input_ids"]
            eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            input_ids = (prompt_ids + completion_ids + eos)[:max_length]
            labels = ([-100] * len(prompt_ids) + completion_ids + eos)[:max_length]
            if not any(label != -100 for label in labels):
                raise ValueError("max_length removed the full supervised completion")
            return {"input_ids": input_ids, "labels": labels}

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in batch)
        pad_id = tokenizer.pad_token_id
        input_ids = []
        labels = []
        attention = []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    stage_metrics: dict[str, dict[str, float]] = {}
    for stage in history:
        stage_seed = _stable_stage_seed(training_seed, stage)
        _seed_everything(torch, stage_seed)
        rows = _read_stage_rows(data_root, stage)
        dataset = CompletionDataset(rows)
        generator = torch.Generator()
        generator.manual_seed(stage_seed)
        loader = DataLoader(
            dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=0,
        )
        stream = _batch_stream(loader)
        optimizer = AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )
        model.train()
        losses: list[float] = []
        stage_steps = int(training_cfg["stage_steps"])
        for _ in range(stage_steps):
            batch = next(stream)
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            if training_cfg.get("max_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_cfg["max_grad_norm"])
                )
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        stage_metrics[stage] = {
            "mean_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "steps": float(stage_steps),
        }

    model.save_pretrained(endpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(endpoint_dir)

    manifest = RunManifest(
        run_id=run_id,
        history=history,
        training_seed=training_seed,
        git_commit=_git_commit(),
        status="trained",
        base_model=base_model,
        base_revision=str(revision) if revision is not None else None,
        stage_artifacts={
            "stage_a_sha256": metadata["sha256"]["stage_a"],
            "stage_b_sha256": metadata["sha256"]["stage_b"],
            "probes_sha256": metadata["sha256"]["probes"],
        },
        config={
            "model": dict(config.model),
            "training": dict(config.training),
            "data": dict(config.data),
        },
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
        },
        artifacts={"endpoint": str(endpoint_dir)},
        notes=[
            "Optimizer reset at each macro stage.",
            "Stage shuffle seeds are independent of macro order.",
        ],
    )
    manifest.write_json(run_dir / "manifest.json")
    (run_dir / "training_metrics.json").write_text(
        json.dumps(stage_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "manifest.sha256").write_text(
        file_sha256(run_dir / "manifest.json") + "  manifest.json\n", encoding="utf-8"
    )
    return run_dir
