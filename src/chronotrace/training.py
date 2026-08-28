"""Matched sequential fine-tuning for ChronoTrace Phase-0 histories."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
    """Make stage randomness independent of where that stage appears in history."""

    offsets = {"A": 17, "B": 43, "C": 71}
    if stage not in offsets:
        raise ValueError(f"unsupported training stage: {stage}")
    return training_seed * 1009 + offsets[stage]


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
    filenames = {"A": "stage_a.jsonl", "B": "stage_b.jsonl", "C": "stage_c.jsonl"}
    if stage not in filenames:
        raise ValueError(f"unsupported training stage: {stage}")
    return read_jsonl(data_root / filenames[stage])


def _batch_stream(loader: Any) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def _configured_stage_steps(training_cfg: dict[str, Any], stage: str) -> int:
    per_stage = training_cfg.get("stage_steps_by_stage")
    if per_stage is not None:
        if stage not in per_stage:
            raise ValueError(f"stage_steps_by_stage is missing stage {stage}")
        steps = int(per_stage[stage])
    else:
        steps = int(training_cfg["stage_steps"])
    if steps <= 0:
        raise ValueError(f"stage {stage} must have a positive training step count")
    return steps


def _validate_phase0_inputs(config: ExperimentConfig, data_root: Path) -> dict[str, Any]:
    """Reject config drift or corrupted generated artifacts before a model is loaded."""

    if config.training.get("reset_optimizer_each_stage") is not True:
        raise ValueError("Phase-0 requires reset_optimizer_each_stage: true")
    if config.training.get("precision") != "fp32":
        raise ValueError("Phase-0 runner currently supports precision: fp32 only")

    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing generated metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    artifact_paths = {
        "stage_a": data_root / "stage_a.jsonl",
        "stage_b": data_root / "stage_b.jsonl",
        "probes": data_root / "probes.jsonl",
    }
    if any("C" in str(history) for history in config.histories):
        artifact_paths["stage_c"] = data_root / "stage_c.jsonl"

    for name, path in artifact_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing generated artifact: {path}")
        expected = metadata["sha256"].get(name)
        if expected is None:
            raise ValueError(f"Generated metadata is missing the hash for {name}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Generated artifact integrity check failed for {path}: {actual} != {expected}"
            )
    return metadata


def _enforce_fp32_model(torch: Any, model: Any, device: Any) -> Any:
    """Make the declared Phase-0 FP32 precision an enforced runtime invariant."""

    model.to(device=device, dtype=torch.float32)
    bad_dtypes = {
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    }
    if bad_dtypes:
        raise TypeError(f"Phase-0 model contains non-FP32 trainable tensors: {sorted(bad_dtypes)}")
    _assert_finite_parameters(torch, model, context="initial model load")
    return model


def _assert_finite_parameters(torch: Any, model: Any, *, context: str) -> None:
    """Fail closed if any floating-point parameter has NaN or Inf values."""

    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError(f"Non-finite parameter detected in {name!r} during {context}")


def train_endpoint(
    config: ExperimentConfig,
    *,
    history: str,
    training_seed: int,
    output_root: str | Path | None = None,
) -> Path:
    """Train one configured history endpoint from the same base checkpoint.

    Optimizer state is reset at every macro stage. Stage A/B/C shuffling uses a
    stage-specific seed that depends on the training seed but not on the stage position,
    so an identical terminal C stage is genuinely identical across matched histories.
    """

    declared_histories = {str(value) for value in config.histories}
    if history not in declared_histories:
        raise ValueError(
            f"history {history!r} is not declared in config.histories={sorted(declared_histories)}"
        )
    unsupported = set(history) - {"A", "B", "C"}
    if unsupported:
        raise ValueError(f"history contains unsupported stages: {sorted(unsupported)}")

    model_cfg = config.model
    training_cfg = config.training
    data_cfg = config.data
    base_model = str(model_cfg["checkpoint"])
    revision = model_cfg.get("revision")
    data_root = Path(data_cfg["root"])
    metadata = _validate_phase0_inputs(config, data_root)

    run_root = Path(output_root or config.artifacts["root"])
    run_id = f"phase0-{history.lower()}-seed{training_seed}"
    run_dir = run_root / run_id
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Run {run_id} already exists at {run_dir}. Remove it explicitly before rerunning."
        )
    endpoint_dir = run_dir / "endpoint"
    endpoint_dir.mkdir(parents=True, exist_ok=False)

    torch, AdamW, loader_types, hf_types = _require_mvp_stack()
    DataLoader, Dataset = loader_types
    AutoModelForCausalLM, AutoTokenizer = hf_types

    device = _device(torch, str(training_cfg.get("device", "auto")))
    _seed_everything(torch, training_seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision)
    model = _enforce_fp32_model(torch, model, device)

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
        preclip_grad_norms: list[float] = []
        stage_steps = _configured_stage_steps(training_cfg, stage)
        for step_index in range(stage_steps):
            batch = next(stream)
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss_value = float(output.loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"Non-finite loss in stage {stage} at step {step_index}: {loss_value}"
                )
            output.loss.backward()

            max_grad_norm = training_cfg.get("max_grad_norm")
            norm_limit = float(max_grad_norm) if max_grad_norm is not None else float("inf")
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                norm_limit,
                error_if_nonfinite=True,
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
            if not math.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm in stage {stage} at step {step_index}"
                )

            optimizer.step()
            losses.append(loss_value)
            preclip_grad_norms.append(grad_norm)

        _assert_finite_parameters(torch, model, context=f"end of stage {stage}")
        stage_metrics[stage] = {
            "mean_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "max_preclip_grad_norm": max(preclip_grad_norms),
            "steps": float(stage_steps),
        }

    _assert_finite_parameters(torch, model, context="before endpoint serialization")
    model.save_pretrained(endpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(endpoint_dir)

    stage_artifacts = {
        "stage_a_sha256": metadata["sha256"]["stage_a"],
        "stage_b_sha256": metadata["sha256"]["stage_b"],
        "probes_sha256": metadata["sha256"]["probes"],
    }
    if "stage_c" in metadata["sha256"]:
        stage_artifacts["stage_c_sha256"] = metadata["sha256"]["stage_c"]

    manifest = RunManifest(
        run_id=run_id,
        history=history,
        training_seed=training_seed,
        git_commit=_git_commit(),
        status="trained",
        base_model=base_model,
        base_revision=str(revision) if revision is not None else None,
        stage_artifacts=stage_artifacts,
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
            "model_dtype": str(next(model.parameters()).dtype),
        },
        artifacts={"endpoint": str(endpoint_dir)},
        notes=[
            "Optimizer reset at each macro stage.",
            "Stage shuffle seeds are independent of macro order and stage position.",
            "FP32 precision is explicitly enforced on all floating-point model parameters.",
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
