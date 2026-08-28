"""Plain-SGD execution helpers for the ChronoTrace Pythia scale gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chronotrace.scale import ScaleTrainingExample


@dataclass(frozen=True)
class ScaleRunMetrics:
    """Numerical evidence from one deterministic stage execution."""

    initial_loss: float
    final_loss: float
    max_gradient_norm: float
    relative_displacement: float
    finite: bool


def _require_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def flatten_parameters(torch: Any, model: Any) -> Any:
    """Return a detached FP32 CPU vector in model parameter order."""

    parts = [parameter.detach().float().reshape(-1).cpu() for parameter in model.parameters()]
    if not parts:
        raise ValueError("model has no parameters")
    return torch.cat(parts)


def load_flat_parameters(torch: Any, model: Any, vector: Any, *, device: Any) -> None:
    """Load a flat vector into a model without changing parameter ordering."""

    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            chunk = vector[cursor : cursor + count]
            if chunk.numel() != count:
                raise ValueError("parameter vector is too short")
            parameter.copy_(chunk.reshape_as(parameter).to(device=device, dtype=parameter.dtype))
            cursor += count
    if cursor != vector.numel():
        raise ValueError("parameter vector is too long")


def completion_batch(
    torch: Any,
    tokenizer: Any,
    examples: Sequence[ScaleTrainingExample],
    device: Any,
) -> dict[str, Any]:
    """Create one deterministic full batch with completion-only labels."""

    rows: list[tuple[list[int], list[int]]] = []
    for example in examples:
        prompt_ids = list(tokenizer.encode(example.prompt, add_special_tokens=False))
        completion_ids = list(tokenizer.encode(example.completion, add_special_tokens=False))
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        input_ids = prompt_ids + completion_ids + eos
        labels = [-100] * len(prompt_ids) + completion_ids + eos
        rows.append((input_ids, labels))
    width = max(len(item[0]) for item in rows)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer requires a pad token")

    input_batch: list[list[int]] = []
    label_batch: list[list[int]] = []
    attention_batch: list[list[int]] = []
    for input_ids, labels in rows:
        pad = width - len(input_ids)
        input_batch.append(input_ids + [pad_id] * pad)
        label_batch.append(labels + [-100] * pad)
        attention_batch.append([1] * len(input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long, device=device),
        "labels": torch.tensor(label_batch, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_batch, dtype=torch.long, device=device),
    }


def _batch_loss(model: Any, batch: dict[str, Any]) -> float:
    model.eval()
    with __import__("torch").no_grad():
        value = float(model(**batch).loss.detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite evaluation loss: {value}")
    return value


def execute_plain_sgd_stage(
    torch: Any,
    model: Any,
    tokenizer: Any,
    examples: Sequence[ScaleTrainingExample],
    *,
    learning_rate: float,
    updates: int,
    device: Any,
    initial_vector: Any | None = None,
) -> tuple[Any, ScaleRunMetrics]:
    """Run one deterministic full-batch stage and return its flat endpoint."""

    if learning_rate <= 0 or updates <= 0:
        raise ValueError("learning_rate and updates must be positive")
    if initial_vector is not None:
        load_flat_parameters(torch, model, initial_vector, device=device)

    base_vector = flatten_parameters(torch, model)
    base_norm = float(torch.linalg.vector_norm(base_vector))
    if base_norm <= 0 or not math.isfinite(base_norm):
        raise FloatingPointError("invalid base parameter norm")

    batch = completion_batch(torch, tokenizer, examples, device)
    initial_loss = _batch_loss(model, batch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=0.0,
        weight_decay=0.0,
    )
    gradient_norms: list[float] = []
    model.train()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = float(output.loss.detach().cpu())
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite scale loss at step {step}: {loss}")
        output.loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float("inf"),
            error_if_nonfinite=True,
        )
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite scale gradient at step {step}")
        optimizer.step()
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point() and not bool(torch.isfinite(parameter.detach()).all()):
                raise FloatingPointError(f"non-finite parameter {name!r} after scale step {step}")
        gradient_norms.append(grad_norm)

    final_loss = _batch_loss(model, batch)
    endpoint = flatten_parameters(torch, model)
    relative_displacement = float(torch.linalg.vector_norm(endpoint - base_vector)) / base_norm
    finite = bool(torch.isfinite(endpoint).all()) and all(
        math.isfinite(value) for value in (initial_loss, final_loss, relative_displacement)
    )
    return endpoint, ScaleRunMetrics(
        initial_loss=initial_loss,
        final_loss=final_loss,
        max_gradient_norm=max(gradient_norms),
        relative_displacement=relative_displacement,
        finite=finite,
    )


def load_scale_model(
    model_id: str,
    revision: str,
    *,
    device_name: str = "cpu",
) -> tuple[Any, Any, Any, Any]:
    """Load one Pythia checkpoint in enforced FP32."""

    torch, AutoModelForCausalLM, AutoTokenizer = _require_stack()
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise TypeError("scale gate requires FP32 model parameters")
    return torch, model, tokenizer, device
