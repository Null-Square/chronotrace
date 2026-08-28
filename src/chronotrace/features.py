"""Fixed-probe feature extraction from trained endpoints."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from chronotrace.config import ExperimentConfig
from chronotrace.data import read_jsonl


_BINDING_FAMILIES = {
    "a_to_b_congruent",
    "a_to_b_incongruent",
    "b_to_a_congruent",
    "b_to_a_incongruent",
}
_CONTROL_FAMILIES = {"a_control", "b_control"}


def _require_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
    }


def _score_sequences(
    model: Any,
    tokenizer: Any,
    torch: Any,
    sequences: list[tuple[str, str]],
    *,
    batch_size: int,
    device: Any,
) -> list[float]:
    """Return summed conditional log probability for each completion."""

    prepared: list[tuple[list[int], list[int]]] = []
    for prompt, completion in sequences:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not prompt_ids or not completion_ids:
            raise ValueError("probe prompt and completion must both tokenize to non-empty sequences")
        prepared.append((prompt_ids, completion_ids))

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            full = [prompt + completion for prompt, completion in chunk]
            width = max(len(ids) for ids in full)
            pad_id = tokenizer.pad_token_id
            input_ids: list[list[int]] = []
            attention: list[list[int]] = []
            completion_masks: list[list[int]] = []
            for (prompt_ids, _completion_ids), ids in zip(chunk, full, strict=True):
                pad = width - len(ids)
                input_ids.append(ids + [pad_id] * pad)
                attention.append([1] * len(ids) + [0] * pad)
                completion_masks.append(
                    [0] * len(prompt_ids) + [1] * (len(ids) - len(prompt_ids)) + [0] * pad
                )

            ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            attention_tensor = torch.tensor(attention, dtype=torch.long, device=device)
            mask_tensor = torch.tensor(completion_masks, dtype=torch.bool, device=device)
            logits = model(input_ids=ids_tensor, attention_mask=attention_tensor).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            labels = ids_tensor[:, 1:]
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            prediction_mask = mask_tensor[:, 1:]
            batch_scores = (token_log_probs * prediction_mask).sum(dim=1)
            scores.extend(float(value) for value in batch_scores.detach().cpu())
    return scores


def extract_run_features(
    config: ExperimentConfig,
    *,
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Extract fixed behavioral features for one trained endpoint."""

    torch, AutoModelForCausalLM, AutoTokenizer = _require_stack()
    run_path = Path(run_dir)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    endpoint = run_path / "endpoint"
    data_root = Path(config.data["root"])
    probes = read_jsonl(data_root / "probes.jsonl")

    device_name = str(config.training.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(endpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(endpoint)
    model.to(device)

    sequences: list[tuple[str, str]] = []
    slices: list[tuple[int, int]] = []
    for probe in probes:
        candidates = [probe["answer"], *probe["decoys"]]
        start = len(sequences)
        sequences.extend((probe["prompt"], completion) for completion in candidates)
        slices.append((start, len(sequences)))

    logps = _score_sequences(
        model,
        tokenizer,
        torch,
        sequences,
        batch_size=int(config.forensics.get("score_batch_size", 16)),
        device=device,
    )

    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    by_world_family: dict[tuple[str, str], float] = {}
    for probe, (start, end) in zip(probes, slices, strict=True):
        candidate_scores = logps[start:end]
        correct = candidate_scores[0]
        decoy_mean = mean(candidate_scores[1:])
        margin = correct - decoy_mean
        row = {
            "probe_id": probe["probe_id"],
            "world_id": probe["world_id"],
            "family": probe["family"],
            "correct_logp": correct,
            "decoy_mean_logp": decoy_mean,
            "margin": margin,
        }
        rows.append(row)
        by_family[probe["family"]].append(margin)
        by_world_family[(probe["world_id"], probe["family"])] = margin

    features: dict[str, float] = {}
    feature_groups: dict[str, list[str]] = {"capability": [], "forensic": []}
    for family, values in sorted(by_family.items()):
        summary = _summarize(values)
        for statistic, value in summary.items():
            name = f"{family}.{statistic}"
            features[name] = value
            if family in _CONTROL_FAMILIES:
                feature_groups["capability"].append(name)
            if family in _BINDING_FAMILIES:
                feature_groups["forensic"].append(name)

    a_to_b_delta: list[float] = []
    b_to_a_delta: list[float] = []
    worlds = sorted({probe["world_id"] for probe in probes})
    for world_id in worlds:
        a_to_b_delta.append(
            by_world_family[(world_id, "a_to_b_congruent")]
            - by_world_family[(world_id, "a_to_b_incongruent")]
        )
        b_to_a_delta.append(
            by_world_family[(world_id, "b_to_a_congruent")]
            - by_world_family[(world_id, "b_to_a_incongruent")]
        )

    for prefix, values in (
        ("binding.a_to_b", a_to_b_delta),
        ("binding.b_to_a", b_to_a_delta),
    ):
        for statistic, value in _summarize(values).items():
            name = f"{prefix}.{statistic}"
            features[name] = value
            feature_groups["forensic"].append(name)

    asymmetry = mean(a_to_b_delta) - mean(b_to_a_delta)
    features["binding.directional_asymmetry"] = asymmetry
    feature_groups["forensic"].append("binding.directional_asymmetry")

    capability = {
        "a_mean_margin": features["a_control.mean"],
        "b_mean_margin": features["b_control.mean"],
        "absolute_gap": abs(features["a_control.mean"] - features["b_control.mean"]),
    }
    payload = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "history": manifest["history"],
        "training_seed": manifest["training_seed"],
        "features": features,
        "feature_groups": feature_groups,
        "capability": capability,
        "probe_rows": rows,
    }
    destination = Path(output_path) if output_path else run_path / "features.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
