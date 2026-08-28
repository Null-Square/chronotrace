"""Discovery-only numerical stability probe for the Phase-0 Pythia fine-tuning stack.

This script is diagnostic by design. It does not create scientific endpoints and must
not use confirmation seeds. It reports exactly when loss/gradient/parameter finiteness
fails for one relation and one learning rate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from chronotrace.config import load_config
from chronotrace.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config("configs/mvp.yaml")
    if args.seed in {int(value) for value in config.seeds["confirmation"]}:
        raise ValueError("stability diagnostics may not use confirmation seeds")

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    checkpoint = str(config.model["checkpoint"])
    revision = config.model.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision)
    model.train()

    data_root = Path(config.data["root"])
    filename = "stage_a.jsonl" if args.stage == "A" else "stage_b.jsonl"
    rows = read_jsonl(data_root / filename)
    max_length = int(config.training["max_length"])

    class CompletionDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = rows[index]
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
            eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            input_ids = (prompt_ids + completion_ids + eos)[:max_length]
            labels = ([-100] * len(prompt_ids) + completion_ids + eos)[:max_length]
            return {"input_ids": input_ids, "labels": labels}

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention = []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    generator = torch.Generator().manual_seed(args.seed * 1009 + (17 if args.stage == "A" else 43))
    loader = DataLoader(
        CompletionDataset(),
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    iterator = iter(loader)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    first_loss = None
    last_finite_loss = None
    max_grad_norm = 0.0
    failure = None
    completed = 0
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = float(output.loss.detach())
        if first_loss is None:
            first_loss = loss
        if not math.isfinite(loss):
            failure = {"kind": "loss", "step": step, "value": str(loss)}
            break

        output.loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm = float(grad_norm_tensor.detach())
        if not math.isfinite(grad_norm):
            failure = {"kind": "gradient", "step": step, "value": str(grad_norm)}
            break
        max_grad_norm = max(max_grad_norm, grad_norm)
        optimizer.step()

        # Full parameter scan is intentionally done in this diagnostic, not production.
        bad_parameters = 0
        for parameter in model.parameters():
            bad_parameters += int((~torch.isfinite(parameter.detach())).sum().item())
        if bad_parameters:
            failure = {
                "kind": "parameter",
                "step": step,
                "nonfinite_parameter_count": bad_parameters,
            }
            break

        completed = step + 1
        last_finite_loss = loss

    result = {
        "stage": args.stage,
        "learning_rate": args.lr,
        "seed": args.seed,
        "requested_steps": args.steps,
        "completed_steps": completed,
        "first_loss": first_loss,
        "last_finite_loss": last_finite_loss,
        "max_preclip_grad_norm": max_grad_norm,
        "failure": failure,
        "stable": failure is None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
