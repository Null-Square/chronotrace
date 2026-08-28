"""Run one Phase-0b design endpoint without touching confirmation infrastructure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from chronotrace.config import ExperimentConfig
from chronotrace.data import generate_dataset
from chronotrace.features import extract_run_features
from chronotrace.training import train_endpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0b_design.yaml")
    parser.add_argument("--c-steps", type=int, required=True, choices=(50, 150, 300))
    parser.add_argument("--history", required=True, choices=("ABC", "BAC"))
    parser.add_argument("--seed", type=int, required=True, choices=(13, 23, 29))
    parser.add_argument("--output-root", default="pilot")
    return parser


def main() -> int:
    args = _parser().parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw["training"]["stage_steps_by_stage"]["C"] = int(args.c_steps)
    candidate_root = Path(args.output_root) / f"c{args.c_steps}"
    raw["artifacts"]["root"] = str(candidate_root)
    config = ExperimentConfig.from_mapping(raw)

    data_root = Path(config.data["root"])
    metadata = generate_dataset(
        data_root,
        seed=int(config.data["seed"]),
        worlds=int(config.data["worlds"]),
        decoys_per_probe=int(config.data["decoys_per_probe"]),
        include_balanced_stage_c=True,
    )
    if metadata["stage_a_examples"] != metadata["stage_b_examples"]:
        raise RuntimeError("A/B corpus sizes are not balanced")
    if metadata["stage_c_examples"] != (
        metadata["stage_a_examples"] + metadata["stage_b_examples"]
    ):
        raise RuntimeError("terminal C corpus is not the exact balanced union")

    run_dir = train_endpoint(
        config,
        history=args.history,
        training_seed=args.seed,
        output_root=candidate_root,
    )
    features_path = extract_run_features(config, run_dir=run_dir)

    metrics = json.loads((run_dir / "training_metrics.json").read_text(encoding="utf-8"))
    features = json.loads(features_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["environment"]["model_dtype"] != "torch.float32":
        raise RuntimeError("pilot endpoint was not trained in enforced FP32")
    for stage in args.history:
        for key in ("mean_loss", "final_loss", "max_preclip_grad_norm"):
            if not math.isfinite(float(metrics[stage][key])):
                raise RuntimeError(f"non-finite {stage}.{key}")
    if not all(math.isfinite(float(value)) for value in features["features"].values()):
        raise RuntimeError("non-finite forensic feature detected")

    print(
        json.dumps(
            {
                "candidate_c_steps": args.c_steps,
                "history": args.history,
                "seed": args.seed,
                "run_dir": str(run_dir),
                "a_control_mean": features["features"]["a_control.mean"],
                "b_control_mean": features["features"]["b_control.mean"],
                "directional_asymmetry": features["features"][
                    "binding.directional_asymmetry"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
