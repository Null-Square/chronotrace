"""Command-line interface for ChronoTrace MVP experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronotrace.config import load_config
from chronotrace.data import generate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chronotrace")
    parser.add_argument("--config", default="configs/mvp.yaml", help="experiment YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="generate deterministic synthetic stage/probe data")

    train = subparsers.add_parser("train", help="train one AB or BA endpoint")
    train.add_argument("--history", choices=("AB", "BA"), required=True)
    train.add_argument("--seed", type=int, required=True)

    features = subparsers.add_parser("features", help="extract fixed probe features")
    features.add_argument("--run-dir", required=True)

    detect = subparsers.add_parser(
        "detect", help="fit discovery detector and evaluate confirmation"
    )
    detect.add_argument("--runs-root", default=None)
    detect.add_argument("--output", default=None)

    matrix = subparsers.add_parser("matrix", help="print all endpoint training commands")
    matrix.add_argument(
        "--split", choices=("all", "discovery", "confirmation"), default="all"
    )
    return parser


def _generate(config_path: str) -> None:
    config = load_config(config_path)
    metadata = generate_dataset(
        config.data["root"],
        seed=int(config.data["seed"]),
        worlds=int(config.data["worlds"]),
        decoys_per_probe=int(config.data.get("decoys_per_probe", 3)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _train(config_path: str, history: str, seed: int) -> None:
    from chronotrace.training import train_endpoint

    config = load_config(config_path)
    run_dir = train_endpoint(config, history=history, training_seed=seed)
    print(run_dir)


def _features(config_path: str, run_dir: str) -> None:
    from chronotrace.features import extract_run_features

    config = load_config(config_path)
    output = extract_run_features(config, run_dir=run_dir)
    print(output)


def _detect(config_path: str, runs_root: str | None, output: str | None) -> None:
    from chronotrace.detector import fit_and_evaluate

    config = load_config(config_path)
    root = runs_root or config.artifacts["root"]
    report = fit_and_evaluate(config, runs_root=root)
    destination = Path(output or Path(root) / "phase0_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


def _matrix(config_path: str, split: str) -> None:
    config = load_config(config_path)
    splits = ("discovery", "confirmation") if split == "all" else (split,)
    for split_name in splits:
        for seed in config.seeds[split_name]:
            for history in config.histories:
                print(
                    "chronotrace --config "
                    f"{config_path} train --history {history} --seed {int(seed)}"
                )


def main(argv: list[str] | None = None) -> int:
    """Run the ChronoTrace CLI."""

    args = _parser().parse_args(argv)
    if args.command == "generate":
        _generate(args.config)
    elif args.command == "train":
        _train(args.config, args.history, args.seed)
    elif args.command == "features":
        _features(args.config, args.run_dir)
    elif args.command == "detect":
        _detect(args.config, args.runs_root, args.output)
    elif args.command == "matrix":
        _matrix(args.config, args.split)
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
