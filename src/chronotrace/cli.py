"""Command-line interface for ChronoTrace MVP experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronotrace.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chronotrace")
    parser.add_argument("--config", default="configs/mvp.yaml", help="experiment YAML")
    parser.add_argument(
        "--lock",
        default="configs/mvp.lock.json",
        help="frozen protocol lock used for drift detection",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="generate and verify frozen synthetic data")
    subparsers.add_parser("verify", help="verify config/data against the frozen protocol lock")
    subparsers.add_parser("discover", help="run the complete discovery split and LOSO report")
    subparsers.add_parser("freeze", help="seal discovery artifacts before confirmation")
    subparsers.add_parser("confirm", help="run the sealed one-shot confirmation split")

    train = subparsers.add_parser("train", help="train one discovery AB or BA endpoint")
    train.add_argument("--history", choices=("AB", "BA"), required=True)
    train.add_argument("--seed", type=int, required=True)

    features = subparsers.add_parser("features", help="extract fixed probe features")
    features.add_argument("--run-dir", required=True)

    matrix = subparsers.add_parser("matrix", help="print discovery endpoint training commands")
    matrix.add_argument("--split", choices=("discovery",), default="discovery")
    return parser


def _generate(config_path: str, lock_path: str) -> None:
    from chronotrace.phase0 import ensure_frozen_dataset

    config = load_config(config_path)
    lock = ensure_frozen_dataset(config, lock_path=lock_path)
    print(json.dumps({"protocol_fingerprint": lock["fingerprint"]}, indent=2))


def _verify(config_path: str, lock_path: str) -> None:
    from chronotrace.protocol import verify_protocol_lock

    config = load_config(config_path)
    data_root = Path(config.data["root"])
    root = data_root if (data_root / "metadata.json").exists() else None
    lock = verify_protocol_lock(config, lock_path=lock_path, data_root=root)
    print(json.dumps({"protocol_fingerprint": lock["fingerprint"], "status": "locked"}, indent=2))


def _discover(config_path: str, lock_path: str) -> None:
    from chronotrace.phase0 import run_discovery

    config = load_config(config_path)
    print(run_discovery(config, lock_path=lock_path))


def _freeze(config_path: str, lock_path: str) -> None:
    from chronotrace.phase0 import freeze_confirmation

    config = load_config(config_path)
    print(freeze_confirmation(config, lock_path=lock_path))


def _confirm(config_path: str, lock_path: str) -> None:
    from chronotrace.phase0 import run_confirmation

    config = load_config(config_path)
    print(run_confirmation(config, lock_path=lock_path))


def _train(config_path: str, lock_path: str, history: str, seed: int) -> None:
    from chronotrace.phase0 import ensure_frozen_dataset
    from chronotrace.training import train_endpoint

    config = load_config(config_path)
    confirmation = {int(value) for value in config.seeds["confirmation"]}
    if seed in confirmation:
        raise PermissionError(
            "Direct training of a confirmation seed is blocked. Use discovery -> freeze -> confirm."
        )
    discovery = {int(value) for value in config.seeds["discovery"]}
    if seed not in discovery:
        raise ValueError("Direct Phase-0 training is limited to declared discovery seeds")
    ensure_frozen_dataset(config, lock_path=lock_path)
    run_dir = train_endpoint(config, history=history, training_seed=seed)
    print(run_dir)


def _features(config_path: str, run_dir: str) -> None:
    from chronotrace.features import extract_run_features

    config = load_config(config_path)
    output = extract_run_features(config, run_dir=run_dir)
    print(output)


def _matrix(config_path: str) -> None:
    config = load_config(config_path)
    for seed in config.seeds["discovery"]:
        for history in config.histories:
            print(
                "chronotrace --config "
                f"{config_path} train --history {history} --seed {int(seed)}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the ChronoTrace CLI."""

    args = _parser().parse_args(argv)
    if args.command == "generate":
        _generate(args.config, args.lock)
    elif args.command == "verify":
        _verify(args.config, args.lock)
    elif args.command == "discover":
        _discover(args.config, args.lock)
    elif args.command == "freeze":
        _freeze(args.config, args.lock)
    elif args.command == "confirm":
        _confirm(args.config, args.lock)
    elif args.command == "train":
        _train(args.config, args.lock, args.history, args.seed)
    elif args.command == "features":
        _features(args.config, args.run_dir)
    elif args.command == "matrix":
        _matrix(args.config)
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
