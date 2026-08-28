"""End-to-end CPU smoke tests using a locally created tiny GPT-NeoX model.

This validates orchestration, training, feature extraction, discovery evaluation, sealing,
confirmation, and the Phase-0b ABC/BAC endpoint path without downloading a model or
consuming scientific confirmation seeds.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from chronotrace.config import ExperimentConfig
from chronotrace.data import generate_dataset, read_jsonl
from chronotrace.features import extract_run_features
from chronotrace.phase0 import freeze_confirmation, run_confirmation, run_discovery
from chronotrace.protocol import write_protocol_lock
from chronotrace.training import train_endpoint

ROOT = Path(__file__).resolve().parents[1]


def _build_local_tiny_model(base_dir: Path, data_root: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

    texts: list[str] = []
    filenames = ["stage_a.jsonl", "stage_b.jsonl", "probes.jsonl"]
    if (data_root / "stage_c.jsonl").exists():
        filenames.append("stage_c.jsonl")
    for filename in filenames:
        for row in read_jsonl(data_root / filename):
            for field in ("prompt", "completion", "answer"):
                if field in row:
                    texts.append(str(row[field]))
            texts.extend(str(item) for item in row.get("decoys", []))

    backend = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[EOS]"],
        min_frequency=1,
    )
    backend.train_from_iterator(texts, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(base_dir)

    model_config = GPTNeoXConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    GPTNeoXForCausalLM(model_config).save_pretrained(base_dir, safe_serialization=True)


def _run_phase0_smoke(root: Path) -> dict[str, float]:
    raw = yaml.safe_load((ROOT / "configs" / "mvp.yaml").read_text(encoding="utf-8"))
    data_root = root / "data"
    runs_root = root / "runs"
    base_dir = root / "tiny-base"

    raw["model"] = {
        "family": "gpt_neox_smoke",
        "checkpoint": str(base_dir),
        "revision": None,
    }
    raw["data"]["root"] = str(data_root)
    raw["data"]["worlds"] = 8
    raw["training"]["stage_steps"] = 2
    raw["training"]["batch_size"] = 4
    raw["training"]["max_length"] = 64
    raw["training"]["device"] = "cpu"
    raw["seeds"] = {"discovery": [1, 2], "confirmation": [3]}
    raw["artifacts"]["root"] = str(runs_root)
    raw["forensics"]["score_batch_size"] = 32
    raw["controls"]["require_capability_matching"] = False
    config = ExperimentConfig.from_mapping(raw)

    metadata = generate_dataset(
        data_root,
        seed=int(config.data["seed"]),
        worlds=int(config.data["worlds"]),
        decoys_per_probe=int(config.data["decoys_per_probe"]),
    )
    _build_local_tiny_model(base_dir, data_root)
    lock_path = root / "smoke.lock.json"
    write_protocol_lock(config, metadata["sha256"], lock_path)

    discovery_path = run_discovery(config, lock_path=lock_path, runs_root=runs_root)
    seal_path = freeze_confirmation(config, lock_path=lock_path, runs_root=runs_root)
    final_path = run_confirmation(config, lock_path=lock_path, runs_root=runs_root)

    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert discovery["split"] == "discovery"
    assert "balanced_accuracy" in discovery["forensic"]["leave_one_seed_out"]["metrics"]
    assert discovery["capability_matching"]["required"] is False
    assert seal_path.exists()
    assert "confirmation" in final["forensic"]
    assert len(list(runs_root.glob("phase0-*/features.json"))) == 6
    return {
        "discovery_balanced_accuracy": discovery["forensic"]["leave_one_seed_out"]["metrics"][
            "balanced_accuracy"
        ],
        "confirmation_balanced_accuracy": final["forensic"]["confirmation"]["metrics"][
            "balanced_accuracy"
        ],
    }


def _run_phase0b_endpoint_smoke(root: Path) -> None:
    raw = yaml.safe_load(
        (ROOT / "configs" / "phase0b_design.yaml").read_text(encoding="utf-8")
    )
    data_root = root / "phase0b-data"
    runs_root = root / "phase0b-runs"
    base_dir = root / "phase0b-tiny-base"

    raw["model"] = {
        "family": "gpt_neox_smoke",
        "checkpoint": str(base_dir),
        "revision": None,
    }
    raw["data"]["root"] = str(data_root)
    raw["data"]["worlds"] = 4
    raw["training"]["stage_steps"] = 1
    raw["training"]["stage_steps_by_stage"] = {"A": 1, "B": 1, "C": 1}
    raw["training"]["batch_size"] = 4
    raw["training"]["max_length"] = 64
    raw["training"]["device"] = "cpu"
    raw["seeds"] = {"discovery": [1], "confirmation": []}
    raw["artifacts"]["root"] = str(runs_root)
    raw["forensics"]["score_batch_size"] = 32
    raw["controls"]["require_capability_matching"] = False
    config = ExperimentConfig.from_mapping(raw)

    metadata = generate_dataset(
        data_root,
        seed=int(config.data["seed"]),
        worlds=int(config.data["worlds"]),
        decoys_per_probe=int(config.data["decoys_per_probe"]),
        include_balanced_stage_c=True,
    )
    assert metadata["stage_c_examples"] == (
        metadata["stage_a_examples"] + metadata["stage_b_examples"]
    )
    _build_local_tiny_model(base_dir, data_root)

    expected_stage_artifacts = {
        "stage_a_sha256",
        "stage_b_sha256",
        "stage_c_sha256",
        "probes_sha256",
    }
    for history in ("ABC", "BAC"):
        run_dir = train_endpoint(
            config,
            history=history,
            training_seed=1,
            output_root=runs_root,
        )
        features_path = extract_run_features(config, run_dir=run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        features = json.loads(features_path.read_text(encoding="utf-8"))
        assert manifest["history"] == history
        assert manifest["environment"]["model_dtype"] == "torch.float32"
        assert set(manifest["stage_artifacts"]) == expected_stage_artifacts
        assert features["history"] == history


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="chronotrace-smoke-") as temporary:
        root = Path(temporary)
        phase0 = _run_phase0_smoke(root)
        _run_phase0b_endpoint_smoke(root)
        print(json.dumps({"status": "ok", "phase0": phase0, "phase0b": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
