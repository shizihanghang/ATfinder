from __future__ import annotations

import argparse
import json
from pathlib import Path

from ours.config import atfinder_config

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one ATfinder checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory with dataset.csv, ESM cache, and splits.json (default: ./data).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=None,
                        help="Protocol JSON used during training; auto-detected for seeded runs.")
    parser.add_argument(
        "--split", choices=("independent", "cv_test"), default="independent"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="CUDA falls back to CPU when unavailable.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    import numpy as np
    import torch

    from ours.data import PeptideDataset, load_multilabel_data, load_protocol
    from ours.features import build_phys_feature_table
    from ours.trainer import (
        _input_fingerprints, _make_loader, build_model, load_esm_cache,
        run_inference, summarize_predictions,
    )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" not in payload:
        raise ValueError("checkpoint is missing its training configuration")
    split_file = args.splits
    if split_file is None and payload["config"].get("paths", {}).get("split_file"):
        saved_path = Path(payload["config"]["paths"]["split_file"])
        local_path = args.checkpoint.resolve().parent.parent / "splits.json"
        split_file = local_path if local_path.is_file() else saved_path
    cfg = atfinder_config(
        data_dir=args.data_dir, device=args.device, split_file=split_file,
        snapshot=payload["config"],
    )
    if payload.get("input_sha256") != _input_fingerprints(cfg):
        raise ValueError(
            "checkpoint dataset, splits, ESM cache, or physicochemical encoding "
            "do not match the evaluation inputs"
        )
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    sequences, targets, _ = load_multilabel_data(cfg)
    splits, independent, _ = load_protocol(cfg, len(sequences))
    fold_number = int(payload.get("fold", 1))
    if not 1 <= fold_number <= len(splits):
        raise ValueError(f"checkpoint fold {fold_number} is not present in splits.json")

    esm_cache = load_esm_cache(cfg, sequences)
    phys_table = build_phys_feature_table(device, cfg.paths.aaindex)
    model = build_model(cfg, esm_cache, phys_table).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    if args.split == "independent":
        indices = independent
    else:
        indices = splits[fold_number - 1][1]
    mask = np.ones((len(indices), len(cfg.model.labels)), dtype=np.float32)
    dataset = PeptideDataset(sequences, targets, mask, indices)
    loader = _make_loader(cfg, dataset, shuffle=False, seed=cfg.train.seed)
    predictions = run_inference(model, loader, device, graph_enabled=True)
    summary = summarize_predictions(cfg, predictions)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "fold": fold_number,
        "split": args.split,
        "n_samples": int(len(indices)),
        "metrics": summary,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
