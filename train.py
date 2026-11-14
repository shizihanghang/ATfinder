from __future__ import annotations

import argparse
from pathlib import Path

from ours.config import atfinder_config

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train ATfinder using the project dataset.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory with dataset.csv and ESM cache (default: ./data).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: ./runs, or ./runs/seedN in seeded mode).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="CUDA falls back to CPU when unavailable.")
    parser.add_argument("--folds", type=int, default=5, choices=(1, 5))
    parser.add_argument("--seed", type=int, default=42,
                        help="Training RNG seed; also used for split generation if --split-seed is omitted.")
    parser.add_argument("--split-mode", choices=("published", "seeded"), default="published",
                        help="Use the original fixed splits.json or generate a new 90/10 outer split and five folds.")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Split RNG seed, only for --split-mode seeded (default: --seed).")
    parser.add_argument("--splits", type=Path, default=None,
                        help="Optional existing splits.json path (only for --split-mode published).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--graph-dropout", type=float, default=0.3)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split_mode == "seeded" and args.splits is not None:
        raise SystemExit("--splits cannot be used with --split-mode seeded")
    if args.split_mode == "published" and args.split_seed is not None:
        raise SystemExit("--split-seed requires --split-mode seeded")

    from ours.trainer import run_experiment
    from ours.data import load_multilabel_data, make_seeded_protocol, save_seeded_protocol

    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.split_mode == "seeded":
        default_output = ROOT / "runs" / f"seed{split_seed}_train{args.seed}"
        output = args.output if args.output is not None else default_output
        split_file = output / "splits.json"
        data_cfg = atfinder_config(data_dir=args.data_dir)
        sequences, _, _ = load_multilabel_data(data_cfg)
        payload = make_seeded_protocol(len(sequences), data_cfg.model.labels, split_seed)
        save_seeded_protocol(payload, split_file)
        print(
            f"[ATfinder] seeded outer split seed={split_seed} "
            f"independent={len(payload['independent'])} "
            f"development={len(sequences)-len(payload['independent'])} "
            f"saved={split_file}",
            flush=True,
        )
    else:
        output = args.output if args.output is not None else ROOT / "runs"
        split_file = args.splits
        print(f"[ATfinder] fixed split={split_file or args.data_dir / 'splits.json'}", flush=True)

    cfg = atfinder_config(
        data_dir=args.data_dir,
        result_root=output,
        split_file=split_file,
        device=args.device,
        folds=args.folds,
        seed=args.seed,
        epochs=args.epochs,
        graph_dropout=args.graph_dropout,
        prior_scale=args.prior_scale,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
