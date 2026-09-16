from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ours.config import atfinder_config
from ours.trainer import run_experiment

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "atfinder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, default=5, choices=(1, 5))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = atfinder_config(result_root=args.output, device=args.device, folds=args.folds)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
