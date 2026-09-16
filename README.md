# ATfinder

Frozen ATfinder model and dataset for five-fold Independent ACP multi-label ranking.

This package contains only the model code, the prepared dataset, and the frozen five-fold checkpoints.

## Contents

- `ours/`: ATfinder model, losses, metrics, and trainer
- `train.py`: single entry point for the frozen configuration
- `data/`: sequences, labels, splits, loss masks, ESM-2 cache, and AAindex table
- `models/fold_1` to `models/fold_5`: frozen checkpoints

## Frozen configuration

- 12 cancer-type labels
- random-initialized downstream network and fixed pretrained ESM-2 cache
- ESM + physicochemical CNN with interaction fusion
- undirected weighted top-3 label co-occurrence GCN
- nnPU + Deficit-TopK + PPG observed-positive auxiliary loss (weight 0.1)
- checkpoint selection: 0.3 * CV F1@3 + 0.3 * CV F1@5 + 0.4 * CV Macro Recall@0.5
- Independent evaluation after CV selection

## Install

    pip install -r requirements.txt

## Train

    python train.py --output runs/atfinder --device cuda

## Data

See `data/README.md`.
