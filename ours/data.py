from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig


def load_multilabel_data(cfg: ExperimentConfig):
    path = cfg.paths.dataset
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    frame = pd.read_csv(path)
    required = ["global_idx", "sequence", *cfg.model.labels]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    sequences = frame["sequence"].astype(str).str.strip().str.upper().tolist()
    targets = frame[list(cfg.model.labels)].to_numpy(dtype=np.float32)
    indices = frame["global_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("dataset global_idx must be 0..n-1 in row order")
    stats = {
        label: {
            "positive_count": int((targets[:, index] > 0.5).sum()),
            "unlabeled_count": int((targets[:, index] <= 0.5).sum()),
        }
        for index, label in enumerate(cfg.model.labels)
    }
    return sequences, targets, stats


def _load_mask(path: Path, labels: tuple[str, ...], expected: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if "global_idx" not in frame.columns:
        raise ValueError(f"mask lacks global_idx: {path}")
    actual = frame["global_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, np.asarray(expected, dtype=np.int64)):
        raise ValueError(f"mask index mismatch: {path}")
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(f"mask {path} is missing labels: {missing}")
    mask = frame[list(labels)].to_numpy(dtype=np.float32)
    if mask.shape != (len(expected), len(labels)):
        raise ValueError(f"unexpected mask shape for {path}: {mask.shape}")
    return mask


def load_protocol(cfg: ExperimentConfig, sample_count: int):
    payload = json.loads(cfg.paths.splits.read_text(encoding="utf-8"))
    folds = []
    for fold in payload["folds"][: cfg.train.folds]:
        fold_number = int(fold["fold"])
        train = np.asarray(fold["train"], dtype=np.int64)
        test = np.asarray(fold["test"], dtype=np.int64)
        if train.size == 0 or test.size == 0 or set(train) & set(test):
            raise ValueError(f"invalid split in fold {fold_number}")
        if train.min() < 0 or test.max() >= sample_count:
            raise ValueError(f"out-of-range split index in fold {fold_number}")
        train_mask = _load_mask(
            cfg.paths.masks / f"fold_{fold_number}_train.csv",
            cfg.model.labels,
            train,
        )
        test_mask = _load_mask(
            cfg.paths.masks / f"fold_{fold_number}_test.csv",
            cfg.model.labels,
            test,
        )
        folds.append((train, test, train_mask, test_mask))
    independent = np.asarray(payload["independent"], dtype=np.int64)
    independent_mask = _load_mask(
        cfg.paths.masks / "independent.csv",
        cfg.model.labels,
        independent,
    )
    return folds, independent, independent_mask


class PeptideDataset(Dataset):
    def __init__(self, sequences, targets, masks, indices):
        indices = np.asarray(indices, dtype=np.int64)
        self.sequences = [sequences[int(index)] for index in indices]
        self.targets = torch.as_tensor(targets[indices], dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.float32)
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        if self.masks.shape != self.targets.shape:
            raise ValueError("mask and target shapes differ")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return {
            "sequence": self.sequences[index],
            "Y": self.targets[index],
            "Mask": self.masks[index],
            "global_idx": self.indices[index],
        }
