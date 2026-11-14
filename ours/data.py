from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .residues import validate_sequences


def load_multilabel_data(cfg: ExperimentConfig):
    path = cfg.paths.dataset
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    frame = pd.read_csv(path)
    required = ["global_idx", "sequence", *cfg.model.labels]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    if frame["sequence"].isna().any():
        raise ValueError("dataset contains missing sequences")
    sequences = frame["sequence"].astype(str).tolist()
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("dataset contains no rows or an empty sequence")
    validate_sequences(sequences)
    if len(set(sequences)) != len(sequences):
        raise ValueError("dataset contains duplicate sequences; resolve them without silently merging labels")
    targets = frame[list(cfg.model.labels)].to_numpy(dtype=np.float32)
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("labels must be 1 (observed positive) or 0 (unlabeled)")
    indices = frame["global_idx"].to_numpy()
    if (not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, np.arange(len(indices)))):
        raise ValueError("dataset global_idx must be 0..n-1 in row order")
    stats = {
        label: {
            "positive_count": int((targets[:, index] > 0.5).sum()),
            "unlabeled_count": int((targets[:, index] <= 0.5).sum()),
        }
        for index, label in enumerate(cfg.model.labels)
    }
    return sequences, targets, stats


def make_seeded_protocol(
    sample_count: int,
    labels: tuple[str, ...],
    seed: int,
    independent_fraction: float = 0.10,
) -> dict:
    """Build an unstratified 90/10 outer split and five disjoint DEV folds.

    The published seed-42 protocol is retained in data/splits.json. This
    function reproduces the *stated random-split procedure*, not necessarily
    the historical sample assignment underlying Appendix Table S6.
    """
    if sample_count < 10:
        raise ValueError("at least ten sequences are needed for a five-fold outer protocol")
    if not 0.0 < independent_fraction < 1.0:
        raise ValueError("independent_fraction must be in (0,1)")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("split seed must be a nonnegative integer")
    if not labels:
        raise ValueError("label names must not be empty")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    n_independent = max(1, min(sample_count - 5, int(round(sample_count * independent_fraction))))
    independent = np.sort(indices[:n_independent])
    development = rng.permutation(indices[n_independent:])
    folds = []
    for number, held_out in enumerate(np.array_split(development, 5), start=1):
        train = np.setdiff1d(development, held_out, assume_unique=True)
        folds.append({
            "fold": number,
            "train": train.astype(int).tolist(),
            "test": np.sort(held_out).astype(int).tolist(),
        })
    return {
        "seed": seed,
        "n_samples": sample_count,
        "labels": list(labels),
        "independent": independent.astype(int).tolist(),
        "folds": folds,
        "generation": {
            "method": "numpy.default_rng random permutation (not stratified)",
            "independent_fraction": independent_fraction,
            "independent_size": n_independent,
            "development_folds": 5,
        },
    }


def save_seeded_protocol(payload: dict, path: Path) -> None:
    """Persist an exact split for checkpoint evaluation; never overwrite silently."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise ValueError(
                f"existing split file differs from the requested seed/protocol: {path}; "
                "choose a different output directory"
            )
        return
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_protocol(cfg: ExperimentConfig, sample_count: int):
    if not cfg.paths.splits.is_file():
        raise FileNotFoundError(f"split file not found: {cfg.paths.splits}")
    payload = json.loads(cfg.paths.splits.read_text(encoding="utf-8"))
    if payload.get("n_samples") != sample_count:
        raise ValueError("split sample count does not match dataset")
    if tuple(payload.get("labels", ())) != tuple(cfg.model.labels):
        raise ValueError("split labels do not match model label order")
    if not isinstance(payload.get("folds"), list) or len(payload["folds"]) != 5:
        raise ValueError("splits.json must contain exactly five development folds")

    def checked_indices(values, name):
        indices = np.asarray(values)
        if (indices.ndim != 1 or indices.size == 0
                or not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError(f"{name} must contain integer indices")
        if indices.min() < 0 or indices.max() >= sample_count:
            raise ValueError(f"out-of-range index in {name}")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"duplicate indices in {name}")
        return indices.astype(np.int64)

    independent = checked_indices(payload["independent"], "independent split")
    independent_set = set(independent)
    development_set = set(range(sample_count)) - independent_set
    held_out_seen = set()
    folds = []
    for fold_number, fold in enumerate(payload["folds"], start=1):
        if type(fold.get("fold")) is not int or fold["fold"] != fold_number:
            raise ValueError("folds must be ordered and numbered from 1")
        train = checked_indices(fold["train"], f"fold {fold_number} train")
        test = checked_indices(fold["test"], f"fold {fold_number} test")
        train_set, test_set = set(train), set(test)
        if train_set & test_set or (train_set | test_set) & independent_set:
            raise ValueError(f"overlapping splits in fold {fold_number}")
        if train_set | test_set != development_set:
            raise ValueError(f"fold {fold_number} does not cover the full development set")
        if held_out_seen & test_set:
            raise ValueError("development samples appear in more than one held-out fold")
        held_out_seen.update(test_set)
        # All unobserved label entries participate in the PU objective.
        train_mask = np.ones((train.size, len(cfg.model.labels)), dtype=np.float32)
        test_mask = np.ones((test.size, len(cfg.model.labels)), dtype=np.float32)
        folds.append((train, test, train_mask, test_mask))
    if held_out_seen != development_set:
        raise ValueError("held-out folds must cover every development sample exactly once")
    independent_mask = np.ones(
        (independent.size, len(cfg.model.labels)), dtype=np.float32
    )
    return folds[: cfg.train.folds], independent, independent_mask


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
