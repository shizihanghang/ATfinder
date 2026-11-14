from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import PHYS_ENCODING_SHA256, ExperimentConfig
from .data import PeptideDataset, load_multilabel_data, load_protocol
from .features import build_phys_feature_table
from .loss import MultiLabelNonNegativePULoss
from .metrics import (
    aggregate_observed_ranking_metrics,
    evaluate_observed_ranking_metrics,
    evaluate_positive_only,
)
from .model import ATfinder
from .residues import ESM_INPUT_POLICY, esm_sequences


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(cfg: ExperimentConfig) -> torch.device:
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_fingerprints(cfg: ExperimentConfig) -> dict[str, str]:
    return {
        "dataset": _sha256(cfg.paths.dataset),
        "splits": _sha256(cfg.paths.splits),
        "aaindex": _sha256(cfg.paths.aaindex),
        "phys_encoding": PHYS_ENCODING_SHA256,
        "esm_cache": _sha256(cfg.paths.esm_cache),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def build_model(
    cfg: ExperimentConfig, esm_cache: torch.Tensor, phys_table: torch.Tensor
) -> ATfinder:
    model = cfg.model
    if model.padding_length is None:
        raise ValueError("resolve the dataset-wide fixed padding length before building the model")
    return ATfinder(
        esm_cache=esm_cache,
        phys_table=phys_table,
        n_labels=len(model.labels),
        phys_input_dim=model.phys_dim,
        cnn_channels=model.cnn_channels,
        cnn_kernels=model.cnn_kernels,
        cnn_output_dim=model.cnn_output_dim,
        model_dim=model.model_dim,
        graph_topk=model.graph_topk,
        graph_lambda=model.graph_lambda,
        base_dropout=model.base_dropout,
        padding_length=model.padding_length,
        graph_dropout=model.graph_dropout,
    )


def _sequence_fingerprint(sequences: List[str]) -> str:
    payload = "\n".join(sequences).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_esm_cache(cfg: ExperimentConfig, sequences: List[str]) -> torch.Tensor:
    """Load frozen mean-pooled ESM-2 features prepared before training."""
    mapped_sequences = esm_sequences(sequences)
    if [len(sequence) for sequence in mapped_sequences] != [len(sequence) for sequence in sequences]:
        raise ValueError("ESM residue mapping must preserve every sequence position")
    if not cfg.paths.esm_cache.is_file():
        raise FileNotFoundError(
            f"ESM cache not found: {cfg.paths.esm_cache}. "
            "Run extract_esm.py to generate the matching ESM cache."
        )
    payload = torch.load(cfg.paths.esm_cache, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not payload.get("sequence_sha256"):
        raise ValueError("ESM cache must include its source sequence_sha256")
    # ESM provenance depends on its token inputs, independently of phys encoding.
    for field, expected in (
        ("esm_input_policy", ESM_INPUT_POLICY),
        ("esm_sequence_sha256", _sequence_fingerprint(mapped_sequences)),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"ESM cache {field} does not match the input token policy")
    if payload.get("model_name") != "esm2_t30_150M_UR50D":
        raise ValueError("ESM cache must use esm2_t30_150M_UR50D")
    if (payload.get("representation_layer") != 30
            or payload.get("pooling") != "mean_residues_excluding_bos_eos"
            or "truncation" not in payload or payload["truncation"] is not None):
        raise ValueError("ESM cache must use full-length layer-30 residue mean pooling")
    for field, expected in (
        ("global_idx", list(range(len(sequences)))),
        ("sequence_lengths", [len(sequence) for sequence in sequences]),
    ):
        if not isinstance(payload.get(field), list) or payload[field] != expected:
            raise ValueError(f"ESM cache {field} does not match the input dataset")
    cache = payload.get("embeddings")
    if not torch.is_tensor(cache) or cache.shape != (len(sequences), cfg.model.esm_dim):
        raise ValueError(f"invalid ESM cache shape: {getattr(cache, 'shape', None)}")
    if not bool(torch.isfinite(cache).all()):
        raise ValueError("ESM cache contains non-finite embeddings")
    if payload["sequence_sha256"] != _sequence_fingerprint(sequences):
        raise ValueError("ESM cache does not match the current sequences")
    return cache.float()


def _make_loader(
    cfg: ExperimentConfig, dataset: PeptideDataset, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def refresh_prototypes(
    model: ATfinder, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    features, targets = [], []
    for batch in loader:
        output = model(
            batch["sequence"],
            batch["global_idx"].to(device),
            graph_enabled=False,
        )
        features.append(output["z0"].detach())
        targets.append(batch["Y"].to(device))
    return model.core.refresh_prototypes(torch.cat(features), torch.cat(targets), None)


@torch.no_grad()
def run_inference(
    model: ATfinder,
    loader: DataLoader,
    device: torch.device,
    graph_enabled: bool = True,
) -> Dict[str, np.ndarray]:
    model.eval()
    storage = {key: [] for key in ("global_idx", "targets", "final")}
    for batch in loader:
        output = model(
            batch["sequence"],
            batch["global_idx"].to(device),
            graph_enabled=graph_enabled,
        )
        storage["global_idx"].append(batch["global_idx"].cpu())
        storage["targets"].append(batch["Y"].cpu())
        storage["final"].append(torch.sigmoid(output["final_logits"]).cpu())
    stacked = {}
    for key, values in storage.items():
        array = torch.cat(values, dim=0).numpy()
        stacked[key] = array.astype(np.int64) if key == "global_idx" else array
    return stacked


def summarize_predictions(cfg: ExperimentConfig, predictions: Dict[str, np.ndarray]) -> Dict:
    targets = predictions["targets"]
    final = predictions["final"]
    positive = evaluate_positive_only(
        targets, final, cfg.model.labels, cfg.train.threshold
    )
    ranking = evaluate_observed_ranking_metrics(
        targets, final, cfg.model.labels, (1, 3, 5)
    )
    # Per-sample scores can be recovered from the saved predictions.
    ranking.pop("per_sample", None)
    macro_recall = float(positive["macro_positive_recall"])
    return {
        "observed_ranking": ranking,
        "positive_only": positive,
        "macro_recall_at_0_5": macro_recall,
        "per_label_recall_at_0_5": [
            float(row["recall"]) for row in positive.get("per_label", [])
        ],
    }


def selection_key(summary: Dict, cfg: ExperimentConfig) -> tuple[float]:
    ranking = summary["observed_ranking"]
    f1 = ranking["f1_at_k"]
    score = (
        cfg.train.selection_f1_3_weight * float(f1["3"])
        + cfg.train.selection_f1_5_weight * float(f1["5"])
        + cfg.train.selection_recall_weight * float(summary["macro_recall_at_0_5"])
    )
    return (float(score),)


def train_fold(
    cfg: ExperimentConfig,
    fold_index: int,
    split,
    independent_indices: np.ndarray,
    independent_mask: np.ndarray,
    sequences: List[str],
    targets: np.ndarray,
    esm_cache: torch.Tensor,
    phys_table: torch.Tensor,
    device: torch.device,
) -> Dict:
    fold_number = fold_index + 1
    set_seed(cfg.train.seed)
    train_indices, test_indices = split[0], split[1]
    all_train_mask = np.ones((len(train_indices), len(cfg.model.labels)), dtype=np.float32)
    all_test_mask = np.ones((len(test_indices), len(cfg.model.labels)), dtype=np.float32)
    all_ind_mask = np.ones((len(independent_indices), len(cfg.model.labels)), dtype=np.float32)
    train_dataset = PeptideDataset(sequences, targets, all_train_mask, train_indices)
    test_dataset = PeptideDataset(sequences, targets, all_test_mask, test_indices)
    independent_dataset = PeptideDataset(
        sequences, targets, all_ind_mask, independent_indices
    )
    train_loader = _make_loader(cfg, train_dataset, True, cfg.train.seed)
    prototype_loader = _make_loader(cfg, train_dataset, False, cfg.train.seed)
    test_loader = _make_loader(cfg, test_dataset, False, cfg.train.seed)
    independent_loader = _make_loader(cfg, independent_dataset, False, cfg.train.seed)

    model = build_model(cfg, esm_cache, phys_table).to(device)
    pu_loss = MultiLabelNonNegativePULoss(
        correction_beta=cfg.loss.pu_correction_beta,
        correction_gamma=cfg.loss.pu_correction_gamma,
    )
    train_targets_tensor = torch.as_tensor(targets[train_indices], dtype=torch.float32)
    empirical_priors = MultiLabelNonNegativePULoss.estimate_priors(train_targets_tensor)
    priors = MultiLabelNonNegativePULoss.estimate_priors(
        train_targets_tensor, prior_scale=cfg.loss.prior_scale
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    history = []
    if cfg.train.initialize_prototypes_before_training:
        refresh_prototypes(model, prototype_loader, device)

    best_state = None
    best_selection = None
    best_key = None
    started = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        sums: Dict[str, float] = {"batches": 0.0}
        for batch in train_loader:
            y = batch["Y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["sequence"],
                batch["global_idx"].to(device),
                graph_enabled=True,
            )
            total, parts = pu_loss(output["final_logits"], y, None, priors)
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite loss: fold={fold_number} epoch={epoch}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.gradient_clip_norm
            )
            optimizer.step()
            parts["loss"] = float(total.detach())
            parts["effective_lambda"] = float(output["effective_lambda"].detach())
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            sums["batches"] += 1.0

        refresh_prototypes(model, prototype_loader, device)
        denominator = max(sums["batches"], 1.0)
        validation = run_inference(model, test_loader, device, graph_enabled=True)
        validation_summary = summarize_predictions(cfg, validation)
        candidate_key = selection_key(validation_summary, cfg)
        ranking = validation_summary["observed_ranking"]
        epoch_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss": sums.get("loss", 0.0) / denominator,
            "selection_f1_at_1": ranking["f1_at_k"]["1"],
            "selection_f1_at_3": ranking["f1_at_k"]["3"],
            "selection_f1_at_5": ranking["f1_at_k"]["5"],
            "selection_macro_recall_at_0_5": validation_summary[
                "macro_recall_at_0_5"
            ],
            "selection_score": candidate_key[0],
            "effective_lambda": float(output["effective_lambda"].detach()),
        }
        epoch_row.update(
            {
                key: value / denominator
                for key, value in sums.items()
                if key not in {"batches", "loss"} and key not in epoch_row
            }
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_selection = {
                "epoch": epoch,
                "metric": cfg.snapshot()["checkpoint_selection"],
                "score": candidate_key[0],
                "f1_3_weight": cfg.train.selection_f1_3_weight,
                "f1_5_weight": cfg.train.selection_f1_5_weight,
                "recall_weight": cfg.train.selection_recall_weight,
                "tie_break": "earliest epoch",
            }
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history.append(epoch_row)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Fold {fold_number}] epoch={epoch:03d} "
                f"loss={epoch_row['loss']:.5f} "
                f"F1@1/3/5={ranking['f1_at_k']['1']:.4f}/"
                f"{ranking['f1_at_k']['3']:.4f}/{ranking['f1_at_k']['5']:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("checkpoint selection produced no model")
    model.load_state_dict(best_state, strict=True)
    training_seconds = time.perf_counter() - started
    test_predictions = run_inference(model, test_loader, device, graph_enabled=True)
    independent_predictions = (
        run_inference(model, independent_loader, device, graph_enabled=True)
        if cfg.train.evaluate_independent
        else None
    )
    test_summary = summarize_predictions(cfg, test_predictions)
    if selection_key(test_summary, cfg) != best_key:
        raise AssertionError("selected CV score does not replay")
    independent_summary = (
        summarize_predictions(cfg, independent_predictions)
        if independent_predictions is not None
        else None
    )

    fold_dir = cfg.result_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(fold_dir / "test_predictions.npz", **test_predictions)
    if independent_predictions is not None:
        _save_npz(fold_dir / "independent_predictions.npz", **independent_predictions)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "pu_priors": priors.detach().cpu(),
        "empirical_pu_priors": empirical_priors,
        "config": cfg.snapshot(),
        "fold": fold_number,
        "selection": best_selection,
        "input_sha256": _input_fingerprints(cfg),
    }
    torch.save(checkpoint, fold_dir / "final_model.pt")
    _save_csv(fold_dir / "training_history.csv", history)
    result = {
        "fold": fold_number,
        "seed": cfg.train.seed,
        "selection": best_selection,
        "test": test_summary,
        "pu_priors": {
            "scale": cfg.loss.prior_scale,
            "empirical": dict(zip(cfg.model.labels, empirical_priors.tolist())),
            "effective": dict(zip(cfg.model.labels, priors.detach().cpu().tolist())),
        },
        "independent_validation": independent_summary,
        "efficiency": {"training_seconds": training_seconds},
        "artifacts": {
            "test_predictions": str(fold_dir / "test_predictions.npz"),
            "independent_predictions": (
                str(fold_dir / "independent_predictions.npz")
                if independent_predictions is not None
                else None
            ),
            "selected_model": str(fold_dir / "final_model.pt"),
            "training_history": str(fold_dir / "training_history.csv"),
        },
    }
    _save_json(fold_dir / "fold_summary.json", result)
    return result


def _aggregate(folds: List[Dict]) -> Dict:
    def rows(section: str):
        return [fold[section]["observed_ranking"] for fold in folds if fold.get(section)]

    result = {"cv_observed_ranking": aggregate_observed_ranking_metrics(rows("test"))}
    cv_recall = [fold["test"]["macro_recall_at_0_5"] for fold in folds]
    result["cv_macro_recall_at_0_5_mean"] = float(np.mean(cv_recall))
    result["cv_macro_recall_at_0_5_std"] = float(
        np.std(cv_recall, ddof=1) if len(cv_recall) > 1 else 0.0
    )
    if any(fold.get("independent_validation") for fold in folds):
        result["independent_observed_ranking"] = aggregate_observed_ranking_metrics(
            rows("independent_validation")
        )
        ind_recall = [
            fold["independent_validation"]["macro_recall_at_0_5"] for fold in folds
        ]
        result["independent_macro_recall_at_0_5_mean"] = float(np.mean(ind_recall))
        result["independent_macro_recall_at_0_5_std"] = float(
            np.std(ind_recall, ddof=1) if len(ind_recall) > 1 else 0.0
        )
    return result


def run_experiment(cfg: ExperimentConfig) -> Dict:
    cfg.validate()
    set_seed(cfg.train.seed)
    device = _device(cfg)
    sequences, targets, data_stats = load_multilabel_data(cfg)
    cfg = replace(cfg, model=replace(cfg.model, padding_length=max(map(len, sequences))))
    cfg.validate()
    splits, independent, independent_mask = load_protocol(cfg, len(sequences))
    esm_cache = load_esm_cache(cfg, sequences)
    phys_table = build_phys_feature_table(device, cfg.paths.aaindex)
    print(f"[ATfinder] device={device} result={cfg.result_dir}")

    folds = [
        train_fold(
            cfg,
            fold_index,
            split,
            independent,
            independent_mask,
            sequences,
            targets,
            esm_cache,
            phys_table,
            device,
        )
        for fold_index, split in enumerate(splits)
    ]
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_name": cfg.name,
        "config": cfg.snapshot(),
        "protocol": {
            "dataset_samples": len(sequences),
            "independent_samples": len(independent),
            "folds": cfg.train.folds,
            "unreported_labels": "treated as unlabeled for nnPU; no sampled loss mask",
        },
        "data_stats": data_stats,
        "input_sha256": _input_fingerprints(cfg),
        "fold_results": folds,
        "aggregate": _aggregate(folds),
    }
    _save_json(cfg.result_dir / "results.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2), flush=True)
    return result
