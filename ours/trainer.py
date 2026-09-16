from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import PeptideDataset, load_multilabel_data, load_protocol
from .features import build_phys_feature_table
from .loss import (
    BalancedObservedPositiveLoss,
    DeficitAwarePositiveTopKLoss,
    MultiLabelNonNegativePULoss,
)
from .metrics import (
    aggregate_fold_metrics,
    aggregate_observed_ranking_metrics,
    evaluate_masked_multilabel,
    evaluate_observed_ranking_metrics,
    evaluate_positive_only,
)
from .model import ATfinder


@dataclass(frozen=True)
class TrainingStage:
    name: str
    graph_enabled: bool
    graph_scale: float


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
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def graph_scale_for_epoch(epoch: int, cfg: ExperimentConfig) -> float:
    if epoch < cfg.train.graph_start_epoch:
        return 0.0
    return min(1.0, (epoch - cfg.train.graph_start_epoch + 1) / cfg.train.graph_ramp_epochs)


def stage_for_epoch(epoch: int, cfg: ExperimentConfig) -> TrainingStage:
    scale = graph_scale_for_epoch(epoch, cfg)
    if scale == 0.0:
        return TrainingStage('base_warmup', False, 0.0)
    if scale < 1.0:
        return TrainingStage('graph_ramp', True, scale)
    return TrainingStage('full_model', True, 1.0)


def ranking_weight_for_epoch(epoch: int, cfg: ExperimentConfig) -> float:
    if cfg.loss.topk_weight == 0.0 or epoch < cfg.loss.topk_start_epoch:
        return 0.0
    progress = min(
        1.0,
        (epoch - cfg.loss.topk_start_epoch + 1) / cfg.loss.topk_ramp_epochs,
    )
    return cfg.loss.topk_weight * progress


def positive_weight_for_epoch(epoch: int, cfg: ExperimentConfig) -> float:
    if cfg.loss.ppg_positive_weight == 0.0 or epoch < cfg.loss.ppg_positive_start_epoch:
        return 0.0
    progress = min(
        1.0,
        (epoch - cfg.loss.ppg_positive_start_epoch + 1)
        / cfg.loss.ppg_positive_ramp_epochs,
    )
    return cfg.loss.ppg_positive_weight * progress


def build_model(cfg: ExperimentConfig, esm_cache: torch.Tensor, phys_table: torch.Tensor) -> ATfinder:
    model = cfg.model
    return ATfinder(
        esm_cache=esm_cache,
        phys_table=phys_table,
        n_labels=len(model.labels),
        phys_input_dim=model.phys_dim,
        cnn_channels=model.cnn_channels,
        cnn_kernels=model.cnn_kernels,
        cnn_output_dim=model.cnn_output_dim,
        model_dim=model.model_dim,
        graph_dim=model.graph_dim,
        graph_topk=model.graph_topk,
        graph_lambda=model.graph_lambda,
        base_dropout=model.base_dropout,
        graph_dropout=model.graph_dropout,
        graph_match_temperature=model.graph_match_temperature,
        graph_residual_gamma=model.graph_residual_gamma,
    )


def _sequence_fingerprint(sequences: List[str]) -> str:
    payload = "\n".join(str(item).strip().upper() for item in sequences).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_esm_cache(cfg: ExperimentConfig, sequences: List[str]) -> torch.Tensor:

    payload = torch.load(cfg.paths.esm_cache, map_location="cpu", weights_only=False)
    cache = payload.get("embeddings") if isinstance(payload, dict) else payload
    if not torch.is_tensor(cache) or cache.shape != (len(sequences), cfg.model.esm_dim):
        raise ValueError(f"invalid ESM cache shape: {getattr(cache, 'shape', None)}")
    if isinstance(payload, dict):
        expected = _sequence_fingerprint(sequences)
        actual = payload.get("sequence_sha256")
        if actual is not None and actual != expected:
            raise ValueError("ESM cache sequence fingerprint does not match dataset")
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
    features, targets, masks = [], [], []
    for batch in loader:
        output = model(
            batch["sequence"],
            batch["global_idx"].to(device),
            graph_enabled=False,
            graph_scale=0.0,
        )
        features.append(output["z0"].detach())
        targets.append(batch["Y"].to(device))
        masks.append(batch["Mask"].to(device))
    return model.core.refresh_prototypes(
        torch.cat(features), torch.cat(targets), torch.cat(masks)
    )


@torch.no_grad()
def run_inference(model: ATfinder, loader: DataLoader, device: torch.device, graph_enabled: bool, graph_scale: float = 1.0) -> Dict[str, np.ndarray]:
    model.eval()
    storage = {key: [] for key in ('global_idx', 'targets', 'mask', 'final', 'base', 'graph', 'base_logits', 'graph_logits', 'z0', 'graph_available', 'effective_graph_lambda')}
    for batch in loader:
        output = model(
            batch['sequence'],
            batch['global_idx'].to(device),
            graph_enabled=graph_enabled,
            graph_scale=graph_scale if graph_enabled else 0.0,
        )
        storage['global_idx'].append(batch['global_idx'].cpu())
        storage['targets'].append(batch['Y'].cpu())
        storage['mask'].append(batch['Mask'].cpu())
        storage['final'].append(torch.sigmoid(output['final_logits']).cpu())
        storage['base'].append(torch.sigmoid(output['base_logits']).cpu())
        storage['graph'].append(torch.sigmoid(output['graph_logits']).cpu())
        storage['base_logits'].append(output['base_logits'].cpu())
        storage['graph_logits'].append(output['graph_logits'].cpu())
        storage['z0'].append(output['z0'].cpu())
        batch_size = output['final_logits'].shape[0]
        storage['graph_available'].append(output['graph_available'].reshape(1).expand(batch_size).cpu())
        storage['effective_graph_lambda'].append(output['effective_graph_lambda'].reshape(1, 1).expand(batch_size, 1).cpu())
    stacked = {}
    for key, values in storage.items():
        array = torch.cat(values, dim=0).numpy()
        stacked[key] = array.astype(np.int64) if key == 'global_idx' else array
    return stacked


def summarize_predictions(cfg: ExperimentConfig, predictions: Dict[str, np.ndarray]) -> Dict:
    targets = predictions["targets"]
    final = predictions["final"]
    decision = predictions.get("decision", final)
    mask = predictions["mask"]
    positive = evaluate_positive_only(
        targets, decision, cfg.model.labels, cfg.train.threshold
    )
    one_to_one = evaluate_masked_multilabel(
        targets, decision, mask, cfg.model.labels, cfg.train.threshold
    )
    base_positive = evaluate_positive_only(
        targets, predictions["base"], cfg.model.labels, cfg.train.threshold
    )
    base_one = evaluate_masked_multilabel(
        targets, predictions["base"], mask, cfg.model.labels, cfg.train.threshold
    )
    graph_available = bool(np.asarray(predictions["graph_available"]).any())
    graph_positive = graph_one = None
    if graph_available:
        graph_positive = evaluate_positive_only(
            targets, predictions["graph"], cfg.model.labels, cfg.train.threshold
        )
        graph_one = evaluate_masked_multilabel(
            targets, predictions["graph"], mask, cfg.model.labels, cfg.train.threshold
        )
    base_norm = np.linalg.norm(predictions["base_logits"], axis=1)
    graph_norm = np.linalg.norm(predictions["graph_logits"], axis=1)
    return {
        "positive_only": positive,
        "one_to_one": one_to_one,
        "observed_ranking": evaluate_observed_ranking_metrics(
            targets, final, cfg.model.labels, (1, 2, 3, 5)
        ),
        "branches": {
            "base_positive_only": base_positive,
            "base_1to1": base_one,
            "graph_positive_only": graph_positive,
            "graph_1to1": graph_one,
        },
        "diagnostics": {
            "graph_available": graph_available,
            "effective_graph_lambda": float(predictions["effective_graph_lambda"].mean()),
            "base_logit_l2_mean": float(base_norm.mean()),
            "graph_logit_l2_mean": float(graph_norm.mean()),
            "final_prediction_positive_rate": one_to_one["prediction_positive_rate"],
            "base_prediction_positive_rate": base_one["prediction_positive_rate"],
            "graph_prediction_positive_rate": (
                graph_one["prediction_positive_rate"] if graph_one else None
            ),
        },
    }


def tail_recall_from_summary(summary: Dict, tail_indices: np.ndarray) -> float:
    per_label = summary["one_to_one"].get("per_label", [])
    recalls = [
        float(per_label[int(index)]["recall"])
        for index in tail_indices
        if int(index) < len(per_label)
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def macro_recall_from_summary(summary: Dict) -> float:
    per_label = summary["one_to_one"].get("per_label", [])
    recalls = [float(item["recall"]) for item in per_label]
    return float(np.mean(recalls)) if recalls else 0.0


def selection_key(summary: Dict, tail_indices: np.ndarray, cfg: ExperimentConfig) -> tuple[float]:
    ranking = summary['observed_ranking']
    f1 = ranking['f1_at_k']
    score = (
        float(cfg.train.selection_f1_3_weight) * float(f1['3'])
        + float(cfg.train.selection_f1_5_weight) * float(f1['5'])
        + float(cfg.train.selection_recall_weight) * macro_recall_from_summary(summary)
    )
    return (score,)


def inference_options(cfg: ExperimentConfig, stage: TrainingStage) -> Dict:
    return {'graph_enabled': stage.graph_enabled, 'graph_scale': stage.graph_scale}


def _classification_objective(cfg, logits, targets, mask, priors, pu_loss):
    classification, stats = pu_loss(logits, targets, mask, priors)
    details = {'objective_final': float(classification.detach()), **stats}
    return classification, details


def _objective(cfg, outputs, targets, mask, priors, pu_loss, topk_loss, positive_loss, topk_weight):
    classification, details = _classification_objective(
        cfg, outputs['final_logits'], targets, mask, priors, pu_loss
    )
    ranking, ranking_stats = topk_loss(outputs['final_logits'], targets)
    positive_auxiliary = classification.new_zeros(())
    positive_weight = positive_weight_for_epoch(int(outputs.get('current_epoch', 1)), cfg)
    graph_auxiliary = outputs.get('graph_aux_logits')
    if graph_auxiliary is not None and positive_weight > 0.0:
        positive_auxiliary, positive_details = positive_loss(
            graph_auxiliary, targets, mask, None, 0.0, None
        )
        details.update({
            'ppg_positive_risk': float(positive_auxiliary.detach()),
            'ppg_positive_valid_labels': positive_details['valid_labels'],
            'ppg_positive_positions': positive_details['positive_positions'],
        })
    total = classification + float(topk_weight) * ranking + float(positive_weight) * positive_auxiliary
    details.update({
        'total': float(total.detach()),
        'multi_k_boundary_loss': float(ranking.detach()),
        'multi_k_boundary_weight': float(topk_weight),
        'ppg_positive_weight': float(positive_weight),
        'multi_k_boundary_valid_samples': ranking_stats['valid_samples'],
        'multi_k_boundary_active_pairs': ranking_stats['active_pairs'],
    })
    for k, value in ranking_stats['active_pairs_by_k'].items():
        details[f'multi_k_boundary_active_pairs_at_{k}'] = value
    return total, details


def _development_objective(cfg, predictions, priors, pu_loss, device):
    probabilities = torch.as_tensor(predictions['final'], device=device)
    epsilon = torch.finfo(probabilities.dtype).eps
    logits = torch.logit(probabilities.clamp(epsilon, 1.0 - epsilon))
    targets = torch.as_tensor(predictions['targets'], device=device)
    mask = torch.as_tensor(predictions['mask'], device=device)
    _, details = _classification_objective(cfg, logits, targets, mask, priors, pu_loss)
    return {f'development_{key}': float(value) for key, value in details.items()}


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
    train_indices, test_indices, train_mask, test_mask = split
    train_dataset = PeptideDataset(sequences, targets, train_mask, train_indices)
    test_dataset = PeptideDataset(sequences, targets, test_mask, test_indices)
    independent_dataset = PeptideDataset(
        sequences, targets, independent_mask, independent_indices
    )
    train_loader = _make_loader(cfg, train_dataset, True, cfg.train.seed)
    prototype_loader = _make_loader(cfg, train_dataset, False, cfg.train.seed)
    test_loader = _make_loader(cfg, test_dataset, False, cfg.train.seed)
    independent_loader = _make_loader(cfg, independent_dataset, False, cfg.train.seed)

    model = build_model(cfg, esm_cache, phys_table).to(device)
    initial_parameters = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }
    gradient_maxima = {name: 0.0 for name, _ in model.named_parameters()}
    pu_loss = MultiLabelNonNegativePULoss(
        cfg.loss.pu_prior_min,
        cfg.loss.pu_prior_max,
        cfg.loss.pu_correction_beta,
        cfg.loss.pu_correction_gamma,
    )
    topk_loss = DeficitAwarePositiveTopKLoss(
        cfg.loss.topk_values, cfg.loss.topk_weights, cfg.loss.topk_margin
    )
    positive_loss = BalancedObservedPositiveLoss()
    train_targets_tensor = torch.as_tensor(targets[train_indices], dtype=torch.float32)
    priors = MultiLabelNonNegativePULoss.estimate_priors(
        train_targets_tensor,
        cfg.loss.pu_prior_min,
        cfg.loss.pu_prior_max,
    ).to(device)
    tail_indices = np.argsort(targets[train_indices].sum(axis=0), kind="stable")[:4]

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.min_learning_rate
    )

    history = []
    prototype_history = []
    if cfg.train.initialize_prototypes_before_training:
        initial_prototype_stats = refresh_prototypes(model, prototype_loader, device)
        initial_prototype_stats["epoch"] = 0
        prototype_history.append(initial_prototype_stats)
    best_state = best_selection = best_key = None
    started = time.perf_counter()
    for epoch in range(1, cfg.train.epochs + 1):
        stage = stage_for_epoch(epoch, cfg)
        topk_weight = ranking_weight_for_epoch(epoch, cfg)
        model.train()
        sums: Dict[str, float] = {"batches": 0.0}
        for batch in train_loader:
            y = batch["Y"].to(device)
            mask = batch["Mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["sequence"],
                batch["global_idx"].to(device),
                **inference_options(cfg, stage),
            )
            output["current_epoch"] = epoch
            total, parts = _objective(
                cfg,
                output,
                y,
                mask,
                priors,
                pu_loss,
                topk_loss,
                positive_loss,
                topk_weight,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss: fold={fold_number} epoch={epoch}")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient_maxima[name] = max(
                        gradient_maxima[name], float(parameter.grad.detach().abs().max())
                    )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.gradient_clip_norm
            )
            optimizer.step()
            parts.update(
                {
                    "effective_graph_lambda": float(output["effective_graph_lambda"].detach()),
                }
            )
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            sums["batches"] += 1.0
        scheduler.step()

        prototype_stats = refresh_prototypes(model, prototype_loader, device)
        model.core.graph_branch.graph_structure_ready.fill_(
            epoch >= cfg.train.graph_start_epoch
        )
        prototype_stats["epoch"] = epoch
        prototype_history.append(prototype_stats)
        denominator = max(sums["batches"], 1.0)
        epoch_row = {
            "epoch": epoch,
            "stage": stage.name,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "graph_scale": stage.graph_scale,
            **{key: value / denominator for key, value in sums.items() if key != "batches"},
        }

        validation = run_inference(
            model,
            test_loader,
            device,
            **inference_options(cfg, stage),
        )
        validation_summary = summarize_predictions(cfg, validation)
        ranking = validation_summary["observed_ranking"]
        candidate_key = selection_key(validation_summary, tail_indices, cfg)
        epoch_row.update(
            _development_objective(
                cfg, validation, priors, pu_loss, device
            )
        )
        epoch_row.update(
            {
                "selection_f1_at_1": ranking["f1_at_k"]["1"],
                "selection_f1_at_2": ranking["f1_at_k"]["2"],
                "selection_f1_at_3": ranking["f1_at_k"]["3"],
                "selection_f1_at_5": ranking["f1_at_k"]["5"],
                "selection_score": candidate_key[0],
                "selection_tail_recall": tail_recall_from_summary(
                    validation_summary, tail_indices
                ),
                "selection_ap_samples": ranking["ap_samples"],
                "selection_map_labels": ranking["map_labels"],
            }
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_selection = {
                "epoch": epoch, "observed_ranking": ranking,
                "metric": cfg.snapshot()["checkpoint_selection"],
                "f1_3_weight": cfg.train.selection_f1_3_weight,
                "f1_5_weight": cfg.train.selection_f1_5_weight,
                "recall_weight": cfg.train.selection_recall_weight,
                "tie_break": "earliest epoch",
                "inference_options": inference_options(cfg, stage),
            }
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        history.append(epoch_row)
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.train.graph_start_epoch:
            print(
                f"[Fold {fold_number}] epoch={epoch:03d} stage={stage.name:<11} "
                f"loss={epoch_row['total']:.5f} F1@1/3/5="
                f"{ranking['f1_at_k']['1']:.4f}/{ranking['f1_at_k']['3']:.4f}/"
                f"{ranking['f1_at_k']['5']:.4f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("checkpoint selection produced no model")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    test_predictions = run_inference(
        model,
        test_loader,
        device,
        **best_selection["inference_options"],
    )
    independent_predictions = run_inference(
        model,
        independent_loader,
        device,
        **best_selection["inference_options"],
    ) if cfg.train.evaluate_independent else None
    test_summary = summarize_predictions(cfg, test_predictions)
    if selection_key(test_summary, tail_indices, cfg) != best_key:
        raise AssertionError("selected CV score does not replay")
    independent_summary = (
        summarize_predictions(cfg, independent_predictions)
        if independent_predictions is not None
        else None
    )

    fold_dir = cfg.result_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _save_json(fold_dir / "parameter_update_audit.json", {
        name: {
            "requires_grad": parameter.requires_grad,
            "max_gradient_during_training": gradient_maxima[name],
            "selected_checkpoint_max_delta": float(
                (parameter.detach().cpu() - initial_parameters[name]).abs().max()
            ),
        } for name, parameter in model.named_parameters()
    })
    _save_npz(fold_dir / "test_predictions.npz", **test_predictions)
    if independent_predictions is not None:
        _save_npz(fold_dir / "independent_predictions.npz", **independent_predictions)
    graph = model.core.graph_branch
    prototypes = graph.prototypes.detach().cpu().numpy()
    _save_npz(
        fold_dir / "graph_state.npz",
        prototypes=prototypes,
        prototype_cosine=prototypes @ prototypes.T,
        adjacency=graph.adjacency.detach().cpu().numpy(),
        positive_counts=graph.positive_counts.detach().cpu().numpy(),
        graph_available=np.asarray(bool(graph.prototypes_ready.item())),
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "pu_priors": priors.detach().cpu(),
        "config": cfg.snapshot(),
        "fold": fold_number,
        "selection": best_selection,
    }
    torch.save(checkpoint, fold_dir / "final_model.pt")
    _save_csv(fold_dir / "training_history.csv", history)
    result = {
        "fold": fold_number,
        "seed": cfg.train.seed,
        "test": test_summary,
        "independent_validation": independent_summary,
        "training_history": history,
        "prototype_history": prototype_history,
        "efficiency": {"training_seconds": training_seconds},
        "artifacts": {
            "test_predictions": str(fold_dir / "test_predictions.npz"),
            "independent_predictions": (
                str(fold_dir / "independent_predictions.npz")
                if independent_predictions is not None
                else None
            ),
            "graph_state": str(fold_dir / "graph_state.npz"),
            "selected_model": str(fold_dir / "final_model.pt"),
        },
    }
    _save_json(fold_dir / "fold_summary.json", result)
    return result


def _combine_oof(folds: List[Dict], output_dir: Path) -> Path:
    bundles = [np.load(fold["artifacts"]["test_predictions"]) for fold in folds]
    common = set(bundles[0].files)
    for bundle in bundles[1:]:
        common.intersection_update(bundle.files)
    combined = {
        key: np.concatenate([bundle[key] for bundle in bundles]) for key in sorted(common)
    }
    combined["fold"] = np.concatenate(
        [
            np.full(bundle["global_idx"].shape[0], index + 1, dtype=np.int16)
            for index, bundle in enumerate(bundles)
        ]
    )
    order = np.argsort(combined["global_idx"], kind="stable")
    for key, value in combined.items():
        if value.ndim and value.shape[0] == len(order):
            combined[key] = value[order]
    path = output_dir / "oof_predictions.npz"
    _save_npz(path, **combined)
    return path


def _aggregate(folds: List[Dict]) -> Dict:
    def rows(section: str, key: str):
        return [fold[section][key] for fold in folds if fold.get(section)]

    result = {
        "cv_positive_only": aggregate_fold_metrics(rows("test", "positive_only")),
        "cv_one_to_one": aggregate_fold_metrics(rows("test", "one_to_one")),
        "cv_observed_ranking": aggregate_observed_ranking_metrics(
            rows("test", "observed_ranking")
        ),
    }
    if any(fold.get("independent_validation") for fold in folds):
        result.update(
            {
                "independent_positive_only": aggregate_fold_metrics(
                    rows("independent_validation", "positive_only")
                ),
                "independent_one_to_one": aggregate_fold_metrics(
                    rows("independent_validation", "one_to_one")
                ),
                "independent_observed_ranking": aggregate_observed_ranking_metrics(
                    rows("independent_validation", "observed_ranking")
                ),
            }
        )
    return result


def run_experiment(cfg: ExperimentConfig) -> Dict:

    cfg.validate()
    set_seed(cfg.train.seed)
    device = _device(cfg)
    sequences, targets, data_stats = load_multilabel_data(cfg)
    splits, independent, independent_mask = load_protocol(cfg, len(sequences))
    esm_cache = load_esm_cache(cfg, sequences)
    phys_table = build_phys_feature_table(device, cfg.paths.aaindex)
    print(f"[ATfinder] device={device} result={cfg.result_dir}")
    print(
        f"[Audit] dataset={_sha256(cfg.paths.dataset)} split={_sha256(cfg.paths.splits)}"
    )

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
    oof_path = _combine_oof(folds, cfg.result_dir)
    result = {
        "experiment_name": cfg.name,
        "config": cfg.snapshot(),
        "protocol": {
            "dataset_samples": len(sequences),
            "independent_samples": len(independent),
            "folds": cfg.train.folds,
        },
        "data_stats": data_stats,
        "fold_results": folds,
        "aggregate": _aggregate(folds),
        "artifacts": {"oof_predictions": str(oof_path)},
    }
    _save_json(cfg.result_dir / "results.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2), flush=True)
    return result
