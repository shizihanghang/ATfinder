from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(function, y_true, values, default=float("nan")):
    try:
        return float(function(y_true, values))
    except (ValueError, ZeroDivisionError):
        return default


def _nanmean(values, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float(default)


def _confusion(yt: np.ndarray, yp: np.ndarray) -> tuple[int, int, int, int]:
    yt = np.asarray(yt, dtype=bool)
    yp = np.asarray(yp, dtype=bool)
    tp = int(np.logical_and(yt, yp).sum())
    tn = int(np.logical_and(~yt, ~yp).sum())
    fp = int(np.logical_and(~yt, yp).sum())
    fn = int(np.logical_and(yt, ~yp).sum())
    return tn, fp, fn, tp


def _mcc(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0


def _ece(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(y_true.size)
    value = 0.0
    for index in range(int(n_bins)):
        left, right = edges[index], edges[index + 1]
        selected = (
            (probabilities >= left) & (probabilities < right)
            if index < int(n_bins) - 1
            else (probabilities >= left) & (probabilities <= right)
        )
        if not bool(selected.any()):
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(y_true[selected].mean())
        value += float(selected.sum()) / total * abs(confidence - accuracy)
    return float(value)


def _samplewise_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float | int]:
    jaccards = []
    exact = []
    true_counts = []
    pred_counts = []
    cardinality_errors = []
    for row in range(y_true.shape[0]):
        selected = mask[row]
        if not bool(selected.any()):
            continue
        yt = y_true[row, selected].astype(bool)
        yp = predictions[row, selected].astype(bool)
        intersection = int(np.logical_and(yt, yp).sum())
        union = int(np.logical_or(yt, yp).sum())
        jaccards.append(1.0 if union == 0 else intersection / union)
        exact.append(float(np.array_equal(yt, yp)))
        true_count = int(yt.sum())
        pred_count = int(yp.sum())
        true_counts.append(true_count)
        pred_counts.append(pred_count)
        cardinality_errors.append(abs(pred_count - true_count))
    if not jaccards:
        return {
            "valid_samples": 0,
            "samples_jaccard": 0.0,
            "subset_accuracy": 0.0,
            "mean_true_labels_per_sample": 0.0,
            "mean_predicted_labels_per_sample": 0.0,
            "label_cardinality_mae": 0.0,
        }
    return {
        "valid_samples": len(jaccards),
        "samples_jaccard": float(np.mean(jaccards)),
        "subset_accuracy": float(np.mean(exact)),
        "mean_true_labels_per_sample": float(np.mean(true_counts)),
        "mean_predicted_labels_per_sample": float(np.mean(pred_counts)),
        "label_cardinality_mae": float(np.mean(cardinality_errors)),
    }


def evaluate_masked_multilabel(
    y_true,
    probabilities,
    mask,
    label_names,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> Dict:
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float32) > 0.5
    if y_true.shape != probabilities.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, probabilities, and mask must have the same shape")
    if y_true.ndim != 2 or y_true.shape[1] != len(label_names):
        raise ValueError("multilabel arrays must have shape [samples, len(label_names)]")

    predictions = probabilities >= float(threshold)
    per_label: List[Dict] = []
    all_true, all_pred, all_prob = [], [], []
    for index, label in enumerate(label_names):
        selected = mask[:, index]
        yt = y_true[selected, index]
        yp = predictions[selected, index]
        pr = probabilities[selected, index]
        tn, fp, fn, tp = _confusion(yt, yp)
        positives = tp + fn
        negatives = tn + fp
        if yt.size == 0:
            precision = recall = f1 = 0.0
            specificity = balanced_accuracy = float("nan")
            auroc = aupr = float("nan")
            brier = float("nan")
            mcc = 0.0
        else:
            precision = float(precision_score(yt, yp, zero_division=0))
            recall = float(recall_score(yt, yp, zero_division=0))
            f1 = float(f1_score(yt, yp, zero_division=0))
            specificity = float(tn / negatives) if negatives > 0 else float("nan")
            sensitivity = float(tp / positives) if positives > 0 else float("nan")
            balanced_accuracy = (
                float((specificity + sensitivity) / 2.0)
                if np.isfinite(specificity) and np.isfinite(sensitivity)
                else float("nan")
            )
            auroc = (
                _safe_metric(roc_auc_score, yt, pr)
                if np.unique(yt).size == 2
                else float("nan")
            )
            aupr = (
                _safe_metric(average_precision_score, yt, pr)
                if positives > 0
                else float("nan")
            )
            brier = float(np.mean((pr - yt) ** 2))
            mcc = _mcc(tn, fp, fn, tp)
            all_true.append(yt)
            all_pred.append(yp)
            all_prob.append(pr)
        per_label.append({
            "label": label,
            "selected": int(selected.sum()),
            "positives": int(positives),
            "negatives": int(negatives),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "mcc": mcc,
            "auroc": auroc,
            "aupr": aupr,
            "brier": brier,
            "prediction_positive_rate": float(yp.mean()) if yp.size else 0.0,
        })

    flat_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int32)
    flat_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=bool)
    flat_prob = np.concatenate(all_prob) if all_prob else np.array([], dtype=np.float64)
    micro_precision = (
        float(precision_score(flat_true, flat_pred, zero_division=0))
        if flat_true.size else 0.0
    )
    micro_recall = (
        float(recall_score(flat_true, flat_pred, zero_division=0))
        if flat_true.size else 0.0
    )
    micro_f1 = (
        float(f1_score(flat_true, flat_pred, zero_division=0))
        if flat_true.size else 0.0
    )
    micro_auroc = (
        _safe_metric(roc_auc_score, flat_true, flat_prob, 0.0)
        if flat_true.size and np.unique(flat_true).size == 2 else 0.0
    )
    micro_aupr = (
        _safe_metric(average_precision_score, flat_true, flat_prob, 0.0)
        if flat_true.size and bool((flat_true > 0).any()) else 0.0
    )
    micro_tn, micro_fp, micro_fn, micro_tp = _confusion(flat_true, flat_pred)
    micro_negatives = micro_tn + micro_fp
    micro_specificity = (
        float(micro_tn / micro_negatives) if micro_negatives > 0 else 0.0
    )
    selected_positions = int(mask.sum())
    samplewise = _samplewise_metrics(y_true, predictions, mask)
    output = {
        "macro_precision": float(np.mean([row["precision"] for row in per_label])),
        "macro_recall": float(np.mean([row["recall"] for row in per_label])),
        "macro_specificity": _nanmean([row["specificity"] for row in per_label]),
        "macro_balanced_accuracy": _nanmean(
            [row["balanced_accuracy"] for row in per_label]
        ),
        "macro_f1": float(np.mean([row["f1"] for row in per_label])),
        "macro_mcc": float(np.mean([row["mcc"] for row in per_label])),
        "macro_auroc": _nanmean([row["auroc"] for row in per_label]),
        "macro_aupr": _nanmean([row["aupr"] for row in per_label]),
        "macro_brier": _nanmean([row["brier"] for row in per_label]),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_specificity": micro_specificity,
        "micro_f1": micro_f1,
        "micro_mcc": _mcc(micro_tn, micro_fp, micro_fn, micro_tp),
        "micro_auroc": micro_auroc,
        "micro_aupr": micro_aupr,
        "hamming_loss": (
            float(np.mean(flat_true != flat_pred)) if flat_true.size else 0.0
        ),
        "brier_score": (
            float(np.mean((flat_prob - flat_true) ** 2)) if flat_true.size else 0.0
        ),
        "ece": _ece(flat_true.astype(float), flat_prob, n_bins=ece_bins),
        "prediction_positive_rate": (
            float(flat_pred.mean()) if flat_pred.size else 0.0
        ),
        "selected_positions": selected_positions,
        "valid_auroc_labels": int(sum(np.isfinite(row["auroc"]) for row in per_label)),
        "valid_aupr_labels": int(sum(np.isfinite(row["aupr"]) for row in per_label)),
        "valid_specificity_labels": int(
            sum(np.isfinite(row["specificity"]) for row in per_label)
        ),
        "per_label": per_label,
        **samplewise,
    }
    return output


def evaluate_positive_only(y_true, probabilities, label_names, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive_mask = y_true > 0
    base = evaluate_masked_multilabel(
        y_true,
        probabilities,
        positive_mask.astype(np.float32),
        label_names,
        threshold=threshold,
    )
    per_label_recalls = []
    for index in range(y_true.shape[1]):
        selected = positive_mask[:, index]
        if bool(selected.any()):
            per_label_recalls.append(
                float((probabilities[selected, index] >= threshold).mean())
            )
        else:
            per_label_recalls.append(0.0)
    known_positive_positions = int(positive_mask.sum())
    hits = int(np.logical_and(positive_mask, probabilities >= threshold).sum())
    micro_recall = hits / known_positive_positions if known_positive_positions else 0.0
    for row in base["per_label"]:
        row["specificity"] = float("nan")
        row["balanced_accuracy"] = float("nan")
        row["mcc"] = 0.0
        row["auroc"] = float("nan")
        row["aupr"] = float("nan")
    base.update({
        "macro_specificity": 0.0,
        "macro_balanced_accuracy": 0.0,
        "macro_mcc": 0.0,
        "macro_auroc": 0.0,
        "macro_aupr": 0.0,
        "micro_specificity": 0.0,
        "micro_mcc": 0.0,
        "micro_auroc": 0.0,
        "micro_aupr": 0.0,
        "valid_auroc_labels": 0,
        "valid_aupr_labels": 0,
        "valid_specificity_labels": 0,
        "macro_positive_recall": float(np.mean(per_label_recalls)),
        "micro_positive_recall": float(micro_recall),
        "positive_miss_rate": float(1.0 - micro_recall) if known_positive_positions else 0.0,
        "known_positive_positions": known_positive_positions,
        "protocol_note": (
            "Positive-only contains no evaluated negatives; interpret it as known-positive "
            "coverage/recall. Specificity, AUROC, and false-positive control require the "
            "fixed masked protocol."
        ),
    })
    return base


def aggregate_fold_metrics(rows: List[Dict], keys=None) -> Dict:
    keys = keys or [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_specificity",
        "macro_balanced_accuracy",
        "macro_mcc",
        "macro_auroc",
        "macro_aupr",
        "micro_aupr",
        "hamming_loss",
        "samples_jaccard",
        "subset_accuracy",
        "brier_score",
        "ece",
        "prediction_positive_rate",
        "mean_predicted_labels_per_sample",
        "macro_positive_recall",
        "micro_positive_recall",
    ]
    result = {}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if row and key in row and row[key] is not None],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        result[f"{key}_mean"] = float(finite.mean()) if finite.size else 0.0
        result[f"{key}_std"] = float(finite.std()) if finite.size else 0.0
    return result


def evaluate_observed_ranking_metrics(
    y_true,
    probabilities,
    label_names: Sequence[str],
    k_values: Iterable[int] = (1, 2, 3, 5),
) -> Dict:

    y = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.shape != scores.shape or y.ndim != 2:
        raise ValueError("ranking inputs must share [samples, labels] shape")
    if y.shape[1] != len(label_names):
        raise ValueError("label_names does not match ranking columns")
    ks = tuple(sorted(set(int(k) for k in k_values)))
    if not ks or any(k < 1 or k > y.shape[1] for k in ks):
        raise ValueError("invalid K values")
    keys = tuple(str(k) for k in ks)
    precision = {key: [] for key in keys}
    recall = {key: [] for key in keys}
    f1 = {key: [] for key in keys}
    sample_ap = []
    per_sample = []

    for sample_index, (target_row, score_row) in enumerate(zip(y, scores)):
        positives = target_row == 1
        count = int(positives.sum())
        row = {
            "sample_index": sample_index,
            "positive_count": count,
            "valid": count > 0,
            "precision_at_k": {key: None for key in keys},
            "recall_at_k": {key: None for key in keys},
            "f1_at_k": {key: None for key in keys},
            "average_precision": None,
        }
        if count == 0:
            per_sample.append(row)
            continue
        order = np.argsort(-score_row, kind="mergesort")
        hits = positives[order].astype(np.float64)
        cumulative = np.cumsum(hits)
        for k, key in zip(ks, keys):
            p = float(cumulative[k - 1] / k)
            r = float(cumulative[k - 1] / count)
            score = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
            precision[key].append(p)
            recall[key].append(r)
            f1[key].append(score)
            row["precision_at_k"][key] = p
            row["recall_at_k"][key] = r
            row["f1_at_k"][key] = score
        precision_by_rank = cumulative / np.arange(1, y.shape[1] + 1, dtype=np.float64)
        ap = float((precision_by_rank * hits).sum() / count)
        row["average_precision"] = ap
        sample_ap.append(ap)
        per_sample.append(row)

    per_label = []
    label_ap = []
    for column, label in enumerate(label_names):
        target = (y[:, column] == 1).astype(np.int32)
        count = int(target.sum())
        ap = float(average_precision_score(target, scores[:, column])) if count else None
        if ap is not None:
            label_ap.append(ap)
        per_label.append({"label": str(label), "positive_count": count, "average_precision": ap})

    return {
        "metric": "Observed-positive multilabel ranking",
        "positive_policy": "target=1 observed positive; target=0 remains unlabeled",
        "candidate_policy": "rank all labels within each peptide for Top-K/AP-samples",
        "k_values": list(ks),
        "valid_samples": len(sample_ap),
        "skipped_samples": int(y.shape[0] - len(sample_ap)),
        "valid_labels": len(label_ap),
        "precision_at_k": {key: float(np.mean(precision[key])) for key in keys},
        "recall_at_k": {key: float(np.mean(recall[key])) for key in keys},
        "f1_at_k": {key: float(np.mean(f1[key])) for key in keys},
        "ap_samples": float(np.mean(sample_ap)) if sample_ap else 0.0,
        "map_labels": float(np.mean(label_ap)) if label_ap else 0.0,
        "per_sample": per_sample,
        "per_label": per_label,
        "protocol_note": (
            "Observed positives are hits; unlabeled entries are non-hits, not "
            "confirmed biological negatives."
        ),
    }


def aggregate_observed_ranking_metrics(rows: Iterable[Dict]) -> Dict:
    rows = [row for row in rows if row]
    if not rows:
        return {}
    result = {"k_values": list(rows[0]["k_values"]), "n_folds": len(rows)}
    for name in ("ap_samples", "map_labels"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
        result[f"{name}_per_fold"] = values.tolist()
    for metric_name in ("precision_at_k", "recall_at_k", "f1_at_k"):
        result[metric_name] = {}
        for k in result["k_values"]:
            values = np.asarray(
                [row[metric_name][str(k)] for row in rows], dtype=np.float64
            )
            result[metric_name][str(k)] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "per_fold": values.tolist(),
            }
    return result
