from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def evaluate_positive_only(
    y_true, probabilities, label_names, threshold: float = 0.5
) -> Dict:
    """Recall at a fixed threshold using observed positives only.

    This is used for checkpoint selection. Unreported entries are not treated
    as confirmed negatives, so the output contains only observed-positive
    recall and per-label counts.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    if y_true.ndim != 2 or y_true.shape[1] != len(label_names):
        raise ValueError("inputs must have shape [samples, len(label_names)]")

    predicted = probabilities > float(threshold)
    per_label = []
    recalls = []
    total_positive = int((y_true > 0).sum())
    total_hits = int(np.logical_and(y_true > 0, predicted).sum())
    for index, label in enumerate(label_names):
        observed = y_true[:, index] > 0
        positives = int(observed.sum())
        hits = int(np.logical_and(observed, predicted[:, index]).sum())
        recall = float(hits / positives) if positives else 0.0
        recalls.append(recall)
        per_label.append(
            {
                "label": str(label),
                "positive_count": positives,
                "hits": hits,
                "recall": recall,
            }
        )
    return {
        "threshold": float(threshold),
        "macro_positive_recall": float(np.mean(recalls)) if recalls else 0.0,
        "known_positive_positions": total_positive,
        "hits": total_hits,
        "prediction_positive_rate": float(predicted.mean()),
        "per_label": per_label,
        "protocol_note": (
            "Only observed positives are evaluated; unreported entries remain unlabeled."
        ),
    }


def evaluate_observed_ranking_metrics(
    y_true,
    probabilities,
    label_names: Sequence[str],
    k_values: Iterable[int] = (1, 3, 5),
) -> Dict:
    """Sample-wise Recall@k, Precision@k, F1@k and AP from Appendix S3."""
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
            f = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
            precision[key].append(p)
            recall[key].append(r)
            f1[key].append(f)
            row["precision_at_k"][key] = p
            row["recall_at_k"][key] = r
            row["f1_at_k"][key] = f
        precision_by_rank = cumulative / np.arange(1, y.shape[1] + 1, dtype=np.float64)
        ap = float((precision_by_rank * hits).sum() / count)
        row["average_precision"] = ap
        sample_ap.append(ap)
        per_sample.append(row)

    return {
        "metric": "Observed-positive multilabel ranking",
        "positive_policy": "target=1 observed positive; target=0 remains unlabeled",
        "candidate_policy": "rank all labels within each peptide",
        "k_values": list(ks),
        "valid_samples": len(sample_ap),
        "skipped_samples": int(y.shape[0] - len(sample_ap)),
        "precision_at_k": {key: float(np.mean(precision[key])) for key in keys},
        "recall_at_k": {key: float(np.mean(recall[key])) for key in keys},
        "f1_at_k": {key: float(np.mean(f1[key])) for key in keys},
        "ap_samples": float(np.mean(sample_ap)) if sample_ap else 0.0,
        "per_sample": per_sample,
        "protocol_note": (
            "Observed positives are hits; unlabeled entries are not confirmed negatives."
        ),
    }


def aggregate_observed_ranking_metrics(rows: Iterable[Dict]) -> Dict:
    rows = [row for row in rows if row]
    if not rows:
        return {}
    result = {"k_values": list(rows[0]["k_values"]), "n_folds": len(rows)}
    values = np.asarray([row["ap_samples"] for row in rows], dtype=np.float64)
    result["ap_samples_mean"] = float(values.mean())
    result["ap_samples_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    result["ap_samples_per_fold"] = values.tolist()
    for metric_name in ("precision_at_k", "recall_at_k", "f1_at_k"):
        result[metric_name] = {}
        for k in result["k_values"]:
            key = str(k)
            values = np.asarray(
                [row[metric_name][key] for row in rows], dtype=np.float64
            )
            result[metric_name][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "per_fold": values.tolist(),
            }
    return result
