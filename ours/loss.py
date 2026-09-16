from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelNonNegativePULoss(nn.Module):


    def __init__(
        self,
        prior_min: float = 1e-3,
        prior_max: float = 0.5,
        correction_beta: float = 0.0,
        correction_gamma: float = 0.75,
        risk_mode: Literal["nnpu", "upu"] = "nnpu",
    ):
        super().__init__()
        if risk_mode not in {"nnpu", "upu"}:
            raise ValueError(f"unknown PU risk mode: {risk_mode}")
        self.prior_min = float(prior_min)
        self.prior_max = float(prior_max)
        self.correction_beta = float(correction_beta)
        self.correction_gamma = float(correction_gamma)
        self.risk_mode = risk_mode

    @staticmethod
    def estimate_priors(
        targets: torch.Tensor, min_value: float = 1e-3, max_value: float = 0.5
    ) -> torch.Tensor:
        if targets.ndim != 2:
            raise ValueError("targets must have shape [samples, labels]")
        return (targets > 0.5).float().mean(dim=0).clamp(min_value, max_value)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        priors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have equal shape")
        valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask > 0.5
        positive = (targets > 0.5) & valid
        unlabeled = (targets <= 0.5) & valid
        priors = priors.to(logits).clamp(self.prior_min, self.prior_max)

        risks = []
        positive_risks = []
        raw_negative_risks = []
        unbiased_risks = []
        nonnegative_risks = []
        corrected_flags = []
        for label in range(logits.shape[1]):
            p_mask = positive[:, label]
            u_mask = unlabeled[:, label]
            if not bool(p_mask.any()) or not bool(u_mask.any()):
                continue
            positive_logits = logits[p_mask, label]
            unlabeled_logits = logits[u_mask, label]
            prior = priors[label]
            positive_risk = prior * F.softplus(-positive_logits).mean()
            raw_negative_risk = (
                F.softplus(unlabeled_logits).mean()
                - prior * F.softplus(positive_logits).mean()
            )
            unbiased_risk = positive_risk + raw_negative_risk
            nonnegative_risk = torch.clamp(raw_negative_risk, min=0.0)
            corrected = self.risk_mode == "nnpu" and bool(
                raw_negative_risk.detach().item() < -self.correction_beta
            )
            if self.risk_mode == "upu":
                risk = unbiased_risk
            else:
                risk = (
                    -self.correction_gamma * raw_negative_risk
                    if corrected
                    else unbiased_risk
                )
            risks.append(risk)
            positive_risks.append(positive_risk.detach())
            raw_negative_risks.append(raw_negative_risk.detach())
            unbiased_risks.append(unbiased_risk.detach())
            nonnegative_risks.append(nonnegative_risk.detach())
            corrected_flags.append(corrected)

        if not risks:
            zero = logits.sum() * 0.0
            return zero, {
                "valid_labels": 0,
                "mean_positive_risk": 0.0,
                "mean_raw_negative_risk": 0.0,
                "mean_unbiased_risk": 0.0,
                "mean_nonnegative_risk": 0.0,
                "clamped_labels": 0,
                "clamp_rate": 0.0,
                "correction_labels": 0,
                "correction_rate": 0.0,
            }
        loss = torch.stack(risks).mean()
        raw = torch.stack(raw_negative_risks)
        valid_labels = len(risks)
        corrected_labels = sum(bool(flag) for flag in corrected_flags)
        clamped_labels = int((raw < 0).sum().item())
        return loss, {
            "valid_labels": valid_labels,
            "mean_positive_risk": float(torch.stack(positive_risks).mean().item()),
            "mean_raw_negative_risk": float(raw.mean().item()),
            "mean_unbiased_risk": float(torch.stack(unbiased_risks).mean().item()),
            "mean_nonnegative_risk": float(torch.stack(nonnegative_risks).mean().item()),
            "clamped_labels": clamped_labels,
            "clamp_rate": float(clamped_labels / valid_labels),
            "correction_labels": corrected_labels,
            "correction_rate": float(corrected_labels / valid_labels),
        }


class BalancedObservedPositiveLoss(nn.Module):


    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
        label_scope: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have equal shape")
        valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask > 0.5
        positive = (targets > 0.5) & valid
        if label_scope is not None:
            if label_scope.shape != (logits.shape[1],):
                raise ValueError("label_scope must have one value per label")
            positive = positive & label_scope.to(
                device=positive.device, dtype=torch.bool
            ).unsqueeze(0)
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if class_weights is not None or focal_gamma > 0.0:
            if class_weights is not None and class_weights.shape != (logits.shape[1],):
                raise ValueError("class_weights must have one value per label")
            if not bool(positive.any()):
                zero = logits.sum() * 0.0
                return zero, {"valid_labels": 0, "positive_positions": 0}
            weights = (
                torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype)
                if class_weights is None
                else class_weights.to(device=logits.device, dtype=logits.dtype)
            )
            values = F.softplus(-logits)
            if focal_gamma > 0.0:
                values = values * (1.0 - torch.sigmoid(logits)).pow(focal_gamma)
            expanded = weights.unsqueeze(0).expand_as(values)
            loss = (values[positive] * expanded[positive]).mean()
            return loss, {
                "valid_labels": int(positive.any(dim=0).sum().item()),
                "positive_positions": int(positive.sum().item()),
            }
        per_label = []
        for label in range(logits.shape[1]):
            label_positive = positive[:, label]
            if bool(label_positive.any()):
                per_label.append(F.softplus(-logits[label_positive, label]).mean())
        if not per_label:
            zero = logits.sum() * 0.0
            return zero, {"valid_labels": 0, "positive_positions": 0}
        loss = torch.stack(per_label).mean()
        return loss, {
            "valid_labels": len(per_label),
            "positive_positions": int(positive.sum().item()),
        }


class DeficitAwarePositiveTopKLoss(nn.Module):


    def __init__(
        self,
        ks=(1, 3, 5),
        k_weights=(0.30, 0.40, 0.30),
        margin: float = 0.02,
    ):
        super().__init__()
        self.ks = tuple(int(k) for k in ks)
        weights = tuple(float(weight) for weight in k_weights)
        if not self.ks or len(self.ks) != len(weights):
            raise ValueError("ks and k_weights must be non-empty and have equal length")
        if any(k < 1 for k in self.ks):
            raise ValueError("all K values must be positive")
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("k_weights must be non-negative with a positive sum")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        total = sum(weights)
        self.k_weights = tuple(weight / total for weight in weights)
        self.margin = float(margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have equal shape")
        positive = targets > 0.5
        detached = logits.detach()
        terms = {k: [] for k in self.ks}
        active = {k: 0 for k in self.ks}
        valid = {k: 0 for k in self.ks}
        deficit_total = {k: 0 for k in self.ks}
        repair_total = {k: 0 for k in self.ks}
        sample_has_term = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)

        for row in range(logits.shape[0]):
            positive_count = int(positive[row].sum().item())
            if positive_count == 0:
                continue
            order = torch.argsort(detached[row], descending=True)
            for k in self.ks:
                top_indices = order[:k]
                outside_indices = order[k:]
                true_positive_count = int(positive[row, top_indices].sum().item())
                deficit = min(positive_count, k) - true_positive_count
                if deficit <= 0:
                    continue
                u_inside = top_indices[~positive[row, top_indices]]
                p_outside = outside_indices[positive[row, outside_indices]]
                repair_count = min(deficit, int(u_inside.numel()), int(p_outside.numel()))
                if repair_count <= 0:
                    continue
                p_scores = torch.topk(
                    logits[row, p_outside], k=repair_count, largest=True
                ).values
                u_scores = torch.topk(
                    detached[row, u_inside], k=repair_count, largest=False
                ).values
                p_scores = torch.sort(p_scores, descending=True).values
                u_scores = torch.sort(u_scores, descending=True).values
                hinges = F.relu(self.margin + u_scores - p_scores)
                terms[k].append(hinges.mean())
                active[k] += int((hinges.detach() > 0).sum().item())
                valid[k] += 1
                deficit_total[k] += deficit
                repair_total[k] += repair_count
                sample_has_term[row] = True

        zero = logits.sum() * 0.0
        total_loss = zero
        mean_by_k = {}
        for k, weight in zip(self.ks, self.k_weights):
            k_loss = torch.stack(terms[k]).mean() if terms[k] else zero
            total_loss = total_loss + weight * k_loss
            mean_by_k[k] = float(k_loss.detach().item())
        return total_loss, {
            "valid_samples": int(sample_has_term.sum().item()),
            "active_pairs": int(sum(active.values())),
            "mean_loss": float(total_loss.detach().item()),
            "active_pairs_by_k": active,
            "valid_samples_by_k": valid,
            "mean_loss_by_k": mean_by_k,
            "deficit_by_k": deficit_total,
            "repair_pairs_by_k": repair_total,
        }
