from __future__ import annotations

import math
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelNonNegativePULoss(nn.Module):
    """Label-wise nnPU objective corresponding to Eqs. (10)--(12)."""

    def __init__(
        self,
        correction_beta: float = 0.0,
        correction_gamma: float = 0.75,
        risk_mode: Literal["nnpu", "upu"] = "nnpu",
    ):
        super().__init__()
        if risk_mode not in {"nnpu", "upu"}:
            raise ValueError(f"unknown PU risk mode: {risk_mode}")
        self.correction_beta = float(correction_beta)
        self.correction_gamma = float(correction_gamma)
        self.risk_mode = risk_mode

    @staticmethod
    def estimate_priors(
        targets: torch.Tensor, prior_scale: float = 1.0
    ) -> torch.Tensor:
        """Return alpha times the observed-positive rate in the training split."""
        if targets.ndim != 2 or targets.shape[0] == 0:
            raise ValueError("targets must have nonempty shape [samples, labels]")
        scale = float(prior_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("prior_scale must be finite and positive")
        priors = (targets > 0.5).float().mean(dim=0) * scale
        if not torch.isfinite(priors).all() or ((priors < 0) | (priors > 1)).any():
            raise ValueError("scaled priors must lie in [0, 1]; reduce prior_scale")
        return priors

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        priors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if logits.ndim != 2 or logits.shape != targets.shape:
            raise ValueError("logits and targets must have equal [samples, labels] shape")
        if logits.shape[1] == 0:
            raise ValueError("the PU objective requires at least one label")
        if priors.ndim != 1 or priors.shape[0] != logits.shape[1]:
            raise ValueError("priors must contain one value per label")
        priors = priors.to(logits)
        if not torch.isfinite(priors).all() or ((priors < 0) | (priors > 1)).any():
            raise ValueError("priors must be finite probabilities in [0, 1]")
        # The paper protocol treats every unreported label as unlabeled.  The
        # optional mask remains only for backwards compatibility with old data.
        valid = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask > 0.5
        positive = (targets > 0.5) & valid
        unlabeled = (targets <= 0.5) & valid
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
                # An unestimable minibatch term contributes zero; retain the
                # full label count M in Eq. (12)'s cross-label normalization.
                continue
            positive_logits = logits[p_mask, label]
            unlabeled_logits = logits[u_mask, label]
            prior = priors[label]
            positive_risk = prior * F.softplus(-positive_logits).mean()
            negative_risk = (
                F.softplus(unlabeled_logits).mean()
                - prior * F.softplus(positive_logits).mean()
            )
            unbiased = positive_risk + negative_risk
            nonnegative = torch.clamp(negative_risk, min=0.0)
            corrected = self.risk_mode == "nnpu" and bool(
                negative_risk.detach().item() < -self.correction_beta
            )
            if self.risk_mode == "upu":
                risk = unbiased
            else:
                risk = (
                    -self.correction_gamma * negative_risk
                    if corrected
                    else unbiased
                )
            risks.append(risk)
            positive_risks.append(positive_risk.detach())
            raw_negative_risks.append(negative_risk.detach())
            unbiased_risks.append(unbiased.detach())
            nonnegative_risks.append(nonnegative.detach())
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

        loss = torch.stack(risks).sum() / logits.shape[1]
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
