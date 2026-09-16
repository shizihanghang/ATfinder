from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import encode_sequences_phys


class InteractionFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.left_norm = nn.LayerNorm(dim)
        self.right_norm = nn.LayerNorm(dim)
        self.interaction = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_aligned = self.left_norm(left)
        right_aligned = self.right_norm(right)
        return self.interaction(
            torch.cat(
                [
                    left_aligned,
                    right_aligned,
                    left_aligned * right_aligned,
                    (left_aligned - right_aligned).abs(),
                ],
                dim=-1,
            )
        )


def adjacency_to_edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    src, dst = torch.nonzero(adjacency > 0, as_tuple=True)
    return torch.stack([src, dst], dim=0)


def adjacency_to_edge_weight(adjacency: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return adjacency[edge_index[0], edge_index[1]]


class PrototypeGraphPropagation(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.layer = GCNConv(
            dim,
            dim,
            improved=False,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        edge_index = adjacency_to_edge_index(adjacency)
        edge_weight = adjacency_to_edge_weight(adjacency, edge_index)
        output = self.layer(nodes, edge_index, edge_weight=edge_weight)
        return self.dropout(F.gelu(output))


class PositivePrototypeGraphBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        graph_dim: int,
        topk: int,
        match_temperature: float,
        graph_dropout: float,
        residual_gamma: float,
    ):
        super().__init__()
        self.n_labels = int(n_labels)
        self.topk = int(topk)
        self.match_temperature = float(match_temperature)
        self.residual_gamma = float(residual_gamma)
        self.prototype_proj = nn.Linear(input_dim, graph_dim)
        self.sample_proj = nn.Linear(input_dim, graph_dim)
        self.pyg_graph = PrototypeGraphPropagation(graph_dim, graph_dropout)
        self.graph_norm = nn.LayerNorm(graph_dim)
        self.register_buffer("prototypes", torch.zeros(n_labels, input_dim))
        self.register_buffer("prototype_center", torch.zeros(input_dim))
        self.register_buffer("adjacency", torch.eye(n_labels))
        self.register_buffer("positive_counts", torch.zeros(n_labels))
        self.register_buffer("prototypes_ready", torch.tensor(False))
        self.register_buffer("graph_structure_ready", torch.tensor(False))

    def refresh_prototypes(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        train_mask: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        valid = torch.ones_like(targets, dtype=torch.bool) if train_mask is None else train_mask > 0.5
        observed_positive = ((targets > 0.5) & valid).to(features.dtype)
        center = features.mean(dim=0)
        working_features = features - center.unsqueeze(0)
        counts = observed_positive.sum(dim=0)
        assignment_mass = observed_positive.sum(dim=0)
        raw = observed_positive.transpose(0, 1) @ working_features
        raw = raw / assignment_mass.clamp(min=1.0).unsqueeze(1)
        positive_rows = observed_positive.sum(dim=1) > 0
        global_center = (
            working_features[positive_rows].mean(dim=0)
            if bool(positive_rows.any())
            else working_features.mean(dim=0)
        )
        raw[counts <= 0] = global_center
        normalized = F.normalize(raw, dim=1)
        cooccurrence = observed_positive.transpose(0, 1) @ observed_positive
        denominator = torch.sqrt(
            counts.clamp_min(1.0).unsqueeze(1) * counts.clamp_min(1.0).unsqueeze(0)
        )
        similarity = cooccurrence / denominator
        selected = torch.zeros_like(similarity, dtype=torch.bool)
        selected.fill_diagonal_(True)
        candidates = similarity.clone()
        candidates.fill_diagonal_(-1.0)
        selected.scatter_(
            1,
            torch.topk(candidates, k=min(self.topk, self.n_labels - 1), dim=1).indices,
            True,
        )
        selected = selected | selected.transpose(0, 1)
        adjacency = similarity.clamp_min(0.0) * selected.to(similarity.dtype)
        self.prototypes.copy_(normalized)
        self.prototype_center.copy_(center)
        if not bool(self.graph_structure_ready.item()):
            self.adjacency.copy_(adjacency)
        self.positive_counts.copy_(counts)
        self.prototypes_ready.fill_(True)
        return {
            "updated_labels": int((counts > 0).sum().item()),
            "graph_edges": int(selected.sum().item()),
            "mean_edge_weight": float(adjacency[selected].mean().item()),
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        prototype_nodes = self.prototype_proj(self.prototypes)
        graph_delta = self.pyg_graph(prototype_nodes, self.adjacency)
        graph_prototypes = self.graph_norm(prototype_nodes + self.residual_gamma * graph_delta)
        match_features = features - self.prototype_center
        sample = self.sample_proj(match_features)
        logits = F.normalize(sample, dim=-1) @ F.normalize(graph_prototypes, dim=-1).transpose(0, 1)
        logits = logits - logits.mean(dim=1, keepdim=True)
        return logits / self.match_temperature


class ClassificationCore(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_labels: int,
        graph_dim: int,
        graph_topk: int,
        graph_lambda: float,
        base_dropout: float,
        graph_dropout: float,
        graph_match_temperature: float,
        graph_residual_gamma: float,
    ):
        super().__init__()
        if not 0.0 < float(graph_lambda) < 1.0:
            raise ValueError("graph_lambda must be strictly between zero and one")
        self.base_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(base_dropout),
            nn.Linear(feature_dim, n_labels),
        )
        initial_gate = math.log(float(graph_lambda) / (1.0 - float(graph_lambda)))
        self.graph_fusion_logit = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))
        self.graph_branch = PositivePrototypeGraphBranch(
            input_dim=feature_dim,
            n_labels=n_labels,
            graph_dim=graph_dim,
            topk=graph_topk,
            match_temperature=graph_match_temperature,
            graph_dropout=graph_dropout,
            residual_gamma=graph_residual_gamma,
        )

    def refresh_prototypes(self, features, targets, train_mask=None):
        return self.graph_branch.refresh_prototypes(features, targets, train_mask)

    def forward(
        self,
        z0: torch.Tensor,
        graph_enabled: bool = True,
        graph_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        base_logits = self.base_classifier(z0)
        scale = min(max(float(graph_scale), 0.0), 1.0)
        graph_ready = bool(self.graph_branch.prototypes_ready.item())
        if graph_enabled and graph_ready and scale > 0.0:
            graph_logits = self.graph_branch(z0)
            effective_lambda = torch.sigmoid(self.graph_fusion_logit) * scale
            final_logits = base_logits + effective_lambda * graph_logits
            graph_available = z0.new_tensor(True)
        else:
            graph_logits = torch.zeros_like(base_logits)
            final_logits = base_logits
            graph_available = z0.new_tensor(False)
        return {
            "z0": z0,
            "base_logits": base_logits,
            "graph_logits": graph_logits,
            "graph_aux_logits": graph_logits if bool(graph_available.item()) else None,
            "final_logits": final_logits,
            "graph_available": graph_available,
            "effective_graph_lambda": torch.sigmoid(self.graph_fusion_logit) * scale,
            "graph_positive_counts": self.graph_branch.positive_counts,
        }


class PhysicochemicalCNN(nn.Module):
    def __init__(self, in_channels: int, channels, kernels, output_dim: int):
        super().__init__()
        layers = []
        current = in_channels
        for out_channels, kernel in zip(channels, kernels):
            layers.extend(
                [
                    nn.Conv1d(current, out_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ]
            )
            current = out_channels
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(current, output_dim)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(features.shape[-1], device=features.device).unsqueeze(0)
        valid = (positions < lengths.unsqueeze(1)).unsqueeze(1).to(features.dtype)
        hidden = self.network(features)
        pooled = (hidden * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
        return self.output(pooled)


class AlignmentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ATfinder(nn.Module):
    def __init__(
        self,
        esm_cache: torch.Tensor,
        phys_table: torch.Tensor,
        n_labels: int,
        phys_input_dim: int,
        cnn_channels,
        cnn_kernels,
        cnn_output_dim: int,
        model_dim: int,
        graph_dim: int,
        graph_topk: int,
        graph_lambda: float,
        base_dropout: float,
        graph_dropout: float,
        graph_match_temperature: float,
        graph_residual_gamma: float,
    ):
        super().__init__()
        self.register_buffer("esm_cache", torch.as_tensor(esm_cache).float(), persistent=False)
        self.register_buffer("phys_table", torch.as_tensor(phys_table).float(), persistent=False)
        self.phys_cnn = PhysicochemicalCNN(phys_input_dim, cnn_channels, cnn_kernels, cnn_output_dim)
        self.phys_align = AlignmentMLP(cnn_output_dim, model_dim)
        self.esm_align = AlignmentMLP(int(self.esm_cache.shape[1]), model_dim)
        self.fusion = InteractionFusion(model_dim)
        self.core = ClassificationCore(
            feature_dim=model_dim,
            n_labels=n_labels,
            graph_dim=graph_dim,
            graph_topk=graph_topk,
            graph_lambda=graph_lambda,
            base_dropout=base_dropout,
            graph_dropout=graph_dropout,
            graph_match_temperature=graph_match_temperature,
            graph_residual_gamma=graph_residual_gamma,
        )

    def extract_initial_features(self, sequences, global_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        indices = global_indices.to(device=self.esm_cache.device, dtype=torch.long)
        esm = self.esm_align(self.esm_cache.index_select(0, indices))
        encoded, lengths = encode_sequences_phys(list(sequences), self.phys_table)
        phys = self.phys_align(self.phys_cnn(encoded.permute(0, 2, 1), lengths))
        z0 = self.fusion(esm, phys)
        return {"z0": z0, "esm": esm, "phys": phys}

    def forward(
        self,
        sequences,
        global_indices: torch.Tensor,
        graph_enabled: bool = True,
        graph_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        features = self.extract_initial_features(sequences, global_indices)
        output = self.core(
            features["z0"],
            graph_enabled=graph_enabled,
            graph_scale=graph_scale,
        )
        output.update({"esm_features": features["esm"], "phys_features": features["phys"]})
        return output
