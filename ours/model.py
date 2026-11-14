from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import PHYS_IN_DIM, encode_sequences_phys
from .residues import CANONICAL_ORDER


class InteractionFusion(nn.Module):
    """Eq. (3): concatenate two views and their interaction terms."""

    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(dim * 4, dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        interaction = torch.cat(
            [left, right, left * right, (left - right).abs()], dim=-1
        )
        return F.gelu(self.projection(interaction))


def adjacency_to_edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    src, dst = torch.nonzero(adjacency > 0, as_tuple=True)
    if src.numel() == 0:
        n = adjacency.shape[0]
        indices = torch.arange(n, device=adjacency.device)
        return torch.stack([indices, indices], dim=0)
    return torch.stack([src, dst], dim=0)


def adjacency_to_edge_weight(
    adjacency: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    return adjacency[edge_index[0], edge_index[1]]


class PrototypeGraphPropagation(nn.Module):
    """One GCN layer as specified in Appendix S1."""

    def __init__(self, dim: int, graph_dropout: float = 0.3):
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
        self.dropout = nn.Dropout(graph_dropout)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        edge_index = adjacency_to_edge_index(adjacency)
        edge_weight = adjacency_to_edge_weight(adjacency, edge_index)
        output = self.layer(nodes, edge_index, edge_weight=edge_weight)
        return self.dropout(F.gelu(output))


class PositivePrototypeGraphBranch(nn.Module):
    """Construct p_m, the top-v label graph, and graph logits g_a."""

    def __init__(
        self,
        input_dim: int,
        n_labels: int,
        topk: int,
        graph_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_labels = int(n_labels)
        self.topk = int(topk)
        self.pyg_graph = PrototypeGraphPropagation(input_dim, graph_dropout)
        self.register_buffer("prototypes", torch.zeros(n_labels, input_dim))
        self.register_buffer("prototype_center", torch.zeros(input_dim))
        self.register_buffer("adjacency", torch.zeros(n_labels, n_labels))
        self.register_buffer("positive_counts", torch.zeros(n_labels))
        self.register_buffer("prototypes_ready", torch.tensor(False))

    @torch.no_grad()
    def refresh_prototypes(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        train_mask: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        valid = (
            torch.ones_like(targets, dtype=torch.bool)
            if train_mask is None
            else train_mask > 0.5
        )
        observed_positive = ((targets > 0.5) & valid).to(features.dtype)
        center = features.mean(dim=0)
        centered = features - center.unsqueeze(0)
        counts = observed_positive.sum(dim=0)
        raw = observed_positive.transpose(0, 1) @ centered
        raw = raw / counts.clamp(min=1.0).unsqueeze(1)
        fallback = centered.mean(dim=0)
        raw[counts <= 0] = fallback
        prototypes = F.normalize(raw, dim=1)

        cooccurrence = observed_positive.transpose(0, 1) @ observed_positive
        denominator = torch.sqrt(
            counts.clamp_min(1.0).unsqueeze(1) * counts.clamp_min(1.0).unsqueeze(0)
        )
        similarity = cooccurrence / denominator
        selected = torch.zeros_like(similarity, dtype=torch.bool)
        candidates = similarity.clone()
        candidates.fill_diagonal_(-1.0)
        neighbor_count = min(self.topk, self.n_labels - 1)
        selected.scatter_(
            1,
            torch.topk(candidates, k=neighbor_count, dim=1).indices,
            True,
        )
        selected = selected | selected.transpose(0, 1)
        adjacency = similarity.clamp_min(0.0) * selected.to(similarity.dtype)

        self.prototypes.copy_(prototypes)
        self.prototype_center.copy_(center)
        self.adjacency.copy_(adjacency)
        self.positive_counts.copy_(counts)
        self.prototypes_ready.fill_(True)
        return {
            "updated_labels": int((counts > 0).sum().item()),
            "graph_edges": int(selected.sum().item()),
            "mean_edge_weight": float(adjacency[selected].mean().item())
            if bool(selected.any())
            else 0.0,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        label_embeddings = self.pyg_graph(self.prototypes, self.adjacency)
        return F.normalize(features, dim=-1) @ F.normalize(
            label_embeddings, dim=-1
        ).transpose(0, 1)


class ClassificationCore(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_labels: int,
        graph_topk: int,
        graph_lambda: float,
        base_dropout: float,
        graph_dropout: float = 0.3,
    ):
        super().__init__()
        if graph_lambda <= 0.0:
            raise ValueError("graph_lambda must be positive")
        # Appendix S1: two-layer MLP with dropout and batch normalization.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(base_dropout),
            nn.Linear(feature_dim, n_labels),
        )
        # The paper initializes the learnable scalar lambda to 0.1 directly.
        self.lambda_gate = nn.Parameter(torch.tensor(float(graph_lambda)))
        self.graph_branch = PositivePrototypeGraphBranch(
            input_dim=feature_dim,
            n_labels=n_labels,
            topk=graph_topk,
            graph_dropout=graph_dropout,
        )

    def refresh_prototypes(self, features, targets, train_mask=None):
        return self.graph_branch.refresh_prototypes(features, targets, train_mask)

    def forward(self, z0: torch.Tensor, graph_enabled: bool = True) -> Dict[str, torch.Tensor]:
        classification_logits = self.classifier(z0)  # g_b in Eq. (8)
        ready = bool(self.graph_branch.prototypes_ready.item())
        if graph_enabled and ready:
            graph_logits = self.graph_branch(z0)  # g_a in Eq. (8)
            final_logits = graph_logits + self.lambda_gate * classification_logits
            available = z0.new_tensor(True)
        else:
            graph_logits = torch.zeros_like(classification_logits)
            final_logits = classification_logits
            available = z0.new_tensor(False)
        return {
            "z0": z0,
            "classification_logits": classification_logits,
            "graph_logits": graph_logits,
            "final_logits": final_logits,
            "graph_available": available,
            "effective_lambda": self.lambda_gate,
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
        hidden = features * valid
        for layer in self.network:
            hidden = layer(hidden)
            if isinstance(layer, nn.GELU):
                # Prevent padded activations entering the next convolution.
                hidden = hidden * valid
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
        graph_topk: int,
        graph_lambda: float,
        base_dropout: float,
        padding_length: int,
        graph_dropout: float = 0.3,
    ):
        super().__init__()
        if isinstance(padding_length, bool) or not isinstance(padding_length, int) or padding_length < 1:
            raise ValueError("padding_length must be a positive integer")
        self.padding_length = padding_length
        if phys_input_dim != PHYS_IN_DIM:
            raise ValueError(f"phys_input_dim must be {PHYS_IN_DIM}")
        expected_table_shape = (len(CANONICAL_ORDER), PHYS_IN_DIM)
        if tuple(phys_table.shape) != expected_table_shape:
            raise ValueError(
                f"physicochemical table shape must be {expected_table_shape}, "
                f"got {tuple(phys_table.shape)}"
            )
        self.register_buffer("esm_cache", torch.as_tensor(esm_cache).float(), persistent=False)
        self.register_buffer("phys_table", torch.as_tensor(phys_table).float(), persistent=False)
        self.phys_cnn = PhysicochemicalCNN(
            phys_input_dim, cnn_channels, cnn_kernels, cnn_output_dim
        )
        self.phys_align = AlignmentMLP(cnn_output_dim, model_dim)
        self.esm_align = AlignmentMLP(int(self.esm_cache.shape[1]), model_dim)
        self.fusion = InteractionFusion(model_dim)
        self.core = ClassificationCore(
            feature_dim=model_dim,
            n_labels=n_labels,
            graph_topk=graph_topk,
            graph_lambda=graph_lambda,
            base_dropout=base_dropout,
            graph_dropout=graph_dropout,
        )

    def extract_initial_features(
        self, sequences, global_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        indices = global_indices.to(device=self.esm_cache.device, dtype=torch.long)
        esm = self.esm_align(self.esm_cache.index_select(0, indices))
        encoded, lengths = encode_sequences_phys(
            list(sequences), self.phys_table, self.padding_length,
        )
        phys = self.phys_align(self.phys_cnn(encoded.permute(0, 2, 1), lengths))
        fused = self.fusion(esm, phys)
        return {"z0": fused, "esm": esm, "phys": phys}

    def forward(
        self,
        sequences,
        global_indices: torch.Tensor,
        graph_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        features = self.extract_initial_features(sequences, global_indices)
        output = self.core(features["z0"], graph_enabled=graph_enabled)
        output.update({"esm_features": features["esm"], "phys_features": features["phys"]})
        return output
