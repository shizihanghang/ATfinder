from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


LABELS = (
    "Breast",
    "Lung",
    "Bowel",
    "Prostate",
    "Cervix",
    "Skin",
    "Myeloid",
    "Lymphoid",
    "Brain",
    "Liver",
    "Ovary",
    "Stomach",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    project: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    results: Path = PROJECT_ROOT / "runs" / "atfinder"

    @property
    def dataset(self) -> Path:
        return self.data / "dataset.csv"

    @property
    def esm_cache(self) -> Path:
        return self.data / "esm2_t30_150m_mean.pt"

    @property
    def splits(self) -> Path:
        return self.data / "splits.json"

    @property
    def aaindex(self) -> Path:
        return self.data / "aaindex.csv"

    @property
    def masks(self) -> Path:
        return self.data / "masks"


@dataclass(frozen=True)
class ModelConfig:
    labels: tuple[str, ...] = LABELS
    max_length: int = 50
    phys_dim: int = 105
    esm_dim: int = 640
    model_dim: int = 256
    cnn_output_dim: int = 128
    cnn_channels: tuple[int, ...] = (128, 256, 128)
    cnn_kernels: tuple[int, ...] = (3, 5, 3)
    fusion_alpha: float = 0.535
    base_dropout: float = 0.30
    graph_dim: int = 128
    graph_topk: int = 3
    graph_dropout: float = 0.20
    graph_match_temperature: float = 0.20
    graph_residual_gamma: float = 1.0
    graph_lambda: float = 0.10


@dataclass(frozen=True)
class LossConfig:
    pu_prior_min: float = 1e-3
    pu_prior_max: float = 0.50
    pu_correction_beta: float = 0.0
    pu_correction_gamma: float = 0.75
    topk_weight: float = 0.015
    topk_margin: float = 0.02
    topk_values: tuple[int, ...] = (1, 3, 5)
    topk_weights: tuple[float, ...] = (0.30, 0.40, 0.30)
    topk_start_epoch: int = 41
    topk_ramp_epochs: int = 10
    ppg_positive_weight: float = 0.10
    ppg_positive_start_epoch: int = 1
    ppg_positive_ramp_epochs: int = 1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    folds: int = 5
    epochs: int = 100
    batch_size: int = 48
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 5e-3
    gradient_clip_norm: float = 1.0
    threshold: float = 0.5
    graph_start_epoch: int = 1
    graph_ramp_epochs: int = 10
    initialize_prototypes_before_training: bool = True
    evaluate_independent: bool = True
    selection_f1_3_weight: float = 0.3
    selection_f1_5_weight: float = 0.3
    selection_recall_weight: float = 0.4


@dataclass(frozen=True)
class ExperimentConfig:
    paths: Paths = Paths()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()

    @property
    def name(self) -> str:
        return f"atfinder_seed{self.train.seed}"

    @property
    def result_dir(self) -> Path:
        return self.paths.results

    def snapshot(self) -> dict:
        payload = asdict(self)
        payload["paths"] = {key: str(value) for key, value in payload["paths"].items()}
        payload["experiment_name"] = self.name
        payload["checkpoint_selection"] = (
            "CV 0.3*F1@3 + 0.3*F1@5 + 0.4*Macro Recall@0.5; earliest tie"
        )
        payload["classification_risk"] = "nnpu"
        payload["unlabeled_policy"] = "positive_unlabeled"
        payload["feature_fusion"] = "interaction_mlp"
        payload["graph_operator"] = "pyg_gcn"
        return payload

    def validate(self) -> None:
        if len(self.model.labels) != 12:
            raise ValueError("ATfinder requires 12 labels")
        if not 1 <= self.model.graph_topk < len(self.model.labels):
            raise ValueError("graph_topk must be between 1 and 11")
        if not 0.0 < self.model.graph_lambda < 1.0:
            raise ValueError("graph_lambda must be strictly between zero and one")
        if abs(sum(self.loss.topk_weights) - 1.0) > 1e-8:
            raise ValueError("topk_weights must sum to one")
        if self.train.epochs < 1:
            raise ValueError("epochs must be positive")


def atfinder_config(
    *,
    result_root: Path | None = None,
    device: str | None = None,
    folds: int | None = None,
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if result_root is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, results=Path(result_root).resolve()))
    if device is not None:
        cfg = replace(cfg, train=replace(cfg.train, device=str(device)))
    if folds is not None:
        cfg = replace(cfg, train=replace(cfg.train, folds=int(folds)))
    cfg.validate()
    return cfg
