from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path

from .residues import PHYS_ENCODING_POLICY


PHYS_ENCODING_SHA256 = hashlib.sha256(PHYS_ENCODING_POLICY.encode("ascii")).hexdigest()

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
    results: Path = PROJECT_ROOT / "runs"
    split_file: Path | None = None

    @property
    def dataset(self) -> Path:
        return self.data / "dataset.csv"

    @property
    def esm_cache(self) -> Path:
        return self.data / "esm2_t30_150m_mean.pt"

    @property
    def splits(self) -> Path:
        return self.split_file if self.split_file is not None else self.data / "splits.json"

    @property
    def aaindex(self) -> Path:
        return self.project / "resources" / "aaindex.csv"

@dataclass(frozen=True)
class ModelConfig:
    labels: tuple[str, ...] = LABELS
    phys_dim: int = 105
    esm_dim: int = 640
    model_dim: int = 256
    cnn_output_dim: int = 128
    cnn_channels: tuple[int, ...] = (128, 256, 128)
    cnn_kernels: tuple[int, ...] = (3, 5, 3)
    base_dropout: float = 0.30
    graph_dropout: float = 0.30
    graph_topk: int = 3
    graph_lambda: float = 0.10
    padding_length: int | None = None


@dataclass(frozen=True)
class LossConfig:
    prior_scale: float = 1.0
    pu_correction_beta: float = 0.0
    pu_correction_gamma: float = 0.75


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    folds: int = 5
    epochs: int = 100
    batch_size: int = 48
    learning_rate: float = 1e-4
    weight_decay: float = 5e-3
    gradient_clip_norm: float = 1.0
    threshold: float = 0.5
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
        payload["implementation_version"] = 6
        payload["phys_encoding_policy"] = PHYS_ENCODING_POLICY
        payload["phys_encoding_sha256"] = PHYS_ENCODING_SHA256
        payload["paths"] = {
            key: str(value) if value is not None else None
            for key, value in payload["paths"].items()
        }
        payload["experiment_name"] = self.name
        payload["checkpoint_selection"] = (
            "CV 0.3*F1@3 + 0.3*F1@5 + 0.4*Macro Recall@0.5; earliest tie"
        )
        return payload

    def validate(self) -> None:
        if len(self.model.labels) != 12:
            raise ValueError("ATfinder requires 12 labels")
        if not 1 <= self.model.graph_topk < len(self.model.labels):
            raise ValueError("graph_topk must be between 1 and 11")
        if self.model.graph_lambda <= 0.0:
            raise ValueError("graph_lambda must be positive")
        if self.model.padding_length is not None and (
            isinstance(self.model.padding_length, bool)
            or not isinstance(self.model.padding_length, int)
            or self.model.padding_length < 1
        ):
            raise ValueError("padding_length must be a positive integer")
        for value in (self.model.base_dropout, self.model.graph_dropout):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError("dropout probabilities must be in [0, 1)")
        if not math.isfinite(self.loss.prior_scale) or self.loss.prior_scale <= 0:
            raise ValueError("prior_scale must be finite and positive")
        if self.train.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.train.folds not in (1, 5):
            raise ValueError("folds must be 1 or 5")


def atfinder_config(
    *,
    data_dir: Path | None = None,
    result_root: Path | None = None,
    device: str | None = None,
    folds: int | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    graph_dropout: float | None = None,
    prior_scale: float | None = None,
    split_file: Path | None = None,
    snapshot: dict | None = None,
) -> ExperimentConfig:
    if snapshot is not None and snapshot.get("implementation_version") != 6:
        raise ValueError(
            "This checkpoint was trained under a different residue-encoding policy; "
            "use a checkpoint matching the current model and input configuration."
        )
    if snapshot is not None and (
        snapshot.get("phys_encoding_policy") != PHYS_ENCODING_POLICY
        or snapshot.get("phys_encoding_sha256") != PHYS_ENCODING_SHA256
    ):
        raise ValueError("checkpoint physicochemical encoding does not match the current model")
    saved = snapshot or {}
    cfg = ExperimentConfig(
        model=ModelConfig(**saved.get("model", {})),
        loss=LossConfig(**saved.get("loss", {})),
        train=TrainConfig(**saved.get("train", {})),
    )
    if snapshot is not None and cfg.model.padding_length is None:
        raise ValueError("checkpoint is missing its fixed padding length")
    if data_dir is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, data=Path(data_dir).resolve()))
    if split_file is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, split_file=Path(split_file).resolve()))
    if result_root is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, results=Path(result_root).resolve()))
    if device is not None:
        cfg = replace(cfg, train=replace(cfg.train, device=str(device)))
    if folds is not None:
        cfg = replace(cfg, train=replace(cfg.train, folds=int(folds)))
    if seed is not None:
        cfg = replace(cfg, train=replace(cfg.train, seed=int(seed)))
    if epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=int(epochs)))
    if graph_dropout is not None:
        cfg = replace(cfg, model=replace(cfg.model, graph_dropout=float(graph_dropout)))
    if prior_scale is not None:
        cfg = replace(cfg, loss=replace(cfg.loss, prior_scale=float(prior_scale)))
    cfg.validate()
    return cfg
