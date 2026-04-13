"""Model wrappers for HIV activity prediction."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.config import Settings, get_settings

from .utils import compute_binary_metrics, ensure_directory, extract_positive_class_probabilities, patch_deepchem_batchnorm, safe_average_precision, safe_roc_auc


MODEL_ALIASES = {
    "rf": "random_forest",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "gc": "graphconv",
    "graphconv": "graphconv",
    "afp": "attentivefp",
    "attentivefp": "attentivefp",
}

AVAILABLE_MODELS = ("random_forest", "graphconv", "attentivefp")


def canonical_model_name(model_name: str) -> str:
    """Normalize a model identifier to a canonical registry name."""

    normalized = model_name.strip().lower()
    if normalized not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model name: {model_name}")
    return MODEL_ALIASES[normalized]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_labels(dataset: Any) -> np.ndarray:
    labels = np.asarray(dataset.y).reshape(-1)
    return labels.astype(int)


def _compute_class_imbalance_ratio(dataset: Any) -> float:
    labels = _flatten_labels(dataset)
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Summary returned after a model has been trained."""

    model_name: str
    metrics: dict[str, float]
    train_history: list[float] = field(default_factory=list)
    valid_history: list[float] = field(default_factory=list)
    checkpoint_dir: Path | None = None


class BaseHIVModel(ABC):
    """Abstract base class shared by all HIV model wrappers."""

    model_name: ClassVar[str]

    @abstractmethod
    def fit(self, train_dataset: Any, valid_dataset: Any, *, epochs: int = 30) -> TrainingResult:
        """Train the model and return the associated metrics."""

    @abstractmethod
    def predict_proba(self, dataset: Any) -> np.ndarray:
        """Return positive-class probabilities for a dataset."""

    @abstractmethod
    def save(self) -> None:
        """Persist the trained model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path, *, settings: Settings | None = None) -> "BaseHIVModel":
        """Restore a model from disk."""


@dataclass(slots=True)
class RandomForestHIV(BaseHIVModel):
    """Random Forest baseline trained on ECFP fingerprints."""

    model_name: ClassVar[str] = "random_forest"

    model_dir: Path
    seed: int
    n_estimators: int = 500
    min_samples_leaf: int = 2
    max_depth: int | None = None
    class_weight: str | dict[str, float] | None = "balanced"
    n_jobs: int = -1
    model: RandomForestClassifier | None = field(default=None, init=False, repr=False)

    @property
    def artifact_path(self) -> Path:
        return self.model_dir / "model.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"

    def fit(self, train_dataset: Any, valid_dataset: Any, *, epochs: int = 1) -> TrainingResult:
        """Fit the Random Forest and evaluate it on train and validation splits."""

        del epochs
        ensure_directory(self.model_dir)
        X_train = np.asarray(train_dataset.X)
        y_train = _flatten_labels(train_dataset)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        self.model.fit(X_train, y_train)
        self.save()

        train_proba = self.predict_proba(train_dataset)
        valid_proba = self.predict_proba(valid_dataset)
        metrics = {
            "train_roc_auc": safe_roc_auc(y_train, train_proba),
            "train_average_precision": safe_average_precision(y_train, train_proba),
            "valid_roc_auc": safe_roc_auc(_flatten_labels(valid_dataset), valid_proba),
            "valid_average_precision": safe_average_precision(_flatten_labels(valid_dataset), valid_proba),
        }
        return TrainingResult(model_name=self.model_name, metrics=metrics, checkpoint_dir=self.model_dir)

    def predict_proba(self, dataset: Any) -> np.ndarray:
        """Predict positive-class probabilities for a featurized dataset."""

        if self.model is None:
            raise RuntimeError("RandomForestHIV is not fitted or loaded")
        probabilities = self.model.predict_proba(np.asarray(dataset.X))
        return np.asarray(probabilities)[:, 1].astype(float)

    def save(self) -> None:
        """Persist the Random Forest and its metadata."""

        if self.model is None:
            raise RuntimeError("RandomForestHIV cannot be saved before fitting")
        ensure_directory(self.model_dir)
        joblib.dump(self.model, self.artifact_path)
        _json_dump(
            self.metadata_path,
            {
                "model_name": self.model_name,
                "seed": self.seed,
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
                "class_weight": self.class_weight,
                "n_jobs": self.n_jobs,
            },
        )

    @classmethod
    def load(cls, model_dir: Path, *, settings: Settings | None = None) -> "RandomForestHIV":
        """Restore a previously trained Random Forest from disk."""

        settings = settings or get_settings()
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        metadata = _json_load(metadata_path)
        instance = cls(
            model_dir=model_dir,
            seed=int(metadata.get("seed", settings.seed)),
            n_estimators=int(metadata.get("n_estimators", settings.rf_estimators)),
            min_samples_leaf=int(metadata.get("min_samples_leaf", settings.rf_min_samples_leaf)),
            max_depth=metadata.get("max_depth"),
            class_weight=metadata.get("class_weight", "balanced"),
            n_jobs=int(metadata.get("n_jobs", -1)),
        )
        if not instance.artifact_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {instance.artifact_path}")
        instance.model = joblib.load(instance.artifact_path)
        return instance


@dataclass(slots=True)
class GraphConvHIV(BaseHIVModel):
    """DeepChem GraphConv model trained on ConvMol features."""

    model_name: ClassVar[str] = "graphconv"

    model_dir: Path
    seed: int
    device: str = "cpu"
    graph_conv_layers: tuple[int, int] = (128, 128)
    dense_layer_size: int = 256
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 128
    class_imbalance_ratio: float | None = None
    model: Any | None = field(default=None, init=False, repr=False)

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"

    def _build_model(self) -> Any:
        from deepchem.models import GraphConvModel

        patch_deepchem_batchnorm()
        return GraphConvModel(
            n_tasks=1,
            mode="classification",
            graph_conv_layers=list(self.graph_conv_layers),
            dense_layer_size=self.dense_layer_size,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            class_imbalance_ratio=self.class_imbalance_ratio,
            device=self.device,
            model_dir=str(self.model_dir),
        )

    def fit(self, train_dataset: Any, valid_dataset: Any, *, epochs: int = 30) -> TrainingResult:
        """Fit the GraphConv model and keep the best checkpoint on disk."""

        ensure_directory(self.model_dir)
        if self.class_imbalance_ratio is None:
            self.class_imbalance_ratio = _compute_class_imbalance_ratio(train_dataset)
        self.model = self._build_model()

        train_history: list[float] = []
        valid_history: list[float] = []
        best_valid = float("-inf")

        for _ in range(epochs):
            self.model.fit(train_dataset, nb_epoch=1, deterministic=True)
            train_metrics = self.evaluate(train_dataset)
            valid_metrics = self.evaluate(valid_dataset)
            train_history.append(train_metrics["roc_auc"])
            valid_history.append(valid_metrics["roc_auc"])
            if valid_metrics["roc_auc"] > best_valid:
                best_valid = valid_metrics["roc_auc"]
                self.model.save_checkpoint(model_dir=str(self.model_dir))

        self.model.restore(model_dir=str(self.model_dir))
        self.save()

        final_train = self.evaluate(train_dataset)
        final_valid = self.evaluate(valid_dataset)
        metrics = {
            "train_roc_auc": final_train["roc_auc"],
            "train_average_precision": final_train["average_precision"],
            "valid_roc_auc": final_valid["roc_auc"],
            "valid_average_precision": final_valid["average_precision"],
        }
        return TrainingResult(model_name=self.model_name, metrics=metrics, train_history=train_history, valid_history=valid_history, checkpoint_dir=self.model_dir)

    def predict_proba(self, dataset: Any) -> np.ndarray:
        """Predict positive-class probabilities for ConvMol inputs."""

        if self.model is None:
            raise RuntimeError("GraphConvHIV is not fitted or loaded")
        predictions = self.model.predict(dataset)
        return extract_positive_class_probabilities(predictions)

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Evaluate ROC-AUC and average precision on a dataset."""

        labels = _flatten_labels(dataset)
        probabilities = self.predict_proba(dataset)
        return {
            "roc_auc": safe_roc_auc(labels, probabilities),
            "average_precision": safe_average_precision(labels, probabilities),
        }

    def save(self) -> None:
        """Persist training metadata for later restoration."""

        ensure_directory(self.model_dir)
        _json_dump(
            self.metadata_path,
            {
                "model_name": self.model_name,
                "seed": self.seed,
                "device": self.device,
                "graph_conv_layers": list(self.graph_conv_layers),
                "dense_layer_size": self.dense_layer_size,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "class_imbalance_ratio": self.class_imbalance_ratio,
            },
        )

    @classmethod
    def load(cls, model_dir: Path, *, settings: Settings | None = None) -> "GraphConvHIV":
        """Restore a GraphConv model from checkpoint metadata."""

        settings = settings or get_settings()
        metadata_path = model_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = _json_load(metadata_path)

        stored_layers = tuple(metadata.get("graph_conv_layers", list(settings.graph_conv_layers)))
        if len(stored_layers) != 2:
            stored_layers = settings.graph_conv_layers

        instance = cls(
            model_dir=model_dir,
            seed=int(metadata.get("seed", settings.seed)),
            device=str(metadata.get("device", settings.device)),
            graph_conv_layers=(int(stored_layers[0]), int(stored_layers[1])),
            dense_layer_size=int(metadata.get("dense_layer_size", settings.graph_dense_layer_size)),
            dropout=float(metadata.get("dropout", 0.2)),
            learning_rate=float(metadata.get("learning_rate", settings.learning_rate)),
            batch_size=int(metadata.get("batch_size", settings.batch_size)),
            class_imbalance_ratio=metadata.get("class_imbalance_ratio"),
        )
        instance.model = instance._build_model()
        instance.model.restore(model_dir=str(model_dir))
        return instance


@dataclass(slots=True)
class AttentiveFPHIV(BaseHIVModel):
    """DeepChem AttentiveFP model trained on graph featurizations."""

    model_name: ClassVar[str] = "attentivefp"

    model_dir: Path
    seed: int
    device: str = "cpu"
    num_layers: int = 3
    num_timesteps: int = 2
    graph_feat_size: int = 200
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 128
    model: Any | None = field(default=None, init=False, repr=False)

    @staticmethod
    def is_available() -> bool:
        """Return True when the optional DGL dependencies are installed."""

        try:
            import importlib

            importlib.import_module("dgl")
            importlib.import_module("dgllife")
        except Exception:
            return False
        return True

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"

    def _build_model(self) -> Any:
        if not self.is_available():
            raise ImportError("AttentiveFP requires optional dependencies: dgl and dgllife")

        from deepchem.models import AttentiveFPModel

        return AttentiveFPModel(
            n_tasks=1,
            mode="classification",
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            graph_feat_size=self.graph_feat_size,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            model_dir=str(self.model_dir),
        )

    def fit(self, train_dataset: Any, valid_dataset: Any, *, epochs: int = 30) -> TrainingResult:
        """Fit the AttentiveFP model and save the best checkpoint."""

        if not self.is_available():
            raise ImportError("AttentiveFP requires optional dependencies: dgl and dgllife")

        ensure_directory(self.model_dir)
        self.model = self._build_model()

        train_history: list[float] = []
        valid_history: list[float] = []
        best_valid = float("-inf")

        for _ in range(epochs):
            self.model.fit(train_dataset, nb_epoch=1, deterministic=True)
            train_metrics = self.evaluate(train_dataset)
            valid_metrics = self.evaluate(valid_dataset)
            train_history.append(train_metrics["roc_auc"])
            valid_history.append(valid_metrics["roc_auc"])
            if valid_metrics["roc_auc"] > best_valid:
                best_valid = valid_metrics["roc_auc"]
                self.model.save_checkpoint(model_dir=str(self.model_dir))

        self.model.restore(model_dir=str(self.model_dir))
        self.save()

        final_train = self.evaluate(train_dataset)
        final_valid = self.evaluate(valid_dataset)
        metrics = {
            "train_roc_auc": final_train["roc_auc"],
            "train_average_precision": final_train["average_precision"],
            "valid_roc_auc": final_valid["roc_auc"],
            "valid_average_precision": final_valid["average_precision"],
        }
        return TrainingResult(model_name=self.model_name, metrics=metrics, train_history=train_history, valid_history=valid_history, checkpoint_dir=self.model_dir)

    def predict_proba(self, dataset: Any) -> np.ndarray:
        """Predict positive-class probabilities for graph inputs."""

        if self.model is None:
            raise RuntimeError("AttentiveFPHIV is not fitted or loaded")
        predictions = self.model.predict(dataset)
        return extract_positive_class_probabilities(predictions)

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Evaluate ROC-AUC and average precision on a dataset."""

        labels = _flatten_labels(dataset)
        probabilities = self.predict_proba(dataset)
        return {
            "roc_auc": safe_roc_auc(labels, probabilities),
            "average_precision": safe_average_precision(labels, probabilities),
        }

    def save(self) -> None:
        """Persist training metadata for later restoration."""

        ensure_directory(self.model_dir)
        _json_dump(
            self.metadata_path,
            {
                "model_name": self.model_name,
                "seed": self.seed,
                "device": self.device,
                "num_layers": self.num_layers,
                "num_timesteps": self.num_timesteps,
                "graph_feat_size": self.graph_feat_size,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
            },
        )

    @classmethod
    def load(cls, model_dir: Path, *, settings: Settings | None = None) -> "AttentiveFPHIV":
        """Restore an AttentiveFP model from checkpoint metadata."""

        settings = settings or get_settings()
        metadata_path = model_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = _json_load(metadata_path)

        instance = cls(
            model_dir=model_dir,
            seed=int(metadata.get("seed", settings.seed)),
            device=str(metadata.get("device", settings.device)),
            num_layers=int(metadata.get("num_layers", settings.attentivefp_layers)),
            num_timesteps=int(metadata.get("num_timesteps", settings.attentivefp_timesteps)),
            graph_feat_size=int(metadata.get("graph_feat_size", settings.attentivefp_graph_feat_size)),
            dropout=float(metadata.get("dropout", 0.2)),
            learning_rate=float(metadata.get("learning_rate", settings.learning_rate)),
            batch_size=int(metadata.get("batch_size", settings.batch_size)),
        )
        instance.model = instance._build_model()
        instance.model.restore(model_dir=str(model_dir))
        return instance


def create_model(model_name: str, *, settings: Settings | None = None) -> BaseHIVModel:
    """Create a model wrapper configured from application settings."""

    settings = settings or get_settings()
    canonical_name = canonical_model_name(model_name)
    model_dir = settings.model_dir_for(canonical_name)
    if canonical_name == "random_forest":
        return RandomForestHIV(
            model_dir=model_dir,
            seed=settings.seed,
            n_estimators=settings.rf_estimators,
            min_samples_leaf=settings.rf_min_samples_leaf,
        )
    if canonical_name == "graphconv":
        return GraphConvHIV(
            model_dir=model_dir,
            seed=settings.seed,
            device=settings.device,
            graph_conv_layers=settings.graph_conv_layers,
            dense_layer_size=settings.graph_dense_layer_size,
            learning_rate=settings.learning_rate,
            batch_size=settings.batch_size,
        )
    if canonical_name == "attentivefp":
        return AttentiveFPHIV(
            model_dir=model_dir,
            seed=settings.seed,
            device=settings.device,
            num_layers=settings.attentivefp_layers,
            num_timesteps=settings.attentivefp_timesteps,
            graph_feat_size=settings.attentivefp_graph_feat_size,
            learning_rate=settings.learning_rate,
            batch_size=settings.batch_size,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def load_trained_model(model_name: str, *, settings: Settings | None = None) -> BaseHIVModel:
    """Load a trained model from its configured checkpoint directory."""

    settings = settings or get_settings()
    canonical_name = canonical_model_name(model_name)
    model_dir = settings.model_dir_for(canonical_name)
    if canonical_name == "random_forest":
        return RandomForestHIV.load(model_dir, settings=settings)
    if canonical_name == "graphconv":
        return GraphConvHIV.load(model_dir, settings=settings)
    if canonical_name == "attentivefp":
        return AttentiveFPHIV.load(model_dir, settings=settings)
    raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(model: BaseHIVModel, dataset: Any) -> dict[str, float]:
    """Evaluate a model on a dataset and return standard binary metrics."""

    labels = _flatten_labels(dataset)
    probabilities = model.predict_proba(dataset)
    return compute_binary_metrics(labels, probabilities)
