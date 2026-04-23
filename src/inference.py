"""Inference helpers for saved HIV models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.config import Settings, get_settings

from .featurizers import featurize_smiles_for_model
from .models import AVAILABLE_MODELS, canonical_model_name, load_trained_model
from .utils import ensure_directory


@dataclass(frozen=True)
class ModelPrediction:
    """Single-molecule prediction returned by the inference layer."""

    smiles: str
    probability: float
    label: int


def _normalize_smiles(smiles: Sequence[str] | str) -> list[str]:
    if isinstance(smiles, str):
        values = [smiles]
    else:
        values = [str(item).strip() for item in smiles]
    values = [item for item in values if item]
    if not values:
        raise ValueError("At least one valid SMILES string is required")
    return values


def _build_prediction_dataset(model_name: str, smiles: Sequence[str], *, settings: Settings) -> Any:
    return featurize_smiles_for_model(smiles, model_name, ecfp_size=settings.ecfp_size, ecfp_radius=settings.ecfp_radius)


def predict_smiles(
    model_name: str,
    smiles: Sequence[str] | str,
    *,
    threshold: float = 0.5,
    settings: Settings | None = None,
) -> list[ModelPrediction]:
    """Predict activity probabilities for one or more SMILES strings."""

    settings = settings or get_settings()
    normalized_smiles = _normalize_smiles(smiles)
    canonical_name = canonical_model_name(model_name)
    model = load_trained_model(canonical_name, settings=settings)
    dataset = _build_prediction_dataset(canonical_name, normalized_smiles, settings=settings)
    probabilities = np.asarray(model.predict_proba(dataset)).reshape(-1)
    labels = (probabilities >= threshold).astype(int)

    if len(probabilities) != len(normalized_smiles):
        raise RuntimeError("Prediction length does not match the number of SMILES inputs")

    return [
        ModelPrediction(smiles=smiles_value, probability=float(probability), label=int(label))
        for smiles_value, probability, label in zip(normalized_smiles, probabilities, labels)
    ]


def _has_checkpoints(model_dir: Any) -> bool:
    """Return True if any checkpoint files exist in the directory."""
    return (
        any(model_dir.glob("checkpoint*"))
        or any(model_dir.glob("*.pt"))
        or any(model_dir.glob("*.ckpt"))
    )


def list_model_statuses(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Report whether trained artifacts are available for each supported model."""

    settings = settings or get_settings()
    statuses: list[dict[str, Any]] = []
    for model_name in AVAILABLE_MODELS:
        model_dir = settings.model_dir_for(model_name)
        if model_name == "random_forest":
            available = (model_dir / "model.joblib").exists()
        else:
            available = _has_checkpoints(model_dir)
        statuses.append(
            {
                "name": model_name,
                "path": str(model_dir),
                "available": available,
                "metadata": str(model_dir / "metadata.json"),
            }
        )
    return statuses


def ensure_runtime_directories(settings: Settings | None = None) -> None:
    """Create the model and artifact directories required by the application."""

    settings = settings or get_settings()
    ensure_directory(settings.models_dir)
    ensure_directory(settings.artifacts_dir)