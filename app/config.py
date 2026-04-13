"""Application settings loaded from environment variables.

The project keeps all runtime paths and hyperparameters in this module so the
training CLI, FastAPI application, and notebook refactor share the same source
of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for the project."""

    project_root: Path
    models_dir: Path
    artifacts_dir: Path
    seed: int
    default_model: str
    device: str
    api_host: str
    api_port: int
    training_epochs: int
    batch_size: int
    learning_rate: float
    rf_estimators: int
    rf_min_samples_leaf: int
    graph_conv_layers: tuple[int, int]
    graph_dense_layer_size: int
    attentivefp_layers: int
    attentivefp_timesteps: int
    attentivefp_graph_feat_size: int
    ecfp_size: int
    ecfp_radius: int

    def model_dir_for(self, model_name: str) -> Path:
        """Return the on-disk directory used to persist a given model."""

        normalized = model_name.strip().lower()
        mapping = {
            "random_forest": self.models_dir / "random_forest",
            "rf": self.models_dir / "random_forest",
            "graphconv": self.models_dir / "graphconv",
            "gc": self.models_dir / "graphconv",
            "attentivefp": self.models_dir / "attentivefp",
            "afp": self.models_dir / "attentivefp",
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown model name: {model_name}")
        return mapping[normalized]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build a cached :class:`Settings` instance from environment variables."""

    project_root = Path(__file__).resolve().parents[1]
    models_dir = (project_root / _env_str("HIV_MODELS_DIR", "models")).resolve()
    artifacts_dir = (project_root / _env_str("HIV_ARTIFACTS_DIR", "artifacts")).resolve()

    return Settings(
        project_root=project_root,
        models_dir=models_dir,
        artifacts_dir=artifacts_dir,
        seed=_env_int("HIV_SEED", 42),
        default_model=_env_str("HIV_DEFAULT_MODEL", "graphconv"),
        device=_env_str("HIV_DEVICE", "cpu"),
        api_host=_env_str("HIV_API_HOST", "0.0.0.0"),
        api_port=_env_int("HIV_API_PORT", 8000),
        training_epochs=_env_int("HIV_EPOCHS", 30),
        batch_size=_env_int("HIV_BATCH_SIZE", 128),
        learning_rate=_env_float("HIV_LEARNING_RATE", 1e-3),
        rf_estimators=_env_int("HIV_RF_ESTIMATORS", 500),
        rf_min_samples_leaf=_env_int("HIV_RF_MIN_SAMPLES_LEAF", 2),
        graph_conv_layers=(128, 128),
        graph_dense_layer_size=256,
        attentivefp_layers=3,
        attentivefp_timesteps=2,
        attentivefp_graph_feat_size=200,
        ecfp_size=1024,
        ecfp_radius=2,
    )
