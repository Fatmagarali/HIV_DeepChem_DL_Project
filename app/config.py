"""Application configuration.

All constants are centralised here and can be overridden via environment
variables or a ``.env`` file (loaded automatically if ``python-dotenv``
is installed).

Example ``.env``::

    SEED=42
    BATCH_SIZE=128
    LEARNING_RATE=0.001
    MODEL_DIR=./models
    HOST=0.0.0.0
    PORT=8000
"""

from __future__ import annotations

import os

# Load .env file if present (requires python-dotenv)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


def _int(key: str, default: int) -> int:
    """Read an integer environment variable with a fallback.

    Args:
        key: Environment variable name.
        default: Default value if the variable is unset or invalid.

    Returns:
        Parsed integer value.
    """
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _float(key: str, default: float) -> float:
    """Read a float environment variable with a fallback.

    Args:
        key: Environment variable name.
        default: Default value if the variable is unset or invalid.

    Returns:
        Parsed float value.
    """
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _str(key: str, default: str) -> str:
    """Read a string environment variable with a fallback.

    Args:
        key: Environment variable name.
        default: Default value if the variable is unset.

    Returns:
        String value.
    """
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED: int = _int("SEED", 42)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

BATCH_SIZE: int = _int("BATCH_SIZE", 128)
LEARNING_RATE: float = _float("LEARNING_RATE", 1e-3)
EPOCHS: int = _int("EPOCHS", 30)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR: str = _str("MODEL_DIR", "./models")
DATA_DIR: str = _str("DATA_DIR", "./data")

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------

HOST: str = _str("HOST", "0.0.0.0")
PORT: int = _int("PORT", 8000)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

THRESHOLD: float = _float("THRESHOLD", 0.5)

# Model descriptions (used by the /models endpoint)
MODEL_DESCRIPTIONS: dict = {
    "random_forest": {
        "name": "Random Forest",
        "description": "Baseline — ECFP4 (1024-bit CircularFingerprint, radius=2) + sklearn RandomForestClassifier.",
        "checkpoint": "random_forest/random_forest.pkl",
    },
    "graphconv": {
        "name": "GraphConv",
        "description": "Spatial Graph Convolutional Network (2 layers × 128 units) via DeepChem.",
        "checkpoint": "graphconv/",
    },
    "attentivefp": {
        "name": "AttentiveFP",
        "description": "Graph + Attention mechanism (3 GNN layers, 2 timesteps) via DeepChem + DGL.",
        "checkpoint": "attentivefp/",
    },
}
