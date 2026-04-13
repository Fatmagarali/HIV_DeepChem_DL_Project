"""Training script for HIV activity prediction models.

Usage::

    python -m src.train --model random_forest --epochs 30
    python -m src.train --model graphconv --epochs 50 --learning-rate 1e-3
    python -m src.train --model attentivefp --epochs 50 --batch-size 64
    python -m src.train --model all --epochs 30

Checkpoints are saved to ``./models/`` by default.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging setup (before any imports that may trigger their own loggers)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: silence TF / DeepChem verbosity early
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_graph_datasets(reload: bool = False) -> Tuple:
    """Load HIV dataset with MolGraphConv featurizer (scaffold split).

    Args:
        reload: Whether to reload a cached featurization.

    Returns:
        Tuple (train_dataset, valid_dataset, test_dataset).
    """
    import deepchem as dc

    logger.info("Loading HIV dataset (graph featurizer) …")
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    _, datasets, _ = dc.molnet.load_hiv(
        featurizer=featurizer, splitter="scaffold", reload=reload
    )
    return datasets  # (train, valid, test)


def _load_ecfp_datasets(graph_datasets: Tuple) -> Tuple:
    """Build ECFP4 datasets re-using scaffold splits from graph_datasets.

    Args:
        graph_datasets: (train, valid, test) from ``_load_graph_datasets``.

    Returns:
        Tuple (train_ecfp, valid_ecfp, test_ecfp).
    """
    import deepchem as dc

    featurizer = dc.feat.CircularFingerprint(size=1024, radius=2)

    def _convert(ds: dc.data.Dataset, name: str) -> dc.data.NumpyDataset:
        smiles = ds.ids
        logger.info("  ECFP featurization — %s: %d molecules", name, len(smiles))
        X = featurizer.featurize(smiles)
        return dc.data.NumpyDataset(X=X, y=ds.y, w=ds.w, ids=smiles)

    train, valid, test = graph_datasets
    return _convert(train, "train"), _convert(valid, "valid"), _convert(test, "test")


def _load_convmol_datasets(graph_datasets: Tuple) -> Tuple:
    """Build ConvMol datasets for GraphConvModel.

    Args:
        graph_datasets: (train, valid, test) from ``_load_graph_datasets``.

    Returns:
        Tuple (train_convmol, valid_convmol, test_convmol).
    """
    import deepchem as dc

    featurizer = dc.feat.ConvMolFeaturizer()

    def _convert(ds: dc.data.Dataset, name: str) -> dc.data.NumpyDataset:
        smiles = ds.ids
        logger.info("  ConvMol featurization — %s: %d molecules", name, len(smiles))
        X = featurizer.featurize(smiles)
        return dc.data.NumpyDataset(X=X, y=ds.y, w=ds.w, ids=smiles)

    train, valid, test = graph_datasets
    return _convert(train, "train"), _convert(valid, "valid"), _convert(test, "test")


# ---------------------------------------------------------------------------
# Individual train functions
# ---------------------------------------------------------------------------

def train_random_forest(
    model_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    graph_datasets: Optional[Tuple] = None,
) -> Dict:
    """Train a Random Forest model and save the checkpoint.

    Args:
        model_dir: Root directory for model checkpoints.
        epochs: Ignored for Random Forest.
        batch_size: Ignored for Random Forest.
        learning_rate: Ignored for Random Forest.
        seed: Random seed.
        graph_datasets: Pre-loaded graph datasets (avoids reloading).

    Returns:
        Dict with evaluation metrics.
    """
    from src.models import RandomForestHIV

    if graph_datasets is None:
        graph_datasets = _load_graph_datasets()
    ecfp_train, ecfp_valid, ecfp_test = _load_ecfp_datasets(graph_datasets)

    rf = RandomForestHIV(model_dir=os.path.join(model_dir, "random_forest"), seed=seed)
    metrics = rf.train(ecfp_train, ecfp_valid, epochs=epochs)

    ckpt_dir = os.path.join(model_dir, "random_forest")
    os.makedirs(ckpt_dir, exist_ok=True)
    rf.save()

    logger.info("Random Forest — valid ROC-AUC: %.4f", metrics["valid_roc_auc"])
    return metrics


def train_graphconv(
    model_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    graph_datasets: Optional[Tuple] = None,
) -> Dict:
    """Train the GraphConv model and save the checkpoint.

    Args:
        model_dir: Root directory for model checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        seed: Random seed.
        graph_datasets: Pre-loaded graph datasets (avoids reloading).

    Returns:
        Dict with training history.
    """
    from src.models import GraphConvHIV

    if graph_datasets is None:
        graph_datasets = _load_graph_datasets()
    convmol_train, convmol_valid, _ = _load_convmol_datasets(graph_datasets)

    gc_dir = os.path.join(model_dir, "graphconv")
    gc = GraphConvHIV(
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_dir=gc_dir,
        seed=seed,
    )
    metrics = gc.train(convmol_train, convmol_valid, epochs=epochs)
    gc.save()

    best_valid = max(metrics["valid_history"]) if metrics["valid_history"] else float("nan")
    logger.info("GraphConv — best valid ROC-AUC: %.4f", best_valid)
    return metrics


def train_attentivefp(
    model_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    graph_datasets: Optional[Tuple] = None,
) -> Dict:
    """Train the AttentiveFP model and save the checkpoint.

    Args:
        model_dir: Root directory for model checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        seed: Random seed.
        graph_datasets: Pre-loaded graph datasets (avoids reloading).

    Returns:
        Dict with training history, or empty dict if DGL unavailable.
    """
    from src.models import AttentiveFPHIV

    afp_dir = os.path.join(model_dir, "attentivefp")
    afp = AttentiveFPHIV(
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_dir=afp_dir,
        seed=seed,
    )

    if not afp.available:
        logger.warning(
            "AttentiveFP skipped — DGL/DGLLife not installed. "
            "Run: pip install dgl dgllife"
        )
        return {}

    if graph_datasets is None:
        graph_datasets = _load_graph_datasets()
    train_ds, valid_ds, _ = graph_datasets

    metrics = afp.train(train_ds, valid_ds, epochs=epochs)
    afp.save()

    best_valid = max(metrics["valid_history"]) if metrics["valid_history"] else float("nan")
    logger.info("AttentiveFP — best valid ROC-AUC: %.4f", best_valid)
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HIV activity prediction models."
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "graphconv", "attentivefp", "all"],
        default="all",
        help="Which model(s) to train (default: all).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for GCN models (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        dest="batch_size",
        help="Batch size (default: 128).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--model-dir",
        default="./models",
        dest="model_dir",
        help="Root directory for checkpoints (default: ./models).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the training CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv).
    """
    args = _parse_args(argv)

    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
    except ImportError:
        pass

    os.makedirs(args.model_dir, exist_ok=True)

    models_to_train: List[str] = (
        ["random_forest", "graphconv", "attentivefp"]
        if args.model == "all"
        else [args.model]
    )

    # Pre-load graph datasets once when training multiple models
    graph_datasets = None
    if len(models_to_train) > 1 or models_to_train[0] in ("graphconv", "attentivefp"):
        graph_datasets = _load_graph_datasets()

    overall_start = time.time()

    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info("Starting training: %s", model_name)
        logger.info("=" * 60)
        t0 = time.time()

        try:
            if model_name == "random_forest":
                train_random_forest(
                    model_dir=args.model_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    graph_datasets=graph_datasets,
                )
            elif model_name == "graphconv":
                train_graphconv(
                    model_dir=args.model_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    graph_datasets=graph_datasets,
                )
            elif model_name == "attentivefp":
                train_attentivefp(
                    model_dir=args.model_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    graph_datasets=graph_datasets,
                )
        except Exception:
            logger.exception("Training failed for model: %s", model_name)
            sys.exit(1)

        elapsed = time.time() - t0
        logger.info("Finished %s in %.1f s", model_name, elapsed)

    total = time.time() - overall_start
    logger.info("All models trained in %.1f s. Checkpoints in: %s", total, args.model_dir)


if __name__ == "__main__":
    main()
