"""Shared utilities for reproducibility, metrics, and plotting."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return its resolved path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and common ML backends for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if hasattr(tf.keras.utils, "set_random_seed"):
            tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def patch_deepchem_batchnorm() -> None:
    """Patch TensorFlow BatchNormalization so older DeepChem code paths keep working."""

    try:
        from tensorflow.keras.layers import BatchNormalization
    except Exception:
        return

    if not hasattr(BatchNormalization, "_original_init_unpatched"):
        BatchNormalization._original_init_unpatched = BatchNormalization.__init__

    if getattr(BatchNormalization.__init__, "_deepchem_fused_patch", False):
        return

    original_init = BatchNormalization._original_init_unpatched

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("fused", None)
        return original_init(self, *args, **kwargs)

    _patched_init._deepchem_fused_patch = True
    BatchNormalization.__init__ = _patched_init


def extract_positive_class_probabilities(predictions: np.ndarray | Sequence[float]) -> np.ndarray:
    """Extract positive-class probabilities from DeepChem or sklearn outputs."""

    array = np.asarray(predictions)
    if array.ndim == 3 and array.shape[-1] >= 2:
        return array[:, 0, 1].astype(float)
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, 1].astype(float)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0].astype(float)
    return array.reshape(-1).astype(float)


def safe_roc_auc(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Compute ROC-AUC and return NaN when only one class is present."""

    try:
        return float(roc_auc_score(np.asarray(y_true).reshape(-1), np.asarray(y_score).reshape(-1)))
    except ValueError:
        return float("nan")


def safe_average_precision(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Compute average precision and return NaN when it cannot be evaluated."""

    try:
        return float(average_precision_score(np.asarray(y_true).reshape(-1), np.asarray(y_score).reshape(-1)))
    except ValueError:
        return float("nan")


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return standard binary classification metrics for a score vector."""

    y_true_array = np.asarray(y_true).reshape(-1).astype(int)
    y_score_array = np.asarray(y_score).reshape(-1).astype(float)
    y_pred = (y_score_array >= threshold).astype(int)

    tp = float(np.logical_and(y_true_array == 1, y_pred == 1).sum())
    tn = float(np.logical_and(y_true_array == 0, y_pred == 0).sum())
    fp = float(np.logical_and(y_true_array == 0, y_pred == 1).sum())
    fn = float(np.logical_and(y_true_array == 1, y_pred == 0).sum())

    precision = float(precision_score(y_true_array, y_pred, zero_division=0))
    recall = float(recall_score(y_true_array, y_pred, zero_division=0))

    return {
        "roc_auc": safe_roc_auc(y_true_array, y_score_array),
        "average_precision": safe_average_precision(y_true_array, y_score_array),
        "precision": precision,
        "recall": recall,
        "f1": float(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def plot_roc_pr_curves(
    results: Mapping[str, Mapping[str, Any]],
    y_true: Sequence[int] | np.ndarray,
    output_path: Path | None = None,
) -> Path | None:
    """Plot ROC and precision-recall curves for several model predictions."""

    y_true_array = np.asarray(y_true).reshape(-1).astype(int)
    baseline_pr = float(y_true_array.mean()) if len(y_true_array) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Random baseline")
    for name, payload in results.items():
        probabilities = np.asarray(payload["proba"]).reshape(-1)
        try:
            fpr, tpr, _ = roc_curve(y_true_array, probabilities)
        except ValueError:
            continue
        ax.plot(
            fpr,
            tpr,
            color=payload.get("color", "#1f77b4"),
            linestyle=payload.get("ls", "-"),
            linewidth=2,
            label=f"{name} (AUC={payload.get('roc', float('nan')):.4f})",
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    ax = axes[1]
    ax.axhline(y=baseline_pr, color="k", linestyle=":", alpha=0.4, label=f"Baseline (P={baseline_pr:.3f})")
    for name, payload in results.items():
        probabilities = np.asarray(payload["proba"]).reshape(-1)
        try:
            precision, recall, _ = precision_recall_curve(y_true_array, probabilities)
        except ValueError:
            continue
        ax.plot(
            recall,
            precision,
            color=payload.get("color", "#1f77b4"),
            linestyle=payload.get("ls", "-"),
            linewidth=2,
            label=f"{name} (AUPRC={payload.get('prc', float('nan')):.4f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    saved_path: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_path = output_path
    plt.close(fig)
    return saved_path


def plot_learning_curves(
    histories: Mapping[str, tuple[Sequence[float], Sequence[float], str]],
    output_path: Path | None = None,
) -> Path | None:
    """Plot training and validation ROC-AUC histories."""

    fig, axes = plt.subplots(1, len(histories), figsize=(6 * max(len(histories), 1), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (name, (train_history, valid_history, color)) in zip(axes, histories.items()):
        if len(train_history) == 0 or len(valid_history) == 0:
            ax.text(0.5, 0.5, f"{name}\nnot trained", ha="center", va="center")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(name)
            ax.grid(alpha=0.3)
            continue

        epochs = np.arange(1, len(train_history) + 1)
        valid_array = np.asarray(valid_history, dtype=float)
        if np.all(np.isnan(valid_array)):
            ax.text(0.5, 0.5, f"{name}\nmetrics unavailable", ha="center", va="center")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(name)
            ax.grid(alpha=0.3)
            continue
        best_epoch = int(np.nanargmax(valid_array) + 1)
        best_value = float(np.nanmax(valid_array))

        ax.plot(epochs, train_history, color=color, alpha=0.5, linestyle="--", label="Train")
        ax.plot(epochs, valid_history, color=color, linewidth=2, label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)
        ax.scatter([best_epoch], [best_value], color=color, s=60, zorder=5)
        ax.set_title(f"{name} (best valid={best_value:.4f} @ ep.{best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim([0.4, 1.0])

    fig.tight_layout()
    saved_path: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_path = output_path
    plt.close(fig)
    return saved_path
