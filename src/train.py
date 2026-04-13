"""Command-line entry point for training HIV models."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from app.config import Settings, get_settings

from .featurizers import HIVSplits, load_all_feature_views
from .inference import ensure_runtime_directories
from .models import AttentiveFPHIV, GraphConvHIV, RandomForestHIV, canonical_model_name
from .utils import compute_binary_metrics, ensure_directory, plot_learning_curves, plot_roc_pr_curves, set_global_seed


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by the training script."""

    parser = argparse.ArgumentParser(description="Train HIV activity prediction models")
    parser.add_argument("--model", default="all", choices=["all", "rf", "random_forest", "graphconv", "gc", "attentivefp", "afp"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs for graph models")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument("--reload-data", action="store_true", help="Force a reload of the HIV dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where reports and plots are saved")
    parser.add_argument("--skip-plots", action="store_true", help="Skip saving figures")
    return parser


def _model_display_name(model_name: str) -> str:
    canonical = canonical_model_name(model_name)
    if canonical == "random_forest":
        return "Random Forest"
    if canonical == "graphconv":
        return "GraphConv"
    if canonical == "attentivefp":
        return "AttentiveFP"
    return canonical


def _train_random_forest(settings: Settings, views: dict[str, HIVSplits]) -> tuple[RandomForestHIV, dict[str, float], np.ndarray]:
    model = RandomForestHIV(
        model_dir=settings.model_dir_for("random_forest"),
        seed=settings.seed,
        n_estimators=settings.rf_estimators,
        min_samples_leaf=settings.rf_min_samples_leaf,
    )
    result = model.fit(views["ecfp"].train, views["ecfp"].valid)
    test_proba = model.predict_proba(views["ecfp"].test)
    test_metrics = compute_binary_metrics(views["ecfp"].test.y, test_proba)
    metrics = {**result.metrics, **{f"test_{key}": value for key, value in test_metrics.items()}}
    return model, metrics, test_proba


def _train_graphconv(settings: Settings, views: dict[str, HIVSplits], epochs: int) -> tuple[GraphConvHIV, dict[str, float], np.ndarray, list[float], list[float]]:
    model = GraphConvHIV(
        model_dir=settings.model_dir_for("graphconv"),
        seed=settings.seed,
        device=settings.device,
        graph_conv_layers=settings.graph_conv_layers,
        dense_layer_size=settings.graph_dense_layer_size,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
    )
    result = model.fit(views["convmol"].train, views["convmol"].valid, epochs=epochs)
    test_proba = model.predict_proba(views["convmol"].test)
    test_metrics = compute_binary_metrics(views["convmol"].test.y, test_proba)
    metrics = {**result.metrics, **{f"test_{key}": value for key, value in test_metrics.items()}}
    return model, metrics, test_proba, result.train_history, result.valid_history


def _train_attentivefp(settings: Settings, views: dict[str, HIVSplits], epochs: int) -> tuple[AttentiveFPHIV | None, dict[str, float], np.ndarray | None, list[float], list[float]]:
    if not AttentiveFPHIV.is_available():
        return None, {"available": 0.0}, None, [], []

    model = AttentiveFPHIV(
        model_dir=settings.model_dir_for("attentivefp"),
        seed=settings.seed,
        device=settings.device,
        num_layers=settings.attentivefp_layers,
        num_timesteps=settings.attentivefp_timesteps,
        graph_feat_size=settings.attentivefp_graph_feat_size,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
    )
    result = model.fit(views["graph"].train, views["graph"].valid, epochs=epochs)
    test_proba = model.predict_proba(views["graph"].test)
    test_metrics = compute_binary_metrics(views["graph"].test.y, test_proba)
    metrics = {**result.metrics, **{f"test_{key}": value for key, value in test_metrics.items()}}
    return model, metrics, test_proba, result.train_history, result.valid_history


def _build_summary(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["Modèle", "ROC-AUC", "AUPRC"])

    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Modèle": model_name,
                "ROC-AUC": round(metrics.get("test_roc_auc", float("nan")), 4),
                "AUPRC": round(metrics.get("test_average_precision", float("nan")), 4),
            }
        )
    summary = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    summary.index = summary.index + 1
    return summary


def _training_settings(settings: Settings, seed: int | None) -> Settings:
    if seed is None:
        return settings
    return replace(settings, seed=seed)


def main(argv: Sequence[str] | None = None) -> int:
    """Train the requested model(s) and save metrics and plots."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    settings = _training_settings(get_settings(), args.seed)

    set_global_seed(settings.seed)
    ensure_runtime_directories(settings)
    output_dir = ensure_directory(args.output_dir or settings.artifacts_dir)
    plots_dir = ensure_directory(output_dir / "plots")

    epochs = args.epochs or settings.training_epochs
    views = load_all_feature_views(reload=args.reload_data, ecfp_size=settings.ecfp_size, ecfp_radius=settings.ecfp_radius)

    selected = {args.model}
    if args.model == "all":
        selected = {"random_forest", "graphconv", "attentivefp"}

    results: dict[str, dict[str, float]] = {}
    plot_payload: dict[str, dict[str, object]] = {}
    histories: dict[str, tuple[Sequence[float], Sequence[float], str]] = {}

    if selected & {"random_forest", "rf"} or args.model == "all":
        _, metrics, test_proba = _train_random_forest(settings, views)
        model_name = _model_display_name("random_forest")
        results[model_name] = metrics
        plot_payload[model_name] = {"proba": test_proba, "roc": metrics["test_roc_auc"], "prc": metrics["test_average_precision"], "color": "#888780", "ls": "--"}

    if selected & {"graphconv", "gc"} or args.model == "all":
        _, metrics, test_proba, train_history, valid_history = _train_graphconv(settings, views, epochs)
        model_name = _model_display_name("graphconv")
        results[model_name] = metrics
        plot_payload[model_name] = {"proba": test_proba, "roc": metrics["test_roc_auc"], "prc": metrics["test_average_precision"], "color": "#378ADD", "ls": "-"}
        histories[model_name] = (train_history, valid_history, "#378ADD")

    if selected & {"attentivefp", "afp"} or args.model == "all":
        attentivefp_model, metrics, test_proba, train_history, valid_history = _train_attentivefp(settings, views, epochs)
        if attentivefp_model is not None and test_proba is not None:
            model_name = _model_display_name("attentivefp")
            results[model_name] = metrics
            plot_payload[model_name] = {"proba": test_proba, "roc": metrics["test_roc_auc"], "prc": metrics["test_average_precision"], "color": "#1D9E75", "ls": "-"}
            histories[model_name] = (train_history, valid_history, "#1D9E75")
        else:
            print("AttentiveFP not trained because optional dependencies are missing.")

    summary = _build_summary(results)
    print("=" * 75)
    print("HIV Deep Learning results")
    print("=" * 75)
    if summary.empty:
        print("No models were trained.")
    else:
        print(summary.to_string(index=True))

    summary_path = output_dir / "metrics_summary.csv"
    summary.to_csv(summary_path)
    (output_dir / "metrics_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    if not args.skip_plots and results:
        y_true = views["graph"].test.y.reshape(-1)
        plot_roc_pr_curves(plot_payload, y_true, output_path=plots_dir / "roc_pr_curves.png")
        if histories:
            plot_learning_curves(histories, output_path=plots_dir / "learning_curves.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
