"""Inference module for HIV activity prediction.

This module can be imported by other modules **or** run from the command line::

    python src/inference.py --smiles "CCO" "c1ccccc1" --model random_forest
    python src/inference.py --smiles-file molecules.txt --model graphconv

It loads pre-trained checkpoints from ``./models/`` (configurable via
``--model-dir`` or the ``MODEL_DIR`` environment variable).

No models are loaded automatically on import — call
:func:`load_model` or :func:`predict_smiles` explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMILES validation helper
# ---------------------------------------------------------------------------

def _validate_smiles(smiles_list: List[str]) -> List[str]:
    """Filter out invalid SMILES strings using RDKit (if available).

    Args:
        smiles_list: Raw input list.

    Returns:
        List of SMILES that RDKit could parse, or the original list if RDKit
        is not installed.
    """
    try:
        from rdkit import Chem

        valid = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                valid.append(smi.strip())
            else:
                logger.warning("Invalid SMILES (skipped): %s", smi)
        return valid
    except ImportError:
        logger.debug("RDKit not available — SMILES validation skipped.")
        return [s.strip() for s in smiles_list if s.strip()]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

# Map model name → class import path
_MODEL_REGISTRY = {
    "random_forest": "src.models.RandomForestHIV",
    "graphconv": "src.models.GraphConvHIV",
    "attentivefp": "src.models.AttentiveFPHIV",
}


def _import_model_class(class_path: str):
    """Dynamically import a model class by dotted path.

    Args:
        class_path: e.g. ``"src.models.RandomForestHIV"``.

    Returns:
        The model class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(model_name: str, model_dir: str = "./models"):
    """Instantiate and load a pre-trained model from disk.

    Args:
        model_name: One of ``random_forest``, ``graphconv``, ``attentivefp``.
        model_dir: Root directory where checkpoints are stored.

    Returns:
        A loaded model instance (subclass of :class:`src.models.HIVModelBase`).

    Raises:
        ValueError: If ``model_name`` is not recognised.
        FileNotFoundError: If the checkpoint does not exist.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    cls = _import_model_class(_MODEL_REGISTRY[model_name])

    # Determine checkpoint path per model type
    if model_name == "random_forest":
        ckpt_dir = os.path.join(model_dir, "random_forest")
        instance = cls(model_dir=ckpt_dir)
    elif model_name == "graphconv":
        ckpt_dir = os.path.join(model_dir, "graphconv")
        instance = cls(model_dir=ckpt_dir)
    else:  # attentivefp
        ckpt_dir = os.path.join(model_dir, "attentivefp")
        instance = cls(model_dir=ckpt_dir)

    instance.load()
    logger.info("Model '%s' loaded from %s", model_name, ckpt_dir)
    return instance


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def predict_smiles(
    smiles_list: List[str],
    model_name: str,
    threshold: float = 0.5,
    model_dir: str = "./models",
    validate: bool = True,
) -> Dict[str, List]:
    """Predict HIV activity for a list of SMILES strings.

    This function loads the model checkpoint from disk on each call.
    For repeated inference, prefer :func:`load_model` once and call
    ``model.predict()`` directly.

    Args:
        smiles_list: Input SMILES strings.
        model_name: ``random_forest``, ``graphconv``, or ``attentivefp``.
        threshold: Decision threshold for binary labels.
        model_dir: Root directory for checkpoints.
        validate: Whether to validate SMILES via RDKit before prediction.

    Returns:
        Dict with keys:
            - ``smiles``: validated input SMILES
            - ``predictions``: list of floats (probability of activity)
            - ``labels``: list of ints (0 or 1)

    Raises:
        ValueError: If no valid SMILES remain after validation.
    """
    if validate:
        smiles_list = _validate_smiles(smiles_list)

    if not smiles_list:
        raise ValueError("No valid SMILES to predict.")

    model = load_model(model_name, model_dir=model_dir)
    return model.predict(smiles_list, threshold=threshold)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using a pre-trained HIV model."
    )

    smiles_group = parser.add_mutually_exclusive_group(required=True)
    smiles_group.add_argument(
        "--smiles",
        nargs="+",
        help="One or more SMILES strings to predict.",
    )
    smiles_group.add_argument(
        "--smiles-file",
        dest="smiles_file",
        help="Path to a text file with one SMILES per line.",
    )

    parser.add_argument(
        "--model",
        choices=list(_MODEL_REGISTRY.keys()),
        default="random_forest",
        help="Model to use for prediction (default: random_forest).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "./models"),
        dest="model_dir",
        help="Root directory for model checkpoints.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results to.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for inference.

    Args:
        argv: Command-line arguments (defaults to sys.argv).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args(argv)

    # Gather SMILES
    if args.smiles:
        smiles_list = args.smiles
    else:
        if not os.path.exists(args.smiles_file):
            logger.error("SMILES file not found: %s", args.smiles_file)
            sys.exit(1)
        with open(args.smiles_file) as fh:
            smiles_list = [line.strip() for line in fh if line.strip()]

    try:
        results = predict_smiles(
            smiles_list=smiles_list,
            model_name=args.model,
            threshold=args.threshold,
            model_dir=args.model_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Prediction failed: %s", exc)
        sys.exit(1)

    # Print results
    print("\n── Prediction Results ─────────────────────────────────────────────")
    print(f"{'SMILES':<40} {'Probability':>12} {'Label':>8}")
    print("-" * 64)
    for smi, prob, label in zip(
        results["smiles"], results["predictions"], results["labels"]
    ):
        status = "ACTIVE" if label == 1 else "inactive"
        print(f"{smi:<40} {prob:>12.4f} {status:>8}")
    print()

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
