"""Model definitions for HIV activity prediction.

Provides three model classes wrapping DeepChem / scikit-learn estimators:
    - RandomForestHIV  (ECFP4 fingerprints + sklearn)
    - GraphConvHIV     (spatial GCN via DeepChem)
    - AttentiveFPHIV   (graph + attention via DeepChem + DGL)

Each class exposes a unified interface:
    train / predict / save / load
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HIVModelBase:
    """Abstract base class shared by all HIV models.

    Args:
        model_dir: Directory used for checkpoint persistence.
    """

    def __init__(self, model_dir: str = "./models") -> None:
        self.model_dir = model_dir
        self._model: Any = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, smiles_list: List[str]) -> Dict[str, List]:
        """Return predictions for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Dict with keys ``smiles``, ``predictions`` (probabilities),
            and ``labels`` (binary, threshold=0.5).

        Raises:
            RuntimeError: If the model has not been trained or loaded yet.
        """
        raise NotImplementedError

    def train(self, train_dataset: Any, valid_dataset: Any, epochs: int = 30) -> Dict:
        """Train the model.

        Args:
            train_dataset: DeepChem NumpyDataset (or compatible) for training.
            valid_dataset: DeepChem NumpyDataset for validation.
            epochs: Number of training epochs (ignored for Random Forest).

        Returns:
            Dict with training history / metrics.
        """
        raise NotImplementedError

    def save(self, path: Optional[str] = None) -> None:
        """Persist model to disk.

        Args:
            path: Override save path. Defaults to ``self.model_dir``.
        """
        raise NotImplementedError

    def load(self, path: Optional[str] = None) -> None:
        """Restore model from disk.

        Args:
            path: Override load path. Defaults to ``self.model_dir``.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_binary(probabilities: np.ndarray, threshold: float = 0.5) -> List[int]:
        return (probabilities >= threshold).astype(int).tolist()


# ---------------------------------------------------------------------------
# Random Forest (ECFP4 fingerprints)
# ---------------------------------------------------------------------------

class RandomForestHIV(HIVModelBase):
    """Random Forest classifier using ECFP4 circular fingerprints.

    Args:
        n_estimators: Number of trees.
        fingerprint_size: Bit-vector length.
        fingerprint_radius: Morgan radius.
        model_dir: Directory for checkpoint persistence.
        seed: Random seed for reproducibility.
    """

    CHECKPOINT_NAME = "random_forest.pkl"

    def __init__(
        self,
        n_estimators: int = 500,
        fingerprint_size: int = 1024,
        fingerprint_radius: int = 2,
        model_dir: str = "./models",
        seed: int = 42,
    ) -> None:
        super().__init__(model_dir)
        self.seed = seed

        try:
            from sklearn.ensemble import RandomForestClassifier
            import deepchem as dc

            self._model = RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
            self._featurizer = dc.feat.CircularFingerprint(
                size=fingerprint_size, radius=fingerprint_radius
            )
        except ImportError as exc:
            logger.warning("scikit-learn or deepchem not available: %s", exc)
            self._model = None
            self._featurizer = None

    # ------------------------------------------------------------------

    def train(self, train_dataset: Any, valid_dataset: Any, epochs: int = 30) -> Dict:
        """Train the Random Forest (epochs parameter is ignored).

        Args:
            train_dataset: DeepChem NumpyDataset with ECFP features.
            valid_dataset: DeepChem NumpyDataset for validation.
            epochs: Ignored (RF does not use iterative training).

        Returns:
            Dict with ``train_roc_auc`` and ``valid_roc_auc``.
        """
        from sklearn.metrics import roc_auc_score

        X_train = train_dataset.X
        y_train = train_dataset.y.flatten().astype(int)
        X_valid = valid_dataset.X
        y_valid = valid_dataset.y.flatten().astype(int)

        logger.info("Training Random Forest on %d samples …", len(X_train))
        self._model.fit(X_train, y_train)

        train_proba = self._model.predict_proba(X_train)[:, 1]
        valid_proba = self._model.predict_proba(X_valid)[:, 1]

        metrics = {
            "train_roc_auc": roc_auc_score(y_train, train_proba),
            "valid_roc_auc": roc_auc_score(y_valid, valid_proba),
        }
        logger.info("RF — train ROC-AUC: %.4f | valid ROC-AUC: %.4f",
                    metrics["train_roc_auc"], metrics["valid_roc_auc"])
        return metrics

    def predict(self, smiles_list: List[str], threshold: float = 0.5) -> Dict[str, List]:
        """Predict HIV activity for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings.
            threshold: Decision threshold for binary labels.

        Returns:
            Dict with ``smiles``, ``predictions``, ``labels``.

        Raises:
            RuntimeError: If model is not trained/loaded.
        """
        if self._model is None or self._featurizer is None:
            raise RuntimeError("RandomForestHIV model is not initialized.")

        X = self._featurizer.featurize(smiles_list)
        proba = self._model.predict_proba(X)[:, 1]
        return {
            "smiles": smiles_list,
            "predictions": proba.tolist(),
            "labels": self._to_binary(proba, threshold),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save the fitted model to a pickle file.

        Args:
            path: Full file path. Defaults to ``<model_dir>/random_forest.pkl``.
        """
        dest = path or os.path.join(self.model_dir, self.CHECKPOINT_NAME)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self._model, fh)
        logger.info("Random Forest saved to %s", dest)

    def load(self, path: Optional[str] = None) -> None:
        """Load model from a pickle file.

        Args:
            path: Full file path. Defaults to ``<model_dir>/random_forest.pkl``.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        src = path or os.path.join(self.model_dir, self.CHECKPOINT_NAME)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Checkpoint not found: {src}")
        with open(src, "rb") as fh:
            self._model = pickle.load(fh)
        logger.info("Random Forest loaded from %s", src)


# ---------------------------------------------------------------------------
# GraphConv (spatial GCN)
# ---------------------------------------------------------------------------

class GraphConvHIV(HIVModelBase):
    """GraphConv model (spatial GCN) via DeepChem.

    Args:
        graph_conv_layers: List of hidden layer sizes.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        model_dir: Directory for checkpoint persistence.
        seed: Random seed.
    """

    def __init__(
        self,
        graph_conv_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        model_dir: str = "./models/graphconv",
        seed: int = 42,
    ) -> None:
        super().__init__(model_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if graph_conv_layers is None:
            graph_conv_layers = [128, 128]

        try:
            import deepchem as dc
            import tensorflow as tf

            tf.random.set_seed(seed)

            self._model = dc.models.GraphConvModel(
                n_tasks=1,
                mode="classification",
                graph_conv_layers=graph_conv_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_dir=model_dir,
            )
            self._featurizer = dc.feat.ConvMolFeaturizer()
        except ImportError as exc:
            logger.warning("DeepChem/TensorFlow not available: %s", exc)
            self._model = None
            self._featurizer = None

    # ------------------------------------------------------------------

    def train(self, train_dataset: Any, valid_dataset: Any, epochs: int = 30) -> Dict:
        """Fit the GraphConv model.

        Args:
            train_dataset: DeepChem NumpyDataset (ConvMol features).
            valid_dataset: DeepChem NumpyDataset for validation.
            epochs: Number of training epochs.

        Returns:
            Dict with ``train_history`` and ``valid_history`` (ROC-AUC per epoch).
        """
        import deepchem as dc
        from sklearn.metrics import roc_auc_score

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        train_history: List[float] = []
        valid_history: List[float] = []

        logger.info("Training GraphConv for %d epochs …", epochs)
        for epoch in range(epochs):
            self._model.fit(train_dataset, nb_epoch=1)
            tr_score = self._model.evaluate(train_dataset, [metric])["roc_auc_score"]
            va_score = self._model.evaluate(valid_dataset, [metric])["roc_auc_score"]
            train_history.append(tr_score)
            valid_history.append(va_score)
            if (epoch + 1) % 5 == 0:
                logger.info("  [%d/%d] train=%.4f  valid=%.4f",
                            epoch + 1, epochs, tr_score, va_score)

        return {"train_history": train_history, "valid_history": valid_history}

    def predict(self, smiles_list: List[str], threshold: float = 0.5) -> Dict[str, List]:
        """Predict HIV activity for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings.
            threshold: Decision threshold.

        Returns:
            Dict with ``smiles``, ``predictions``, ``labels``.
        """
        import deepchem as dc

        X = self._featurizer.featurize(smiles_list)
        dataset = dc.data.NumpyDataset(X=X, ids=smiles_list)
        y_pred = self._model.predict(dataset)

        if y_pred.ndim == 3:
            proba = y_pred[:, 0, 1]
        elif y_pred.ndim == 2:
            proba = y_pred[:, 1]
        else:
            proba = y_pred.flatten()

        return {
            "smiles": smiles_list,
            "predictions": proba.tolist(),
            "labels": self._to_binary(proba, threshold),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save GraphConv checkpoint.

        Args:
            path: Override directory. Defaults to ``self.model_dir``.
        """
        dest = path or self.model_dir
        os.makedirs(dest, exist_ok=True)
        self._model.save_checkpoint(model_dir=dest)
        logger.info("GraphConv checkpoint saved to %s", dest)

    def load(self, path: Optional[str] = None) -> None:
        """Restore GraphConv checkpoint.

        Args:
            path: Override directory. Defaults to ``self.model_dir``.

        Raises:
            FileNotFoundError: If directory does not exist.
        """
        src = path or self.model_dir
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Checkpoint directory not found: {src}")
        self._model.restore(model_dir=src)
        logger.info("GraphConv loaded from %s", src)


# ---------------------------------------------------------------------------
# AttentiveFP (GCN + attention)
# ---------------------------------------------------------------------------

class AttentiveFPHIV(HIVModelBase):
    """AttentiveFP model (graph + attention) via DeepChem + DGL.

    Args:
        num_layers: Number of GNN layers.
        num_timesteps: Attention timesteps.
        graph_feat_size: Node embedding dimension.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        model_dir: Directory for checkpoint persistence.
        seed: Random seed.
    """

    def __init__(
        self,
        num_layers: int = 3,
        num_timesteps: int = 2,
        graph_feat_size: int = 200,
        dropout: float = 0.2,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        model_dir: str = "./models/attentivefp",
        seed: int = 42,
    ) -> None:
        super().__init__(model_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        try:
            import dgl  # noqa: F401
            import dgllife  # noqa: F401

            # ⚠️  Security warning: all known versions of dgl (<= 2.4.0) are
            # affected by a Remote Code Execution vulnerability via pickle
            # deserialization (no patched version exists).
            # Only use AttentiveFP in isolated, trusted environments.
            # See requirements-attentivefp.txt for details.
            logger.warning(
                "dgl is installed. All known dgl versions (<= 2.4.0) carry a "
                "Remote Code Execution risk via pickle deserialization. "
                "Use AttentiveFP only in isolated, trusted environments and "
                "never expose it via a public API endpoint."
            )

            import deepchem as dc
            import torch

            torch.manual_seed(seed)

            self._model = dc.models.AttentiveFPModel(
                n_tasks=1,
                mode="classification",
                num_layers=num_layers,
                num_timesteps=num_timesteps,
                graph_feat_size=graph_feat_size,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_dir=model_dir,
            )
            self._featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        except ImportError as exc:
            logger.warning("DGL/DGLLife/DeepChem not available: %s", exc)
            self._model = None
            self._featurizer = None

    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return True if DGL and DeepChem were importable at init time."""
        return self._model is not None

    def train(self, train_dataset: Any, valid_dataset: Any, epochs: int = 30) -> Dict:
        """Fit the AttentiveFP model.

        Args:
            train_dataset: DeepChem NumpyDataset (MolGraphConv features).
            valid_dataset: DeepChem NumpyDataset for validation.
            epochs: Number of training epochs.

        Returns:
            Dict with ``train_history`` and ``valid_history`` (ROC-AUC per epoch).
        """
        if not self.available:
            raise RuntimeError("AttentiveFP requires DGL and DGLLife. Install them first.")

        import deepchem as dc

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        train_history: List[float] = []
        valid_history: List[float] = []

        logger.info("Training AttentiveFP for %d epochs …", epochs)
        for epoch in range(epochs):
            self._model.fit(train_dataset, nb_epoch=1)
            tr_score = self._model.evaluate(train_dataset, [metric])["roc_auc_score"]
            va_score = self._model.evaluate(valid_dataset, [metric])["roc_auc_score"]
            train_history.append(tr_score)
            valid_history.append(va_score)
            if (epoch + 1) % 5 == 0:
                logger.info("  [%d/%d] train=%.4f  valid=%.4f",
                            epoch + 1, epochs, tr_score, va_score)

        return {"train_history": train_history, "valid_history": valid_history}

    def predict(self, smiles_list: List[str], threshold: float = 0.5) -> Dict[str, List]:
        """Predict HIV activity for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings.
            threshold: Decision threshold.

        Returns:
            Dict with ``smiles``, ``predictions``, ``labels``.

        Raises:
            RuntimeError: If DGL is not installed.
        """
        if not self.available:
            raise RuntimeError("AttentiveFP requires DGL and DGLLife.")

        import deepchem as dc

        X = self._featurizer.featurize(smiles_list)
        dataset = dc.data.NumpyDataset(X=X, ids=smiles_list)
        y_pred = self._model.predict(dataset)

        if y_pred.ndim == 3:
            proba = y_pred[:, 0, 1]
        elif y_pred.ndim == 2:
            proba = y_pred[:, 1]
        else:
            proba = y_pred.flatten()

        return {
            "smiles": smiles_list,
            "predictions": proba.tolist(),
            "labels": self._to_binary(proba, threshold),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save AttentiveFP checkpoint.

        Args:
            path: Override directory. Defaults to ``self.model_dir``.
        """
        if not self.available:
            raise RuntimeError("AttentiveFP requires DGL and DGLLife.")
        dest = path or self.model_dir
        os.makedirs(dest, exist_ok=True)
        self._model.save_checkpoint(model_dir=dest)
        logger.info("AttentiveFP checkpoint saved to %s", dest)

    def load(self, path: Optional[str] = None) -> None:
        """Restore AttentiveFP checkpoint.

        Args:
            path: Override directory. Defaults to ``self.model_dir``.

        Raises:
            FileNotFoundError: If directory does not exist.
            RuntimeError: If DGL is not installed.
        """
        if not self.available:
            raise RuntimeError("AttentiveFP requires DGL and DGLLife.")
        src = path or self.model_dir
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Checkpoint directory not found: {src}")
        self._model.restore(model_dir=src)
        logger.info("AttentiveFP loaded from %s", src)
