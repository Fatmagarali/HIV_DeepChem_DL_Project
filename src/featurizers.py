"""HIV dataset loading and featurization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class FeaturizedBatch:
    """Lightweight batch container used for runtime inference."""

    X: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class HIVSplits:
    """Container for a train/validation/test split of the HIV dataset."""

    tasks: tuple[str, ...]
    train: Any
    valid: Any
    test: Any
    transformers: tuple[Any, ...]
    featurizer_name: str

    def as_tuple(self) -> tuple[Any, Any, Any]:
        """Return the dataset splits in train/validation/test order."""

        return self.train, self.valid, self.test


def _load_hiv_dataset(featurizer: Any, featurizer_name: str, *, reload: bool = False) -> HIVSplits:
    import deepchem as dc

    tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=featurizer, splitter="scaffold", reload=reload)
    train, valid, test = datasets
    return HIVSplits(tasks=tuple(tasks), train=train, valid=valid, test=test, transformers=tuple(transformers), featurizer_name=featurizer_name)


def _featurize_split(split: Any, featurizer: Any, split_name: str) -> Any:
    import deepchem as dc

    smiles = getattr(split, "ids", None)
    if smiles is None:
        raise ValueError(f"Split {split_name} does not expose molecule identifiers")
    features = featurizer.featurize(list(smiles))
    return dc.data.NumpyDataset(X=features, y=split.y, w=split.w, ids=np.asarray(smiles, dtype=object))


def load_graph_splits(*, reload: bool = False) -> HIVSplits:
    """Load HIV splits featurized with MolGraphConvFeaturizer."""

    import deepchem as dc

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    return _load_hiv_dataset(featurizer, "graph", reload=reload)


def load_ecfp_splits(graph_splits: HIVSplits, *, size: int = 1024, radius: int = 2) -> HIVSplits:
    """Create ECFP features using the SMILES from already scaffold-split datasets."""

    import deepchem as dc

    featurizer = dc.feat.CircularFingerprint(size=size, radius=radius)
    return HIVSplits(
        tasks=graph_splits.tasks,
        train=_featurize_split(graph_splits.train, featurizer, "train"),
        valid=_featurize_split(graph_splits.valid, featurizer, "valid"),
        test=_featurize_split(graph_splits.test, featurizer, "test"),
        transformers=graph_splits.transformers,
        featurizer_name="ecfp",
    )


def load_convmol_splits(graph_splits: HIVSplits) -> HIVSplits:
    """Create ConvMol features using the SMILES from scaffold-split datasets."""

    import deepchem as dc

    featurizer = dc.feat.ConvMolFeaturizer()
    return HIVSplits(
        tasks=graph_splits.tasks,
        train=_featurize_split(graph_splits.train, featurizer, "train"),
        valid=_featurize_split(graph_splits.valid, featurizer, "valid"),
        test=_featurize_split(graph_splits.test, featurizer, "test"),
        transformers=graph_splits.transformers,
        featurizer_name="convmol",
    )


def load_all_feature_views(*, reload: bool = False, ecfp_size: int = 1024, ecfp_radius: int = 2) -> dict[str, HIVSplits]:
    """Load the HIV dataset once and derive all three feature views from it."""

    graph = load_graph_splits(reload=reload)
    return {
        "graph": graph,
        "ecfp": load_ecfp_splits(graph, size=ecfp_size, radius=ecfp_radius),
        "convmol": load_convmol_splits(graph),
    }


def featurize_smiles_for_model(
    smiles: Sequence[str],
    model_name: str,
    *,
    ecfp_size: int = 1024,
    ecfp_radius: int = 2,
) -> Any:
    """Featurize a batch of SMILES strings for a specific model family."""

    smiles_list = [str(item).strip() for item in smiles]
    if not smiles_list:
        raise ValueError("At least one SMILES string is required")

    normalized = model_name.strip().lower()
    if normalized in {"random_forest", "rf"}:
        features = np.zeros((len(smiles_list), ecfp_size), dtype=np.float32)
        for row_index, smiles_value in enumerate(smiles_list):
            molecule = Chem.MolFromSmiles(smiles_value)
            if molecule is None:
                raise ValueError(f"Invalid SMILES string: {smiles_value}")
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=ecfp_radius, nBits=ecfp_size)
            DataStructs.ConvertToNumpyArray(fingerprint, features[row_index])
        return FeaturizedBatch(X=features, ids=np.asarray(smiles_list, dtype=object))
    elif normalized == "graphconv":
        import deepchem as dc

        featurizer = dc.feat.ConvMolFeaturizer()
    elif normalized == "attentivefp":
        import deepchem as dc

        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    import deepchem as dc

    features = featurizer.featurize(smiles_list)

    if hasattr(features, 'toarray'):
        features = features.toarray()

    return dc.data.NumpyDataset(X=features, ids=np.asarray(smiles_list, dtype=object))