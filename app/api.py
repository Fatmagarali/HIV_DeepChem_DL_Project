"""FastAPI application for HIV activity prediction.

Start the server::

    python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

Interactive docs available at:  http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import MODEL_DIR, MODEL_DESCRIPTIONS, THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HIV Activity Prediction API",
    description=(
        "Deep Learning models for anti-HIV activity prediction "
        "(Random Forest, GraphConv, AttentiveFP) trained on the "
        "MolNet HIV dataset."
    ),
    version="1.0.0",
    contact={
        "name": "HIV DeepChem DL Project",
        "url": "https://github.com/Fatmagarali/HIV_DeepChem_DL_Project",
    },
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""

    smiles: List[str] = Field(
        ...,
        min_length=1,
        description="List of SMILES strings to predict.",
        examples=[["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]],
    )
    model: str = Field(
        default="random_forest",
        description="Model name: random_forest | graphconv | attentivefp.",
    )
    threshold: float = Field(
        default=THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold for binary labels (default 0.5).",
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        available = list(MODEL_DESCRIPTIONS.keys())
        if v not in available:
            raise ValueError(f"Unknown model '{v}'. Available: {available}")
        return v

    @field_validator("smiles")
    @classmethod
    def _validate_smiles_not_empty(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("smiles list must contain at least one non-empty string.")
        return cleaned


class PredictionResponse(BaseModel):
    """Response body from the /predict endpoint."""

    smiles: List[str] = Field(..., description="Validated input SMILES.")
    predictions: List[float] = Field(..., description="Probability of anti-HIV activity.")
    labels: List[int] = Field(..., description="Binary labels (0 = inactive, 1 = active).")
    model: str = Field(..., description="Model used for prediction.")
    threshold: float = Field(..., description="Threshold applied.")


class ModelInfo(BaseModel):
    """Metadata for a single available model."""

    name: str
    description: str
    checkpoint: str
    available: bool


class ModelsResponse(BaseModel):
    """Response body for the /models endpoint."""

    models: Dict[str, ModelInfo]


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_dir: str
    available_models: List[str]


# ---------------------------------------------------------------------------
# Model cache (load once per process)
# ---------------------------------------------------------------------------

_model_cache: Dict[str, object] = {}


def _get_model(model_name: str):
    """Return a cached, loaded model instance.

    Args:
        model_name: One of the keys in ``MODEL_DESCRIPTIONS``.

    Returns:
        Loaded model instance.

    Raises:
        HTTPException 503: If the model checkpoint is not found on disk.
        HTTPException 503: If a required dependency (DGL, TensorFlow…) is absent.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from src.inference import load_model

        model = load_model(model_name, model_dir=MODEL_DIR)
        _model_cache[model_name] = model
        logger.info("Model '%s' loaded and cached.", model_name)
        return model
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{model_name}' checkpoint not found. "
                f"Train it first with: python -m src.train --model {model_name}. "
                f"Details: {exc}"
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' unavailable: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check() -> HealthResponse:
    """Health-check endpoint.

    Returns the API status and which model checkpoints are present on disk.
    """
    available = [
        name
        for name, info in MODEL_DESCRIPTIONS.items()
        if os.path.exists(os.path.join(MODEL_DIR, info["checkpoint"].rstrip("/")))
    ]
    return HealthResponse(
        status="ok",
        model_dir=MODEL_DIR,
        available_models=available,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Meta"])
async def list_models() -> ModelsResponse:
    """List all available models with their metadata.

    A model is marked as ``available`` if its checkpoint exists on disk.
    """
    models: Dict[str, ModelInfo] = {}
    for key, info in MODEL_DESCRIPTIONS.items():
        ckpt_path = os.path.join(MODEL_DIR, info["checkpoint"].rstrip("/"))
        models[key] = ModelInfo(
            name=info["name"],
            description=info["description"],
            checkpoint=info["checkpoint"],
            available=os.path.exists(ckpt_path),
        )
    return ModelsResponse(models=models)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict anti-HIV activity for a list of SMILES strings.

    The endpoint validates each SMILES (via RDKit when available),
    loads the requested model checkpoint, and returns probabilities
    together with binary labels at the requested threshold.

    Example request body::

        {
          "smiles": ["CCO", "c1ccccc1"],
          "model": "random_forest",
          "threshold": 0.5
        }
    """
    model = _get_model(request.model)

    try:
        from src.inference import _validate_smiles

        valid_smiles = _validate_smiles(request.smiles)
        if not valid_smiles:
            raise HTTPException(
                status_code=422,
                detail="None of the provided SMILES are valid.",
            )
        result = model.predict(valid_smiles, threshold=request.threshold)
    except (ValueError, RuntimeError) as exc:
        logger.exception("Prediction error for model '%s'", request.model)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        smiles=result["smiles"],
        predictions=result["predictions"],
        labels=result["labels"],
        model=request.model,
        threshold=request.threshold,
    )


# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run("app.api:app", host=HOST, port=PORT, reload=True)
