"""FastAPI application exposing HIV prediction endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional 

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import Settings, get_settings

from src.inference import list_model_statuses, predict_smiles


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class PredictRequest(BaseModel):
    """Request payload for the /predict endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    smiles: list[str] = Field(min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("smiles")
    @classmethod
    def _validate_smiles(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty SMILES string is required")
        return cleaned


class PredictionItem(BaseModel):
    """Single prediction returned by the API."""

    smiles: str
    probability: float
    label: int


class PredictResponse(BaseModel):
    """Response payload for the /predict endpoint."""

    model: str
    threshold: float
    predictions: list[PredictionItem]


class ModelStatus(BaseModel):
    """Availability information for a saved checkpoint."""

    name: str
    path: str
    available: bool
    metadata: str


class HealthResponse(BaseModel):
    """Health check response returned by the API."""

    status: str
    seed: int
    default_model: str
    models_dir: str
    artifacts_dir: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = settings or get_settings()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="HIV DeepChem API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _attach_settings(request: Request, call_next):
        request.state.settings = settings
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Render the single-page front-end."""

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "default_model": settings.default_model,
                "models": list_model_statuses(settings),
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return a simple health payload."""

        return HealthResponse(
            status="ok",
            seed=settings.seed,
            default_model=settings.default_model,
            models_dir=str(settings.models_dir),
            artifacts_dir=str(settings.artifacts_dir),
        )

    @app.get("/models", response_model=list[ModelStatus])
    async def models() -> list[ModelStatus]:
        """List the supported models and whether their checkpoints are available."""

        return [ModelStatus(**item) for item in list_model_statuses(settings)]

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest) -> PredictResponse:
        """Predict activity labels and probabilities for a batch of SMILES strings."""

        try:
            model_name = payload.model or settings.default_model
            predictions = predict_smiles(
                model_name,
                payload.smiles,
                threshold=payload.threshold,
                settings=settings,
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictResponse(
            model=model_name,
            threshold=payload.threshold,
            predictions=[
                PredictionItem(smiles=item.smiles, probability=item.probability, label=item.label)
                for item in predictions
            ],
        )

    app.state.settings = settings
    return app


app = create_app()
