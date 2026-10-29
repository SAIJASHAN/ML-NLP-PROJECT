"""FastAPI service for ComplaintOps intent classification."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from complaint_ops.config import (
    API_MAX_BATCH_SIZE,
    API_MAX_TEXT_LENGTH,
    LOW_CONFIDENCE_THRESHOLD,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    TOP_K_PREDICTIONS,
)
from complaint_ops.models.pipeline import predict_query
from complaint_ops.tracking import load_registered_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., description="Customer complaint or query text")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text must be a non-empty string")
        if len(value) > API_MAX_TEXT_LENGTH:
            raise ValueError(
                f"text exceeds maximum length of {API_MAX_TEXT_LENGTH} characters"
            )
        return value.strip()


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., description="List of complaint texts")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("texts list cannot be empty")
        if len(values) > API_MAX_BATCH_SIZE:
            raise ValueError(
                f"batch size exceeds maximum of {API_MAX_BATCH_SIZE}"
            )

        cleaned = []
        for index, text in enumerate(values):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] is empty or whitespace")
            if len(text) > API_MAX_TEXT_LENGTH:
                raise ValueError(
                    f"texts[{index}] exceeds maximum length of "
                    f"{API_MAX_TEXT_LENGTH}"
                )
            cleaned.append(text.strip())

        return cleaned


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    is_low_confidence: bool
    routing_action: str
    top_candidates: list[dict]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    latency_ms: float


# ---------------------------------------------------------------------------
# FastAPI app and model loading
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ComplaintOps Prediction Service",
    version="0.1.0",
    description="Banking customer complaint intent classification API.",
)

app.state.pipeline = None
app.state.categories = []
app.state.model_loaded = False
app.state.model_source = None


def _extract_categories(pipeline: Any) -> list[str]:
    """Extract class labels from the fitted sklearn pipeline."""
    if hasattr(pipeline, "classes_"):
        return [str(item) for item in pipeline.classes_]

    named_steps = getattr(pipeline, "named_steps", {})
    for step in reversed(list(named_steps.values())):
        if hasattr(step, "classes_"):
            return [str(item) for item in step.classes_]

    return []


def _load_local_baseline() -> Any:
    """Load the saved baseline pipeline from the project model directory."""
    # Works for both source execution and the provided Dockerfile layout.
    candidates = [
        Path(__file__).resolve().parents[2]
        / "models"
        / "baseline_v1"
        / "model.joblib",
        Path("/app/models/baseline_v1/model.joblib"),
    ]

    for model_path in candidates:
        if model_path.is_file():
            logger.info("Loading local baseline model from %s", model_path)
            return joblib.load(model_path)

    raise FileNotFoundError(
        "Baseline model not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _initialize_model(pipeline: Any, source: str) -> bool:
    """Validate and store a loaded model."""
    categories = _extract_categories(pipeline)

    if not categories:
        raise RuntimeError("Loaded model exposes no class categories")

    app.state.pipeline = pipeline
    app.state.categories = categories
    app.state.model_source = source
    app.state.model_loaded = True

    logger.info(
        "Model loaded from %s with %d categories",
        source,
        len(categories),
    )
    return True


@app.on_event("startup")
def load_model() -> None:
    """Prefer MLflow; fall back to the local baseline model."""
    app.state.pipeline = None
    app.state.categories = []
    app.state.model_loaded = False
    app.state.model_source = None

    # Try the MLflow registry first.
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        pipeline = load_registered_model(
            model_name=MLFLOW_MODEL_NAME,
            version="1",
        )
        _initialize_model(pipeline, "mlflow_registry")
        return
    except Exception:
        logger.exception(
            "MLflow model unavailable; trying the local baseline."
        )

    # The local baseline is self-contained in the Docker image.
    try:
        pipeline = _load_local_baseline()
        _initialize_model(pipeline, "local_fallback")
    except Exception:
        logger.exception("Failed to initialize the local baseline model.")
        app.state.pipeline = None
        app.state.categories = []
        app.state.model_source = None
        app.state.model_loaded = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Return service and model status."""
    return {
        "status": "ok" if app.state.model_loaded else "degraded",
        "model_loaded": app.state.model_loaded,
        "model_source": app.state.model_source,
    }


@app.get("/metadata")
def metadata() -> dict:
    """Return model metadata and supported categories."""
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": "1",
        "model_source": app.state.model_source,
        "supported_intents": app.state.categories,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "top_k_predictions": TOP_K_PREDICTIONS,
    }


def _predict_one(text: str) -> dict:
    """Run one prediction using the shared prediction helper."""
    return predict_query(
        pipeline=app.state.pipeline,
        query=text,
        categories=app.state.categories,
        top_k=TOP_K_PREDICTIONS,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the intent of one complaint."""
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        result = _predict_one(request.text)
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal model error.",
        ) from exc

    latency = (time.perf_counter() - start) * 1000

    return PredictResponse(
        intent=result["intent"],
        confidence=float(result["confidence"]),
        is_low_confidence=bool(result["is_low_confidence"]),
        routing_action=result["routing_action"],
        top_candidates=result["top_candidates"],
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict the intent of multiple complaints."""
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    predictions: list[PredictResponse] = []

    try:
        for text in request.texts:
            item_start = time.perf_counter()
            result = _predict_one(text)
            item_latency = (time.perf_counter() - item_start) * 1000

            predictions.append(
                PredictResponse(
                    intent=result["intent"],
                    confidence=float(result["confidence"]),
                    is_low_confidence=bool(result["is_low_confidence"]),
                    routing_action=result["routing_action"],
                    top_candidates=result["top_candidates"],
                    latency_ms=round(item_latency, 2),
                )
            )
    except Exception as exc:
        logger.exception("Batch prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed due to an internal model error.",
        ) from exc

    total_latency = (time.perf_counter() - start) * 1000
    return BatchPredictResponse(
        predictions=predictions,
        latency_ms=round(total_latency, 2),
    )
