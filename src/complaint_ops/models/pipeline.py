"""Baseline Scikit-learn Pipeline and Model Bundle utilities."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from complaint_ops.config import (
    LR_C,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_SOLVER,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)


def build_baseline_pipeline(
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
    C: float = LR_C,
    max_iter: int = LR_MAX_ITER,
    class_weight: Optional[str] = LR_CLASS_WEIGHT,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline for multi-class classification."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
    )
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver=LR_SOLVER,
        random_state=random_state,
    )
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])
    return pipeline


def save_model_bundle(
    pipeline: Pipeline,
    categories: List[str],
    target_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model pipeline, category list, and metadata to directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Save categories
    categories_path = target_dir / "categories.json"
    with open(categories_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)

    # Save metadata
    tfidf_step = pipeline.named_steps.get("tfidf")
    clf_step = pipeline.named_steps.get("clf")
    vocab_size = len(tfidf_step.vocabulary_) if hasattr(tfidf_step, "vocabulary_") else None

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "TF-IDF + LogisticRegression",
        "num_classes": len(categories),
        "vocab_size": vocab_size,
        "hyperparameters": {
            "tfidf_ngram_range": tfidf_step.ngram_range if tfidf_step else None,
            "tfidf_max_features": tfidf_step.max_features if tfidf_step else None,
            "lr_C": clf_step.C if clf_step else None,
            "lr_class_weight": clf_step.class_weight if clf_step else None,
            "lr_max_iter": clf_step.max_iter if clf_step else None,
        },
        "metrics": metrics or {},
        "extra": extra_metadata or {},
    }
    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return target_dir


def load_model_bundle(model_dir: Path) -> Tuple[Pipeline, List[str], Dict[str, Any]]:
    """Load model pipeline, categories, and metadata from directory."""
    model_path = model_dir / "model.joblib"
    categories_path = model_dir / "categories.json"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found at {categories_path}")

    pipeline = joblib.load(model_path)
    with open(categories_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, categories, metadata


def predict_query(
    pipeline: Pipeline,
    query: str,
    categories: List[str],
    top_k: int = 3,
    low_confidence_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Predict intent with calibrated probabilities and low-confidence handling."""
    probas = pipeline.predict_proba([query])[0]
    classes = pipeline.classes_

    # Map pipeline internal class order
    ranked_indices = np.argsort(probas)[::-1]
    top_ranked = []
    for idx in ranked_indices[:top_k]:
        cls_name = str(classes[idx])
        conf = float(probas[idx])
        top_ranked.append({"intent": cls_name, "confidence": round(conf, 4)})

    primary_intent = top_ranked[0]["intent"]
    primary_confidence = top_ranked[0]["confidence"]
    is_low_confidence = primary_confidence < low_confidence_threshold

    return {
        "intent": primary_intent,
        "confidence": primary_confidence,
        "is_low_confidence": is_low_confidence,
        "routing_action": "HUMAN_AGENT_REVIEW" if is_low_confidence else "DIRECT_ROUTING",
        "top_candidates": top_ranked,
    }
