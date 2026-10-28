"""Unit tests for baseline pipeline, training, serialization, and evaluation."""

import numpy as np
import pytest
from complaint_ops.models.evaluate import evaluate_predictions
from complaint_ops.models.pipeline import (
    build_baseline_pipeline,
    load_model_bundle,
    predict_query,
    save_model_bundle,
)


def test_build_baseline_pipeline():
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=500, C=1.0)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    assert pipeline.named_steps["tfidf"].max_features == 500


def test_fit_and_predict_pipeline(sample_train_df, sample_test_df):
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=500, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])

    preds = pipeline.predict(sample_test_df["text"])
    assert len(preds) == len(sample_test_df)
    assert set(preds).issubset(set(sample_train_df["category"].unique()))


def test_predict_query_schema(sample_train_df, sample_categories):
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=500, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])

    result = predict_query(
        pipeline=pipeline,
        query="When is my card arriving?",
        categories=sample_categories,
        top_k=3,
        low_confidence_threshold=0.35,
    )

    assert "intent" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert "is_low_confidence" in result
    assert "routing_action" in result
    assert result["routing_action"] in ["DIRECT_ROUTING", "HUMAN_AGENT_REVIEW"]
    assert len(result["top_candidates"]) == 3
    assert result["top_candidates"][0]["intent"] == result["intent"]


def test_low_confidence_routing_trigger(sample_train_df, sample_categories):
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=500, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])

    # High threshold forces human review
    result = predict_query(
        pipeline=pipeline,
        query="random gibberish xyz abc 12345",
        categories=sample_categories,
        low_confidence_threshold=0.99,
    )
    assert result["is_low_confidence"] is True
    assert result["routing_action"] == "HUMAN_AGENT_REVIEW"


def test_model_bundle_serialization(sample_train_df, sample_categories, tmp_path):
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=500, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])

    save_dir = tmp_path / "model_bundle"
    saved_path = save_model_bundle(
        pipeline=pipeline,
        categories=sample_categories,
        target_dir=save_dir,
        metrics={"macro_f1": 0.85},
    )
    assert saved_path.exists()
    assert (saved_path / "model.joblib").exists()
    assert (saved_path / "categories.json").exists()
    assert (saved_path / "metadata.json").exists()

    loaded_pipe, loaded_cats, loaded_meta = load_model_bundle(save_dir)
    assert loaded_cats == sample_categories
    assert loaded_meta["metrics"]["macro_f1"] == 0.85

    # Verification: predictions must match exactly
    test_q = "When will my card arrive?"
    orig_pred = pipeline.predict([test_q])[0]
    loaded_pred = loaded_pipe.predict([test_q])[0]
    assert orig_pred == loaded_pred


def test_evaluate_predictions_metrics():
    y_true = ["cat_a", "cat_b", "cat_c", "cat_a"]
    y_pred = ["cat_a", "cat_b", "cat_a", "cat_a"]
    labels = ["cat_a", "cat_b", "cat_c"]

    result = evaluate_predictions(y_true, y_pred, labels=labels)
    summary = result["summary"]

    assert "accuracy" in summary
    assert "macro_f1" in summary
    assert "per_class" in summary
    assert "top_10_confused_pairs" in summary
    assert summary["accuracy"] == 0.75
    assert summary["total_test_samples"] == 4
    assert len(summary["per_class"]) == 3


def test_real_saved_baseline_bundle():
    from complaint_ops.config import BASELINE_MODEL_DIR, FIGURES_DIR, METRICS_DIR

    # Verify artifacts
    assert (BASELINE_MODEL_DIR / "model.joblib").exists()
    assert (BASELINE_MODEL_DIR / "categories.json").exists()
    assert (BASELINE_MODEL_DIR / "metadata.json").exists()
    assert (METRICS_DIR / "baseline_eval_report.json").exists()
    assert (FIGURES_DIR / "confusion_matrix.png").exists()

    pipeline, categories, metadata = load_model_bundle(BASELINE_MODEL_DIR)
    assert len(categories) == 77
    assert metadata["metrics"]["macro_f1"] >= 0.85
    assert metadata["metrics"]["accuracy"] >= 0.85

    # Run inference test
    sample_query = "I lost my debit card today, how do I cancel or block it immediately?"
    pred = predict_query(pipeline, sample_query, categories)
    assert pred["intent"] in ["lost_or_stolen_card", "card_arrival", "pin_blocked", "compromised_card"]
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["routing_action"] in ["DIRECT_ROUTING", "HUMAN_AGENT_REVIEW"]

