"""Automated tests for MLflow experiment tracking, artifact consistency, and model registry."""

import json
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.tracking import MlflowClient

from complaint_ops.models.evaluate import evaluate_predictions
from complaint_ops.models.pipeline import build_baseline_pipeline
from complaint_ops.tracking import (
    export_per_class_csv,
    get_model_registry_history,
    init_mlflow,
    load_model_from_run,
    load_registered_model,
    log_training_run,
)


@pytest.fixture
def test_mlflow_env(tmp_path):
    """Provide an isolated MLflow SQLite tracking URI and experiment name."""
    db_file = (tmp_path / "test_mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_file}"
    exp_name = "pytest-complaintops"
    return tracking_uri, exp_name


def test_init_mlflow_creates_experiment(test_mlflow_env):
    tracking_uri, exp_name = test_mlflow_env
    exp = init_mlflow(tracking_uri=tracking_uri, experiment_name=exp_name)
    assert exp is not None
    assert exp.name == exp_name
    assert exp.experiment_id is not None

    # Idempotence check: calling again retrieves the existing experiment
    exp2 = init_mlflow(tracking_uri=tracking_uri, experiment_name=exp_name)
    assert exp2.experiment_id == exp.experiment_id


def test_export_per_class_csv(tmp_path):
    per_class = {
        "card_arrival": {"precision": 0.9, "recall": 0.85, "f1_score": 0.87, "support": 40},
        "pin_blocked": {"precision": 0.95, "recall": 0.92, "f1_score": 0.93, "support": 40},
    }
    csv_path = tmp_path / "per_class_test.csv"
    res = export_per_class_csv(per_class, csv_path)
    assert res.exists()

    with open(res, "r", encoding="utf-8") as f:
        content = f.read()
    assert "category,precision,recall,f1_score,support" in content
    assert "card_arrival" in content
    assert "pin_blocked" in content


def test_log_training_run_metrics_and_artifacts(
    sample_train_df, sample_test_df, sample_categories, test_mlflow_env, tmp_path
):
    tracking_uri, exp_name = test_mlflow_env

    # Fit small pipeline
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=300, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])
    preds = pipeline.predict(sample_test_df["text"])

    eval_result = evaluate_predictions(
        sample_test_df["category"].tolist(), preds.tolist(), labels=sample_categories
    )
    summary = eval_result["summary"]
    summary["train_duration_sec"] = 0.5
    summary["eval_duration_sec"] = 0.05

    run_id, version = log_training_run(
        pipeline=pipeline,
        summary_metrics=summary,
        categories=sample_categories,
        run_name="pytest-baseline-run",
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
        metrics_dir=tmp_path,
        figures_dir=tmp_path,
    )

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) == 32

    # Verify run logged in MLflow
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["model_type"] == "TF-IDF + LogisticRegression"
    assert run.data.params["num_classes"] == "5"
    assert "accuracy" in run.data.metrics
    assert "macro_f1" in run.data.metrics
    assert "class_f1/card_arrival" in run.data.metrics

    # Verify artifacts list
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "evaluation" in artifact_paths

    # Verify logged model entity in MLflow 3
    logged_models = client.search_logged_models(experiment_ids=[run.info.experiment_id])
    assert len(logged_models) >= 1
    assert any(m.source_run_id == run_id for m in logged_models)


def test_model_registration_and_versioning(
    sample_train_df, sample_test_df, sample_categories, test_mlflow_env
):
    tracking_uri, exp_name = test_mlflow_env
    model_name = "TestBankingIntentModel"

    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=300, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])
    preds = pipeline.predict(sample_test_df["text"])

    eval_result = evaluate_predictions(
        sample_test_df["category"].tolist(), preds.tolist(), labels=sample_categories
    )
    summary = eval_result["summary"]

    run_id, version = log_training_run(
        pipeline=pipeline,
        summary_metrics=summary,
        categories=sample_categories,
        run_name="pytest-registered-run",
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
        register_model_name=model_name,
    )

    assert version is not None
    assert str(version) == "1"

    history = get_model_registry_history(model_name=model_name, tracking_uri=tracking_uri)
    assert len(history) == 1
    assert history[0]["version"] == "1"
    assert history[0]["run_id"] == run_id


def test_model_artifact_consistency(
    sample_train_df, sample_test_df, sample_categories, test_mlflow_env, tmp_path
):
    tracking_uri, exp_name = test_mlflow_env
    pipeline = build_baseline_pipeline(ngram_range=(1, 2), max_features=300, C=1.0)
    pipeline.fit(sample_train_df["text"], sample_train_df["category"])

    eval_result = evaluate_predictions(
        sample_test_df["category"].tolist(),
        pipeline.predict(sample_test_df["text"]).tolist(),
        labels=sample_categories,
    )

    run_id, _ = log_training_run(
        pipeline=pipeline,
        summary_metrics=eval_result["summary"],
        categories=sample_categories,
        run_name="consistency-check-run",
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
        metrics_dir=tmp_path,
        figures_dir=tmp_path,
    )

    # Load logged pipeline from MLflow
    mlflow.set_tracking_uri(tracking_uri)
    loaded_pipeline = load_model_from_run(run_id)

    # Test query predictions must be 100% numerically and categorically identical
    test_queries = [
        "When will my debit card arrive?",
        "Please help unblock my PIN immediately",
        "What are your currency exchange rates?",
    ]
    original_preds = pipeline.predict(test_queries)
    loaded_preds = loaded_pipeline.predict(test_queries)
    assert list(original_preds) == list(loaded_preds)

    original_probas = pipeline.predict_proba(test_queries)
    loaded_probas = loaded_pipeline.predict_proba(test_queries)
    import numpy as np
    np.testing.assert_allclose(original_probas, loaded_probas, rtol=1e-5, atol=1e-5)
