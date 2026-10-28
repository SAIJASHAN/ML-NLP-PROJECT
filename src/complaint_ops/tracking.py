"""MLflow experiment tracking, artifact logging, and model registry management."""

import csv
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient

from complaint_ops.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)


def init_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> Experiment:
    """Initialize MLflow tracking URI and retrieve or create the active experiment."""
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI set to: {uri}")

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(exp_id)
        logger.info(f"Created new MLflow experiment '{experiment_name}' (ID: {exp_id})")
    else:
        logger.info(f"Using existing MLflow experiment '{experiment_name}' (ID: {experiment.experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment


def export_per_class_csv(per_class_dict: Dict[str, Dict[str, Any]], output_path: Path) -> Path:
    """Convert per-class metrics dictionary to a CSV file for artifact logging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat_name, metrics in sorted(per_class_dict.items()):
        rows.append({
            "category": cat_name,
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "support": metrics.get("support", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def log_training_run(
    pipeline: Pipeline,
    summary_metrics: Dict[str, Any],
    categories: List[str],
    run_name: str = "baseline-tfidf-logistic-regression",
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    register_model_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    model_bundle_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Tuple[str, Optional[Any]]:
    """Log parameters, overall metrics, per-class metrics, artifacts, and optionally register model."""
    init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    target_metrics_dir = metrics_dir or METRICS_DIR
    target_figures_dir = figures_dir or FIGURES_DIR

    tfidf_step = pipeline.named_steps.get("tfidf")
    clf_step = pipeline.named_steps.get("clf")

    params = {
        "model_type": "TF-IDF + LogisticRegression",
        "num_classes": len(categories),
        "total_test_samples": summary_metrics.get("total_test_samples", 0),
        "train_duration_sec": summary_metrics.get("train_duration_sec", 0.0),
        "eval_duration_sec": summary_metrics.get("eval_duration_sec", 0.0),
    }

    if tfidf_step:
        params["tfidf_ngram_range"] = str(tfidf_step.ngram_range)
        params["tfidf_max_features"] = tfidf_step.max_features
        params["tfidf_sublinear_tf"] = tfidf_step.sublinear_tf
        params["vocab_size"] = len(tfidf_step.vocabulary_) if hasattr(tfidf_step, "vocabulary_") else 0

    if clf_step:
        params["lr_C"] = clf_step.C
        params["lr_max_iter"] = clf_step.max_iter
        params["lr_class_weight"] = str(clf_step.class_weight)
        params["lr_solver"] = getattr(clf_step, "solver", "lbfgs")

    run_tags = {
        "dataset": "BANKING77",
        "task": "77-class intent classification",
        "framework": "scikit-learn",
    }
    if tags:
        run_tags.update(tags)

    registered_version = None

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: ID={run_id}")

        mlflow.set_tags(run_tags)
        mlflow.log_params(params)

        # 1. Summary metrics
        overall_metrics = {
            "accuracy": summary_metrics.get("accuracy", 0.0),
            "macro_f1": summary_metrics.get("macro_f1", 0.0),
            "weighted_f1": summary_metrics.get("weighted_f1", 0.0),
            "macro_precision": summary_metrics.get("macro_precision", 0.0),
            "macro_recall": summary_metrics.get("macro_recall", 0.0),
        }
        mlflow.log_metrics(overall_metrics)

        # 2. Per-class metrics
        per_class = summary_metrics.get("per_class", {})
        for cat_name, metrics in per_class.items():
            sanitized_name = re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", str(cat_name))
            sanitized_name = re.sub(r"_+", "_", sanitized_name).strip("_")
            mlflow.log_metric(f"class_f1/{sanitized_name}", metrics.get("f1_score", 0.0))
            mlflow.log_metric(f"class_prec/{sanitized_name}", metrics.get("precision", 0.0))
            mlflow.log_metric(f"class_rec/{sanitized_name}", metrics.get("recall", 0.0))

        # 3. Artifacts: Reports & CSVs
        per_class_csv_path = target_metrics_dir / "per_class_metrics.csv"
        export_per_class_csv(per_class, per_class_csv_path)
        if per_class_csv_path.exists():
            mlflow.log_artifact(str(per_class_csv_path), artifact_path="evaluation")

        eval_json_path = target_metrics_dir / "baseline_eval_report.json"
        if eval_json_path.exists():
            mlflow.log_artifact(str(eval_json_path), artifact_path="evaluation")

        cm_png_path = target_figures_dir / "confusion_matrix.png"
        if cm_png_path.exists():
            mlflow.log_artifact(str(cm_png_path), artifact_path="figures")

        if model_bundle_dir and model_bundle_dir.exists():
            mlflow.log_artifacts(str(model_bundle_dir), artifact_path="model_bundle")

        # 4. Log Scikit-learn Pipeline Model
        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=register_model_name if register_model_name else None,
        )

        # If registered via registered_model_name, retrieve version info
        if register_model_name:
            client = MlflowClient(tracking_uri=tracking_uri or MLFLOW_TRACKING_URI)
            try:
                versions = client.search_model_versions(f"name='{register_model_name}'")
                if versions:
                    sorted_v = sorted(versions, key=lambda x: int(x.version), reverse=True)
                    registered_version = str(sorted_v[0].version)
                    logger.info(
                        f"Model registered as '{register_model_name}', version: {registered_version}"
                    )
            except Exception as e:
                logger.warning(f"Could not retrieve registered version: {e}")

        logger.info(f"MLflow Run {run_id} logging complete.")

    return run_id, registered_version


def load_model_from_run(run_id: str, model_subpath: str = "model") -> Pipeline:
    """Load logged scikit-learn model pipeline from an MLflow run ID."""
    model_uri = f"runs:/{run_id}/{model_subpath}"
    logger.info(f"Loading pipeline from MLflow URI: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def load_registered_model(
    model_name: str = MLFLOW_MODEL_NAME,
    version: Optional[str] = None,
    alias: Optional[str] = None,
) -> Pipeline:
    """Load model pipeline from MLflow Model Registry by version or alias."""
    if alias:
        model_uri = f"models:/{model_name}@{alias}"
    elif version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading registered model from URI: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def get_model_registry_history(
    model_name: str = MLFLOW_MODEL_NAME,
    tracking_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve full version history and metadata from the MLflow model registry."""
    client = MlflowClient(tracking_uri=tracking_uri or MLFLOW_TRACKING_URI)
    try:
        registered_model = client.get_registered_model(model_name)
    except Exception as e:
        logger.warning(f"Model '{model_name}' not found in registry: {e}")
        return []

    history = []
    # Search all versions
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        history.append({
            "version": str(v.version),
            "creation_timestamp": v.creation_timestamp,
            "current_stage": getattr(v, "current_stage", "None"),
            "description": v.description or "",
            "source": v.source,
            "run_id": v.run_id,
            "status": v.status,
            "aliases": getattr(v, "aliases", []),
            "tags": dict(v.tags) if v.tags else {},
        })
    return history
