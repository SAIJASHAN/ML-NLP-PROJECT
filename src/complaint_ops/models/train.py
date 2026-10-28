"""Baseline training pipeline with evaluation, artifact generation, and MLflow tracking."""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
from sklearn.pipeline import Pipeline

from complaint_ops.config import (
    BASELINE_MODEL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    RAW_DATA_DIR,
)
from complaint_ops.data.ingest import inspect_dataset, load_banking77
from complaint_ops.models.evaluate import evaluate_predictions, export_evaluation_report
from complaint_ops.models.pipeline import build_baseline_pipeline, save_model_bundle
from complaint_ops.tracking import log_training_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_model_dir: Path = BASELINE_MODEL_DIR,
    use_mlflow: bool = False,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    register_model: bool = False,
    registered_model_name: Optional[str] = None,
    run_name: str = "baseline-tfidf-logistic-regression",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Execute complete reproducible training, evaluation, and optional MLflow tracking."""
    logger.info("Step 1: Loading official BANKING77 train and test splits...")
    train_df, test_df, categories = load_banking77(raw_dir=raw_data_dir)
    inspection = inspect_dataset(train_df, test_df, categories)
    logger.info(
        f"Dataset loaded: {inspection['train_samples']} train samples, "
        f"{inspection['test_samples']} test samples, {inspection['num_categories']} categories."
    )

    X_train = train_df["text"].tolist()
    y_train = train_df["category"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["category"].tolist()

    logger.info("Step 2: Constructing baseline Pipeline (TF-IDF + Logistic Regression)...")
    pipeline = build_baseline_pipeline()

    logger.info("Step 3: Fitting baseline pipeline on 10,003 training queries...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_duration_sec = round(time.time() - t0, 2)
    logger.info(f"Model fitting complete in {train_duration_sec}s.")

    logger.info("Step 4: Evaluating baseline model on official 3,080 test queries...")
    t1 = time.time()
    y_pred = pipeline.predict(X_test)
    eval_duration_sec = round(time.time() - t1, 2)

    eval_result = evaluate_predictions(y_test, y_pred, labels=categories)
    summary = eval_result["summary"]
    summary["train_duration_sec"] = train_duration_sec
    summary["eval_duration_sec"] = eval_duration_sec

    logger.info("=== Baseline Test Results ===")
    logger.info(f"  Macro-F1:  {summary['macro_f1']:.4f}")
    logger.info(f"  Accuracy:  {summary['accuracy']:.4f}")
    logger.info(f"  Weighted-F1: {summary['weighted_f1']:.4f}")
    logger.info(f"  Macro-Precision: {summary['macro_precision']:.4f}")
    logger.info(f"  Macro-Recall: {summary['macro_recall']:.4f}")

    logger.info("Step 5: Exporting evaluation reports and confusion matrix...")
    metrics_path = METRICS_DIR / "baseline_eval_report.json"
    figure_path = FIGURES_DIR / "confusion_matrix.png"
    export_evaluation_report(eval_result, metrics_path=metrics_path, figure_path=figure_path)

    logger.info(f"Step 6: Saving serialized model bundle to {output_model_dir}...")
    save_model_bundle(
        pipeline=pipeline,
        categories=categories,
        target_dir=output_model_dir,
        metrics=summary,
        extra_metadata={"dataset_inspection": inspection},
    )

    if use_mlflow:
        reg_name = (registered_model_name or MLFLOW_MODEL_NAME) if register_model else None
        logger.info(
            f"Step 7: Tracking experiment with MLflow (experiment='{experiment_name}', register={register_model})..."
        )
        run_id, version = log_training_run(
            pipeline=pipeline,
            summary_metrics=summary,
            categories=categories,
            run_name=run_name,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri or MLFLOW_TRACKING_URI,
            register_model_name=reg_name,
            model_bundle_dir=output_model_dir,
        )
        summary["mlflow_run_id"] = run_id
        summary["mlflow_registered_version"] = version
        logger.info(f"MLflow Run ID: {run_id}, Registered Version: {version}")

    return pipeline, summary


def main():
    parser = argparse.ArgumentParser(description="Train ComplaintOps baseline model.")
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register trained model in MLflow model registry.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MLFLOW_MODEL_NAME,
        help="Name under which to register the model in MLflow Model Registry.",
    )
    args = parser.parse_args()

    train_and_evaluate(
        use_mlflow=args.use_mlflow,
        register_model=args.register_model,
        experiment_name=args.experiment_name,
        registered_model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
