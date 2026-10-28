"""Baseline training pipeline with evaluation and model artifact generation."""

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
    RAW_DATA_DIR,
)
from complaint_ops.data.ingest import inspect_dataset, load_banking77
from complaint_ops.models.evaluate import evaluate_predictions, export_evaluation_report
from complaint_ops.models.pipeline import build_baseline_pipeline, save_model_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_model_dir: Path = BASELINE_MODEL_DIR,
    use_mlflow: bool = False,
    experiment_name: str = "ComplaintOps-Banking77",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Execute complete reproducible training and evaluation cycle."""
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

    # Optional MLflow tracking
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name="baseline-tfidf-logistic-regression")
            logger.info(f"MLflow Run started: ID={mlflow_run.info.run_id}")

            tfidf_step = pipeline.named_steps["tfidf"]
            clf_step = pipeline.named_steps["clf"]
            mlflow.log_params({
                "model_type": "TF-IDF + LogisticRegression",
                "tfidf_ngram_range": str(tfidf_step.ngram_range),
                "tfidf_max_features": tfidf_step.max_features,
                "tfidf_sublinear_tf": tfidf_step.sublinear_tf,
                "lr_C": clf_step.C,
                "lr_max_iter": clf_step.max_iter,
                "lr_class_weight": str(clf_step.class_weight),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "num_classes": len(categories),
            })
        except ImportError:
            logger.warning("MLflow not installed, skipping MLflow logging.")
            use_mlflow = False

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

    logger.info(f"=== Baseline Test Results ===")
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

    if use_mlflow and mlflow_run:
        import mlflow

        mlflow.log_metrics({
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "weighted_f1": summary["weighted_f1"],
            "macro_precision": summary["macro_precision"],
            "macro_recall": summary["macro_recall"],
            "train_duration_sec": train_duration_sec,
        })
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(figure_path), artifact_path="figures")
        try:
            mlflow.sklearn.log_model(pipeline, name="model")
        except TypeError:
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.end_run()
        logger.info("MLflow logging completed.")

    return pipeline, summary


def main():
    parser = argparse.ArgumentParser(description="Train ComplaintOps baseline model.")
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking.",
    )
    args = parser.parse_args()
    train_and_evaluate(use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
