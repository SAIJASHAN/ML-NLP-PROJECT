"""Model evaluation suite: Macro-F1, accuracy, per-class metrics, confusion matrix."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from complaint_ops.config import FIGURES_DIR, METRICS_DIR


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multi-class metrics."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    per_class_metrics = {}
    for label in labels:
        if label in report_dict:
            per_class_metrics[label] = {
                "precision": round(float(report_dict[label]["precision"]), 4),
                "recall": round(float(report_dict[label]["recall"]), 4),
                "f1_score": round(float(report_dict[label]["f1-score"]), 4),
                "support": int(report_dict[label]["support"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Top confused pairs (excluding true positives on diagonal)
    confused_pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_intent": labels[i],
                    "pred_intent": labels[j],
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    summary = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "total_test_samples": len(y_true),
        "num_classes": len(labels),
        "per_class": per_class_metrics,
        "top_10_confused_pairs": confused_pairs[:10],
        "confusion_matrix_shape": list(cm.shape),
    }

    return {
        "summary": summary,
        "confusion_matrix": cm,
        "labels": labels,
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: Path,
    top_n_display: int = 25,
) -> Path:
    """Render and save a readable confusion matrix figure for top-N classes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For readability when num_classes == 77, slice to top-N most frequent or display subset
    n_classes = min(top_n_display, len(labels))
    sub_cm = cm[:n_classes, :n_classes]
    sub_labels = [l.replace("_", " ") for l in labels[:n_classes]]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sub_cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(sub_cm.shape[1]),
        yticks=np.arange(sub_cm.shape[0]),
        xticklabels=sub_labels,
        yticklabels=sub_labels,
        title=f"BANKING77 Intent Confusion Matrix (Top {n_classes} Classes)",
        ylabel="True Intent",
        xlabel="Predicted Intent",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Loop over data dimensions and create text annotations
    thresh = sub_cm.max() / 2.0
    for i in range(sub_cm.shape[0]):
        for j in range(sub_cm.shape[1]):
            val = sub_cm[i, j]
            if val > 0:
                ax.text(
                    j,
                    i,
                    format(val, "d"),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def export_evaluation_report(
    eval_result: Dict[str, Any],
    metrics_path: Optional[Path] = None,
    figure_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist metrics JSON and figure to reports directory."""
    if metrics_path is None:
        metrics_path = METRICS_DIR / "baseline_eval_report.json"
    if figure_path is None:
        figure_path = FIGURES_DIR / "confusion_matrix.png"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    summary = eval_result["summary"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_and_save_confusion_matrix(
        eval_result["confusion_matrix"], eval_result["labels"], figure_path
    )

    return summary
