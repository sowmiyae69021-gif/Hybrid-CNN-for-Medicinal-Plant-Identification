"""
Evaluation and reporting utilities.

This module computes classification metrics and generates
reports required to validate experimental results.

Outputs produced:
- classification_report.txt
- confusion_matrix.png
- metrics_summary.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.

    Returns a dictionary of results.
    """

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

    return metrics


def save_classification_report(y_true, y_pred, class_names, save_path):
    """
    Save classification report as text.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    with open(save_path, "w") as f:
        f.write(report)

    return report


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Generate and save confusion matrix figure.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_json(metrics, save_path):
    """
    Save metrics summary to JSON.
    """

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate_predictions(
    y_true,
    y_pred,
    class_names,
    report_dir
):
    """
    Complete evaluation pipeline.
    """

    metrics = compute_metrics(y_true, y_pred)

    report_text = save_classification_report(
        y_true,
        y_pred,
        class_names,
        f"{report_dir}/classification_report.txt"
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        f"{report_dir}/confusion_matrix.png"
    )

    save_metrics_json(
        metrics,
        f"{report_dir}/metrics_summary.json"
    )

    return metrics, report_text
