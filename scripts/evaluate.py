"""
Model evaluation script.

This script:
1. Loads the trained model
2. Runs inference on the test dataset
3. Computes classification metrics
4. Saves evaluation reports

Outputs generated:
reports/
    classification_report.txt
    confusion_matrix.png
    metrics_summary.json
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils import load_config, ensure_dir
from src.data import create_datasets
from src.metrics import evaluate_predictions


CONFIG_PATH = "configs/default.yaml"


def main():

    cfg = load_config(CONFIG_PATH)

    image_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    splits_dir = Path(cfg["dataset"]["splits_dir"])

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    # Build datasets (no augmentation)
    train_ds, val_ds, test_ds, class_to_index, index_to_class = create_datasets(
        train_csv,
        val_csv,
        test_csv,
        image_size,
        batch_size,
        augment_fn=None
    )

    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    # Load trained model
    model_path = Path(cfg["paths"]["checkpoints"]) / "best_model.h5"

    print("Loading model:", model_path)

    model = tf.keras.models.load_model(model_path)

    # Run inference
    y_true = []
    y_pred = []

    for images, labels in test_ds:

        preds = model.predict(images, verbose=0)

        preds = np.argmax(preds, axis=1)

        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    # Create reports directory
    report_dir = cfg["paths"]["reports"]
    ensure_dir(report_dir)

    metrics, report_text = evaluate_predictions(
        y_true,
        y_pred,
        class_names,
        report_dir
    )

    print("Evaluation completed.")

    print("Accuracy:", metrics["accuracy"])
    print("F1 score:", metrics["f1_macro"])


if __name__ == "__main__":
    main()
