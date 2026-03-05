"""
Cross-validation experiment script.

Runs stratified K-fold cross-validation and reports
mean and standard deviation of performance metrics.

Outputs:
reports/crossval_results.json
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.utils import load_config, ensure_dir
from src.data import build_dataset
from src.model import build_from_config
from src.metrics import compute_metrics
from src.reproducibility import prepare_reproducible_environment


CONFIG_PATH = "configs/default.yaml"


def load_full_dataset(processed_dir):
    rows = []

    for class_dir in Path(processed_dir).iterdir():

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for img in class_dir.glob("*"):

            rows.append({
                "image_path": str(img),
                "label": label
            })

    df = pd.DataFrame(rows)

    return df


def encode_labels(labels):
    classes = sorted(list(set(labels)))

    class_to_index = {c: i for i, c in enumerate(classes)}

    encoded = [class_to_index[l] for l in labels]

    return np.array(encoded), class_to_index


def main():

    cfg = load_config(CONFIG_PATH)

    seed = cfg["seed"]

    prepare_reproducible_environment(seed)

    processed_dir = cfg["dataset"]["processed_dir"]
    image_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    epochs = cfg["training"]["epochs"]

    df = load_full_dataset(processed_dir)

    labels_encoded, class_to_index = encode_labels(df["label"])

    image_paths = df["image_path"].values

    skf = StratifiedKFold(
        n_splits=cfg["cross_validation"]["folds"],
        shuffle=True,
        random_state=seed
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels_encoded)):

        print(f"Running fold {fold+1}")

        train_paths = image_paths[train_idx]
        val_paths = image_paths[val_idx]

        train_labels = labels_encoded[train_idx]
        val_labels = labels_encoded[val_idx]

        train_ds = build_dataset(
            train_paths,
            train_labels,
            image_size,
            batch_size,
            shuffle=True
        )

        val_ds = build_dataset(
            val_paths,
            val_labels,
            image_size,
            batch_size,
            shuffle=False
        )

        model, _, _ = build_from_config(cfg)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg["training"]["learning_rate"]
        )

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )

        y_true = []
        y_pred = []

        for images, labels in val_ds:

            preds = model.predict(images, verbose=0)

            preds = np.argmax(preds, axis=1)

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

        metrics = compute_metrics(y_true, y_pred)

        results.append(metrics)

    # Aggregate results
    metrics_df = pd.DataFrame(results)

    summary = {

        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "accuracy_std": float(metrics_df["accuracy"].std()),

        "precision_mean": float(metrics_df["precision_macro"].mean()),
        "precision_std": float(metrics_df["precision_macro"].std()),

        "recall_mean": float(metrics_df["recall_macro"].mean()),
        "recall_std": float(metrics_df["recall_macro"].std()),

        "f1_mean": float(metrics_df["f1_macro"].mean()),
        "f1_std": float(metrics_df["f1_macro"].std())
    }

    report_dir = cfg["paths"]["reports"]

    ensure_dir(report_dir)

    output_path = Path(report_dir) / "crossval_results.json"

    metrics_df.to_json(Path(report_dir) / "crossval_all_folds.json", indent=4)

    with open(output_path, "w") as f:

        import json
        json.dump(summary, f, indent=4)

    print("Cross-validation completed.")
    print(summary)


if __name__ == "__main__":
    main()
