"""
Ablation study script.

Trains and evaluates different architecture variants to quantify
the contribution of each component of the proposed model.

Variants evaluated:
- InceptionV3 only
- ResNet50 only
- Hybrid without feature fusion
- Full hybrid model

Outputs:
reports/ablation_results.csv
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.utils import load_config, ensure_dir
from src.data import create_datasets
from src.metrics import compute_metrics
from src.model import build_model, ModelOptions
from src.reproducibility import prepare_reproducible_environment


CONFIG_PATH = "configs/default.yaml"


def evaluate_model(model, dataset):

    y_true = []
    y_pred = []

    for images, labels in dataset:

        preds = model.predict(images, verbose=0)

        preds = np.argmax(preds, axis=1)

        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    return compute_metrics(y_true, y_pred)


def main():

    cfg = load_config(CONFIG_PATH)

    prepare_reproducible_environment(cfg["seed"])

    image_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    num_classes = cfg["dataset"]["num_classes"]
    epochs = cfg["training"]["epochs"]

    splits_dir = Path(cfg["dataset"]["splits_dir"])

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    train_ds, val_ds, test_ds, class_to_index, index_to_class = create_datasets(
        train_csv,
        val_csv,
        test_csv,
        image_size,
        batch_size,
        augment_fn=None
    )

    experiments = [

        ("InceptionV3", ModelOptions(
            image_size=image_size,
            num_classes=num_classes,
            mode="inception_only"
        )),

        ("ResNet50", ModelOptions(
            image_size=image_size,
            num_classes=num_classes,
            mode="resnet_only"
        )),

        ("Hybrid_No_Fusion", ModelOptions(
            image_size=image_size,
            num_classes=num_classes,
            mode="hybrid_no_fusion"
        )),

        ("Hybrid_Full", ModelOptions(
            image_size=image_size,
            num_classes=num_classes,
            mode="hybrid_concat"
        )),
    ]

    results = []

    for name, opts in experiments:

        print(f"\nRunning experiment: {name}")

        model, _ = build_model(opts)

        model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=cfg["training"]["learning_rate"]
            ),

            loss="sparse_categorical_crossentropy",

            metrics=["accuracy"]
        )

        model.fit(

            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1
        )

        metrics = evaluate_model(model, test_ds)

        metrics["model"] = name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    report_dir = cfg["paths"]["reports"]

    ensure_dir(report_dir)

    output_file = Path(report_dir) / "ablation_results.csv"

    results_df.to_csv(output_file, index=False)

    print("\nAblation study completed.")
    print(results_df)


if __name__ == "__main__":
    main()
