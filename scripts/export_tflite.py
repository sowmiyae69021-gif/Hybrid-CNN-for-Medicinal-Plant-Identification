"""
Export trained model to TensorFlow Lite format for mobile inference.

Outputs:
exports/
    model.tflite
    labels.txt
"""

import tensorflow as tf
from pathlib import Path

from src.utils import load_config, ensure_dir, export_labels
from src.data import create_datasets


CONFIG_PATH = "configs/default.yaml"


def main():

    cfg = load_config(CONFIG_PATH)

    image_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    splits_dir = Path(cfg["dataset"]["splits_dir"])

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    # Load datasets only to retrieve label mapping
    train_ds, val_ds, test_ds, class_to_index, index_to_class = create_datasets(
        train_csv,
        val_csv,
        test_csv,
        image_size,
        batch_size,
        augment_fn=None
    )

    # Load trained model
    checkpoint_dir = Path(cfg["paths"]["checkpoints"])
    model_path = checkpoint_dir / "best_model.h5"

    print("Loading trained model:", model_path)

    model = tf.keras.models.load_model(model_path)

    # Create export directory
    export_dir = Path(cfg["paths"]["exports"])
    ensure_dir(export_dir)

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save model
    model_path = export_dir / "model.tflite"

    with open(model_path, "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved:", model_path)

    # Export labels
    labels_path = export_dir / "labels.txt"

    export_labels(index_to_class, labels_path)

    print("Labels saved:", labels_path)

    print("Model export completed.")


if __name__ == "__main__":
    main()
