"""
Training script for Hybrid CNN medicinal plant classifier.

Pipeline:
1. Load configuration
2. Initialize reproducible environment
3. Load dataset splits
4. Build TensorFlow datasets
5. Construct hybrid CNN model
6. Train model
7. Save checkpoints and training history
"""

import tensorflow as tf
from pathlib import Path

from src.reproducibility import prepare_reproducible_environment
from src.utils import load_config, ensure_dir, save_training_history
from src.data import create_datasets
from src.augment import augment
from src.model import build_from_config


CONFIG_PATH = "configs/default.yaml"


def build_optimizer(cfg):
    lr = cfg["training"]["learning_rate"]
    opt_name = cfg["training"]["optimizer"].lower()

    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)

    raise ValueError(f"Unsupported optimizer: {opt_name}")


def main():

    # Load config
    cfg = load_config(CONFIG_PATH)

    seed = cfg["seed"]

    prepare_reproducible_environment(seed)

    image_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    splits_dir = Path(cfg["dataset"]["splits_dir"])

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    # Build datasets
    train_ds, val_ds, test_ds, class_to_index, index_to_class = create_datasets(
        train_csv,
        val_csv,
        test_csv,
        image_size,
        batch_size,
        augment_fn=augment
    )

    # Build model
    model, parts, opts = build_from_config(cfg)

    optimizer = build_optimizer(cfg)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Create directories
    checkpoint_dir = cfg["paths"]["checkpoints"]
    logs_dir = cfg["paths"]["logs"]

    ensure_dir(checkpoint_dir)
    ensure_dir(logs_dir)

    # Callbacks
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir
        )
    ]

    # Train
    history = model.fit(

        train_ds,
        validation_data=val_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks
    )

    # Save final model
    model.save(f"{checkpoint_dir}/final_model")

    # Save history
    save_training_history(history, f"{logs_dir}/training_history.json")

    print("Training completed.")


if __name__ == "__main__":
    main()
