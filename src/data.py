"""
Dataset loading utilities for the Hybrid CNN medicinal plant project.

Responsibilities:
- Read train/val/test split CSV files
- Build TensorFlow datasets
- Decode and preprocess images
- Return tf.data pipelines ready for training/evaluation
"""

import pandas as pd
import tensorflow as tf
from pathlib import Path


def load_split_csv(csv_path):
    """
    Load a dataset split CSV.

    Expected CSV format:
    image_path,label
    path/to/image.jpg,class_name
    """

    df = pd.read_csv(csv_path)

    image_paths = df["image_path"].values
    labels = df["label"].values

    return image_paths, labels


def build_label_encoder(labels):
    """
    Build label mapping dictionary.
    """

    classes = sorted(list(set(labels)))
    class_to_index = {c: i for i, c in enumerate(classes)}
    index_to_class = {i: c for c, i in class_to_index.items()}

    return class_to_index, index_to_class


def encode_labels(labels, class_to_index):
    """
    Convert string labels to integer labels.
    """

    return [class_to_index[l] for l in labels]


def decode_image(filename, label, image_size):
    """
    Read and decode an image.
    """

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def build_dataset(
    image_paths,
    labels,
    image_size,
    batch_size,
    shuffle=False,
    augment_fn=None
):
    """
    Build tf.data dataset.
    """

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    ds = ds.map(
        lambda x, y: decode_image(x, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment_fn is not None:
        ds = ds.map(
            lambda x, y: (augment_fn(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_datasets(
    train_csv,
    val_csv,
    test_csv,
    image_size,
    batch_size,
    augment_fn=None
):
    """
    Create train, validation, and test datasets.
    """

    train_paths, train_labels = load_split_csv(train_csv)
    val_paths, val_labels = load_split_csv(val_csv)
    test_paths, test_labels = load_split_csv(test_csv)

    class_to_index, index_to_class = build_label_encoder(train_labels)

    train_labels = encode_labels(train_labels, class_to_index)
    val_labels = encode_labels(val_labels, class_to_index)
    test_labels = encode_labels(test_labels, class_to_index)

    train_ds = build_dataset(
        train_paths,
        train_labels,
        image_size,
        batch_size,
        shuffle=True,
        augment_fn=augment_fn
    )

    val_ds = build_dataset(
        val_paths,
        val_labels,
        image_size,
        batch_size,
        shuffle=False
    )

    test_ds = build_dataset(
        test_paths,
        test_labels,
        image_size,
        batch_size,
        shuffle=False
    )

    return train_ds, val_ds, test_ds, class_to_index, index_to_class
