"""
Hybrid CNN model definition: InceptionV3 + ResNet50 feature fusion.

Implements:
- InceptionV3 backbone (ImageNet pretrained)
- ResNet50 backbone (ImageNet pretrained)
- GlobalAveragePooling (vector features)
- Feature fusion via concatenation
- Dense classifier head (1024 -> 512) + Dropout
- Softmax output for multi-class classification

Also supports ablation variants:
- single backbone only (inceptionv3 OR resnet50)
- hybrid with/without fusion (controlled by mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tensorflow as tf


@dataclass
class ModelOptions:
    image_size: int = 224
    num_classes: int = 38
    dropout: float = 0.5
    dense_units: Tuple[int, int] = (1024, 512)

    # Backbones
    backbone1: str = "inceptionv3"  # inceptionv3 | resnet50 | none
    backbone2: str = "resnet50"     # resnet50 | inceptionv3 | none
    pretrained: bool = True

    # Training control
    freeze_backbones: bool = True
    # If fine_tune_at is not None, unfreeze from this layer index onward (for each backbone)
    fine_tune_at: Optional[int] = None

    # Mode
    # "hybrid_concat" = both backbones + concatenation fusion (main model)
    # "inception_only" = only inception backbone
    # "resnet_only" = only resnet backbone
    # "hybrid_no_fusion" = both backbones, but classifier uses only backbone1 features
    mode: str = "hybrid_concat"


def _get_backbone(name: str, image_size: int, pretrained: bool) -> tf.keras.Model:
    """Return a Keras application backbone with pooled vector output."""
    name = (name or "").lower()

    if name == "inceptionv3":
        return tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )

    if name == "resnet50":
        return tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )

    raise ValueError(f"Unsupported backbone: {name}")


def _freeze_or_finetune(backbone: tf.keras.Model, freeze: bool, fine_tune_at: Optional[int]) -> None:
    """
    Freeze full backbone, or optionally fine-tune from a given layer index onward.
    """
    backbone.trainable = True  # enable toggling at layer-level

    if freeze:
        for layer in backbone.layers:
            layer.trainable = False
        return

    # Not frozen: optionally fine-tune from fine_tune_at
    if fine_tune_at is None:
        for layer in backbone.layers:
            layer.trainable = True
        return

    # Fine-tune only deeper layers
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= fine_tune_at)


def build_model(opts: ModelOptions) -> Tuple[tf.keras.Model, Dict[str, tf.keras.Model]]:
    """
    Build the requested model variant.

    Returns:
        model: compiled-uncompiled Keras model (compile happens in training script)
        parts: dict with backbone references for logging / fine-tuning control
    """
    inputs = tf.keras.Input(shape=(opts.image_size, opts.image_size, 3), name="image")

    # NOTE: Our data pipeline already normalizes to [0,1].
    # Keras applications expect specific preprocessing; for reproducibility we keep it explicit here.
    # We map [0,1] -> [-1,1] for InceptionV3, and [0,1] -> [0,255] + mean subtraction for ResNet50 is typical.
    # To keep the hybrid consistent and simple, we use a shared normalization to [-1,1] which works well in practice.
    x = tf.keras.layers.Lambda(lambda t: (t * 2.0) - 1.0, name="normalize_minus1_plus1")(inputs)

    parts: Dict[str, tf.keras.Model] = {}

    mode = (opts.mode or "").lower().strip()

    if mode not in {"hybrid_concat", "inception_only", "resnet_only", "hybrid_no_fusion"}:
        raise ValueError(f"Unsupported mode: {opts.mode}")

    # Backbone selection based on mode
    use_inception = (mode in {"hybrid_concat", "hybrid_no_fusion", "inception_only"})
    use_resnet = (mode in {"hybrid_concat", "resnet_only"})  # resnet only or hybrid concat

    feat_list = []

    if use_inception:
        inc = _get_backbone("inceptionv3", opts.image_size, opts.pretrained)
        _freeze_or_finetune(inc, opts.freeze_backbones, opts.fine_tune_at)
        parts["inceptionv3"] = inc
        f_inc = inc(x)
        feat_list.append(tf.keras.layers.BatchNormalization(name="bn_inception")(f_inc))

    if use_resnet:
        res = _get_backbone("resnet50", opts.image_size, opts.pretrained)
        _freeze_or_finetune(res, opts.freeze_backbones, opts.fine_tune_at)
        parts["resnet50"] = res
        f_res = res(x)
        feat_list.append(tf.keras.layers.BatchNormalization(name="bn_resnet")(f_res))

    if mode == "hybrid_no_fusion":
        # Both backbones can be present, but we intentionally do NOT fuse;
        # classifier uses only the first feature stream (Inception feature if enabled).
        if not feat_list:
            raise RuntimeError("No features available for hybrid_no_fusion.")
        fused = feat_list[0]
    else:
        if len(feat_list) == 1:
            fused = feat_list[0]
        else:
            fused = tf.keras.layers.Concatenate(name="feature_concat")(feat_list)

    # Classifier head
    h = fused
    for i, units in enumerate(opts.dense_units, start=1):
        h = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(h)
        h = tf.keras.layers.BatchNormalization(name=f"bn_head_{i}")(h)
        h = tf.keras.layers.Dropout(opts.dropout, name=f"dropout_{i}")(h)

    outputs = tf.keras.layers.Dense(opts.num_classes, activation="softmax", name="classifier")(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"hybrid_cnn_{mode}")

    return model, parts


def build_from_config(cfg: dict) -> Tuple[tf.keras.Model, Dict[str, tf.keras.Model], ModelOptions]:
    """
    Convenience builder that reads configs/default.yaml structure.

    Expected keys:
      dataset.image_size
      dataset.num_classes
      training.dropout
      model.dense_units
      model.backbone1/backbone2 (informational)
      model.pretrained
      model.fusion
    """
    image_size = int(cfg["dataset"]["image_size"])
    num_classes = int(cfg["dataset"]["num_classes"])
    dropout = float(cfg["training"]["dropout"])
    dense_units = tuple(cfg["model"]["dense_units"])

    backbone1 = str(cfg["model"].get("backbone1", "inceptionv3")).lower()
    backbone2 = str(cfg["model"].get("backbone2", "resnet50")).lower()
    pretrained = bool(cfg["model"].get("pretrained", True))
    fusion = str(cfg["model"].get("fusion", "concatenate")).lower()

    # Map config to modes:
    # - default uses hybrid_concat
    # - other modes are handled by scripts/ablation.py
    mode = "hybrid_concat" if fusion == "concatenate" else "hybrid_concat"

    opts = ModelOptions(
        image_size=image_size,
        num_classes=num_classes,
        dropout=dropout,
        dense_units=(int(dense_units[0]), int(dense_units[1])),
        backbone1=backbone1,
        backbone2=backbone2,
        pretrained=pretrained,
        freeze_backbones=True,
        fine_tune_at=None,
        mode=mode,
    )

    model, parts = build_model(opts)
    return model, parts, opts
