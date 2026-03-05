"""
Image augmentation pipeline for training.

This module implements the augmentation strategy described in the
manuscript. It uses TensorFlow native operations so it integrates
directly with the tf.data pipeline.

Augmentations implemented:
- horizontal flip
- rotation
- zoom
- translation
- brightness adjustment
"""

import tensorflow as tf


def random_flip(image):
    """Random horizontal flip."""
    image = tf.image.random_flip_left_right(image)
    return image


def random_rotation(image, max_angle=20):
    """
    Random rotation in degrees.
    """

    angle = tf.random.uniform([], -max_angle, max_angle)
    angle = angle * 3.14159265 / 180.0

    image = tfa_image_rotate(image, angle)
    return image


def random_zoom(image, zoom_range=0.2):
    """
    Random zoom by cropping and resizing.
    """

    scales = tf.random.uniform([1], 1 - zoom_range, 1 + zoom_range)
    scale = scales[0]

    new_size = tf.cast(scale * tf.cast(tf.shape(image)[:2], tf.float32), tf.int32)

    image = tf.image.resize_with_crop_or_pad(image, new_size[0], new_size[1])
    image = tf.image.resize(image, (224, 224))

    return image


def random_shift(image, width_shift=0.1, height_shift=0.1):
    """
    Random translation shift.
    """

    h, w = tf.shape(image)[0], tf.shape(image)[1]

    dx = tf.cast(tf.random.uniform([], -width_shift, width_shift) * tf.cast(w, tf.float32), tf.int32)
    dy = tf.cast(tf.random.uniform([], -height_shift, height_shift) * tf.cast(h, tf.float32), tf.int32)

    image = tf.roll(image, shift=[dy, dx], axis=[0, 1])

    return image


def random_brightness(image, brightness_range=(0.8, 1.2)):
    """
    Random brightness scaling.
    """

    factor = tf.random.uniform([], brightness_range[0], brightness_range[1])
    image = tf.clip_by_value(image * factor, 0.0, 1.0)

    return image


def augment(image):
    """
    Combined augmentation pipeline.

    Applied sequentially during training.
    """

    image = random_flip(image)
    image = random_rotation(image)
    image = random_zoom(image)
    image = random_shift(image)
    image = random_brightness(image)

    return image


def tfa_image_rotate(image, angle):
    """
    Lightweight rotation helper using projective transform.
    Avoids requiring tensorflow-addons dependency.
    """

    return tf.keras.layers.RandomRotation(
        factor=angle / 3.14159265
    )(tf.expand_dims(image, 0))[0]
