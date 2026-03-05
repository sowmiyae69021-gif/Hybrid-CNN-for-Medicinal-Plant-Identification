"""
Utilities to enforce experiment reproducibility.

This module centralizes seed initialization and deterministic
execution settings for Python, NumPy, and TensorFlow so that
experiments can be reproduced across runs as closely as possible.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42):
    """
    Set seeds for Python, NumPy, and TensorFlow.

    Args:
        seed (int): Random seed value.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_deterministic_ops():
    """
    Enable TensorFlow deterministic operations where supported.

    This reduces nondeterminism in GPU operations and improves
    experiment reproducibility.
    """

    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older TF versions may not support this API
        pass


def configure_gpu_memory_growth():
    """
    Prevent TensorFlow from allocating all GPU memory at once.
    This allows multiple experiments to run without memory conflicts.
    """

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass


def prepare_reproducible_environment(seed: int = 42):
    """
    Configure the full reproducible environment.

    This function should be called at the beginning of training scripts.

    Args:
        seed (int): Random seed value.
    """

    set_global_seed(seed)
    enable_deterministic_ops()
    configure_gpu_memory_growth()
