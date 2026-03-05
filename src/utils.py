"""
General utility functions used across training, evaluation, and export scripts.

Responsibilities:
- Load YAML configuration files
- Create experiment directories
- Save training history
- Export label files for inference
"""

import os
import yaml
import json


def load_config(config_path):
    """
    Load YAML configuration file.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def save_json(data, path):
    """
    Save dictionary as JSON.
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_training_history(history, path):
    """
    Save training history from Keras training.
    """

    hist = history.history

    with open(path, "w") as f:
        json.dump(hist, f, indent=4)


def export_labels(index_to_class, path):
    """
    Export labels file for inference/mobile usage.
    """

    with open(path, "w") as f:
        for i in range(len(index_to_class)):
            f.write(index_to_class[i] + "\n")
