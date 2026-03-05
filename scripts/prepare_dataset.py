"""
Dataset preparation script.

This script merges the PlantCLEF and Indian Medicinal Plants datasets
into a unified directory structure used by the training pipeline.

Steps performed:
1. Scan dataset directories
2. Validate images
3. Standardize class folder structure
4. Copy valid images to processed dataset
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def is_valid_image(path):
    """
    Check if an image file is valid.
    """

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def process_dataset(dataset_dir, output_dir):
    """
    Process dataset folder.

    Each class folder will be copied into the processed directory.
    """

    dataset_dir = Path(dataset_dir)

    for class_dir in dataset_dir.iterdir():

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        target_class_dir = Path(output_dir) / class_name

        target_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*"))

        for img_path in tqdm(images, desc=f"Processing {class_name}"):

            if not img_path.is_file():
                continue

            if not is_valid_image(img_path):
                continue

            dst = target_class_dir / img_path.name

            shutil.copy(img_path, dst)


def main():

    raw_dir = Path(RAW_DATA_DIR)
    processed_dir = Path(PROCESSED_DIR)

    processed_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in raw_dir.iterdir() if d.is_dir()]

    if len(datasets) == 0:
        print("No datasets found in data/raw/")
        return

    for dataset in datasets:

        print(f"Processing dataset: {dataset.name}")

        process_dataset(dataset, processed_dir)

    print("Dataset preparation completed.")
    print(f"Processed dataset available at: {processed_dir}")


if __name__ == "__main__":
    main()
