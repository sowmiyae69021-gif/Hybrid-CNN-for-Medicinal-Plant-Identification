"""
Dataset split generation script.

Creates reproducible train / validation / test splits
from the processed dataset directory.

Outputs:
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


CONFIG_PATH = "configs/default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def collect_dataset(processed_dir):
    """
    Scan dataset directory and build dataframe.
    """

    rows = []

    for class_dir in Path(processed_dir).iterdir():

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for img in class_dir.glob("*"):

            if img.is_file():

                rows.append({
                    "image_path": str(img),
                    "label": label
                })

    df = pd.DataFrame(rows)

    return df


def create_splits(df, train_ratio, val_ratio, test_ratio, seed):

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["label"],
        random_state=seed
    )

    val_size = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df["label"],
        random_state=seed
    )

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, splits_dir):

    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print("Dataset splits created:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


def main():

    cfg = load_config()

    processed_dir = cfg["dataset"]["processed_dir"]
    splits_dir = cfg["dataset"]["splits_dir"]

    train_ratio = cfg["splits"]["train_ratio"]
    val_ratio = cfg["splits"]["val_ratio"]
    test_ratio = cfg["splits"]["test_ratio"]

    seed = cfg["seed"]

    df = collect_dataset(processed_dir)

    train_df, val_df, test_df = create_splits(
        df,
        train_ratio,
        val_ratio,
        test_ratio,
        seed
    )

    save_splits(train_df, val_df, test_df, splits_dir)


if __name__ == "__main__":
    main()
