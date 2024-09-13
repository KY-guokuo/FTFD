import os

import pandas as pd
from datasets import Dataset

label2id = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8":8}
id2label = {id: label for label, id in label2id.items()}


def load_training_dataset(model_type: str = "") -> Dataset:
    """Load dataset."""
    SFC_dataset_df = pd.read_csv(
        "../dataset/training_dataset.csv",
        header=None,
        names=["label", "text"],
    )

    SFC_dataset_df["label"] = SFC_dataset_df["label"].astype(str)
    # Convert labels to integers
    SFC_dataset_df["label"] = SFC_dataset_df["label"].map(
        label2id
    )

    SFC_dataset_df["text"] = SFC_dataset_df["text"].astype(str)
    dataset = Dataset.from_pandas(SFC_dataset_df)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset


def load_test_dataset(model_type: str = "") -> Dataset:
    """Load dataset."""
    SFC_dataset_df = pd.read_csv(
        "../dataset/test_dataset.csv",
        header=None,
        names=["label", "text"],
    )

    SFC_dataset_df["label"] = SFC_dataset_df["label"].astype(str)
    SFC_dataset_df["label"] = SFC_dataset_df["label"].map(
        label2id
    )

    SFC_dataset_df["text"] = SFC_dataset_df["text"].astype(str)
    dataset = Dataset.from_pandas(SFC_dataset_df)
    dataset = dataset.shuffle(seed=42)

    return dataset
