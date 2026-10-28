"""Dataset ingestion and verification for BANKING77."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import pandas as pd

from complaint_ops.config import (
    BANKING77_CATEGORIES_URL,
    BANKING77_TEST_URL,
    BANKING77_TRAIN_URL,
    EXPECTED_INTENT_COUNT,
    EXPECTED_TEST_SAMPLES,
    EXPECTED_TRAIN_SAMPLES,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> Path:
    """Download a remote file if not already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size == 0:
        logger.info(f"Downloading {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        logger.info(f"Successfully downloaded {destination.name} ({destination.stat().st_size} bytes)")
    return destination


def download_banking77(raw_dir: Path = RAW_DATA_DIR) -> Tuple[Path, Path, Path]:
    """Download official BANKING77 train, test, and category mapping files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    categories_path = raw_dir / "categories.json"

    download_file(BANKING77_TRAIN_URL, train_path)
    download_file(BANKING77_TEST_URL, test_path)
    download_file(BANKING77_CATEGORIES_URL, categories_path)

    return train_path, test_path, categories_path


def load_banking77(
    raw_dir: Path = RAW_DATA_DIR, force_download: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load and validate the official BANKING77 train and test splits."""
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    categories_path = raw_dir / "categories.json"

    if (
        force_download
        or not train_path.exists()
        or not test_path.exists()
        or not categories_path.exists()
    ):
        download_banking77(raw_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    with open(categories_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    # Standardize column naming: expect 'text' and 'category'
    for df in (train_df, test_df):
        if "label" in df.columns and "category" not in df.columns:
            df.rename(columns={"label": "category"}, inplace=True)
        if "text" not in df.columns or "category" not in df.columns:
            raise ValueError(f"Expected columns ['text', 'category'], found: {list(df.columns)}")

    # Verify official dataset dimensions
    if len(train_df) != EXPECTED_TRAIN_SAMPLES:
        logger.warning(
            f"Train split has {len(train_df)} samples, expected {EXPECTED_TRAIN_SAMPLES}"
        )
    if len(test_df) != EXPECTED_TEST_SAMPLES:
        logger.warning(
            f"Test split has {len(test_df)} samples, expected {EXPECTED_TEST_SAMPLES}"
        )
    if len(categories) != EXPECTED_INTENT_COUNT:
        logger.warning(
            f"Found {len(categories)} categories, expected {EXPECTED_INTENT_COUNT}"
        )

    return train_df, test_df, categories


def inspect_dataset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, categories: List[str]
) -> Dict:
    """Inspect data characteristics, distributions, nulls, and label coverage."""
    train_cats = set(train_df["category"].unique())
    test_cats = set(test_df["category"].unique())
    all_cats = set(categories)

    train_lengths = train_df["text"].str.split().str.len()
    test_lengths = test_df["text"].str.split().str.len()

    inspection_summary = {
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "total_samples": int(len(train_df) + len(test_df)),
        "num_categories": len(categories),
        "train_unique_categories": len(train_cats),
        "test_unique_categories": len(test_cats),
        "unseen_test_categories": list(test_cats - train_cats),
        "categories_missing_in_train": list(all_cats - train_cats),
        "categories_missing_in_test": list(all_cats - test_cats),
        "train_null_text": int(train_df["text"].isnull().sum()),
        "test_null_text": int(test_df["text"].isnull().sum()),
        "train_tokens_mean": float(round(train_lengths.mean(), 2)),
        "train_tokens_median": float(train_lengths.median()),
        "train_tokens_max": int(train_lengths.max()),
        "test_tokens_mean": float(round(test_lengths.mean(), 2)),
        "test_tokens_median": float(test_lengths.median()),
        "test_tokens_max": int(test_lengths.max()),
        "train_class_min_samples": int(train_df["category"].value_counts().min()),
        "train_class_max_samples": int(train_df["category"].value_counts().max()),
        "test_class_min_samples": int(test_df["category"].value_counts().min()),
        "test_class_max_samples": int(test_df["category"].value_counts().max()),
    }
    return inspection_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tr_df, te_df, cats = load_banking77()
    summary = inspect_dataset(tr_df, te_df, cats)
    print(json.dumps(summary, indent=2))
