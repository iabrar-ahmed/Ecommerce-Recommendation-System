"""
recommendation/preprocessing.py
---------------------------------
Step 2: Clean the raw dataset and build the combined "content" text field
used for training.

Reads:  ../dataset/products_dataset.csv
Writes: ../dataset/products_processed.csv

Usage:
    python preprocessing.py
"""

import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR.parent / "dataset" / "products_dataset.csv"
OUT_PATH = BASE_DIR.parent / "dataset" / "products_processed.csv"


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/extra spaces from a text field."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows.")

    text_cols = ["name", "category", "subcategory", "brand", "description"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    numeric_defaults = {"price": df["price"].median() if "price" in df else 0,
                         "rating": 0,
                         "num_reviews": 0,
                         "stock_quantity": 0}
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    df = df[df["price"] > 0]
    df["rating"] = df["rating"].clip(1.0, 5.0)
    df["stock_quantity"] = df["stock_quantity"].clip(lower=0)

    for col in text_cols:
        df[col + "_clean"] = df[col].apply(clean_text)

    df["content"] = (
        df["name_clean"] + " " +
        (df["brand_clean"] + " ") * 2 +
        (df["category_clean"] + " ") * 2 +
        (df["subcategory_clean"] + " ") * 2 +
        df["description_clean"]
    ).str.strip()

    df = df.drop(columns=[c for c in df.columns if c.endswith("_clean")])
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw dataset: {raw_df.shape}")

    processed_df = preprocess(raw_df)
    print(f"Processed dataset: {processed_df.shape}")
    print("\nSample content field (used for similarity):")
    print(processed_df[["name", "content"]].head(5).to_string())

    processed_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUT_PATH}")