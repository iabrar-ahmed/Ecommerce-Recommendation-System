"""
recommendation/inspect_dataset.py
----------------------------------
Step 1: Inspect the raw dataset. Run this first, read the output, before
touching preprocessing.

Run from anywhere — paths are resolved relative to this file's location,
not your terminal's current directory.

Usage (from the recommendation/ folder, or the project root — both work):
    python inspect_dataset.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # .../recommendation/
DATA_PATH = BASE_DIR.parent / "dataset" / "products_dataset.csv"


def inspect(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. SHAPE & BASIC INFO")
    print("=" * 70)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.info())

    print("\n" + "=" * 70)
    print("2. FIRST FEW ROWS")
    print("=" * 70)
    print(df.head(5).to_string())

    print("\n" + "=" * 70)
    print("3. MISSING VALUES PER COLUMN")
    print("=" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}))

    print("\n" + "=" * 70)
    print("4. DUPLICATE CHECKS")
    print("=" * 70)
    print(f"Fully duplicated rows: {df.duplicated().sum()}")
    if "product_id" in df.columns:
        print(f"Duplicate product_ids: {df['product_id'].duplicated().sum()}")
    if "name" in df.columns:
        print(f"Duplicate product names: {df['name'].duplicated().sum()}")

    print("\n" + "=" * 70)
    print("5. NUMERIC COLUMN STATS")
    print("=" * 70)
    print(df.describe())

    print("\n" + "=" * 70)
    print("6. CATEGORICAL COLUMN CARDINALITY")
    print("=" * 70)
    for col in ["category", "subcategory", "brand"]:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")

    print("\n" + "=" * 70)
    print("7. CATEGORY DISTRIBUTION")
    print("=" * 70)
    print(df["category"].value_counts())

    print("\n" + "=" * 70)
    print("8. SUBCATEGORY DISTRIBUTION (top 15)")
    print("=" * 70)
    print(df["subcategory"].value_counts().head(15))

    print("\n" + "=" * 70)
    print("9. PRICE / RATING SANITY CHECKS")
    print("=" * 70)
    print(f"Negative or zero prices: {(df['price'] <= 0).sum()}")
    print(f"Ratings outside [1,5]: {((df['rating'] < 1) | (df['rating'] > 5)).sum()}")
    print(f"Negative stock: {(df['stock_quantity'] < 0).sum()}")

    print("\n" + "=" * 70)
    print("10. TEXT FIELD LENGTH (description) — useful before TF-IDF")
    print("=" * 70)
    desc_len = df["description"].str.split().str.len()
    print(desc_len.describe())
    print(f"Empty descriptions: {(desc_len.fillna(0) == 0).sum()}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    inspect(df)
