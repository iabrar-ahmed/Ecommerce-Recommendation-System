"""
utils/seed_database.py
------------------------
Loads dataset/products_processed.csv into the SQL database. Run this once
after preprocessing.py has produced products_processed.csv, and again any
time the dataset changes.

Usage (from project root):
    python -m utils.seed_database
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # project root
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from app import create_app
from database.db import db
from database.models import Product

DATA_PATH = BASE_DIR / "dataset" / "products_processed.csv"


def seed():
    app = create_app()
    with app.app_context():
        db.create_all()

        df = pd.read_csv(DATA_PATH)
        print(f"Loading {len(df)} products into the database...")

        # clear existing rows so re-running this script is safe/idempotent
        Product.query.delete()

        products = [
            Product(
                product_id=int(row["product_id"]),
                name=row["name"],
                category=row["category"],
                subcategory=row["subcategory"],
                brand=row["brand"],
                price=float(row["price"]),
                rating=float(row["rating"]),
                num_reviews=int(row["num_reviews"]),
                stock_quantity=int(row["stock_quantity"]),
                description=row["description"],
            )
            for _, row in df.iterrows()
        ]

        db.session.bulk_save_objects(products)
        db.session.commit()

        count = Product.query.count()
        print(f"Done. {count} products now in the database.")


if __name__ == "__main__":
    seed()
