"""
database/models.py
--------------------
Database schema. One table for now (Product) — this is the DB-backed
source of truth for browsing/searching/filtering products in the web app.

Note: the recommendation engine (recommendation/recommender.py) does NOT
read from this database — it reads directly from dataset/products_processed.csv
and the precomputed similarity matrix, which is faster for that specific job.
This DB is used for everything else: listing, searching, filtering, product pages.
"""

from database.db import db


class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)               # DB's own internal id
    product_id = db.Column(db.Integer, unique=True, nullable=False, index=True)  # matches dataset's product_id
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), index=True)
    subcategory = db.Column(db.String(100), index=True)
    brand = db.Column(db.String(100), index=True)
    price = db.Column(db.Float, nullable=False)
    rating = db.Column(db.Float)
    num_reviews = db.Column(db.Integer)
    stock_quantity = db.Column(db.Integer)
    description = db.Column(db.Text)

    def to_dict(self):
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "price": self.price,
            "rating": self.rating,
            "num_reviews": self.num_reviews,
            "stock_quantity": self.stock_quantity,
            "description": self.description,
        }

    def __repr__(self):
        return f"<Product {self.product_id}: {self.name}>"
