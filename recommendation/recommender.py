"""
recommendation/recommender.py
--------------------------------
Production recommender module — this is what app.py imports.
Inference-only: loads the artifacts saved by train_model.py, no training here.

Usage inside app.py (project root):

    from recommendation.recommender import ProductRecommender
    recommender = ProductRecommender()   # load once at startup, not per-request

    @app.route("/product/<int:product_id>")
    def product_detail(product_id):
        product = recommender.get_product(product_id)
        related = recommender.recommend(product_id, top_n=5)
        return render_template("product.html", product=product, related=related)

Standalone test:
    python recommender.py
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "dataset" / "products_processed.csv"

MODEL_DIR = BASE_DIR / "model"
SIMILARITY_PATH = MODEL_DIR / "similarity_matrix.joblib"
INDEX_PATH = MODEL_DIR / "product_index.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"


class ProductRecommender:
    def __init__(
        self,
        data_path=DATA_PATH,
        similarity_path=SIMILARITY_PATH,
        index_path=INDEX_PATH,
        vectorizer_path=VECTORIZER_PATH,
        matrix_path=MATRIX_PATH,
    ):
        self.df = pd.read_csv(data_path)
        self.similarity_matrix = joblib.load(similarity_path)
        self.product_index = joblib.load(index_path)
        # Needed for free-text fallback search (query -> nearest product by meaning)
        self.vectorizer = joblib.load(vectorizer_path)
        self.tfidf_matrix = joblib.load(matrix_path)

    def recommend(self, product_id: int, top_n: int = 5):
        """Return top_n products most similar to the given product_id."""
        if product_id not in self.product_index:
            raise ValueError(f"product_id {product_id} not found in catalog.")

        idx = self.product_index[product_id]
        sim_row = self.similarity_matrix[idx]
        sim_row = sim_row.toarray().flatten() if hasattr(sim_row, "toarray") else np.asarray(sim_row).flatten()

        top_indices = np.argsort(sim_row)[::-1]
        top_indices = [i for i in top_indices if i != idx][:top_n]

        results = self.df.iloc[top_indices][
            ["product_id", "name", "category", "subcategory", "brand", "price", "rating"]
        ].copy()
        results["similarity_score"] = sim_row[top_indices].round(3)

        return results.to_dict(orient="records")

    def get_product(self, product_id: int):
        """Fetch a single product's own details (for the product detail page)."""
        idx = self.product_index[product_id]
        return self.df.iloc[idx].to_dict()

    def find_product_by_name(self, name: str):
        """
        Find the product_id whose name best matches the given search text.
        Tries an exact/substring match first (fast, predictable), then falls
        back to fuzzy matching for typos or partial/rephrased queries.
        Returns None if nothing reasonable is found.
        """
        query = name.lower().strip()
        if not query:
            return None

        # 1. Substring match — e.g. "sneakers" matches "Draper & Co Elegant Sneakers"
        mask = self.df["name"].str.lower().str.contains(query, na=False, regex=False)
        matches = self.df[mask]
        if not matches.empty:
            # prefer the shortest matching name — usually the closest/most direct match
            best = matches.loc[matches["name"].str.len().idxmin()]
            return int(best["product_id"])

        # 2. Semantic fallback — uses the same TF-IDF model the recommender was
        # trained with, so "laptop bag" can match "Backpack" even with zero
        # words in common with the exact product name. Much better than plain
        # string-similarity matching, which can latch onto an unrelated product
        # just because the letters happen to look similar.
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = int(sims.argmax())
        if sims[best_idx] > 0:  # at least some real overlap in meaning
            return int(self.df.iloc[best_idx]["product_id"])

        return None

    def recommend_by_name(self, name: str, top_n: int = 5):
        """
        The main entry point for name-based search: give it a product name
        (full, partial, or approximate), get back the matched product plus
        its top_n most similar products. No product_id needed from the caller.

        Raises ValueError if no reasonably close product name is found.
        """
        product_id = self.find_product_by_name(name)
        if product_id is None:
            raise ValueError(f"No product found matching '{name}'")

        matched_product = self.get_product(product_id)
        recommendations = self.recommend(product_id, top_n=top_n)
        return matched_product, recommendations

    def search(self, query: str, top_n: int = 20):
        """Simple keyword search over product names (for the site's search bar)."""
        query = query.lower().strip()
        mask = self.df["name"].str.lower().str.contains(query, na=False)
        return self.df[mask].head(top_n).to_dict(orient="records")


if __name__ == "__main__":
    recommender = ProductRecommender()

    # --- Name-based search demo (no product_id needed) ---
    search_terms = ["sneakers", "fitness band", "laptop bagg"]  # last one has a typo, on purpose
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        try:
            matched_product, recommendations = recommender.recommend_by_name(term, top_n=5)
            print(f"Matched: {matched_product['name']}  ({matched_product['category']} / {matched_product['subcategory']})")
            print("Recommended:")
            for item in recommendations:
                print(f"   - {item['name']}  |  ${item['price']}  |  sim={item['similarity_score']}")
        except ValueError as e:
            print(f"   {e}")
