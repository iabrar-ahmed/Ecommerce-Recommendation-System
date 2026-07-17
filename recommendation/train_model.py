"""
recommendation/train_model.py
--------------------------------
Step 3: Train the content-based recommendation model (TF-IDF + cosine similarity).

Reads:  ../dataset/products_processed.csv
Writes: ./model/tfidf_vectorizer.joblib
        ./model/tfidf_matrix.joblib
        ./model/similarity_matrix.joblib
        ./model/product_index.joblib

Usage:
    python train_model.py
"""

import pandas as pd
import joblib
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "dataset" / "products_processed.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
SIMILARITY_PATH = MODEL_DIR / "similarity_matrix.joblib"
INDEX_PATH = MODEL_DIR / "product_index.joblib"


def train():
    df = pd.read_csv(DATA_PATH)
    print(f"Training on {len(df)} products.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1
    )

    t0 = time.time()
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}  (fit in {time.time()-t0:.2f}s)")

    t0 = time.time()
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
    print(f"Similarity matrix shape: {similarity_matrix.shape}  (computed in {time.time()-t0:.2f}s)")

    product_index = pd.Series(df.index, index=df["product_id"]).to_dict()

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, MATRIX_PATH)
    joblib.dump(similarity_matrix, SIMILARITY_PATH)
    joblib.dump(product_index, INDEX_PATH)
    print(f"\nSaved model artifacts to {MODEL_DIR}/")

    return df, vectorizer, tfidf_matrix, similarity_matrix, product_index


def quick_sanity_check(df, similarity_matrix, product_index, sample_size=3):
    import numpy as np
    print("\n" + "=" * 70)
    print("SANITY CHECK — sample recommendations")
    print("=" * 70)

    sample_ids = df["product_id"].sample(sample_size, random_state=1).tolist()
    for pid in sample_ids:
        idx = product_index[pid]
        sim_row = similarity_matrix[idx].toarray().flatten()
        top_idx = np.argsort(sim_row)[::-1][1:6]

        print(f"\nProduct: {df.loc[idx, 'name']}  [{df.loc[idx, 'category']} / {df.loc[idx, 'subcategory']}]")
        print("Recommended:")
        for i in top_idx:
            print(f"   - {df.loc[i, 'name']}  ({df.loc[i, 'category']} / {df.loc[i, 'subcategory']})  sim={sim_row[i]:.3f}")


if __name__ == "__main__":
    df, vectorizer, tfidf_matrix, similarity_matrix, product_index = train()
    quick_sanity_check(df, similarity_matrix, product_index)
