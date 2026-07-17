# 🛍️ E-Commerce Recommendation System

A full-stack e-commerce web application with a built-in **product recommendation engine**. Browse products, search by name, and get "related products" suggestions powered by content-based machine learning — similar to Amazon's "customers also viewed" feature.

Built as a group project using **Flask**, **SQLAlchemy**, and **scikit-learn**.

---

## ✨ Features

- 🔍 **Smart search** — search by product name (even partial or slightly misspelled) and get both matching products *and* AI-recommended related items
- 🎯 **Content-based recommendations** — every product page shows a "You might also like" section powered by TF-IDF + cosine similarity
- 🗂️ **Category browsing** — filter the catalog by category
- 💾 **SQLite database** — product catalog served from a proper relational database via SQLAlchemy
- 💰 **INR currency formatting** — prices displayed in ₹ (Indian Rupees)
- 📱 Clean, responsive product grid UI

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask |
| Database | SQLite + SQLAlchemy (Flask-SQLAlchemy) |
| Recommendation Engine | scikit-learn (TF-IDF, cosine similarity), pandas, numpy |
| Frontend | HTML, Jinja2 templates, vanilla CSS |
| Data | 4,000-product synthetic e-commerce catalog |

---

## 📁 Project Structure

```
Ecommerce-Recommendation-System/
├── app.py                        # Main Flask application (routes, currency filter)
├── requirements.txt
├── .gitignore
│
├── database/
│   ├── db.py                     # Shared SQLAlchemy instance
│   └── models.py                 # Product model / schema
│
├── dataset/
│   ├── products_dataset.csv      # Raw product data
│   └── products_processed.csv    # Cleaned data (output of preprocessing.py)
│
├── recommendation/
│   ├── inspect_dataset.py        # Step 1: EDA on raw data
│   ├── preprocessing.py          # Step 2: Cleaning + feature engineering
│   ├── train_model.py            # Step 3: Trains TF-IDF + cosine similarity model
│   ├── recommender.py            # Step 4: Production module used by app.py
│   └── model/                    # Saved model artifacts (.joblib files)
│
├── templates/
│   ├── base.html                 # Shared layout (nav + search bar)
│   ├── index.html                # Homepage
│   ├── product.html              # Product detail + related products
│   ├── search_results.html       # Search / category results + related products
│   └── 404.html
│
├── static/
│   ├── css/style.css
│   ├── js/
│   └── images/
│
└── utils/
    └── seed_database.py          # Loads dataset into the SQL database
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Ecommerce-Recommendation-System
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the recommendation model

Run these once (and again any time the dataset changes):

```bash
python recommendation/preprocessing.py
python recommendation/train_model.py
```

### 5. Seed the database

```bash
python -m utils.seed_database
```

### 6. Run the app

```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

---

## 🧠 How the Recommendation Engine Works

1. **Preprocessing** combines each product's name, brand, category, subcategory, and description into a single weighted text field.
2. **Training** runs TF-IDF vectorization over that text (unigrams + bigrams), then computes cosine similarity between every pair of products.
3. **Inference** (`recommender.py`) looks up a product and instantly returns the most similar products from the precomputed similarity matrix — no recomputation needed per request.
4. **Name-based search** (`recommend_by_name()`) lets users search by typing a product name instead of needing a product ID — it matches the closest product by name, then falls back to semantic (meaning-based) matching using the same TF-IDF model if there's no exact match.

This is a **content-based filtering** approach — it works from day one with zero user behavior data, and can later be extended with collaborative filtering if user interaction data (views/purchases) is added.

---

## 🔄 Retraining the Model

If the product dataset changes, re-run:

```bash
python recommendation/preprocessing.py
python recommendation/train_model.py
```

⚠️ If you see a `KeyError` or mismatch error from `recommender.py`, it means the model artifacts in `recommendation/model/` are out of sync with `dataset/products_processed.csv` — re-running the two commands above fixes it.

---

## 👥 Team & Roles

| Role | Responsibilities |
|---|---|
| **Recommendation Engine** | Dataset preprocessing, TF-IDF model training, `recommender.py` |
| **Backend & Database** | Flask routes, SQLAlchemy models, database seeding |
| **Frontend & Integration** | HTML templates, styling, Git management, deployment |

---

## 🗺️ Roadmap / Possible Next Steps

- [ ] "Frequently bought together" feature using association rule mining
- [ ] User accounts + purchase history → collaborative filtering
- [ ] Shopping cart & checkout flow
- [ ] Deploy to Render / Railway
- [ ] Product filtering by price range and rating

---

## 📄 License

This project was built for educational purposes as part of a group coursework project.
