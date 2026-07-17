"""
app.py
-------
Main Flask application. Routes:

  /                          -> homepage, featured products + category list
  /category/<name>           -> products filtered by category
  /search?q=...              -> search results
  /product/<int:product_id>  -> product detail page + "related products"

Run:
    python app.py
Then visit http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, abort
from pathlib import Path

from database.db import db
from database.models import Product
from recommendation.recommender import ProductRecommender

BASE_DIR = Path(__file__).resolve().parent


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR / 'database' / 'ecommerce.db'}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    return app


app = create_app()

# Recommender is loaded once at startup (not per-request) — it's read-only
# inference over precomputed artifacts, so this is fast and safe to share.
recommender = ProductRecommender()

# Dataset prices are in USD. Convert + format as rupees everywhere via a
# Jinja filter, instead of editing the underlying data.
USD_TO_INR = 83

@app.template_filter("inr")
def format_inr(value):
    try:
        value = float(value) * USD_TO_INR
    except (TypeError, ValueError):
        return value
    return f"₹{value:,.0f}"


@app.route("/")
def home():
    categories = [row[0] for row in db.session.query(Product.category).distinct()]
    # simple "featured" selection: highest-rated products, a few per category feel is nice
    # but a straightforward top-rated list is a fine starting point
    featured = Product.query.order_by(Product.rating.desc(), Product.num_reviews.desc()).limit(12).all()
    return render_template("index.html", categories=categories, products=featured)


@app.route("/category/<string:category_name>")
def category_view(category_name):
    products = Product.query.filter_by(category=category_name).limit(60).all()
    return render_template("search_results.html", products=products,
                            heading=f'Category: "{category_name}"', related=[])


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return render_template("search_results.html", products=[], heading="Search", related=[])

    results = Product.query.filter(Product.name.ilike(f"%{query}%")).limit(60).all()

    # Also show "related to your search" using the recommender's name-based
    # lookup — matches the closest product by name/meaning, then returns
    # products similar to it. No product_id required from the user.
    related = []
    matched_name = None
    try:
        matched_product, related = recommender.recommend_by_name(query, top_n=5)
        matched_name = matched_product["name"]
    except ValueError as e:
        print(f"[recommend_by_name() found no match for '{query}']: {e}")

    return render_template("search_results.html", products=results,
                            heading=f'Search results for "{query}"',
                            related=related, matched_name=matched_name)


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = Product.query.filter_by(product_id=product_id).first()
    if product is None:
        abort(404)

    try:
        related = recommender.recommend(product_id, top_n=5)
    except Exception as e:
        # Widened from ValueError to Exception, and now logs to the terminal,
        # so any real bug shows up instead of silently returning an empty list.
        print(f"[recommend() failed for product_id={product_id}]: {e}")
        related = []

    return render_template("product.html", product=product, related=related)


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)