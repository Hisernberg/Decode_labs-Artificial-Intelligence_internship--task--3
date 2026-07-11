"""
Data generation & loading utilities for the e-commerce recommendation system.

- generate_synthetic_ecommerce: builds a rich, controlled synthetic dataset
  (products, users, interactions) for walkthrough & ablation.
- load_real_ecommerce: attempts to download the UCI Online Retail dataset
  and convert it to user-item ratings. Falls back gracefully if offline.
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic E-commerce Data Generator
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Books", "Beauty",
    "Sports", "Toys", "Grocery", "Automotive", "Music",
]

BRANDS = {
    "Electronics": ["TechNova", "ElectroMax", "VoltCore", "PixelPro"],
    "Fashion":     ["VogueLine", "Threadly", "UrbanWeave", "ClassyCot"],
    "Home & Kitchen": ["HomeHue", "KitchenKraft", "CozyNest", "PurePan"],
    "Books":       ["PagePress", "LitHouse", "BookHive", "Inkwell"],
    "Beauty":      ["GlowLab", "PureSkin", "Beautea", "LushAura"],
    "Sports":      ["FitForge", "ProStride", "IronWill", "FlexFit"],
    "Toys":        ["PlayPioneer", "ToyTinker", "FunForge", "KidoCraft"],
    "Grocery":     ["FreshCart", "PureBite", "EcoGrain", "HarvestHue"],
    "Automotive":  ["AutoArc", "DriveDyn", "MotoMate", "CarCore"],
    "Music":       ["SoundScape", "BeatLab", "TuneTech", "AudioArc"],
}

TAGS = {
    "Electronics": ["wireless", "smart", "portable", "fast-charging", "bluetooth"],
    "Fashion":     ["cotton", "slim-fit", "casual", "trendy", "premium"],
    "Home & Kitchen": ["durable", "eco-friendly", "non-stick", "compact", "rust-free"],
    "Books":       ["bestseller", "hardcover", "self-help", "fiction", "educational"],
    "Beauty":      ["organic", "vegan", "paraben-free", "hydrating", "anti-aging"],
    "Sports":      ["lightweight", "breathable", "durable", "grip", "waterproof"],
    "Toys":        ["eco-friendly", "non-toxic", "educational", "battery-operated", "colorful"],
    "Grocery":     ["organic", "gluten-free", "low-sugar", "natural", "vegan"],
    "Automotive":  ["heavy-duty", "weatherproof", "easy-install", "universal", "rust-proof"],
    "Music":       ["wireless", "noise-cancelling", "portable", "bass-boost", "bluetooth"],
}


@dataclass
class EcommerceData:
    products: pd.DataFrame
    users: pd.DataFrame
    interactions: pd.DataFrame


def generate_synthetic_ecommerce(
    n_products: int = 500,
    n_users: int = 300,
    n_interactions: int = 8000,
    seed: int = 42,
) -> EcommerceData:
    """Generate a rich, reproducible synthetic e-commerce dataset.

    Products get a category, brand, price, baseline rating, popularity and
    a bag-of-tags. Users get demographic + preference vectors. Interactions
    are sampled with a preference-aligned bias so similar-minded users end up
    liking similar items (this is what the recommenders must recover).
    """
    rng = np.random.default_rng(seed)

    # --- Products -----------------------------------------------------------
    product_ids = [f"P{str(i).zfill(4)}" for i in range(n_products)]
    cats = rng.choice(CATEGORIES, size=n_products)
    brands = [rng.choice(BRANDS[c]) for c in cats]
    # Price: log-normal, varies by category
    base_price = {"Electronics": 200, "Fashion": 50, "Home & Kitchen": 80,
                  "Books": 20, "Beauty": 30, "Sports": 70, "Toys": 40,
                  "Grocery": 25, "Automotive": 150, "Music": 120}
    prices = np.array([
        max(5.0, rng.lognormal(mean=np.log(base_price[c]), sigma=0.5))
        for c in cats
    ]).round(2)
    # Baseline rating: 3.0–5.0 skewed toward 4
    base_ratings = np.clip(rng.normal(loc=4.0, scale=0.6, size=n_products), 2.5, 5.0).round(1)
    # Popularity (latent): some items are just more popular
    popularity = rng.exponential(scale=1.0, size=n_products)
    popularity = popularity / popularity.max()
    # Tags: 2–4 per product, sampled from category tag pool
    tag_lists = []
    for c in cats:
        n_tags = rng.integers(2, 5)
        tag_lists.append(list(rng.choice(TAGS[c], size=n_tags, replace=False)))

    products = pd.DataFrame({
        "product_id": product_ids,
        "name": [f"{b} {c} #{i}" for b, c, i in zip(brands, cats, range(n_products))],
        "category": cats,
        "brand": brands,
        "price": prices,
        "avg_rating": base_ratings,
        "popularity": popularity,
        "tags": ["|".join(t) for t in tag_lists],
    })

    # --- Users --------------------------------------------------------------
    user_ids = [f"U{str(i).zfill(4)}" for i in range(n_users)]
    ages = rng.integers(18, 65, size=n_users)
    genders = rng.choice(["M", "F"], size=n_users)
    # Each user has a latent preference vector over the 10 categories
    pref_vectors = rng.dirichlet(alpha=np.ones(len(CATEGORIES)) * 0.5, size=n_users)
    users = pd.DataFrame({
        "user_id": user_ids,
        "age": ages,
        "gender": genders,
    })
    for i, c in enumerate(CATEGORIES):
        users[f"pref_{c}"] = pref_vectors[:, i]

    # --- Interactions -------------------------------------------------------
    # Bias sampling: users more likely to interact with items in their preferred categories
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    product_cat_idx = np.array([cat_to_idx[c] for c in cats])
    # P(user interacts with product) ∝ user_pref[product_cat] * product_popularity
    interactions = []
    ratings = []
    user_seen = {u: set() for u in user_ids}

    target_per_user = max(1, n_interactions // n_users)
    for u_idx, u in enumerate(user_ids):
        pref = pref_vectors[u_idx]
        # item affinity score
        affinity = pref[product_cat_idx] * popularity
        affinity = affinity / (affinity.sum() + 1e-9)
        n_u = rng.integers(max(3, target_per_user - 5), target_per_user + 10)
        n_u = min(n_u, n_products)
        chosen_idx = rng.choice(n_products, size=n_u, replace=False, p=affinity)
        for p_idx in chosen_idx:
            # Rating: base + noise + alignment bonus
            align = pref[product_cat_idx[p_idx]] * 2.0  # up to ~2 points bonus
            noise = rng.normal(0, 0.7)
            r = base_ratings[p_idx] * 0.4 + 2.5 + align * 1.5 + noise
            r = float(np.clip(r, 1.0, 5.0))
            interactions.append({
                "user_id": u,
                "product_id": product_ids[p_idx],
                "rating": round(r, 1),
                "timestamp": int(time.time()) - rng.integers(0, 90 * 24 * 3600),
            })

    interactions = pd.DataFrame(interactions)
    # Shuffle
    interactions = interactions.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return EcommerceData(products=products, users=users, interactions=interactions)


# ---------------------------------------------------------------------------
# Real E-commerce Dataset Loader (UCI Online Retail)
# ---------------------------------------------------------------------------

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/"
    "Online%20Retail.xlsx"
)


def load_real_ecommerce(
    cache_path: str = "data/online_retail.xlsx",
    max_users: int = 500,
    max_products: int = 500,
    seed: int = 42,
) -> EcommerceData:
    """Load the UCI Online Retail dataset and convert it to user-item ratings.

    The raw dataset has transactions (InvoiceNo, StockCode, Description,
    Quantity, InvoiceDate, UnitPrice, CustomerID, Country). We:
      1. Filter out returns (Quantity < 1) and missing CustomerIDs.
      2. Group by (CustomerID, StockCode) and sum Quantity.
      3. Convert purchase frequency to a 1–5 implicit rating via log-scaling.
      4. Subsample to max_users x max_products for tractability on Colab T4.

    If the download fails, returns None — the caller should fall back to
    the synthetic generator.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if not os.path.exists(cache_path):
        try:
            print(f"Downloading UCI Online Retail from {UCI_URL} ...")
            urllib.request.urlretrieve(UCI_URL, cache_path)
            print(f"Saved to {cache_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    try:
        df = pd.read_excel(cache_path)
    except Exception as e:
        print(f"Read failed: {e}")
        return None

    # Clean
    df = df.dropna(subset=["CustomerID", "StockCode", "Description"])
    df = df[df["Quantity"] > 0]
    df["CustomerID"] = "U" + df["CustomerID"].astype(int).astype(str)
    df["StockCode"] = "P" + df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str).str.strip()

    # Aggregate to user-item purchase counts
    agg = df.groupby(["CustomerID", "StockCode"]).agg(
        quantity=("Quantity", "sum"),
        description=("Description", "first"),
        unit_price=("UnitPrice", "mean"),
    ).reset_index()

    # Subsample: pick top-N most active users and top-N most popular products
    user_activity = agg["CustomerID"].value_counts()
    prod_popularity = agg["StockCode"].value_counts()
    top_users = user_activity.head(max_users).index
    top_products = prod_popularity.head(max_products).index
    agg = agg[agg["CustomerID"].isin(top_users) & agg["StockCode"].isin(top_products)]

    if len(agg) == 0:
        return None

    # Rename columns to the canonical names used everywhere else
    agg = agg.rename(columns={"CustomerID": "user_id", "StockCode": "product_id"})

    # Convert quantity to 1-5 rating via log scale
    agg["rating"] = np.clip(
        np.log1p(agg["quantity"]) / np.log1p(agg["quantity"].max()) * 5.0,
        1.0, 5.0,
    ).round(1)

    # Build products table
    products = agg.groupby("product_id").agg(
        name=("description", "first"),
        price=("unit_price", "mean"),
    ).reset_index()
    # Assign pseudo-categories by keyword matching on description
    cat_keywords = {
        "Home & Kitchen": ["bottle", "cup", "mug", "plate", "bowl", "kitchen", "cook"],
        "Fashion": ["bag", "purse", "scarf", "coat", "shirt", "dress"],
        "Decor": ["candle", "frame", "vase", "ornament", "decoration"],
        "Stationery": ["card", "paper", "pen", "pencil", "notebook"],
        "Toys": ["toy", "game", "puzzle", "doll"],
    }
    def categorize(desc: str) -> str:
        d = desc.lower()
        for cat, kws in cat_keywords.items():
            if any(kw in d for kw in kws):
                return cat
        return "Misc"
    products["category"] = products["name"].apply(categorize)
    products["brand"] = "Generic"
    products["avg_rating"] = agg.groupby("product_id")["rating"].mean().values.round(1)
    products["popularity"] = agg.groupby("product_id")["rating"].count().values
    products["popularity"] = (products["popularity"] / products["popularity"].max()).round(3)
    products["tags"] = ""

    # Build users table
    users = agg.groupby("user_id").agg(
        n_purchases=("product_id", "count"),
    ).reset_index()
    users["age"] = np.random.default_rng(seed).integers(18, 70, size=len(users))
    users["gender"] = np.random.default_rng(seed).choice(["M", "F"], size=len(users))
    # Per-category preference (computed from actual purchases)
    user_cat = agg.merge(products[["product_id", "category"]], on="product_id")
    user_pref = user_cat.groupby(["user_id", "category"])["rating"].mean().unstack(fill_value=0)
    for c in user_pref.columns:
        users[f"pref_{c}"] = users["user_id"].map(user_pref[c]).fillna(0).values

    # Build interactions (timestamp = first invoice date per user-item pair)
    first_dates = (
        df.groupby(["CustomerID", "StockCode"])["InvoiceDate"].first()
        .reset_index().rename(columns={"CustomerID": "user_id", "StockCode": "product_id"})
    )
    interactions = agg[["user_id", "product_id", "rating"]].merge(
        first_dates, on=["user_id", "product_id"], how="left"
    )
    interactions["timestamp"] = (
        pd.to_datetime(interactions["InvoiceDate"]).astype("int64") // 10**9
    )
    interactions = interactions[["user_id", "product_id", "rating", "timestamp"]]

    return EcommerceData(
        products=products.reset_index(drop=True),
        users=users.reset_index(drop=True),
        interactions=interactions.reset_index(drop=True),
    )
