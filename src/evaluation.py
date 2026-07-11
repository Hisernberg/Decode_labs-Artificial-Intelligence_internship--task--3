"""
Offline evaluation metrics for recommendation systems.

Two families:
  1. Ranking metrics (for top-K recommendation quality):
     precision@k, recall@k, MAP@k, NDCG@k
  2. Rating-prediction metrics:
     RMSE, MAE
  3. Beyond-accuracy metrics:
     catalog_coverage, novelty, diversity
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Fraction of top-K recommended items that are relevant."""
    if not recommended:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Average precision @ K — summary statistic for a single recommendation list."""
    if not relevant:
        return 0.0
    topk = recommended[:k]
    score, hits = 0.0, 0
    for i, item in enumerate(topk):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    DCG = sum_{i=1..k} (2^rel_i - 1) / log2(i+1); IDCG is the ideal (sorted)
    ranking; NDCG = DCG / IDCG.
    """
    if not relevant:
        return 0.0
    topk = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(topk):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------
def rmse(predictions: list[float], actuals: list[float]) -> float:
    """Root Mean Squared Error."""
    p, a = np.asarray(predictions), np.asarray(actuals)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def mae(predictions: list[float], actuals: list[float]) -> float:
    """Mean Absolute Error."""
    p, a = np.asarray(predictions), np.asarray(actuals)
    return float(np.mean(np.abs(p - a)))


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------
def catalog_coverage(
    all_recommendations: list[list[str]],
    catalog: list[str],
) -> float:
    """Fraction of catalog items ever recommended across all users."""
    catalog_set = set(catalog)
    recommended_set = set()
    for recs in all_recommendations:
        recommended_set.update(recs)
    return len(recommended_set & catalog_set) / len(catalog_set)


def novelty(
    all_recommendations: list[list[str]],
    item_popularity: dict[str, float],
) -> float:
    """Average novelty (negative log popularity) of recommended items.

    Higher = more novel/surprising. Lower = system keeps recommending popular items.
    """
    eps = 1e-9
    total, count = 0.0, 0
    for recs in all_recommendations:
        for item in recs:
            pop = item_popularity.get(item, eps)
            total += -np.log2(pop + eps)
            count += 1
    return total / max(count, 1)


def diversity(
    all_recommendations: list[list[str]],
    item_similarity: dict[tuple[str, str], float],
) -> float:
    """Average intra-list diversity = 1 - mean pairwise similarity.

    item_similarity is a dict keyed by (item_a, item_b) tuples.
    """
    totals = []
    for recs in all_recommendations:
        if len(recs) < 2:
            continue
        sims = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                key = (recs[i], recs[j])
                sims.append(item_similarity.get(key, item_similarity.get((recs[j], recs[i]), 0.0)))
        if sims:
            totals.append(1.0 - float(np.mean(sims)))
    return float(np.mean(totals)) if totals else 0.0


# ---------------------------------------------------------------------------
# Convenience: evaluate a fitted recommender on a test set
# ---------------------------------------------------------------------------
def evaluate_recommender(
    model,
    test_interactions: "pd.DataFrame",
    train_interactions: "pd.DataFrame",
    k: int = 10,
    n_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """Run a full offline evaluation of a fitted recommender.

    Returns a dict with precision@k, recall@k, MAP@k, NDCG@k, RMSE, MAE.
    """
    import pandas as pd

    # Build per-user relevant set from test
    relevant_by_user = (
        test_interactions[test_interactions["rating"] >= 4.0]
        .groupby("user_id")["product_id"].apply(set).to_dict()
    )

    rng = np.random.default_rng(seed)
    users = list(relevant_by_user.keys())
    if n_samples is not None and len(users) > n_samples:
        users = list(rng.choice(users, size=n_samples, replace=False))

    p_list, r_list, map_list, ndcg_list = [], [], [], []
    pred_ratings, true_ratings = [], []

    for u in users:
        recs = model.recommend(u, top_k=k, exclude_seen=True)
        rec_items = [pid for pid, _ in recs]
        relevant = relevant_by_user[u]
        p_list.append(precision_at_k(rec_items, relevant, k))
        r_list.append(recall_at_k(rec_items, relevant, k))
        map_list.append(average_precision_at_k(rec_items, relevant, k))
        ndcg_list.append(ndcg_at_k(rec_items, relevant, k))

        # RMSE: predict ratings for held-out (user, item) pairs
        u_test = test_interactions[test_interactions["user_id"] == u]
        for _, row in u_test.iterrows():
            try:
                p = model.predict_rating(u, row["product_id"])
                pred_ratings.append(p)
                true_ratings.append(row["rating"])
            except Exception:
                pass

    return {
        f"Precision@{k}": float(np.mean(p_list)) if p_list else 0.0,
        f"Recall@{k}": float(np.mean(r_list)) if r_list else 0.0,
        f"MAP@{k}": float(np.mean(map_list)) if map_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "RMSE": rmse(pred_ratings, true_ratings) if pred_ratings else 0.0,
        "MAE": mae(pred_ratings, true_ratings) if pred_ratings else 0.0,
        "n_users_evaluated": len(users),
    }
