"""
Similarity metrics for recommendation systems.

All functions accept dense numpy arrays and return dense similarity matrices
(small/medium data sizes; sparse versions would be used at production scale).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Cosine similarity = dot(a, b) / (||a|| ||b||).

    Range: [-1, 1] for general vectors, [0, 1] for non-negative feature vectors.
    Best for: content-based filtering where direction matters more than magnitude.
    """
    if Y is None:
        Y = X
    Xn = normalize(X, norm="l2", axis=1)
    Yn = normalize(Y, norm="l2", axis=1)
    return Xn @ Yn.T


def jaccard_similarity_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Jaccard similarity = |A ∩ B| / |A ∪ B| for binary matrices.

    Range: [0, 1]. Best for: tag/attribute sets where presence/absence matters
    but magnitude does not.
    """
    if Y is None:
        Y = X
    Xb = (X > 0).astype(np.float32)
    Yb = (Y > 0).astype(np.float32)
    intersection = Xb @ Yb.T
    x_card = Xb.sum(axis=1, keepdims=True)
    y_card = Yb.sum(axis=1, keepdims=True).T
    union = x_card + y_card - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def pearson_correlation_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlation = covariance / (std_x * std_y).

    Range: [-1, 1]. Best for: ratings data where users have different rating
    scales (some rate high, some rate low). Pearson cancels out mean shifts.
    """
    if Y is None:
        Y = X
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Xn = normalize(Xc, norm="l2", axis=1)
    Yn = normalize(Yc, norm="l2", axis=1)
    sim = Xn @ Yn.T
    # Rows/cols with zero variance produce NaN — replace with 0
    return np.nan_to_num(sim, nan=0.0)


def euclidean_similarity_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Euclidean similarity = 1 / (1 + euclidean_distance).

    Range: (0, 1]. Best for: low-dimensional numerical features where absolute
    distance matters (e.g., price, age). Computed via the algebraic identity
    ||x-y||² = ||x||² + ||y||² - 2 x·y for vectorized speed.
    """
    if Y is None:
        Y = X
    xx = (X * X).sum(axis=1)[:, None]
    yy = (Y * Y).sum(axis=1)[None, :]
    dist_sq = np.maximum(xx + yy - 2 * (X @ Y.T), 0.0)
    dist = np.sqrt(dist_sq)
    return 1.0 / (1.0 + dist)


def hamming_similarity_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Hamming similarity = 1 - (hamming_distance / n_features).

    Range: [0, 1]. Best for: categorical/binary features where the fraction of
    matching positions matters (e.g., one-hot encoded categories).
    """
    if Y is None:
        Y = X
    n = X.shape[1]
    if n == 0:
        return np.zeros((X.shape[0], Y.shape[0]))
    # For each pair, count agreements
    sim = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
    for i in range(X.shape[0]):
        # Agreement on positions where both equal
        sim[i] = (X[i] == Y).mean(axis=1)
    return sim


METRIC_REGISTRY = {
    "cosine": cosine_similarity_matrix,
    "jaccard": jaccard_similarity_matrix,
    "pearson": pearson_correlation_matrix,
    "euclidean": euclidean_similarity_matrix,
    "hamming": hamming_similarity_matrix,
}
