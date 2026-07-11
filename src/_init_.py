"""
AI Recommendation Logic — DecodeLabs Project 3
================================================
Modular source code for the e-commerce recommendation system.
The Jupyter notebook (notebooks/AI_Recommendation_Logic.ipynb) is self-contained
and re-implements these utilities inline so it runs standalone on Google Colab.
This package is provided for code organization, testing, and reuse.

Author: DecodeLabs Trainee (Batch 2026)
Project: AI Recommendation Logic
"""

from .data import generate_synthetic_ecommerce, load_real_ecommerce
from .similarity import (
    cosine_similarity_matrix,
    jaccard_similarity_matrix,
    pearson_correlation_matrix,
    euclidean_similarity_matrix,
    hamming_similarity_matrix,
)
from .models import (
    ContentBasedRecommender,
    UserCFRecommender,
    ItemCFRecommender,
    SVDRecommender,
    NMFRecommender,
    NeuralCFRecommender,
    HybridRecommender,
)
from .evaluation import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
    rmse,
    mae,
    catalog_coverage,
    novelty,
    diversity,
    evaluate_recommender,
)

__version__ = "1.0.0"
__all__ = [
    "generate_synthetic_ecommerce",
    "load_real_ecommerce",
    "cosine_similarity_matrix",
    "jaccard_similarity_matrix",
    "pearson_correlation_matrix",
    "euclidean_similarity_matrix",
    "hamming_similarity_matrix",
    "ContentBasedRecommender",
    "UserCFRecommender",
    "ItemCFRecommender",
    "SVDRecommender",
    "NMFRecommender",
    "NeuralCFRecommender",
    "HybridRecommender",
    "precision_at_k",
    "recall_at_k",
    "average_precision_at_k",
    "ndcg_at_k",
    "rmse",
    "mae",
    "catalog_coverage",
    "novelty",
    "diversity",
    "evaluate_recommender",
]
