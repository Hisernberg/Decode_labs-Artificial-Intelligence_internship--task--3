"""
Recommender models for the e-commerce recommendation system.

Models implemented:
  - ContentBasedRecommender   : item-attribute similarity to user profile
  - UserCFRecommender         : user-based collaborative filtering
  - ItemCFRecommender         : item-based collaborative filtering
  - SVDRecommender            : matrix factorization via TruncatedSVD
  - NMFRecommender            : non-negative matrix factorization
  - NeuralCFRecommender       : NCF (GMF + MLP) in PyTorch (uses T4 GPU if available)
  - HybridRecommender         : weighted blend of any subset of the above
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import normalize

from .similarity import cosine_similarity_matrix, jaccard_similarity_matrix


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseRecommender:
    """Common interface every recommender implements."""

    name: str = "base"

    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame | None = None,
            users: pd.DataFrame | None = None) -> "BaseRecommender":
        raise NotImplementedError

    def recommend(self, user_id: str, top_k: int = 10,
                  exclude_seen: bool = True) -> list[tuple[str, float]]:
        """Return list of (product_id, score) sorted desc."""
        raise NotImplementedError

    def predict_rating(self, user_id: str, product_id: str) -> float:
        """Predicted rating for a (user, product) pair."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Content-Based Filtering
# ---------------------------------------------------------------------------
class ContentBasedRecommender(BaseRecommender):
    """Recommend items whose attribute vector is most similar to the user's
    profile (TF weighted average of rated item feature vectors)."""

    name = "Content-Based"

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.user_profiles: dict[str, np.ndarray] = {}

    def _build_product_features(self, products: pd.DataFrame) -> np.ndarray:
        """One-hot encode category + brand + tags, append normalized price & rating."""
        cats = pd.get_dummies(products["category"], prefix="cat")
        brands = pd.get_dummies(products["brand"], prefix="brand")
        # Tags: bag-of-words
        tag_lists = products["tags"].fillna("").str.split("|")
        all_tags = sorted({t for ts in tag_lists for t in ts if t})
        tag_mat = np.zeros((len(products), len(all_tags)), dtype=np.float32)
        tag_to_idx = {t: i for i, t in enumerate(all_tags)}
        for i, ts in enumerate(tag_lists):
            for t in ts:
                if t in tag_to_idx:
                    tag_mat[i, tag_to_idx[t]] = 1.0
        # Numerical features (normalized)
        price = products["price"].to_numpy().reshape(-1, 1)
        rating = products["avg_rating"].to_numpy().reshape(-1, 1)
        price_n = (price - price.min()) / (price.max() - price.min() + 1e-9)
        rating_n = (rating - rating.min()) / (rating.max() - rating.min() + 1e-9)
        feats = np.hstack([cats.values, brands.values, tag_mat, price_n, rating_n]).astype(np.float32)
        return feats

    def fit(self, interactions, products=None, users=None):
        assert products is not None, "Content-Based needs a products table"
        self.products = products.reset_index(drop=True)
        self.product_ids = self.products["product_id"].to_numpy()
        self.features = self._build_product_features(self.products)
        self.pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}

        # Build user profile = weighted avg of features of items they rated
        self.user_profiles = {}
        for uid, grp in interactions.groupby("user_id"):
            idxs = [self.pid_to_idx[p] for p in grp["product_id"] if p in self.pid_to_idx]
            if not idxs:
                continue
            weights = grp["rating"].to_numpy().reshape(-1, 1)
            feats = self.features[idxs]
            # Weighted average
            prof = (feats * weights).sum(axis=0) / (weights.sum() + 1e-9)
            self.user_profiles[uid] = prof
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in self.user_profiles:
            # Cold start: use mean profile
            prof = self.features.mean(axis=0)
        else:
            prof = self.user_profiles[user_id]
        prof = prof.reshape(1, -1)
        if self.metric == "cosine":
            sims = cosine_similarity_matrix(prof, self.features)[0]
        elif self.metric == "jaccard":
            sims = jaccard_similarity_matrix((prof > 0).astype(np.float32),
                                              (self.features > 0).astype(np.float32))[0]
        else:
            sims = cosine_similarity_matrix(prof, self.features)[0]
        order = np.argsort(-sims)
        if exclude_seen:
            seen = set()  # caller can pass via attr
        else:
            seen = set()
        results = []
        for i in order:
            if self.product_ids[i] in seen:
                continue
            results.append((self.product_ids[i], float(sims[i])))
            if len(results) >= top_k:
                break
        return results

    def predict_rating(self, user_id, product_id):
        if user_id not in self.user_profiles or product_id not in self.pid_to_idx:
            return 3.0  # global mean fallback
        prof = self.user_profiles[user_id].reshape(1, -1)
        feat = self.features[self.pid_to_idx[product_id]].reshape(1, -1)
        sim = cosine_similarity_matrix(prof, feat)[0, 0]
        # Map [-1,1] -> [1,5]
        return float(3.0 + 2.0 * sim)


# ---------------------------------------------------------------------------
# User-Based Collaborative Filtering
# ---------------------------------------------------------------------------
class UserCFRecommender(BaseRecommender):
    name = "User-CF"
    def __init__(self, k=20, metric="cosine"):
        self.k = k
        self.metric = metric
    def fit(self, interactions, products=None, users=None):
        self.user_item = interactions.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0.0
        )
        self.user_ids = self.user_item.index.to_numpy()
        self.product_ids = self.user_item.columns.to_numpy()
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        M = self.user_item.values.astype(np.float32)
        # Mean-center for Pearson-like behavior
        self.user_means = M.sum(axis=1) / np.maximum((M > 0).sum(axis=1), 1)
        Mc = M - self.user_means.reshape(-1, 1) * (M > 0)
        self.Mc = Mc
        self.sim = cosine_similarity_matrix(Mc)
        np.fill_diagonal(self.sim, 0.0)
        return self
    def recommend(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in self.uid_to_idx:
            # Cold start: recommend most popular items
            pop = (self.user_item.values > 0).sum(axis=0)
            order = np.argsort(-pop)
            return [(self.product_ids[i], float(pop[i])) for i in order[:top_k]]
        u_idx = self.uid_to_idx[user_id]
        sims = self.sim[u_idx]
        top_users = np.argsort(-sims)[: self.k]
        # Weighted sum over neighbors' ratings
        scores = np.zeros(len(self.product_ids))
        for v in top_users:
            scores += sims[v] * self.user_item.values[v]
        denom = np.abs(sims[top_users]).sum() + 1e-9
        scores = scores / denom
        # Exclude items the user already interacted with
        seen_mask = self.user_item.values[u_idx] > 0
        scores[seen_mask] = -np.inf
        order = np.argsort(-scores)
        return [(self.product_ids[i], float(scores[i])) for i in order[:top_k]]
    def predict_rating(self, user_id, product_id):
        if user_id not in self.uid_to_idx or product_id not in self.pid_to_idx:
            return 3.0
        u_idx = self.uid_to_idx[user_id]
        sims = self.sim[u_idx]
        top_users = np.argsort(-sims)[: self.k]
        p_idx = self.pid_to_idx[product_id]
        num, den = 0.0, 0.0
        for v in top_users:
            r = self.user_item.values[v, p_idx]
            if r > 0:
                num += sims[v] * (r - self.user_means[v])
                den += abs(sims[v])
        if den == 0:
            return float(self.user_means[u_idx])
        return float(self.user_means[u_idx] + num / den)


# ---------------------------------------------------------------------------
# Item-Based Collaborative Filtering
# ---------------------------------------------------------------------------
class ItemCFRecommender(BaseRecommender):
    name = "Item-CF"
    def __init__(self, k=20, metric="cosine"):
        self.k = k
        self.metric = metric
    def fit(self, interactions, products=None, users=None):
        self.user_item = interactions.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0.0
        )
        self.user_ids = self.user_item.index.to_numpy()
        self.product_ids = self.user_item.columns.to_numpy()
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        M = self.user_item.values.astype(np.float32)
        self.M = M
        # Item-item similarity
        self.sim = cosine_similarity_matrix(M.T)
        np.fill_diagonal(self.sim, 0.0)
        return self
    def recommend(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in self.uid_to_idx:
            pop = (self.user_item.values > 0).sum(axis=0)
            order = np.argsort(-pop)
            return [(self.product_ids[i], float(pop[i])) for i in order[:top_k]]
        u_idx = self.uid_to_idx[user_id]
        user_vec = self.user_item.values[u_idx]
        # For each candidate item, score = sum over user's items of sim * rating
        scores = user_vec @ self.sim  # shape (n_items,)
        # Normalize by sum of similarities to user's items
        seen_mask = user_vec > 0
        sim_sums = self.sim[:, seen_mask].sum(axis=1)
        sim_sums[sim_sums == 0] = 1e-9
        scores = scores / sim_sums
        scores[seen_mask] = -np.inf
        order = np.argsort(-scores)
        return [(self.product_ids[i], float(scores[i])) for i in order[:top_k]]
    def predict_rating(self, user_id, product_id):
        if user_id not in self.uid_to_idx or product_id not in self.pid_to_idx:
            return 3.0
        u_idx = self.uid_to_idx[user_id]
        user_vec = self.user_item.values[u_idx]
        p_idx = self.pid_to_idx[product_id]
        sims = self.sim[p_idx]
        top_items = np.argsort(-sims)[: self.k]
        num, den = 0.0, 0.0
        for j in top_items:
            r = user_vec[j]
            if r > 0:
                num += sims[j] * r
                den += abs(sims[j])
        if den == 0:
            return 3.0
        return float(num / den)


# ---------------------------------------------------------------------------
# Matrix Factorization — SVD
# ---------------------------------------------------------------------------
class SVDRecommender(BaseRecommender):
    name = "SVD"
    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
    def fit(self, interactions, products=None, users=None):
        self.user_item = interactions.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0.0
        )
        self.user_ids = self.user_item.index.to_numpy()
        self.product_ids = self.user_item.columns.to_numpy()
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        M = self.user_item.values.astype(np.float32)
        n_comp = min(self.n_factors, min(M.shape) - 1)
        self.model = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.U = self.model.fit_transform(M)
        self.V = self.model.components_
        self.pred = self.U @ self.V
        self.pred = np.clip(self.pred, 1.0, 5.0)
        return self
    def recommend(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in self.uid_to_idx:
            pop = (self.user_item.values > 0).sum(axis=0)
            order = np.argsort(-pop)
            return [(self.product_ids[i], float(pop[i])) for i in order[:top_k]]
        u_idx = self.uid_to_idx[user_id]
        scores = self.pred[u_idx].copy()
        seen_mask = self.user_item.values[u_idx] > 0
        scores[seen_mask] = -np.inf
        order = np.argsort(-scores)
        return [(self.product_ids[i], float(scores[i])) for i in order[:top_k]]
    def predict_rating(self, user_id, product_id):
        if user_id not in self.uid_to_idx or product_id not in self.pid_to_idx:
            return 3.0
        return float(self.pred[self.uid_to_idx[user_id], self.pid_to_idx[product_id]])


# ---------------------------------------------------------------------------
# Matrix Factorization — NMF
# ---------------------------------------------------------------------------
class NMFRecommender(BaseRecommender):
    name = "NMF"
    def __init__(self, n_factors=15, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
    def fit(self, interactions, products=None, users=None):
        self.user_item = interactions.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0.0
        )
        self.user_ids = self.user_item.index.to_numpy()
        self.product_ids = self.user_item.columns.to_numpy()
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        M = self.user_item.values.astype(np.float32)
        n_comp = min(self.n_factors, min(M.shape) - 1)
        self.model = NMF(n_components=n_comp, init="nndsvda",
                         max_iter=300, random_state=self.random_state)
        self.U = self.model.fit_transform(M)
        self.V = self.model.components_
        self.pred = self.U @ self.V
        self.pred = np.clip(self.pred, 1.0, 5.0)
        return self
    def recommend(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in self.uid_to_idx:
            pop = (self.user_item.values > 0).sum(axis=0)
            order = np.argsort(-pop)
            return [(self.product_ids[i], float(pop[i])) for i in order[:top_k]]
        u_idx = self.uid_to_idx[user_id]
        scores = self.pred[u_idx].copy()
        seen_mask = self.user_item.values[u_idx] > 0
        scores[seen_mask] = -np.inf
        order = np.argsort(-scores)
        return [(self.product_ids[i], float(scores[i])) for i in order[:top_k]]
    def predict_rating(self, user_id, product_id):
        if user_id not in self.uid_to_idx or product_id not in self.pid_to_idx:
            return 3.0
        return float(self.pred[self.uid_to_idx[user_id], self.pid_to_idx[product_id]])


# ---------------------------------------------------------------------------
# Neural Collaborative Filtering (PyTorch)
# ---------------------------------------------------------------------------
class NeuralCFRecommender(BaseRecommender):
    """Neural CF (He et al. 2017): combines Generalized MF (GMF) branch with
    an MLP branch over user/item embeddings. Trains on T4 GPU if available.
    """
    name = "Neural-CF"
    def __init__(self, n_factors=32, n_hidden=(64, 32, 16),
                 epochs=10, batch_size=256, lr=1e-3, device="auto"):
        self.n_factors = n_factors
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def _build_model(self, n_users, n_items):
        import torch
        import torch.nn as nn

        class NCF(nn.Module):
            def __init__(self, n_users, n_items, k, hidden):
                super().__init__()
                self.user_emb_gmf = nn.Embedding(n_users, k)
                self.item_emb_gmf = nn.Embedding(n_items, k)
                self.user_emb_mlp = nn.Embedding(n_users, k)
                self.item_emb_mlp = nn.Embedding(n_items, k)
                nn.init.normal_(self.user_emb_gmf.weight, std=0.01)
                nn.init.normal_(self.item_emb_gmf.weight, std=0.01)
                nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
                nn.init.normal_(self.item_emb_mlp.weight, std=0.01)
                layers = []
                d_in = 2 * k
                for d_out in hidden:
                    layers += [nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(0.2)]
                    d_in = d_out
                self.mlp = nn.Sequential(*layers)
                self.head = nn.Linear(d_in + k, 1)
            def forward(self, u, i):
                gmf = self.user_emb_gmf(u) * self.item_emb_gmf(i)
                mlp_in = torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=-1)
                mlp_out = self.mlp(mlp_in)
                h = torch.cat([gmf, mlp_out], dim=-1)
                return self.head(h).squeeze(-1)
        return NCF(n_users, n_items, self.n_factors, self.n_hidden)

    def fit(self, interactions, products=None, users=None):
        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [NCF] training on device: {self.device}")

        # Index users & products
        self.user_ids = interactions["user_id"].unique()
        self.product_ids = interactions["product_id"].unique()
        self.uid_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.pid_to_idx = {p: i for i, p in enumerate(self.product_ids)}
        n_users = len(self.user_ids)
        n_items = len(self.product_ids)

        # Build training tensor
        u_idx = interactions["user_id"].map(self.uid_to_idx).to_numpy()
        p_idx = interactions["product_id"].map(self.pid_to_idx).to_numpy()
        r = interactions["rating"].to_numpy().astype(np.float32)
        # Normalize ratings to [0,1]
        r_norm = (r - 1.0) / 4.0

        u_t = torch.tensor(u_idx, dtype=torch.long, device=self.device)
        p_t = torch.tensor(p_idx, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r_norm, dtype=torch.float32, device=self.device)

        self.model = self._build_model(n_users, n_items).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        n = len(u_t)
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0
            self.model.train()
            for s in range(0, n, self.batch_size):
                idx = perm[s: s + self.batch_size]
                opt.zero_grad()
                pred = self.model(u_t[idx], p_t[idx])
                loss = loss_fn(pred, r_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            avg = total_loss / n
            print(f"  [NCF] epoch {epoch+1}/{self.epochs}  loss={avg:.4f}")

        # Precompute full prediction matrix for fast recommend()
        self.model.eval()
        with torch.no_grad():
            self.pred = np.zeros((n_users, n_items), dtype=np.float32)
            bs = 1024
            for s in range(0, n_users, bs):
                e = min(s + bs, n_users)
                u_block = torch.arange(s, e, device=self.device).repeat_interleave(n_items)
                i_block = torch.arange(n_items, device=self.device).repeat(e - s)
                p = self.model(u_block, i_block).view(e - s, n_items).cpu().numpy()
                # Map [0,1] back to [1,5]
                self.pred[s:e] = 1.0 + 4.0 * p

        # Save user_item for exclude_seen
        self.user_item = interactions.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0.0
        )
        # Reindex to align with self.user_ids / self.product_ids
        self.user_item = self.user_item.reindex(index=self.user_ids, columns=self.product_ids, fill_value=0.0)
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in self.uid_to_idx:
            pop = (self.user_item.values > 0).sum(axis=0)
            order = np.argsort(-pop)
            return [(self.product_ids[i], float(pop[i])) for i in order[:top_k]]
        u_idx = self.uid_to_idx[user_id]
        scores = self.pred[u_idx].copy()
        seen_mask = self.user_item.values[u_idx] > 0
        scores[seen_mask] = -np.inf
        order = np.argsort(-scores)
        return [(self.product_ids[i], float(scores[i])) for i in order[:top_k]]

    def predict_rating(self, user_id, product_id):
        if user_id not in self.uid_to_idx or product_id not in self.pid_to_idx:
            return 3.0
        return float(self.pred[self.uid_to_idx[user_id], self.pid_to_idx[product_id]])


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------
class HybridRecommender(BaseRecommender):
    """Weighted score fusion across multiple base recommenders."""
    name = "Hybrid"
    def __init__(self, recommenders, weights=None):
        self.recommenders = recommenders
        if weights is None:
            weights = [1.0 / len(recommenders)] * len(recommenders)
        assert len(weights) == len(recommenders)
        self.weights = weights

    def fit(self, interactions, products=None, users=None):
        for r in self.recommenders:
            r.fit(interactions, products, users)
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=True):
        # Collect per-recommender top-K with scores, normalize, fuse
        all_items = set()
        per_rec_scores = []
        for r in self.recommenders:
            recs = r.recommend(user_id, top_k=max(top_k * 3, 30), exclude_seen=exclude_seen)
            scores = {pid: s for pid, s in recs}
            per_rec_scores.append(scores)
            all_items.update(scores.keys())
        # Min-max normalize each recommender's scores to [0,1]
        norm_scores = []
        for scores in per_rec_scores:
            if not scores:
                norm_scores.append({})
                continue
            vals = np.array(list(scores.values()))
            vmin, vmax = vals.min(), vals.max()
            denom = vmax - vmin if vmax > vmin else 1.0
            norm_scores.append({k: (v - vmin) / denom for k, v in scores.items()})
        # Weighted sum
        fused = {}
        for item in all_items:
            s = 0.0
            for ns, w in zip(norm_scores, self.weights):
                if item in ns:
                    s += w * ns[item]
            fused[item] = s
        order = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return order

    def predict_rating(self, user_id, product_id):
        # Weighted average of base predictions
        s, wsum = 0.0, 0.0
        for r, w in zip(self.recommenders, self.weights):
            try:
                p = r.predict_rating(user_id, product_id)
                s += w * p
                wsum += w
            except Exception:
                pass
        return float(s / wsum) if wsum > 0 else 3.0
