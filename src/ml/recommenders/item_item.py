# project/src/ml/recommenders/item_item.py

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemRecommender:
    """
    Item–Item Collaborative Filtering (cosine on co-occurrence)
    - fit() builds the user×item matrix and item–item similarity
    - recommend_for_user(user_id, k)
    - similar_items(product_id, k)
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.products = None
        self.reviews = None
        self.user2idx = {}
        self.idx2user = None
        self.item2idx = {}
        self.idx2item = None
        self.R = None  # csr_matrix users × items
        self.item_item_sim = None  # np.ndarray (items × items)
        self.prod_name_map = {}

    def _load(self):
        dd = self.data_dir
        self.products = pd.read_csv(dd / "products.csv")
        self.reviews = pd.read_csv(dd / "reviews.csv")
        self.products["product_id"] = self.products["product_id"].astype(str)
        self.reviews["user_id"] = self.reviews["user_id"].astype(str)
        self.reviews["product_id"] = self.reviews["product_id"].astype(str)

        self.prod_name_map = self.products.set_index("product_id")["product_name"].to_dict()

    def fit(self):
        self._load()

        # Build implicit interactions (dedup)
        interactions = self.reviews[["user_id", "product_id"]].dropna().drop_duplicates()

        # Encode IDs → indices
        user_ids = interactions["user_id"].unique()
        item_ids = interactions["product_id"].unique()

        self.user2idx = {u: i for i, u in enumerate(user_ids)}
        self.idx2user = np.array(user_ids)

        self.item2idx = {p: j for j, p in enumerate(item_ids)}
        self.idx2item = np.array(item_ids)

        ui = interactions["user_id"].map(self.user2idx).values
        ii = interactions["product_id"].map(self.item2idx).values
        vv = np.ones(len(interactions), dtype=np.float32)

        # Sparse user×item matrix
        self.R = csr_matrix((vv, (ui, ii)), shape=(len(self.user2idx), len(self.item2idx)))

        # Item–item cosine similarity (on columns)
        self.item_item_sim = cosine_similarity(self.R.T)  # (n_items, n_items)
        np.fill_diagonal(self.item_item_sim, 0.0)  # don't recommend itself

        return self

    def _pname(self, pid: str) -> str:
        return self.prod_name_map.get(str(pid), str(pid))

    def recommend_for_user(self, user_id: str, k: int = 10, exclude_seen: bool = True):
        if user_id not in self.user2idx:
            return []

        uidx = self.user2idx[user_id]
        user_row = self.R.getrow(uidx)
        seen = user_row.indices
        if seen.size == 0:
            return []

        # score = sum of similarities to seen items
        scores = self.item_item_sim[:, seen].sum(axis=1)
        if exclude_seen:
            scores[seen] = -np.inf

        k = min(k, scores.size)
        topk = np.argpartition(scores, -k)[-k:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        out = []
        for j in topk:
            pid = self.idx2item[j]
            out.append(
                {
                    "product_id": pid,
                    "score": float(scores[j]),
                    "product_name": self._pname(pid),
                }
            )
        return out

    def similar_items(self, product_id: str, k: int = 10):
        pid = str(product_id)
        if pid not in self.item2idx:
            return []
        j = self.item2idx[pid]
        sims = self.item_item_sim[j]

        k = min(k, sims.size - 1)
        if k <= 0:
            return []

        topk = np.argpartition(sims, -k)[-k:]
        topk = topk[np.argsort(sims[topk])[::-1]]

        out = []
        for jj in topk:
            if jj == j:
                continue
            pid2 = self.idx2item[jj]
            out.append(
                {
                    "product_id": pid2,
                    "similarity": float(sims[jj]),
                    "product_name": self._pname(pid2),
                }
            )
        return out
