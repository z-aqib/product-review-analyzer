#!/usr/bin/env python
# coding: utf-8

# notebooks/item-item-collabfiltering.py

# In[1]:


"""
===============================================================
Item–Item Collaborative Filtering (Cosine) — Baseline Recommender
===============================================================
Inputs:
  - products.csv
  - users.csv
  - product_categories.csv
  - reviews.csv  (one row per user_id–product_id review)

Outputs / Capabilities:
  - Build sparse user×item interaction matrix from reviews (implicit=1)
  - Compute item–item cosine similarity matrix
  - Recommend for a user: recommend_for_user(user_id, k=10)
  - Similar items for a product: similar_items(product_id, k=10)

Notes:
  - With ~1.3K products, full cosine matrix is fine (fast).
  - This is an implicit baseline (no numeric ratings needed).
  - Next steps: add ALS + hybrid re-rank.
"""


# # imports

# In[2]:


# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# # load data

# In[3]:


# ----------------------------
# 1) Load data
# ----------------------------
print("Loading CSVs...")
products = pd.read_csv("/kaggle/input/mlops-amazon-2/products.csv")
users = pd.read_csv("/kaggle/input/mlops-amazon-2/users.csv")
reviews = pd.read_csv("/kaggle/input/mlops-amazon-2/reviews.csv")


# In[4]:


# We only need (user_id, product_id) for implicit CF
if "user_id" not in reviews.columns or "product_id" not in reviews.columns:
    raise ValueError("reviews.csv must contain 'user_id' and 'product_id' columns.")

# De-duplicate in case there are repeated user→product rows
interactions = reviews[["user_id", "product_id"]].dropna().drop_duplicates()

print(f"Loaded: {len(products)} products, {len(users)} users, {len(interactions)} interactions.")


# # encode IDs

# In[5]:


# ----------------------------
# 2) Encode IDs → indices
# ----------------------------
# We create contiguous indices so we can build a compact CSR matrix.
print("Encoding IDs...")

user_ids = interactions["user_id"].astype(str).unique()
prod_ids = interactions["product_id"].astype(str).unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
idx2user = np.array(user_ids)

prod2idx = {p: j for j, p in enumerate(prod_ids)}
idx2prod = np.array(prod_ids)

# map to indices
ui = interactions["user_id"].astype(str).map(user2idx)
pi = interactions["product_id"].astype(str).map(prod2idx)


# # build sparse matrix

# In[6]:


# ----------------------------
# 3) Build sparse interaction matrix R (users × items)
# ----------------------------
# Implicit signal: 1 if user appeared in reviews for that product
print("Building sparse user×item matrix...")
data = np.ones(len(interactions), dtype=np.float32)
R = csr_matrix((data, (ui.values, pi.values)), shape=(len(user2idx), len(prod2idx)))
print(f"R shape: {R.shape} (users × items)")


# # compute item-item cosine similarity

# In[7]:


# ----------------------------
# 4) Compute item–item cosine similarity
# ----------------------------
# For cosine on items, we compare columns of R → compute cosine on R.T
print("Computing item–item cosine similarities...")
# cosine_similarity on sparse = OK; may densify for small matrices; here items~1.3k → fine
item_item_sim = cosine_similarity(R.T)  # shape = (n_items, n_items)

# Zero self-similarity if you prefer (so it won’t appear in top-K)
np.fill_diagonal(item_item_sim, 0.0)


# # helper functions

# In[8]:


# ----------------------------
# 5) Helper lookups (id → name)
# ----------------------------
prod_name_map = (
    products.assign(product_id=products["product_id"].astype(str))
    .set_index("product_id")["product_name"]
    .to_dict()
)


def get_product_name(pid: str) -> str:
    return prod_name_map.get(str(pid), str(pid))


# # recommend user

# In[9]:


# ----------------------------
# 6) Recommend for a user
# ----------------------------
def recommend_for_user(user_id: str, k: int = 10, exclude_seen: bool = True):
    """
    Score each item for the user as a weighted sum of similarities to the items they have interacted with.
    Steps:
      - Get items user has interacted with → I_u
      - Score vector s = sum(sim[:, i] for i in I_u)
      - Optionally zero out items already seen
      - Return top-K (product_id, score, product_name)
    """
    if user_id not in user2idx:
        print(f"[WARN] Unknown user_id={user_id}. Returning popular/similar fallback soon.")
        return []

    uidx = user2idx[user_id]
    # Indices of items this user interacted with
    user_row = R.getrow(uidx)  # 1 × n_items
    interacted_item_indices = user_row.indices

    if interacted_item_indices.size == 0:
        print(f"[INFO] User {user_id} has no interactions. Consider cold-start strategy.")
        return []

    # Sum similarities: fast vector add over columns
    # item_item_sim shape: (n_items, n_items)
    # We take the columns corresponding to interacted items, then sum across them
    scores = item_item_sim[:, interacted_item_indices].sum(axis=1)

    # Exclude already seen items if requested
    if exclude_seen:
        scores[interacted_item_indices] = -np.inf

    # Top-K indices
    topk_idx = np.argpartition(scores, -k)[-k:]
    # Sort by score desc
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    # Build results
    results = []
    for j in topk_idx:
        pid = idx2prod[j]
        results.append(
            {
                "product_id": pid,
                "score": float(scores[j]),
                "product_name": get_product_name(pid),
            }
        )
    return results


# In[10]:


# ----------------------------
# 7) Similar items (by product)
# ----------------------------
def similar_items(product_id: str, k: int = 10):
    """
    Return top-K products with highest cosine similarity to the given product_id.
    """
    pid = str(product_id)
    if pid not in prod2idx:
        print(f"[WARN] Unknown product_id={product_id}")
        return []

    j = prod2idx[pid]
    sims = item_item_sim[j]  # similarities from this item to all others

    # Top-K indices
    topk_idx = np.argpartition(sims, -k)[-k:]
    topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]

    results = []
    for jj in topk_idx:
        if jj == j:  # should already be zero from diag fill, but keep guard
            continue
        pid2 = idx2prod[jj]
        results.append(
            {
                "product_id": pid2,
                "similarity": float(sims[jj]),
                "product_name": get_product_name(pid2),
            }
        )
    return results[:k]


# # demo

# In[16]:


# ----------------------------
# 8) Quick demo
# ----------------------------
if __name__ == "__main__":
    # Pick an example user that exists
    sample_user = interactions["user_id"].astype(str).iloc[421]
    print(f"\n=== Recommendations for user {sample_user} ===")
    recs = recommend_for_user(sample_user, k=10)
    for r in recs:
        print(f"- [{r['product_id']}] {r['product_name'][:80]} ... | score={r['score']:.4f}")

    # Pick an example product that exists
    sample_prod = interactions["product_id"].astype(str).iloc[10]
    print(f"\n=== Items similar to product {sample_prod}: {get_product_name(sample_prod)[:80]} ===")
    sims = similar_items(sample_prod, k=10)
    for s in sims:
        print(f"- [{s['product_id']}] {s['product_name'][:80]} ... | sim={s['similarity']:.4f}")
