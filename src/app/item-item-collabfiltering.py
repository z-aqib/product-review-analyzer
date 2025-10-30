"""
==================================================================
Item–Item Collaborative Filtering (Cosine) — Baseline Recommender
==================================================================

Inputs:
  - products.csv
  - users.csv
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

# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# 1) Load data
# ----------------------------
print("Loading CSVs...")
products = pd.read_csv('data/processed/products.csv')
reviews = pd.read_csv('data/processed/reviews.csv')
users = pd.read_csv('data/processed/users.csv')


# We only need (user_id, product_id) for implicit CF
if "user_id" not in reviews.columns or "product_id" not in reviews.columns:
    raise ValueError("reviews.csv must contain 'user_id' and 'product_id' columns.")

# De-duplicate in case there are repeated user→product rows
interactions = reviews[["user_id", "product_id"]].dropna().drop_duplicates()

print(f"Loaded: {len(products)} products, {len(users)} users, {len(interactions)} interactions.")


# ----------------------------
# 2) Encode IDs → indices
# ----------------------------
print("Encoding IDs...")

user_ids = interactions["user_id"].astype(str).unique()
prod_ids = interactions["product_id"].astype(str).unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
idx2user = np.array(user_ids)

prod2idx = {p: j for j, p in enumerate(prod_ids)}
idx2prod = np.array(prod_ids)

# Map to indices
ui = interactions["user_id"].astype(str).map(user2idx)
pi = interactions["product_id"].astype(str).map(prod2idx)


# ----------------------------
# 3) Build sparse interaction matrix R (users × items)
# ----------------------------
print("Building sparse user×item matrix...")
data = np.ones(len(interactions), dtype=np.float32)
R = csr_matrix((data, (ui.values, pi.values)), shape=(len(user2idx), len(prod2idx)))
print(f"R shape: {R.shape} (users × items)")


# ----------------------------
# 4) Compute item–item cosine similarity
# ----------------------------
print("Computing item–item cosine similarities...")
item_item_sim = cosine_similarity(R.T)  # shape = (n_items, n_items)
np.fill_diagonal(item_item_sim, 0.0)  # remove self-similarity


# ----------------------------
# 5) Helper lookups (id → name)
# ----------------------------
prod_name_map = (
    products.assign(product_id=products["product_id"].astype(str))
            .set_index("product_id")["product_name"]
            .to_dict()
)

def get_product_name(pid: str) -> str:
    """Return readable product name or ID if not found."""
    return prod_name_map.get(str(pid), str(pid))


# ----------------------------
# 6) Recommend for a user
# ----------------------------
def recommend_for_user(user_id: str, k: int = 10, exclude_seen: bool = True):
    """
    Recommend top-K items for a given user based on item–item similarities.
    """
    if user_id not in user2idx:
        print(f"[WARN] Unknown user_id={user_id}. Returning empty list.")
        return []

    uidx = user2idx[user_id]
    user_row = R.getrow(uidx)
    interacted_item_indices = user_row.indices

    if interacted_item_indices.size == 0:
        print(f"[INFO] User {user_id} has no interactions.")
        return []

    scores = item_item_sim[:, interacted_item_indices].sum(axis=1)

    if exclude_seen:
        scores[interacted_item_indices] = -np.inf

    topk_idx = np.argpartition(scores, -k)[-k:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    results = []
    for j in topk_idx:
        pid = idx2prod[j]
        results.append({
            "product_id": pid,
            "score": float(scores[j]),
            "product_name": get_product_name(pid)
        })
    return results


# ----------------------------
# 7) Similar items (by product)
# ----------------------------
def similar_items(product_id: str, k: int = 10):
    """
    Return top-K products most similar to the given product_id.
    """
    pid = str(product_id)
    if pid not in prod2idx:
        print(f"[WARN] Unknown product_id={product_id}")
        return []

    j = prod2idx[pid]
    sims = item_item_sim[j]

    topk_idx = np.argpartition(sims, -k)[-k:]
    topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]

    results = []
    for jj in topk_idx:
        if jj == j:
            continue
        pid2 = idx2prod[jj]
        results.append({
            "product_id": pid2,
            "similarity": float(sims[jj]),
            "product_name": get_product_name(pid2)
        })
    return results[:k]


# ----------------------------
# 8) Demo run
# ----------------------------
if __name__ == "__main__":
    # Example user and product from the dataset
    sample_user = interactions["user_id"].astype(str).iloc[421]
    print(f"\n=== Recommendations for user {sample_user} ===")
    recs = recommend_for_user(sample_user, k=10)
    for r in recs:
        print(f"- [{r['product_id']}] {r['product_name'][:80]} ... | score={r['score']:.4f}")

    sample_prod = interactions["product_id"].astype(str).iloc[10]
    print(f"\n=== Items similar to product {sample_prod}: {get_product_name(sample_prod)[:80]} ===")
    sims = similar_items(sample_prod, k=10)
    for s in sims:
        print(f"- [{s['product_id']}] {s['product_name'][:80]} ... | sim={s['similarity']:.4f}")

# ----------------------------
# 9) Save model artifacts
# ----------------------------
    import joblib
    import os

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(R, "artifacts/user_item_sparse.pkl")
    joblib.dump(item_item_sim, "artifacts/item_item_sim.pkl")
    joblib.dump(prod_name_map, "artifacts/prod_name_map.pkl")
    joblib.dump(idx2prod, "artifacts/idx2prod.pkl")
    joblib.dump(prod2idx, "artifacts/prod2idx.pkl")
    joblib.dump(user2idx, "artifacts/user2idx.pkl")
    joblib.dump(idx2user, "artifacts/idx2user.pkl")

    print("Model artifacts saved in /artifacts directory.")