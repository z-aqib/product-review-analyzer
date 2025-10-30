# from fastapi import FastAPI
from fastapi import FastAPI
import joblib
import numpy as np
from scipy.sparse import csr_matrix

# Path to artifacts (2 folders up from src/app/)
ARTIFACT_DIR = "C:/Users/Tech Fever/mlops/project/product-review-analyzer/artifacts/"

# Load artifacts
print("Loading model artifacts...")
user2idx = joblib.load(f"{ARTIFACT_DIR}user2idx.pkl")
idx2user = joblib.load(f"{ARTIFACT_DIR}idx2user.pkl")
prod2idx = joblib.load(f"{ARTIFACT_DIR}prod2idx.pkl")
idx2prod = joblib.load(f"{ARTIFACT_DIR}idx2prod.pkl")
prod_name_map = joblib.load(f"{ARTIFACT_DIR}prod_name_map.pkl")

item_item_sim = joblib.load(f"{ARTIFACT_DIR}item_item_sim.pkl")
user_item_sparse = joblib.load(f"{ARTIFACT_DIR}user_item_sparse.pkl")  # csr_matrix saved as pickle

# FastAPI app
app = FastAPI()

def get_product_name(pid: str) -> str:
    return prod_name_map.get(str(pid), str(pid))

def recommend_for_user(user_id: str, k: int = 10, exclude_seen: bool = True):
    if user_id not in user2idx:
        return []

    uidx = user2idx[user_id]
    user_row = user_item_sparse.getrow(uidx)
    interacted_items = user_row.indices

    if interacted_items.size == 0:
        return []

    scores = item_item_sim[:, interacted_items].sum(axis=1)

    if exclude_seen:
        scores[interacted_items] = -np.inf

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(user_id: str, k: int = 5):
    recs = recommend_for_user(user_id, k)
    return recs if recs else {"message": "No recommendations"}
