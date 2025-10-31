# # from fastapi import FastAPI
# from fastapi import FastAPI
# import joblib
# import numpy as np
# from scipy.sparse import csr_matrix
# import os
# from prometheus_fastapi_instrumentator import Instrumentator

# ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "../../artifacts/")

# user2idx = joblib.load(os.path.join(ARTIFACT_DIR, "user2idx.pkl"))
# idx2user = joblib.load(os.path.join(ARTIFACT_DIR, "idx2user.pkl"))
# prod2idx = joblib.load(os.path.join(ARTIFACT_DIR, "prod2idx.pkl"))
# idx2prod = joblib.load(os.path.join(ARTIFACT_DIR, "idx2prod.pkl"))
# item_item_sim = joblib.load(os.path.join(ARTIFACT_DIR, "item_item_sim.pkl"))
# prod_name_map = joblib.load(os.path.join(ARTIFACT_DIR, "prod_name_map.pkl"))
# user_item_sparse = joblib.load(os.path.join(ARTIFACT_DIR, "user_item_sparse.pkl"))

# # FastAPI app
# app = FastAPI()

# # Instrumentator for Prometheus metrics
# instrumentator = Instrumentator()
# instrumentator.instrument(app).expose(app)

# def get_product_name(pid: str) -> str:
#     return prod_name_map.get(str(pid), str(pid))

# def recommend_for_user(user_id: str, k: int = 10, exclude_seen: bool = True):
#     if user_id not in user2idx:
#         return []

#     uidx = user2idx[user_id]
#     user_row = user_item_sparse.getrow(uidx)
#     interacted_items = user_row.indices

#     if interacted_items.size == 0:
#         return []

#     scores = item_item_sim[:, interacted_items].sum(axis=1)

#     if exclude_seen:
#         scores[interacted_items] = -np.inf

#     topk_idx = np.argpartition(scores, -k)[-k:]
#     topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

#     results = []
#     for j in topk_idx:
#         pid = idx2prod[j]
#         results.append({
#             "product_id": pid,
#             "score": float(scores[j]),
#             "product_name": get_product_name(pid)
#         })
#     return results

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.get("/recommend")
# def recommend(user_id: str, k: int = 5):
#     recs = recommend_for_user(user_id, k)
#     return recs if recs else {"message": "No recommendations"}
# from fastapi import FastAPI
from fastapi import FastAPI
import joblib
import numpy as np
from scipy.sparse import csr_matrix
import os
from prometheus_fastapi_instrumentator import Instrumentator

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "../../artifacts/")

user2idx = joblib.load(os.path.join(ARTIFACT_DIR, "user2idx.pkl"))
idx2user = joblib.load(os.path.join(ARTIFACT_DIR, "idx2user.pkl"))
prod2idx = joblib.load(os.path.join(ARTIFACT_DIR, "prod2idx.pkl"))
idx2prod = joblib.load(os.path.join(ARTIFACT_DIR, "idx2prod.pkl"))
item_item_sim = joblib.load(os.path.join(ARTIFACT_DIR, "item_item_sim.pkl"))
prod_name_map = joblib.load(os.path.join(ARTIFACT_DIR, "prod_name_map.pkl"))
user_item_sparse = joblib.load(os.path.join(ARTIFACT_DIR, "user_item_sparse.pkl"))

# FastAPI app
app = FastAPI()

# Prometheus metrics setup
from prometheus_client import (
    Counter,
    Histogram,
    start_http_server,
    REGISTRY,
    generate_latest,
)
from fastapi.responses import PlainTextResponse
import time

# Custom metrics
RECOMMENDATIONS_COUNTER = Counter(
    "recommendations_total", "Total number of recommendations made"
)
RECOMMENDATION_DURATION = Histogram(
    "recommendation_duration_seconds", "Time spent generating recommendations"
)
EMPTY_RECOMMENDATIONS = Counter(
    "empty_recommendations_total", "Number of times no recommendations were found"
)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(content=generate_latest(REGISTRY).decode())


instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
)
instrumentator.instrument(app).expose(app, "/metrics")


def get_product_name(pid: str) -> str:
    return prod_name_map.get(str(pid), str(pid))


def recommend_for_user(user_id: str, k: int = 10, exclude_seen: bool = True):
    start_time = time.time()
    RECOMMENDATIONS_COUNTER.inc()

    if user_id not in user2idx:
        EMPTY_RECOMMENDATIONS.inc()
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
        results.append(
            {
                "product_id": pid,
                "score": float(scores[j]),
                "product_name": get_product_name(pid),
            }
        )
    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 5):
    start_time = time.time()
    recs = recommend_for_user(user_id, k)
    duration = time.time() - start_time
    RECOMMENDATION_DURATION.observe(duration)
    return recs if recs else {"message": "No recommendations"}
