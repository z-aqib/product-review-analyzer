# src/ml/service.py
from .recommenders.item_item import ItemItemRecommender

model = ItemItemRecommender("data/processed").fit()


def get_ml_candidates_for_user(user_id: str, k=5):
    return model.recommend_for_user(user_id=user_id, k=k, exclude_seen=True)
