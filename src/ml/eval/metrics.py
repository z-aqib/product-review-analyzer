# project/src/ml/eval/metrics.py

from __future__ import annotations
from math import log2
from typing import Iterable, Set


def recall_at_k(actual_item: str, predicted_items: list[str]) -> float:
    return 1.0 if actual_item in predicted_items else 0.0


def ndcg_at_k(actual_item: str, predicted_items: list[str]) -> float:
    if actual_item in predicted_items:
        rank = predicted_items.index(actual_item) + 1
        return 1.0 / log2(rank + 1)
    return 0.0


def catalog_coverage(all_candidates: Iterable[str], catalog: Set[str]) -> float:
    """
    % of catalog items that ever appear in any user's recommendation list
    """
    recommended_set = set(all_candidates)
    if not catalog:
        return 0.0
    return len(recommended_set & catalog) / len(catalog)
