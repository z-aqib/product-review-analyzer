from src.ml.eval.metrics import recall_at_k, ndcg_at_k, catalog_coverage


def test_recall_and_ndcg_when_item_missing():
    predicted = ["a", "b", "c"]
    actual = "z"

    assert recall_at_k(actual, predicted) == 0.0
    assert ndcg_at_k(actual, predicted) == 0.0


def test_ndcg_value_for_known_rank():
    predicted = ["p1", "p2", "p3", "p4"]
    actual = "p3"  # rank = 3 (1-based)

    value = ndcg_at_k(actual, predicted)
    # rank 3 → 1 / log2(3 + 1) = 1 / log2(4) = 1 / 2 = 0.5
    assert abs(value - 0.5) < 1e-9


def test_catalog_coverage_empty_catalog():
    # no catalog → defined as 0.0
    assert catalog_coverage(["a", "b"], set()) == 0.0


def test_catalog_coverage_partial_overlap():
    catalog = {"a", "b", "c", "d"}
    all_candidates = ["a", "b", "x", "y"]

    cov = catalog_coverage(all_candidates, catalog)

    # only "a" and "b" are in catalog → 2 / 4 = 0.5
    assert abs(cov - 0.5) < 1e-9
