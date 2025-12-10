import pandas as pd

from src.ml.recommenders.item_item import ItemItemRecommender


def _make_small_dataset(tmp_path):
    data_dir = tmp_path

    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3", "p4"],
            "product_name": ["Prod 1", "Prod 2", "Prod 3", "Prod 4"],
        }
    )
    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "product_id": ["p1", "p2", "p2", "p3", "p3", "p4"],
        }
    )

    products.to_csv(data_dir / "products.csv", index=False)
    reviews.to_csv(data_dir / "reviews.csv", index=False)

    return data_dir


def test_fit_builds_matrices(tmp_path):
    data_dir = _make_small_dataset(tmp_path)
    model = ItemItemRecommender(data_dir).fit()

    # basic objects built
    assert model.R is not None
    assert model.item_item_sim is not None
    assert model.idx2item is not None
    assert model.idx2user is not None

    # shapes and mappings are consistent
    assert model.R.shape[0] == len(model.user2idx)
    assert model.R.shape[1] == len(model.item2idx)
    assert set(model.idx2item) == {"p1", "p2", "p3", "p4"}
    assert set(model.idx2user) == {"u1", "u2", "u3"}


def test_recommend_for_known_user_excludes_seen(tmp_path):
    data_dir = _make_small_dataset(tmp_path)
    model = ItemItemRecommender(data_dir).fit()

    # u1 has seen p1 and p2; unseen are p3 and p4
    # Ask for k=2 (the number of unseen items), so the model does not
    # need to "fill" results with any seen items, even if their score is -inf.
    recs = model.recommend_for_user("u1", k=2, exclude_seen=True)

    seen = {"p1", "p2"}

    # No seen items in recommendations
    for rec in recs:
        assert rec["product_id"] not in seen
        assert "score" in rec
        assert "product_name" in rec

    # At most k items and all must be from unseen set {p3, p4}
    assert len(recs) <= 2
    product_ids = {r["product_id"] for r in recs}
    assert product_ids.issubset({"p3", "p4"})


def test_recommend_for_unknown_user_returns_empty(tmp_path):
    data_dir = _make_small_dataset(tmp_path)
    model = ItemItemRecommender(data_dir).fit()

    assert model.recommend_for_user("unknown-user", k=5) == []


def test_similar_items_basic(tmp_path):
    data_dir = _make_small_dataset(tmp_path)
    model = ItemItemRecommender(data_dir).fit()

    sims = model.similar_items("p2", k=3)

    # should not include itself
    for rec in sims:
        assert rec["product_id"] != "p2"
        assert "similarity" in rec
        assert "product_name" in rec

    # at most k similar items
    assert len(sims) <= 3


def test_similar_items_for_unknown_returns_empty(tmp_path):
    data_dir = _make_small_dataset(tmp_path)
    model = ItemItemRecommender(data_dir).fit()

    assert model.similar_items("does-not-exist", k=5) == []
