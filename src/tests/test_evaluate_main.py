# src/tests/test_evaluate_main.py

import json
import sys
from pathlib import Path

import pandas as pd
import src.evaluate as evaluate_module


def test_evaluate_main_writes_summary_and_user_level(monkeypatch, tmp_path):
    """
    Smoke test for evaluate.main():
    - Uses fake build_leave_one_out and fake ItemItemRecommender
    - Writes eval_user_level...csv and eval_summary...json
    - Summary has expected keys and non-negative values
    """
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "eval"
    data_dir.mkdir()

    # Minimal products.csv
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["Prod 1", "Prod 2", "Prod 3"],
        }
    )
    products.to_csv(data_dir / "products.csv", index=False)

    # Fake LOO split: 2 evaluated users, 1 skipped
    def fake_build_leave_one_out(_data_dir, seed: int = 42):
        train_df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "product_id": ["p1", "p2", "p2"],
            }
        )
        test_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "product_id": ["p2", "p3"],
            }
        )
        skipped = ["u3"]
        return train_df, test_df, skipped

    class DummyModel:
        def __init__(self, data_dir_path: Path):
            self.data_dir_path = data_dir_path

        def fit(self):
            return self

        def recommend_for_user(self, user_id: str, k: int, exclude_seen: bool = True):
            # Always recommend p2 for simplicity
            return [{"product_id": "p2"}]

    monkeypatch.setattr(evaluate_module, "build_leave_one_out", fake_build_leave_one_out)
    monkeypatch.setattr(evaluate_module, "ItemItemRecommender", DummyModel)

    # Prepare argv for argparse inside evaluate.main()
    argv_backup = sys.argv
    sys.argv = [
        "evaluate.py",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--k",
        "1",
    ]

    try:
        evaluate_module.main()
    finally:
        # restore original argv
        sys.argv = argv_backup

    # Check that user-level CSV was written
    user_level_path = out_dir / "eval_user_level_item_item_k1.csv"
    assert user_level_path.exists()
    per_user = pd.read_csv(user_level_path)
    # We had 2 users in test_df
    assert len(per_user) == 2
    assert "user_id" in per_user.columns
    assert "true_item" in per_user.columns
    assert "recall@1" in per_user.columns
    assert "ndcg@1" in per_user.columns

    # Check that summary JSON was written
    summary_path = out_dir / "eval_summary_item_item_k1.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["model"] == "item_item"
    assert summary["k"] == 1
    assert summary["users_evaluated"] == 2
    assert summary["users_skipped_low_activity"] == 1
    assert "recall@1" in summary
    assert "ndcg@1" in summary
    assert "catalog_coverage" in summary
    # Basic sanity: metrics in [0, 1]
    assert 0.0 <= summary["catalog_coverage"] <= 1.0
