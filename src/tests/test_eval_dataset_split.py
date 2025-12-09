import pandas as pd

from src.ml.eval.eval_dataset import build_leave_one_out


def test_leave_one_out_basic(tmp_path):
    data_dir = tmp_path

    # u1: 3 items, u2: 1 item, u3: 2 items
    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u3", "u3"],
            "product_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
        }
    )
    reviews.to_csv(data_dir / "reviews.csv", index=False)

    train_df, test_df, skipped = build_leave_one_out(data_dir)

    # Column sanity
    assert set(train_df.columns) == {"user_id", "product_id"}
    assert set(test_df.columns) == {"user_id", "product_id"}

    # Users with >=2 interactions appear in test
    assert set(test_df["user_id"]) == {"u1", "u3"}
    assert "u2" not in set(test_df["user_id"])

    # Users with <2 interactions are skipped and all their items are in train
    assert skipped == ["u2"]
    assert (train_df["user_id"] == "u2").sum() == 1
    assert not (test_df["user_id"] == "u2").any()

    # For each user with >=2 items: exactly 1 test row, rest in train
    for uid, total in (("u1", 3), ("u3", 2)):
        train_count = (train_df["user_id"] == uid).sum()
        test_count = (test_df["user_id"] == uid).sum()
        assert test_count == 1
        assert train_count + test_count == total


def test_leave_one_out_deterministic_with_seed(tmp_path):
    data_dir = tmp_path

    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "product_id": ["p1", "p2", "p3", "p4"],
        }
    )
    reviews.to_csv(data_dir / "reviews.csv", index=False)

    train1, test1, skipped1 = build_leave_one_out(data_dir, seed=123)
    train2, test2, skipped2 = build_leave_one_out(data_dir, seed=123)

    # same seed → identical splits
    pd.testing.assert_frame_equal(
        train1.sort_values(["user_id", "product_id"]).reset_index(drop=True),
        train2.sort_values(["user_id", "product_id"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        test1.sort_values(["user_id", "product_id"]).reset_index(drop=True),
        test2.sort_values(["user_id", "product_id"]).reset_index(drop=True),
    )
    assert skipped1 == skipped2


def test_leave_one_out_all_users_too_small(tmp_path):
    data_dir = tmp_path

    reviews = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "product_id": ["p1", "p2"],
        }
    )
    reviews.to_csv(data_dir / "reviews.csv", index=False)

    train_df, test_df, skipped = build_leave_one_out(data_dir)

    # No user has >=2 interactions → no test rows
    assert test_df.empty
    assert set(skipped) == {"u1", "u2"}
    # All interactions go to train
    assert len(train_df) == 2
