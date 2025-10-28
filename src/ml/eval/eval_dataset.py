# project/src/ml/eval/eval_dataset.py

from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np


def build_leave_one_out(data_dir: str | Path, seed: int = 42):
    """
    Returns:
      train_df: (user_id, product_id)
      test_df : (user_id, product_id) one row per user with >=2 interactions
      skipped : users with <2 interactions
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    reviews = pd.read_csv(
        data_dir / "reviews.csv", dtype={"user_id": str, "product_id": str}
    )

    # implicit interactions
    inter = reviews[["user_id", "product_id"]].dropna().drop_duplicates()

    # group users by list of items
    items_by_user = inter.groupby("user_id")["product_id"].apply(list)

    test_rows = []
    train_rows = []
    skipped = []

    for uid, items in items_by_user.items():
        if len(items) < 2:
            skipped.append(uid)
            # put all into train to keep matrix shape consistent
            for pid in items:
                train_rows.append((uid, pid))
            continue

        # choose one test item at random (or your own policy)
        t_idx = rng.integers(0, len(items))
        test_pid = items[t_idx]

        # rest go to train
        for i, pid in enumerate(items):
            if i == t_idx:
                continue
            train_rows.append((uid, pid))

        test_rows.append((uid, test_pid))

    train_df = pd.DataFrame(train_rows, columns=["user_id", "product_id"])
    test_df = pd.DataFrame(test_rows, columns=["user_id", "product_id"])

    return train_df, test_df, skipped
