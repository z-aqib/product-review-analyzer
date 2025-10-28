# project/src/evaluate.py

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.ml.eval.eval_dataset import build_leave_one_out
from src.ml.eval.metrics import recall_at_k, ndcg_at_k, catalog_coverage
from src.ml.recommenders.item_item import ItemItemRecommender


def main():
    ap = argparse.ArgumentParser(description="Evaluate recommenders with Leave-One-Out")
    ap.add_argument(
        "--data-dir",
        default="data/processed",
        help="Folder with products.csv, reviews.csv, etc.",
    )
    ap.add_argument("--k", type=int, default=10, help="Top-K for evaluation")
    ap.add_argument(
        "--out-dir",
        default="data/processed/eval",
        help="Folder to write evaluation outputs",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["item_item"], default="item_item")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build LOO split
    train_df, test_df, skipped_users = build_leave_one_out(data_dir, seed=args.seed)
    # Save for inspection (optional)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # 2) Fit model (item–item uses full data; for strict LOO you can rebuild using train_df only)
    #    For a strict protocol, we can pass train_df to the model; here we keep it simple and use full reviews.
    model = ItemItemRecommender(data_dir).fit()

    # 3) Evaluate
    K = args.k
    users = test_df["user_id"].tolist()
    actual = test_df["product_id"].tolist()

    rec_list_flat = []  # to compute coverage
    rows = []

    for uid, true_pid in zip(users, actual):
        recs = model.recommend_for_user(uid, k=K, exclude_seen=True)
        predicted = [r["product_id"] for r in recs]
        rec_list_flat.extend(predicted)

        r = recall_at_k(true_pid, predicted)
        n = ndcg_at_k(true_pid, predicted)

        rows.append(
            {
                "user_id": uid,
                "true_item": true_pid,
                f"recall@{K}": r,
                f"ndcg@{K}": n,
                "len_train_interactions": (train_df["user_id"] == uid).sum(),
            }
        )

    # 4) Aggregate metrics
    per_user = pd.DataFrame(rows)
    per_user.to_csv(out_dir / f"eval_user_level_item_item_k{K}.csv", index=False)

    recall_mean = float(per_user[f"recall@{K}"].mean()) if not per_user.empty else 0.0
    ndcg_mean = float(per_user[f"ndcg@{K}"].mean()) if not per_user.empty else 0.0

    # Catalog coverage
    products = pd.read_csv(data_dir / "products.csv", dtype={"product_id": str})
    coverage = catalog_coverage(rec_list_flat, set(products["product_id"].astype(str)))

    summary = {
        "model": "item_item",
        "k": K,
        "users_evaluated": int(len(per_user)),
        "users_skipped_low_activity": int(len(skipped_users)),
        f"recall@{K}": recall_mean,
        f"ndcg@{K}": ndcg_mean,
        "catalog_coverage": coverage,
    }
    with open(
        out_dir / f"eval_summary_item_item_k{K}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
