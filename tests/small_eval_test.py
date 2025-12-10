import pandas as pd


def test_small_eval_exists():
    df = pd.read_csv("data/raw/small_eval.csv")
    assert len(df) > 0
