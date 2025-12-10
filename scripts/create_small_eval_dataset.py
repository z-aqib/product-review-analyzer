import pandas as pd

df = pd.read_csv("data/raw/amazon.csv")
small = df.sample(10, random_state=42)
small.to_csv("data/small_eval.csv", index=False)
