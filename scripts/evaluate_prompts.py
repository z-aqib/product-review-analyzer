import pandas as pd
from prompts.summarize import generate_summary

df = pd.read_csv("data/small_eval.csv")


def evaluate():
    results = []
    for review in df["review_content"].fillna(""):
        summary = generate_summary(review)
        results.append(
            {
                "input_len": len(review),
                "summary_len": len(summary),
            }
        )
    return results


if __name__ == "__main__":
    r = evaluate()
    assert len(r) > 0, "Evaluation produced no output!"
    print("Evaluation OK:", r[:3])
