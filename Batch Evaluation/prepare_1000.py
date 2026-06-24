"""
prepare_1000.py — Create a balanced 1000-essay dataset from ASAP2_train_sourcetexts.csv
~166 essays per score tier (1-6). Score 6 has only 200 so all are taken.
Output: data/asap_balanced_1000.csv
"""

import os
import pandas as pd

BATCH_DIR   = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BATCH_DIR, "..", "Datasets", "ASAP2_train_sourcetexts.csv")
OUT_FILE    = os.path.join(BATCH_DIR, "data", "asap_balanced_1000.csv")

TARGET = 1000

def main():
    print("Loading ASAP2_train_sourcetexts.csv ...")
    df = pd.read_csv(
        SOURCE_FILE,
        usecols=["essay_id", "score", "full_text", "assignment", "prompt_name"],
        dtype={"essay_id": str, "score": float}
    )
    df = df.dropna(subset=["full_text", "score"])
    df["score"] = df["score"].astype(int)
    df["full_text"] = df["full_text"].astype(str).str.strip()
    df["word_count"] = df["full_text"].apply(lambda t: len(t.split()))
    df = df[df["word_count"] >= 50].copy()

    score_tiers = sorted(df["score"].unique())
    n_tiers     = len(score_tiers)
    per_tier    = TARGET // n_tiers      # 166
    remainder   = TARGET % n_tiers       # 4 extra spread to highest tiers

    allocation = {s: per_tier for s in score_tiers}
    for i, s in enumerate(reversed(score_tiers)):
        if i < remainder:
            allocation[s] += 1

    print(f"\nScore tiers: {score_tiers}  |  target per tier: {per_tier}  (remainder {remainder})")
    print("\nDistribution in source:")
    for s in score_tiers:
        print(f"  Score {s}: {len(df[df['score']==s]):>5,}  →  sampling {min(allocation[s], len(df[df['score']==s]))}")

    parts = []
    for s in score_tiers:
        pool   = df[df["score"] == s]
        n_take = min(allocation[s], len(pool))
        parts.append(pool.sample(n=n_take, random_state=42))

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    out = pd.DataFrame({
        "essay_id":         result["essay_id"],
        "dataset":          "ASAP2-balanced-1000",
        "human_score_raw":  result["score"],
        "human_score_norm": result["score"],   # raw 1-6 == normed 1-6 for ASAP2
        "essay_text":       result["full_text"],
        "essay_prompt":     result["assignment"].fillna(result["prompt_name"].fillna("")),
        "word_count":       result["word_count"],
    })

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"\nFinal distribution:")
    for s, c in out["human_score_norm"].value_counts().sort_index().items():
        print(f"  Score {s}: {c}  {'█' * (c // 3)}")
    print(f"\n✅  Saved {len(out)} essays → {OUT_FILE}")


if __name__ == "__main__":
    main()
