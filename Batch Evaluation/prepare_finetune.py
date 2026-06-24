"""
prepare_finetune.py — Prepare balanced 1500-essay fine-tuning dataset from ASAP-2
Output: data/finetune_1500.json  (ready to upload to Kaggle)
"""

import os
import json
import pandas as pd

BATCH_DIR   = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BATCH_DIR, "..", "Datasets", "ASAP2_train_sourcetexts.csv")
OUT_FILE    = os.path.join(BATCH_DIR, "data", "finetune_1500.json")

TARGET      = 1500
PER_TIER    = TARGET // 6  # 250 per score tier (1-6)

RUBRIC = """Score the following student essay on a scale of 1 to 6 based on these criteria:
- Score 1: Very poor. Minimal response, off-topic, or incomprehensible.
- Score 2: Poor. Limited development, weak organization, many errors.
- Score 3: Below average. Some development but lacks clarity and coherence.
- Score 4: Average. Adequate development, reasonable organization, some errors.
- Score 5: Good. Clear development, good organization, minor errors.
- Score 6: Excellent. Strong development, excellent organization, very few errors.

Respond with ONLY a single integer from 1 to 6."""


def format_prompt(essay_text: str, score: int) -> dict:
    return {
        "input": f"{RUBRIC}\n\nEssay:\n{essay_text.strip()}",
        "output": str(score)
    }


def main():
    print("Loading ASAP2_train_sourcetexts.csv ...")
    df = pd.read_csv(
        SOURCE_FILE,
        usecols=["essay_id", "score", "full_text"],
        dtype={"essay_id": str, "score": float}
    )

    df = df.dropna(subset=["full_text", "score"])
    df["score"]     = df["score"].astype(int)
    df["full_text"] = df["full_text"].astype(str).str.strip()
    df["word_count"] = df["full_text"].apply(lambda t: len(t.split()))

    # Remove very short essays
    df = df[df["word_count"] >= 50].copy()

    # Keep essays under 600 words to stay within T4 token limits
    df = df[df["word_count"] <= 600].copy()

    score_tiers = sorted(df["score"].unique())
    print(f"\nScore tiers: {score_tiers}")
    print(f"Target: {PER_TIER} per tier = {TARGET} total\n")

    parts = []
    for s in score_tiers:
        pool   = df[df["score"] == s]
        n_take = min(PER_TIER, len(pool))
        sampled = pool.sample(n=n_take, random_state=42)
        parts.append(sampled)
        print(f"  Score {s}: {len(pool):>5,} available  →  sampled {n_take}")

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal essays: {len(result)}")

    # Format as instruction tuning dataset
    dataset = []
    for _, row in result.iterrows():
        dataset.append(format_prompt(row["full_text"], row["score"]))

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSample entry:")
    print(f"  Input (first 200 chars): {dataset[0]['input'][:200]}...")
    print(f"  Output: {dataset[0]['output']}")
    print(f"\n✅ Saved {len(dataset)} examples → {OUT_FILE}")
    print(f"\nNext step: upload data/finetune_1500.json to Kaggle as a dataset")


if __name__ == "__main__":
    main()
