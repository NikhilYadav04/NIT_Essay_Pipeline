"""
prepare_balanced.py — Create a balanced 100-essay dataset from ASAP2_train_sourcetexts.csv
===========================================================================================

Stratified sampling: ~14-17 essays per score tier, covering all score values present in the data.
Normalises raw scores to 0-6 scale (same as existing asap_clean.csv).
Outputs: data/asap_balanced_100.csv
"""

import os
import sys
import pandas as pd
import numpy as np

BATCH_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(BATCH_DIR)

SOURCE_FILE = os.path.join(ROOT, "Datasets", "ASAP2_train_sourcetexts.csv")
OUT_FILE    = os.path.join(BATCH_DIR, "data", "asap_balanced_100.csv")

# ── ASAP score → normalised 0-6 mapping ────────────────────────────────────────
# ASAP2 raw scores range 1-6 (holistic), but in this dataset scores 1-5 appear.
# We normalise by mapping raw → (raw-1)/max_raw * 6, rounded to nearest int.
# OR we can just use direct mapping if we know the rubric max.
#
# Looking at the raw data: score column appears to be already on 0-6 or 1-5.
# We'll detect the range automatically and map to 0-6.

def normalise_score(raw_score: int, raw_min: int, raw_max: int) -> int:
    """Map raw score linearly to 0-6 scale."""
    if raw_max == raw_min:
        return 3
    normed = (raw_score - raw_min) / (raw_max - raw_min) * 6
    return int(round(normed))


def main():
    print(f"\nLoading {os.path.basename(SOURCE_FILE)} ...")
    
    # Read only needed columns for speed (full file is 200MB)
    df = pd.read_csv(
        SOURCE_FILE,
        usecols=["essay_id", "score", "full_text", "assignment", "prompt_name"],
        dtype={"essay_id": str, "score": float}
    )
    
    # Drop any rows with missing text or score
    df = df.dropna(subset=["full_text", "score"])
    df["score"] = df["score"].astype(int)
    df["full_text"] = df["full_text"].astype(str).str.strip()
    
    # Remove very short essays (< 50 words)
    df["word_count"] = df["full_text"].apply(lambda t: len(t.split()))
    df = df[df["word_count"] >= 50]

    raw_min = df["score"].min()
    raw_max = df["score"].max()
    print(f"Raw score range: {raw_min} – {raw_max}  |  Total essays: {len(df):,}")
    
    # Compute normalised score
    df["human_score_norm"] = df["score"].apply(lambda s: normalise_score(s, raw_min, raw_max))
    df["human_score_raw"]  = df["score"]
    
    # Score distribution
    print("\nScore distribution in full dataset:")
    dist = df.groupby("human_score_norm").size()
    for score, count in dist.items():
        print(f"  Score {score}: {count:>6,} essays")
    
    # ── Stratified sampling ─────────────────────────────────────────────────────
    # Target: 100 essays total, roughly equal per score tier.
    # Scores with fewer essays get all, others get capped.
    
    score_values = sorted(df["human_score_norm"].unique())
    n_scores     = len(score_values)
    target_total = 100
    per_score    = target_total // n_scores
    remainder    = target_total % n_scores
    
    print(f"\nScore tiers found: {score_values}")
    print(f"Target per tier  : {per_score} (remainder {remainder} distributed to highest tiers)")
    
    # Build allocation: start even, then add remainder to higher scores
    allocation = {s: per_score for s in score_values}
    for i, s in enumerate(reversed(score_values)):
        if i < remainder:
            allocation[s] += 1
    
    # Sample
    sampled_parts = []
    actual_counts = {}
    
    for score in score_values:
        pool  = df[df["human_score_norm"] == score]
        n_take = min(allocation[score], len(pool))
        if n_take == 0:
            continue
        sampled = pool.sample(n=n_take, random_state=42)
        sampled_parts.append(sampled)
        actual_counts[score] = n_take
    
    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    print("\nActual samples per score tier:")
    for score, count in actual_counts.items():
        bar = "█" * count
        print(f"  Score {score}: {count:>3}  {bar}")
    
    print(f"\nTotal sampled: {len(result)}")
    
    # ── Build output in the same schema as asap_clean.csv ──────────────────────
    # Required cols: essay_id, dataset, human_score_raw, human_score_norm,
    #                essay_text, essay_prompt, word_count
    
    result_out = pd.DataFrame({
        "essay_id":         result["essay_id"],
        "dataset":          "ASAP2-balanced",
        "human_score_raw":  result["human_score_raw"],
        "human_score_norm": result["human_score_norm"],
        "essay_text":       result["full_text"],
        "essay_prompt":     result["assignment"].fillna(result["prompt_name"].fillna("")),
        "word_count":       result["word_count"],
    })
    
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    result_out.to_csv(OUT_FILE, index=False)
    
    print(f"\n✅ Saved → {OUT_FILE}")
    print(f"   Rows: {len(result_out)}  |  Columns: {list(result_out.columns)}")
    print("\nTo run Hybrid pipeline on this dataset:")
    print(f"  python batch_runner.py --pipeline hybrid --backend ollama --model llama3.1:8b --data asap_balanced_100.csv")


if __name__ == "__main__":
    main()
