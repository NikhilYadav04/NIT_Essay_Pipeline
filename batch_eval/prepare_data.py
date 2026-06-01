"""
prepare_data.py — Dataset Preprocessing for ASAP Batch Evaluation
=================================================================

Loads ASAP_100.csv, normalises scores from the original 1-5 range to 0-6
(matching the rubric used by both MAGIC and AutoSCORE), drops bad rows,
and saves a clean CSV ready for the batch runner.

Score normalisation:
    ASAP scores 1-5 → mapped linearly to 0-6:
        1 → 0
        2 → 2   (1.5 rounded)
        3 → 3
        4 → 5   (4.5 rounded)
        5 → 6

Run:
    cd "n:\\Dev\\Summer Intern"
    python batch_eval/prepare_data.py
"""

import os
import sys
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # n:\Dev\Summer Intern
DATA_IN  = os.path.join(ROOT, "datasets", "ASAP_100.csv")
DATA_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "asap_clean.csv")


def normalise_asap_score(raw_score: int) -> int:
    """Map ASAP 1-5 scores to rubric 0-6 scale."""
    return round((raw_score - 1) / 4 * 6)


def main():
    print("=" * 60)
    print("  ASAP Dataset Preparation")
    print("=" * 60)

    # ── Load ───────────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_IN):
        print(f"[ERROR] Dataset not found: {DATA_IN}")
        sys.exit(1)

    df = pd.read_csv(DATA_IN)
    print(f"\n[1] Loaded {len(df)} rows from {os.path.basename(DATA_IN)}")
    print(f"    Columns: {df.columns.tolist()}")

    # ── Select and rename relevant columns ────────────────────────────────────
    clean = pd.DataFrame()
    clean["essay_id"]     = df["essay_id"]
    clean["essay_text"]   = df["full_text"].str.strip()
    clean["essay_prompt"] = df["assignment"].str.strip()
    clean["human_score_raw"]  = df["score"]
    clean["human_score_norm"] = df["score"].apply(normalise_asap_score)
    clean["dataset"]      = "ASAP"

    # ── Quality filters ────────────────────────────────────────────────────────
    before = len(clean)

    # Drop blank essays
    clean = clean[clean["essay_text"].notna()]
    clean = clean[clean["essay_text"].str.len() > 0]

    # Drop essays with fewer than 30 words (too short to evaluate)
    clean["word_count"] = clean["essay_text"].str.split().str.len()
    clean = clean[clean["word_count"] >= 30]

    after = len(clean)
    print(f"\n[2] Quality filter: kept {after} / {before} essays (removed {before - after})")

    # ── Score distribution ─────────────────────────────────────────────────────
    print("\n[3] Score distribution (after normalisation):")
    dist = clean.groupby(["human_score_raw", "human_score_norm"]).size().reset_index(name="count")
    print(f"    {'Raw':>5}  {'→ Norm':>7}  {'Count':>7}")
    for _, row in dist.iterrows():
        bar = "█" * int(row["count"])
        print(f"    {int(row['human_score_raw']):>5}  {'→ ' + str(int(row['human_score_norm'])):>7}  {int(row['count']):>7}   {bar}")

    print(f"\n    Word count stats:")
    print(f"    Min: {clean['word_count'].min()}  |  Max: {clean['word_count'].max()}  |  Mean: {clean['word_count'].mean():.0f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    out_cols = ["essay_id", "dataset", "essay_text", "essay_prompt", "human_score_raw", "human_score_norm", "word_count"]
    clean[out_cols].to_csv(DATA_OUT, index=False)

    print(f"\n[4] Saved clean dataset → {DATA_OUT}")
    print(f"    Rows: {len(clean)}  |  Columns: {out_cols}")
    print("\n✅ Preparation complete. Ready for batch_runner.py\n")


if __name__ == "__main__":
    main()
