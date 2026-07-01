"""
Generate a synthetic HARS questionnaire dataset.

The real questionnaire responses contain sensitive student data and cannot be
shared. This script produces a fully synthetic dataset with the SAME structure
(14 ordinal HARS items scored 0-4) and a comparable five-class severity
distribution, so that the full pipeline can be executed and every table
reproduced end-to-end without any real participant data.

Usage:
    python generate_synthetic_data.py
Output:
    Anxiety_Synthetic.xlsx
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 306  # number of synthetic respondents

SYMPTOMS = [
    "Anxious Mood", "Tension", "Fears", "Insomnia", "Intellectual (Cognitive)",
    "Depressed Mood", "Somatic (Muscular)", "Somatic (Sensory)",
    "Cardiovascular Symptoms", "Respiratory Symptoms", "Gastrointestinal Symptoms",
    "Genitourinary Symptoms", "Autonomic Symptoms", "Behaviour at Interview",
]
# Column names contain the marker "pilih skor 0 - 4" so the pipeline detects them.
ITEM_COLUMNS = [f"Q{i+1} {s} (pilih skor 0 - 4)" for i, s in enumerate(SYMPTOMS)]

# Target severity bands (same thresholds as the study) and target proportions
# (approximately matching the real class distribution).
BANDS = [
    ("No anxiety",         0, 13, 0.095),
    ("Mild anxiety",       14, 20, 0.150),
    ("Moderate anxiety",   21, 27, 0.297),
    ("Severe anxiety",     28, 41, 0.395),
    ("Very severe anxiety", 42, 56, 0.063),
]


def distribute_total(total, n_items=14, max_score=4, rng=None):
    """Split a target total into n_items ordinal scores in [0, max_score]."""
    scores = np.zeros(n_items, dtype=int)
    for _ in range(int(total)):
        available = np.where(scores < max_score)[0]
        if len(available) == 0:
            break
        j = rng.choice(available)
        scores[j] += 1
    return scores


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    # number of respondents per band
    counts = [max(1, int(round(p * N))) for _, _, _, p in BANDS]
    # adjust to sum exactly to N
    while sum(counts) < N:
        counts[3] += 1
    while sum(counts) > N:
        counts[3] -= 1

    rows = []
    for (name, lo, hi, _), k in zip(BANDS, counts):
        for _ in range(k):
            total = rng.integers(lo, hi + 1)
            rows.append(distribute_total(total, len(ITEM_COLUMNS), 4, rng))

    items = np.array(rows)
    rng.shuffle(items)

    df = pd.DataFrame(items, columns=ITEM_COLUMNS)
    # Synthetic identifier column (analogous to the real "NPM"); not a real ID.
    df.insert(0, "NPM", [f"SYN{100000 + i}" for i in range(len(df))])

    out = "Anxiety_Synthetic.xlsx"
    df.to_excel(out, index=False)

    # Report resulting class distribution
    def band(s):
        for i, (_, lo, hi, _) in enumerate(BANDS):
            if lo <= s <= hi:
                return i
        return len(BANDS) - 1
    totals = items.sum(axis=1)
    labels = np.array([band(s) for s in totals])
    print(f"Wrote {out} with {len(df)} synthetic respondents.")
    for i, (name, _, _, _) in enumerate(BANDS):
        print(f"  {name:20s}: {int((labels == i).sum())}")


if __name__ == "__main__":
    main()
