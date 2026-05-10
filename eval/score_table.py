"""
Aggregate every variant's per-row JSON into a single comparison table.

Usage:
    python eval/score_table.py --results_dir data/results
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval.common import summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="data/results")
    ap.add_argument("--out_csv",     default="data/results/scoreboard.csv")
    args = ap.parse_args()

    rows = []
    for jp in sorted(Path(args.results_dir).glob("*.json")):
        if jp.name == "scoreboard.json":
            continue
        preds = json.loads(jp.read_text())
        rows.append(summarise(preds, jp.stem))

    df = pd.DataFrame(rows)
    cols = ["variant", "n_phish", "n_benign", "tp", "fp",
            "precision", "recall", "identification_rate", "mean_latency_s"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 80)
    print("SCOREBOARD")
    print("=" * 80)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
