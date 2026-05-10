"""
Run the original Phishpedia (Faster-RCNN + Siamese ResNetV2) on the sample.

Usage (from repo root):
    python eval/run_original.py --sample data/sample.json --out data/results/original.json
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.json")
    ap.add_argument("--out",    default="data/results/original.json")
    args = ap.parse_args()

    from phishpedia import PhishpediaWrapper
    pp = PhishpediaWrapper()

    def predict(row: dict) -> dict:
        screenshot = str(Path(row["folder"]) / "shot.png")
        html       = Path(row["folder"]) / "html.txt"
        html_arg   = str(html) if html.exists() else None
        url        = row.get("url", "")

        cat, brand, _domain, _vis, conf, _boxes, _t1, _t2 = \
            pp.test_orig_phishpedia(url, screenshot, html_arg)

        return {
            "pred_phish": int(cat) == 1,
            "pred_brand": brand,
            "pred_score": float(conf) if conf is not None else 0.0,
        }

    run_pipeline(args.sample, predict, args.out, "original")


if __name__ == "__main__":
    main()
