"""
B4: CLIP backbone + VLM verifier.

Same as run_clip.py but pages flagged by CLIP must also be confirmed by a VLM
before being reported as phishing.

Three VLM backends, see eval/vlm_backends.py:
  --backend gemini     Gemini Flash free tier (recommended)
  --backend local      Qwen2-VL-2B local on Colab T4 (no API key)
  --backend anthropic  Claude vision (paid)

Usage:
    python eval/run_combined.py --backend gemini --thr 0.65
    python eval/run_combined.py --backend local  --thr 0.65
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common       import (run_pipeline, load_logo_detector, detect_logo,
                               load_targetlist, load_domain_map,
                               domain_matches_brand)
from eval.run_clip     import ClipEncoder, build_brand_index
from eval.vlm_backends import get_verifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",     default="data/sample.json")
    ap.add_argument("--out",        default="data/results/combined.json")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--backend",    choices=["gemini", "local", "anthropic"],
                    default="gemini")
    ap.add_argument("--vlm_model",  default=None,
                    help="Override default VLM model id for the chosen backend")
    ap.add_argument("--thr",        type=float, default=0.65)
    ap.add_argument("--targetlist", default="models/expand_targetlist")
    ap.add_argument("--domain_map", default="models/domain_map.pkl")
    args = ap.parse_args()

    detector   = load_logo_detector()
    targetlist = load_targetlist(args.targetlist)
    domain_map = load_domain_map(args.domain_map)
    encoder    = ClipEncoder(args.clip_model)
    verifier   = get_verifier(args.backend, args.vlm_model)

    cache = Path("data/_cache") / f"clip_{Path(args.clip_model).name}.npz"
    brands, protos = build_brand_index(encoder, targetlist, cache)
    print(f"indexed {len(brands)} brands")

    def predict(row: dict) -> dict:
        screenshot = str(Path(row["folder"]) / "shot.png")
        url        = row.get("url", "")

        det = detect_logo(detector, screenshot)
        if det is None:
            return {"pred_phish": False, "pred_brand": None, "pred_score": 0.0}

        _bbox, crop = det
        emb  = encoder.encode(crop)
        sims = protos @ emb
        idx  = int(np.argmax(sims))
        score, brand = float(sims[idx]), brands[idx]

        clip_phish = (score >= args.thr) and not domain_matches_brand(url, brand, domain_map)
        if not clip_phish:
            return {"pred_phish": False, "pred_brand": None, "pred_score": score,
                    "extra": {"clip_phish": False}}

        try:
            confirmed = verifier.verify(screenshot, url, brand)
            err = None
        except Exception as e:
            confirmed, err = True, str(e)

        return {"pred_phish": bool(confirmed),
                "pred_brand": brand if confirmed else None,
                "pred_score": score,
                "extra": {"clip_phish": True, "vlm_confirmed": confirmed,
                          "vlm_error": err}}

    run_pipeline(args.sample, predict, args.out,
                 f"combined:clip+{args.backend}")


if __name__ == "__main__":
    main()
