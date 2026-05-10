"""
B3: original Phishpedia + VLM verifier.

For each page Phishpedia flags as phishing, ask a VLM whether the brand is
actually being impersonated (not just an SSO icon, ad, or content image).
Final phish = Phishpedia AND VLM agreement.

Three backends, all selected via --backend:

  --backend gemini     Gemini Flash (free tier - recommended).
                       Set GEMINI_API_KEY: https://aistudio.google.com/apikey
  --backend local      Qwen2-VL-2B local on Colab T4 (no API key, ~5 GB VRAM).
  --backend anthropic  Claude vision (paid). Set ANTHROPIC_API_KEY.

Usage:
    python eval/run_vlm.py --backend gemini
    python eval/run_vlm.py --backend local
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common       import run_pipeline
from eval.vlm_backends import get_verifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",  default="data/sample.json")
    ap.add_argument("--out",     default="data/results/vlm.json")
    ap.add_argument("--backend", choices=["gemini", "local", "anthropic"],
                    default="gemini")
    ap.add_argument("--model",   default=None,
                    help="Override default model id for the chosen backend")
    args = ap.parse_args()

    verifier = get_verifier(args.backend, args.model)

    from phishpedia import PhishpediaWrapper
    pp = PhishpediaWrapper()

    def predict(row: dict) -> dict:
        screenshot = str(Path(row["folder"]) / "shot.png")
        html       = Path(row["folder"]) / "html.txt"
        html_arg   = str(html) if html.exists() else None
        url        = row.get("url", "")

        cat, brand, _domain, _vis, conf, _boxes, _t1, _t2 = \
            pp.test_orig_phishpedia(url, screenshot, html_arg)

        if int(cat) != 1 or not brand:
            return {"pred_phish": False, "pred_brand": None,
                    "pred_score": float(conf) if conf is not None else 0.0,
                    "extra": {"phishpedia_phish": False}}

        try:
            confirmed = verifier.verify(screenshot, url, brand)
            err = None
        except Exception as e:
            confirmed, err = True, str(e)   # fail open

        return {"pred_phish": bool(confirmed),
                "pred_brand": brand if confirmed else None,
                "pred_score": float(conf) if conf is not None else 0.0,
                "extra": {"phishpedia_phish": True, "vlm_confirmed": confirmed,
                          "vlm_error": err}}

    run_pipeline(args.sample, predict, args.out, f"vlm:{args.backend}")


if __name__ == "__main__":
    main()
