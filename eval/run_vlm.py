"""
B3: original Phishpedia + VLM verifier (Claude vision).

For each page Phishpedia flags as phishing, ask Claude whether the brand
is actually being impersonated (not just an SSO icon, ad, or content image).
Final phish = Phishpedia AND VLM agreement.

Set ANTHROPIC_API_KEY before running:
    import os; os.environ["ANTHROPIC_API_KEY"] = "sk-..."

Usage:
    python eval/run_vlm.py \\
        --sample data/sample.json \\
        --out    data/results/vlm.json \\
        --model  claude-haiku-4-5
"""
from __future__ import annotations

import argparse, base64, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import run_pipeline


VLM_PROMPT = """You are reviewing a webpage screenshot to decide whether it is impersonating a brand.

URL: {url}
Brand allegedly impersonated: {brand}

Reply YES only if ALL of:
1. The {brand} logo is the page's main identity (header / login form / title), NOT a third-party SSO button, ad banner, or content thumbnail.
2. The page asks for credentials, payment info, or other sensitive data (login form, password field, card form, etc.).
3. The URL clearly does NOT belong to {brand}.

Otherwise reply NO. Reply with exactly one word: YES or NO."""


def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type)."""
    data = Path(path).read_bytes()
    media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return base64.standard_b64encode(data).decode(), media


def vlm_verify(client, model: str, screenshot_path: str, url: str, brand: str) -> bool:
    b64, media = encode_image(screenshot_path)
    msg = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text",
                 "text": VLM_PROMPT.format(url=url, brand=brand)},
            ],
        }],
    )
    txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return txt.strip().upper().startswith("YES")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.json")
    ap.add_argument("--out",    default="data/results/vlm.json")
    ap.add_argument("--model",  default="claude-haiku-4-5")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[error] set ANTHROPIC_API_KEY first")

    import anthropic
    client = anthropic.Anthropic()

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
            confirmed = vlm_verify(client, args.model, screenshot, url, brand)
        except Exception as e:
            confirmed = True   # fail open: trust Phishpedia if VLM errors
            err = str(e)
        else:
            err = None

        return {"pred_phish": bool(confirmed),
                "pred_brand": brand if confirmed else None,
                "pred_score": float(conf) if conf is not None else 0.0,
                "extra": {"phishpedia_phish": True, "vlm_confirmed": confirmed,
                          "vlm_error": err}}

    run_pipeline(args.sample, predict, args.out, f"vlm:{args.model}")


if __name__ == "__main__":
    main()
