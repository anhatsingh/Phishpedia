"""
B2: same idea as run_clip.py but with SigLIP encoder.

Usage:
    python eval/run_siglip.py \\
        --sample data/sample.json \\
        --out    data/results/siglip.json \\
        --model  google/siglip-base-patch16-224 \\
        --thr    0.55
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (run_pipeline, load_logo_detector, detect_logo,
                         load_targetlist, load_domain_map, domain_matches_brand)


class SiglipEncoder:
    def __init__(self, model_id: str):
        from transformers import AutoModel, AutoProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.proc  = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode(self, img: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        inputs = self.proc(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]


def build_brand_index(encoder, targetlist, cache_path):
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return list(cached["brands"]), cached["protos"]
    brands, protos = [], []
    n_skipped, first_err = 0, None
    for brand, paths in tqdm(targetlist.items(), desc="indexing brands"):
        vecs = []
        for p in paths:
            try:
                vecs.append(encoder.encode(Image.open(p)))
            except Exception as e:
                n_skipped += 1
                if first_err is None:
                    first_err = (str(p), repr(e))
                continue
        if not vecs:
            continue
        proto = np.mean(np.stack(vecs), axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        brands.append(brand)
        protos.append(proto)
    if not protos:
        raise RuntimeError(
            f"build_brand_index produced 0 brand prototypes "
            f"(skipped {n_skipped} images). First error was on "
            f"{first_err[0] if first_err else '<none>'}: "
            f"{first_err[1] if first_err else '<none>'}"
        )
    if n_skipped:
        print(f"[warn] indexed {len(brands)} brands; "
              f"skipped {n_skipped} images. First skipped: {first_err}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, brands=np.array(brands), protos=np.stack(protos))
    return brands, np.stack(protos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.json")
    ap.add_argument("--out",    default="data/results/siglip.json")
    ap.add_argument("--model",  default="google/siglip-base-patch16-224")
    ap.add_argument("--thr",    type=float, default=0.55)
    ap.add_argument("--targetlist", default="models/expand_targetlist")
    ap.add_argument("--domain_map", default="models/domain_map.pkl")
    args = ap.parse_args()

    detector   = load_logo_detector()
    targetlist = load_targetlist(args.targetlist)
    domain_map = load_domain_map(args.domain_map)
    encoder    = SiglipEncoder(args.model)

    cache = Path("data/_cache") / f"siglip_{Path(args.model).name}.npz"
    brands, protos = build_brand_index(encoder, targetlist, cache)
    print(f"indexed {len(brands)} brands, dim={protos.shape[1]}")

    def predict(row: dict) -> dict:
        screenshot = str(Path(row["folder"]) / "shot.png")
        url        = row.get("url", "")

        det = detect_logo(detector, screenshot)
        if det is None:
            return {"pred_phish": False, "pred_brand": None, "pred_score": 0.0}

        _bbox, crop = det
        emb = encoder.encode(crop)
        sims = protos @ emb
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        brand = brands[idx]

        is_phish = (score >= args.thr) and not domain_matches_brand(url, brand, domain_map)
        return {"pred_phish": is_phish,
                "pred_brand": brand if is_phish else None,
                "pred_score": score}

    run_pipeline(args.sample, predict, args.out, f"siglip:{Path(args.model).name}")


if __name__ == "__main__":
    main()
