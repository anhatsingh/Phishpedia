"""
B1: replace the Siamese ResNetV2 with a CLIP image encoder.
Faster-RCNN logo detector is reused (same as original Phishpedia).

Usage:
    python eval/run_clip.py \\
        --sample data/sample.json \\
        --out    data/results/clip.json \\
        --model  openai/clip-vit-base-patch32 \\
        --thr    0.65
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


# ---------- CLIP image encoder ----------

class ClipEncoder:
    def __init__(self, model_id: str):
        from transformers import CLIPModel, CLIPProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc  = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode(self, img: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        inputs = self.proc(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]


# ---------- Brand prototype index ----------

def build_brand_index(encoder: ClipEncoder, targetlist: dict[str, list[Path]],
                      cache_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (brand_names, prototypes [N, D])."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return list(cached["brands"]), cached["protos"]

    brands, protos = [], []
    for brand, paths in tqdm(targetlist.items(), desc="indexing brands"):
        vecs = []
        for p in paths:
            try:
                vecs.append(encoder.encode(Image.open(p)))
            except Exception:
                continue
        if not vecs:
            continue
        proto = np.mean(np.stack(vecs), axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        brands.append(brand)
        protos.append(proto)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, brands=np.array(brands), protos=np.stack(protos))
    return brands, np.stack(protos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.json")
    ap.add_argument("--out",    default="data/results/clip.json")
    ap.add_argument("--model",  default="openai/clip-vit-base-patch32")
    ap.add_argument("--thr",    type=float, default=0.65)
    ap.add_argument("--targetlist",  default="models/expand_targetlist")
    ap.add_argument("--domain_map",  default="models/domain_map.pkl")
    args = ap.parse_args()

    detector   = load_logo_detector()
    targetlist = load_targetlist(args.targetlist)
    domain_map = load_domain_map(args.domain_map)
    encoder    = ClipEncoder(args.model)

    cache = Path("data/_cache") / f"clip_{Path(args.model).name}.npz"
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

    run_pipeline(args.sample, predict, args.out, f"clip:{Path(args.model).name}")


if __name__ == "__main__":
    main()
