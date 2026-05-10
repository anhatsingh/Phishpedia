"""
Inference variant for Wang et al. 2024.

Loads the checkpoint produced by eval/train_wang.py, runs the model on each
sample.json row, and emits data/results/wang.json so eval/score_table.py
picks it up alongside original/clip/siglip/vlm/combined.

Usage:
    python eval/run_wang.py --thr 0.5
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import (run_pipeline, load_domain_map,
                         domain_matches_brand, normalize_brand)
from eval.wang.model import attention_net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",     default="data/sample.json")
    ap.add_argument("--out",        default="data/results/wang.json")
    ap.add_argument("--ckpt",       default="models/wang_v1.pth")
    ap.add_argument("--brands",     default="models/wang_v1.brands.json")
    ap.add_argument("--domain_map", default="models/domain_map.pkl")
    ap.add_argument("--thr",        type=float, default=0.5,
                    help="Softmax-prob threshold for the predicted brand")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. load class index
    cls_meta = json.loads(Path(args.brands).read_text())
    idx_to_class = {int(k): v for k, v in cls_meta["idx_to_class"].items()}

    # 2. load model
    ckpt = torch.load(args.ckpt, map_location=device)
    net = attention_net(
        num_classes=ckpt["num_classes"],
        topN=ckpt.get("topN", 6),
        cat_num=ckpt.get("cat_num", 1),
        proposal_num=ckpt.get("topN", 6),
    ).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    input_size = ckpt.get("input_size", 448)
    print(f"loaded {args.ckpt}: {ckpt['num_classes']} classes, "
          f"input_size={input_size}")

    # 3. preprocessor
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size), Image.BILINEAR),
        transforms.ToTensor(),
        norm,
    ])

    domain_map = load_domain_map(args.domain_map)

    @torch.no_grad()
    def predict(row: dict) -> dict:
        shot = Path(row["folder"]) / "shot.png"
        if not shot.exists():
            return {"pred_phish": False, "pred_brand": None, "pred_score": 0.0}
        img = tf(Image.open(shot).convert("RGB")).unsqueeze(0).to(device)
        _, concat_logits, _, _, _ = net(img)
        probs = torch.softmax(concat_logits[0], dim=-1)
        idx = int(probs.argmax())
        score = float(probs[idx])
        brand = idx_to_class.get(idx)
        url = row.get("url", "")

        is_phish = (score >= args.thr) and brand and \
                   not domain_matches_brand(url, brand, domain_map)
        return {"pred_phish": bool(is_phish),
                "pred_brand": brand if is_phish else None,
                "pred_score": score}

    run_pipeline(args.sample, predict, args.out, "wang")


if __name__ == "__main__":
    main()
