"""
Shared utilities for the eval scripts. Run from the repo root, e.g.:
    python -m eval.run_original

Provides:
- Logo detector (Phishpedia's Faster-RCNN, reused by every variant)
- Target-list & domain-map loaders
- Domain matching helper
- Eval harness that takes `predict_fn(row) -> dict` and writes results JSON
"""
from __future__ import annotations

import json, pickle, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the repo root importable (configs.py, phishpedia.py, etc. live there)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------------------- Logo detector (Faster-RCNN) --------------------------

def load_logo_detector():
    """Load Phishpedia's Faster-RCNN. Returns the Detectron2 predictor object."""
    from configs import load_config
    ELE_MODEL, _SIA_THRE, _SIA_MODEL, _LOGO_FEATS, _LOGO_FILES, _DOMAIN_MAP = load_config()
    return ELE_MODEL


def detect_logo(predictor, image_path: str):
    """Top-confidence logo as (bbox_xyxy, RGB crop) or None if no logo found."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    out = predictor(img)["instances"].to("cpu")
    if hasattr(out, "pred_classes"):
        keep = (out.pred_classes == 0)            # class 0 = logo
        boxes  = out.pred_boxes.tensor[keep].numpy()
        scores = out.scores[keep].numpy()
    else:
        boxes  = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
    if len(boxes) == 0:
        return None
    top = int(np.argmax(scores))
    x1, y1, x2, y2 = boxes[top].astype(int).tolist()
    crop_bgr = img[max(y1, 0):y2, max(x1, 0):x2]
    if crop_bgr.size == 0:
        return None
    return boxes[top], crop_bgr[:, :, ::-1]       # RGB


# -------------------------- Brand-name normalization --------------------------

def normalize_brand(name: str | None) -> str | None:
    """
    Apply Phishpedia's brand_converter so that target-list folder names,
    domain_map keys, and dataset gt_brand labels live in a single namespace.
    Phishpedia normalises 'Adobe Inc.' -> 'Adobe', 'Google Inc.' -> 'Google',
    etc. Without this, every CLIP/SigLIP match fails the domain check
    (domain_map lookup misses) and the identification metric undercounts.
    """
    if not name:
        return None
    try:
        from utils import brand_converter   # provided by upstream Phishpedia
        return brand_converter(name)
    except Exception:
        return name


# -------------------------- Target list & domain map --------------------------

def load_targetlist(targetlist_dir: str) -> dict[str, list[Path]]:
    """{brand: [paths to logo variants]} from <targetlist_dir>/<brand>/*.{png,jpg,jpeg}."""
    root = Path(targetlist_dir)
    out: dict[str, list[Path]] = {}
    for brand_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted([*brand_dir.glob("*.png"), *brand_dir.glob("*.jpg"),
                       *brand_dir.glob("*.jpeg")])
        if imgs:
            out[brand_dir.name] = imgs
    return out


def load_domain_map(domain_map_pkl: str) -> dict[str, list[str]]:
    with open(domain_map_pkl, "rb") as f:
        return pickle.load(f)


def url_domain(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return host[4:] if host.startswith("www.") else host


def domain_matches_brand(url: str, brand: str, domain_map: dict) -> bool:
    host = url_domain(url)
    if not host:
        return False
    for d in domain_map.get(brand, []):
        d = d.lower().lstrip(".")
        if d.startswith("www."):
            d = d[4:]
        if host == d or host.endswith("." + d):
            return True
    return False


# -------------------------- Eval harness --------------------------

@dataclass
class Prediction:
    folder: str
    url: str
    label: str                 # "phish" | "benign"
    gt_brand: Optional[str]
    pred_phish: bool
    pred_brand: Optional[str]
    pred_score: float
    latency_s: float
    extra: Optional[dict] = None


def run_pipeline(sample_json: str, predict_fn: Callable[[dict], dict],
                 out_json: str, variant_name: str) -> dict:
    """
    `predict_fn(row)` must return:
        {"pred_phish": bool, "pred_brand": str|None,
         "pred_score": float, "extra": dict (optional)}
    """
    rows = json.loads(Path(sample_json).read_text())
    preds = []
    for row in tqdm(rows, desc=variant_name):
        t0 = time.time()
        try:
            r = predict_fn(row)
        except Exception as e:
            r = {"pred_phish": False, "pred_brand": None, "pred_score": 0.0,
                 "extra": {"error": str(e)}}
        latency = time.time() - t0
        preds.append(asdict(Prediction(
            folder=row["folder"], url=row.get("url", ""),
            label=row["label"], gt_brand=row.get("brand"),
            pred_phish=bool(r["pred_phish"]),
            pred_brand=r.get("pred_brand"),
            pred_score=float(r.get("pred_score", 0.0)),
            latency_s=latency,
            extra=r.get("extra"),
        )))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(preds, indent=2))
    return summarise(preds, variant_name)


# -------------------------- Metrics --------------------------

def summarise(preds: list[dict], variant: str) -> dict:
    df = pd.DataFrame(preds)
    phish, benign = df[df.label == "phish"], df[df.label == "benign"]

    n_phish_total   = len(phish)
    n_phish_flagged = int(phish.pred_phish.sum())
    # Compare normalised brand names so 'Adobe Inc.' (gt) matches 'Adobe' (pred).
    norm_gt   = phish.gt_brand.map(normalize_brand)
    norm_pred = phish.pred_brand.map(normalize_brand)
    n_brand_correct = int((phish.pred_phish & (norm_pred == norm_gt) &
                           norm_gt.notna()).sum())
    n_benign_total  = len(benign)
    n_fp            = int(benign.pred_phish.sum())

    tp, fp = n_phish_flagged, n_fp
    precision = tp / (tp + fp)        if (tp + fp)        else 0.0
    recall    = tp / n_phish_total    if n_phish_total    else 0.0
    ident     = n_brand_correct / n_phish_flagged if n_phish_flagged else 0.0
    mean_lat  = float(df.latency_s.mean())

    summary = {
        "variant": variant,
        "n_phish": n_phish_total, "n_benign": n_benign_total,
        "tp": tp, "fp": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "identification_rate": round(ident, 4),
        "mean_latency_s": round(mean_lat, 4),
    }
    print("\n=== {} ===".format(variant))
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary
