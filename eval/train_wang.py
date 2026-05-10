"""
Train Wang et al. 2024's NTS-Net-based phishing classifier.

Two phases:
  1. (one-time) Extract a training set from data/_raw/phish.zip into
     data/train/phish/, EXCLUDING any folder already in data/sample.json
     (the eval set). Skipped if data/train/phish/ is already populated.
  2. Train attention_net for N epochs and save the checkpoint to
     models/wang_v1.pth + a brand-index map to models/wang_v1.brands.json.

Typical Colab usage (~30-60 min for 5 epochs on T4 with 5K training images):

    !python eval/train_wang.py \\
        --phish_zip data/_raw/phish.zip \\
        --train_size 5000 --epochs 5 --batch_size 16

After training:
    !python eval/run_wang.py --thr 0.5
    !python eval/score_table.py
"""
from __future__ import annotations

import argparse, ast, json, random, sys, time, zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common import normalize_brand
from eval.wang.model import attention_net, list_loss, ranking_loss


# ---------- training-set extraction ----------

def _parse_info(text: str) -> dict:
    """Same logic as eval/prepare_data.py:parse_info, inlined here."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            d = ast.literal_eval(text)
            if isinstance(d, dict):
                out = {}
                if isinstance(d.get("url"), str):  out["url"]   = d["url"].strip()
                if isinstance(d.get("brand"), str): out["brand"] = d["brand"].strip()
                if out.get("url"):
                    return out
        except (ValueError, SyntaxError):
            pass
    lines = text.splitlines()
    out = {"url": lines[0].strip() if lines else ""}
    if len(lines) >= 2 and lines[1].strip():
        out["brand"] = lines[1].strip()
    return out


def extract_training_set(phish_zip: Path, dst_dir: Path, n_train: int,
                         exclude_folders: set[str], seed: int) -> None:
    """Stream-extract n_train phishing folders that aren't in the eval set."""
    if dst_dir.exists() and len(list(dst_dir.iterdir())) >= n_train:
        print(f"[skip] {dst_dir} already has >= {n_train} folders")
        return

    print(f"[index] {phish_zip} ...")
    by_parent: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    has_shot, has_info = set(), set()
    with zipfile.ZipFile(phish_zip) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            posix = PurePosixPath(info.filename)
            parent = posix.parent.as_posix()
            name = posix.name.lower()
            by_parent[parent].append(info)
            if name == "shot.png":  has_shot.add(parent)
            elif name == "info.txt": has_info.add(parent)

    page_parents = sorted(has_shot & has_info)
    page_map = {p.split("/")[-1] or p: p for p in page_parents}
    candidates = [fid for fid in page_map if fid not in exclude_folders]
    print(f"  total page folders: {len(page_map)}; "
          f"excluding {len(exclude_folders)} eval folders -> "
          f"{len(candidates)} candidates")

    chosen = random.Random(seed).sample(candidates, min(n_train, len(candidates)))
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {len(chosen)} folders -> {dst_dir}")
    with zipfile.ZipFile(phish_zip) as z:
        for fid in tqdm(chosen, desc="extracting"):
            parent = page_map[fid]
            tgt = dst_dir / fid
            tgt.mkdir(parents=True, exist_ok=True)
            for info in by_parent[parent]:
                fname = PurePosixPath(info.filename).name
                if not fname:
                    continue
                with z.open(info) as src, open(tgt / fname, "wb") as dst:
                    dst.write(src.read())


# ---------- dataset ----------

class PhishingFolderDataset(Dataset):
    """
    Each phishing folder = one (image, brand_idx) pair.
    Folders whose normalized brand isn't in `class_to_idx` are filtered out.
    """

    def __init__(self, folders: list[Path], class_to_idx: dict[str, int],
                 transform):
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        for f in folders:
            info_txt = f / "info.txt"
            shot = f / "shot.png"
            if not (info_txt.exists() and shot.exists()):
                continue
            try:
                meta = _parse_info(info_txt.read_text(errors="ignore"))
            except Exception:
                continue
            brand = normalize_brand(meta.get("brand"))
            if not brand or brand not in class_to_idx:
                continue
            self.items.append((shot, class_to_idx[brand]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_class_index(folders: list[Path], min_samples: int) -> dict[str, int]:
    """Brand -> idx, keeping only brands with >= min_samples folders."""
    counts: dict[str, int] = defaultdict(int)
    for f in folders:
        info_txt = f / "info.txt"
        if not info_txt.exists():
            continue
        try:
            meta = _parse_info(info_txt.read_text(errors="ignore"))
        except Exception:
            continue
        brand = normalize_brand(meta.get("brand"))
        if brand:
            counts[brand] += 1
    kept = sorted(b for b, c in counts.items() if c >= min_samples)
    return {b: i for i, b in enumerate(kept)}


# ---------- training ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phish_zip",  default="data/_raw/phish.zip")
    ap.add_argument("--sample_json", default="data/sample.json",
                    help="Eval-set folders to EXCLUDE from training")
    ap.add_argument("--train_dir",  default="data/train/phish")
    ap.add_argument("--train_size", type=int, default=5000)
    ap.add_argument("--epochs",     type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--wd",         type=float, default=1e-4)
    ap.add_argument("--topN",       type=int, default=6)
    ap.add_argument("--cat_num",    type=int, default=1)
    ap.add_argument("--input_size", type=int, default=448)
    ap.add_argument("--min_samples_per_brand", type=int, default=3)
    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out_ckpt",   default="models/wang_v1.pth")
    ap.add_argument("--out_brands", default="models/wang_v1.brands.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. extract training set (idempotent)
    eval_rows = json.loads(Path(args.sample_json).read_text())
    exclude = {Path(r["folder"]).name for r in eval_rows if r["label"] == "phish"}
    extract_training_set(Path(args.phish_zip), Path(args.train_dir),
                         args.train_size, exclude, args.seed)

    # 2. enumerate folders, build class index
    folders = sorted(p for p in Path(args.train_dir).iterdir() if p.is_dir())
    class_to_idx = build_class_index(folders, args.min_samples_per_brand)
    if not class_to_idx:
        sys.exit("[error] No brands met --min_samples_per_brand. "
                 "Lower it or extract a bigger --train_size.")
    idx_to_class = {i: b for b, i in class_to_idx.items()}
    print(f"classes: {len(class_to_idx)} brands "
          f"(>= {args.min_samples_per_brand} samples each)")

    Path(args.out_brands).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_brands).write_text(json.dumps({
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }, indent=2))

    # 3. transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), Image.BILINEAR),
        transforms.ToTensor(),
        norm,
    ])

    # 4. split
    rng = random.Random(args.seed)
    rng.shuffle(folders)
    n_val = max(1, int(len(folders) * args.val_frac))
    val_folders, train_folders = folders[:n_val], folders[n_val:]

    train_ds = PhishingFolderDataset(train_folders, class_to_idx, train_tf)
    val_ds   = PhishingFolderDataset(val_folders,   class_to_idx, val_tf)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")
    if len(train_ds) == 0:
        sys.exit("[error] empty training set after filtering")

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device == "cuda"))
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=(device == "cuda"))

    # 5. model + optimizers (one per parameter group, paper convention)
    net = attention_net(num_classes=len(class_to_idx),
                        topN=args.topN, cat_num=args.cat_num,
                        proposal_num=args.topN).to(device)
    ce = torch.nn.CrossEntropyLoss()

    opts = [torch.optim.SGD(g.parameters(), lr=args.lr, momentum=0.9,
                            weight_decay=args.wd)
            for g in (net.pretrained_model, net.proposal_net,
                      net.concat_net, net.partcls_net)]

    # 6. train
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running = 0.0
        bar = tqdm(train_dl, desc=f"epoch {ep}")
        for img, label in bar:
            img, label = img.to(device), label.to(device)
            for o in opts: o.zero_grad()

            raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
            B = img.size(0)
            part_loss = list_loss(
                part_logits.view(B * args.topN, -1),
                label.unsqueeze(1).repeat(1, args.topN).view(-1),
            ).view(B, args.topN)
            raw_loss     = ce(raw_logits, label)
            concat_loss  = ce(concat_logits, label)
            rank_loss    = ranking_loss(top_n_prob, part_loss, args.topN)
            partcls_loss = ce(part_logits.view(B * args.topN, -1),
                              label.unsqueeze(1).repeat(1, args.topN).view(-1))
            total = raw_loss + rank_loss + concat_loss + partcls_loss
            total.backward()
            for o in opts: o.step()

            running += total.item() * B
            bar.set_postfix(loss=f"{total.item():.3f}")
        train_loss = running / max(1, len(train_ds))

        # val
        net.eval()
        correct, total_n = 0, 0
        with torch.no_grad():
            for img, label in val_dl:
                img, label = img.to(device), label.to(device)
                _, concat_logits, _, _, _ = net(img)
                pred = concat_logits.argmax(dim=1)
                correct += int((pred == label).sum())
                total_n += img.size(0)
        val_acc = correct / max(1, total_n)
        print(f"epoch {ep}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({time.time() - t0:.1f}s)")

    # 7. save
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "num_classes": len(class_to_idx),
                "topN": args.topN, "cat_num": args.cat_num,
                "input_size": args.input_size},
               args.out_ckpt)
    print(f"saved {args.out_ckpt}  ({Path(args.out_ckpt).stat().st_size / 1e6:.0f} MB)")
    print(f"saved {args.out_brands}")


if __name__ == "__main__":
    main()
