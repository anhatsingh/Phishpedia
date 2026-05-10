"""
Download + sample the official Phishpedia test datasets.

Output layout:
    data/
        phish/<folder>/{shot.png, info.txt, html.txt}
        benign/<folder>/{shot.png, info.txt}
        sample.json   # list of {folder, url, label, brand?}

Usage:
    python prepare_data.py --n_phish 500 --n_benign 500
"""
import argparse, json, os, random, shutil, zipfile
from pathlib import Path

import gdown

# Official dataset Google Drive IDs (from https://sites.google.com/view/phishpedia-site/home)
PHISH_30K_ID  = "12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g"   # 29,496 phishing pages
BENIGN_30K_ID = "1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_"   # 30,649 Alexa benign pages


def download_zip(file_id: str, out_zip: Path) -> None:
    if out_zip.exists():
        print(f"[skip] {out_zip} already exists")
        return
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(out_zip), quiet=False)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[skip] {out_dir} already populated")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def find_page_folders(root: Path) -> list[Path]:
    """A page folder is any directory that contains shot.png and info.txt."""
    folders = []
    for p in root.rglob("shot.png"):
        if (p.parent / "info.txt").exists():
            folders.append(p.parent)
    return folders


def parse_info(info_txt: Path) -> dict:
    """info.txt: line 1 = URL, line 2 (optional) = brand label."""
    lines = info_txt.read_text(errors="ignore").strip().splitlines()
    out = {"url": lines[0].strip() if lines else ""}
    if len(lines) >= 2 and lines[1].strip():
        out["brand"] = lines[1].strip()
    return out


def sample(folders: list[Path], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if n >= len(folders):
        return folders
    return rng.sample(folders, n)


def copy_sample(src_folders: list[Path], dst_root: Path, label: str) -> list[dict]:
    dst_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in src_folders:
        name = src.name
        dst = dst_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        info = parse_info(dst / "info.txt")
        rows.append({"folder": str(dst), "label": label, **info})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_phish",  type=int, default=500)
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir  = data_dir / "_raw"

    # 1. download
    download_zip(PHISH_30K_ID,  raw_dir / "phish_30k.zip")
    download_zip(BENIGN_30K_ID, raw_dir / "benign_30k.zip")

    # 2. extract
    extract_zip(raw_dir / "phish_30k.zip",  raw_dir / "phish_30k")
    extract_zip(raw_dir / "benign_30k.zip", raw_dir / "benign_30k")

    # 3. find pages
    phish_pages  = find_page_folders(raw_dir / "phish_30k")
    benign_pages = find_page_folders(raw_dir / "benign_30k")
    print(f"phish pages:  {len(phish_pages)}")
    print(f"benign pages: {len(benign_pages)}")

    # 4. sample
    phish_sample  = sample(phish_pages,  args.n_phish,  args.seed)
    benign_sample = sample(benign_pages, args.n_benign, args.seed + 1)

    # 5. copy + record
    rows  = copy_sample(phish_sample,  data_dir / "phish",  "phish")
    rows += copy_sample(benign_sample, data_dir / "benign", "benign")

    out_json = data_dir / "sample.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_json} with {len(rows)} pages "
          f"({sum(r['label']=='phish' for r in rows)} phish + "
          f"{sum(r['label']=='benign' for r in rows)} benign)")


if __name__ == "__main__":
    main()
