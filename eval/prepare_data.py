"""
Download + sample the official Phishpedia test datasets.

Output layout:
    data/
        phish/<folder>/{shot.png, info.txt, html.txt}
        benign/<folder>/{shot.png, info.txt}
        sample.json   # list of {folder, url, label, brand?}

------------------------------------------------------------------------------
DOWNLOAD QUOTA WORKAROUND (the public Drive files often hit "Too many users"):

The default IDs hit Google's anti-quota guard for popular files. Three fixes,
in order of effort:

(1) [easiest] Use the smaller 5-brand subset (~3K phish):
        python eval/prepare_data.py --use_5brand --n_phish 500 --n_benign 500

(2) [recommended for full dataset] Make your own Drive copy:
      a. Open the file in your browser (signed in to Google):
         - Phish 30K:  https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g
         - Benign 30K: https://drive.google.com/file/d/1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_
      b. Right-click the file -> "Make a copy" (puts it in your My Drive).
      c. Right-click the copy -> "Share" -> "Anyone with the link" -> copy URL.
         The new ID is the long string in the URL.
      d. Pass the new IDs:
           python eval/prepare_data.py \\
                --phish_id  <YOUR_PHISH_ID> \\
                --benign_id <YOUR_BENIGN_ID>

(3) [manual] Download the zip from your browser, upload it to Colab/Kaggle,
    and pass the path:
        python eval/prepare_data.py \\
                --phish_zip  /content/phish_30k.zip \\
                --benign_zip /content/benign_30k.zip
------------------------------------------------------------------------------
"""
import argparse, json, random, shutil, sys, zipfile
from pathlib import Path

# Default IDs (heavy quota; expect failures)
PHISH_30K_ID  = "12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g"
BENIGN_30K_ID = "1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_"

# Smaller 5-brand subset (BoA, Chase, DHL, Microsoft, PayPal). Much less quota pressure.
PHISH_5BRAND_ID = "1EJnx9oX9wQieF7UPQJeTVg850nZsuxTi"


def gdown_download(file_id: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} already exists")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
        gdown.download(f"https://drive.google.com/uc?id={file_id}",
                       str(out), quiet=False, fuzzy=True)
        return out.exists() and out.stat().st_size > 0
    except Exception as e:
        print(f"[FAIL] gdown {file_id}: {e}", file=sys.stderr)
        return False


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[skip] {out_dir} already populated")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def find_page_folders(root: Path) -> list[Path]:
    """A page folder = any directory containing both shot.png and info.txt."""
    return sorted({p.parent for p in root.rglob("shot.png")
                   if (p.parent / "info.txt").exists()})


def parse_info(info_txt: Path) -> dict:
    """info.txt: line 1 = URL, line 2 (optional) = brand label."""
    lines = info_txt.read_text(errors="ignore").strip().splitlines()
    out = {"url": lines[0].strip() if lines else ""}
    if len(lines) >= 2 and lines[1].strip():
        out["brand"] = lines[1].strip()
    return out


def sample(folders: list[Path], n: int, seed: int) -> list[Path]:
    return folders if n >= len(folders) else random.Random(seed).sample(folders, n)


def copy_sample(src_folders: list[Path], dst_root: Path, label: str) -> list[dict]:
    dst_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in src_folders:
        dst = dst_root / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        rows.append({"folder": str(dst), "label": label,
                     **parse_info(dst / "info.txt")})
    return rows


def acquire_zip(label: str, zip_path_arg: str | None,
                file_id_arg: str | None, default_id: str,
                cache_zip: Path) -> Path:
    """Returns the local path to the dataset zip, however we acquire it."""
    if zip_path_arg:
        p = Path(zip_path_arg)
        if not p.exists():
            sys.exit(f"[error] --{label}_zip path does not exist: {p}")
        print(f"[use] {label} zip from --{label}_zip: {p}")
        return p

    fid = file_id_arg or default_id
    print(f"[gdown] {label}: trying file id {fid}")
    if gdown_download(fid, cache_zip):
        return cache_zip

    sys.exit(
        f"\n[FAIL] Could not download {label} dataset.\n"
        f"This is almost certainly Google's quota guard on a popular file.\n"
        f"See the docstring at the top of this file for three fixes:\n"
        f"  1) --use_5brand for a smaller subset\n"
        f"  2) --{label}_id <your-own-drive-copy-id>\n"
        f"  3) --{label}_zip <path-to-manually-uploaded-zip>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_phish",   type=int, default=500)
    ap.add_argument("--n_benign",  type=int, default=500)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--data_dir",  default="data")

    ap.add_argument("--use_5brand", action="store_true",
                    help="Use the 5-brand phish subset (lower quota pressure)")

    ap.add_argument("--phish_id",   help="Override Drive ID for phishing zip")
    ap.add_argument("--benign_id",  help="Override Drive ID for benign zip")
    ap.add_argument("--phish_zip",  help="Path to a locally-uploaded phishing zip")
    ap.add_argument("--benign_zip", help="Path to a locally-uploaded benign zip")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir  = data_dir / "_raw"

    phish_default = PHISH_5BRAND_ID if args.use_5brand else PHISH_30K_ID

    # Acquire zips
    phish_zip  = acquire_zip("phish",  args.phish_zip,  args.phish_id,
                             phish_default, raw_dir / "phish.zip")
    benign_zip = acquire_zip("benign", args.benign_zip, args.benign_id,
                             BENIGN_30K_ID, raw_dir / "benign.zip")

    # Extract
    extract_zip(phish_zip,  raw_dir / "phish_extracted")
    extract_zip(benign_zip, raw_dir / "benign_extracted")

    # Find page folders
    phish_pages  = find_page_folders(raw_dir / "phish_extracted")
    benign_pages = find_page_folders(raw_dir / "benign_extracted")
    print(f"\nphish pages:  {len(phish_pages)}")
    print(f"benign pages: {len(benign_pages)}")
    if not phish_pages or not benign_pages:
        sys.exit("[error] No page folders found. The zip layout differs from "
                 "expected (each page = a folder with shot.png + info.txt).")

    # Sample
    phish_sample  = sample(phish_pages,  args.n_phish,  args.seed)
    benign_sample = sample(benign_pages, args.n_benign, args.seed + 1)

    # Copy + record
    rows  = copy_sample(phish_sample,  data_dir / "phish",  "phish")
    rows += copy_sample(benign_sample, data_dir / "benign", "benign")

    out_json = data_dir / "sample.json"
    out_json.write_text(json.dumps(rows, indent=2))
    n_phish = sum(r['label'] == 'phish' for r in rows)
    n_benign = sum(r['label'] == 'benign' for r in rows)
    n_with_brand = sum(bool(r.get("brand")) for r in rows if r["label"] == "phish")
    print(f"\nWrote {out_json}: {n_phish} phish + {n_benign} benign")
    print(f"  phish rows with brand label: {n_with_brand}/{n_phish}")
    if n_with_brand == 0:
        print("[warn] no brand labels found in info.txt line 2 - "
              "the dataset format may differ; identification rate will be 0.")


if __name__ == "__main__":
    main()
