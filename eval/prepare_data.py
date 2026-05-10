"""
Stream-sample the official Phishpedia test datasets without ever extracting
the full ~30K-page zips (which blow past Colab's disk limit).

Strategy:
  1. Open the zip and read its file list.
  2. Group entries by their "page folder" (the dir containing shot.png+info.txt).
  3. Randomly sample N folders.
  4. Extract ONLY the files inside those N folders, straight to data/{phish,benign}/.

Disk footprint = zip size + sampled output (~hundreds of MB), instead of
zip + full extraction (~50-70 GB).

Output layout:
    data/
        phish/<folder>/{shot.png, info.txt, html.txt}
        benign/<folder>/{shot.png, info.txt}
        sample.json   # list of {folder, url, label, brand?}

------------------------------------------------------------------------------
DOWNLOAD QUOTA WORKAROUND (the public Drive files often hit "Too many users"):

(1) [easiest] Smaller 5-brand subset (~3K phish):
        python eval/prepare_data.py --use_5brand --n_phish 500 --n_benign 500

(2) [recommended] Drive-API download via your authenticated Colab session:
        # cell 1
        from google.colab import auth; auth.authenticate_user()
        # cell 2
        import sys; sys.path.insert(0, '/content/Phishpedia/eval')
        from drive_download import download
        download('12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g', 'data/_raw/phish.zip')
        download('1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_', 'data/_raw/benign.zip')
        # cell 3
        !python eval/prepare_data.py \\
            --phish_zip  data/_raw/phish.zip \\
            --benign_zip data/_raw/benign.zip

(3) Manual Make-a-copy in your own Drive, then:
        python eval/prepare_data.py --phish_id <ID> --benign_id <ID>
------------------------------------------------------------------------------
"""
import argparse, json, random, sys, zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


# Default IDs (heavy quota; expect failures)
PHISH_30K_ID    = "12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g"
BENIGN_30K_ID   = "1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_"
PHISH_5BRAND_ID = "1EJnx9oX9wQieF7UPQJeTVg850nZsuxTi"


def gdown_download(file_id: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} already exists ({out.stat().st_size / 1e9:.2f} GB)")
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


def acquire_zip(label: str, zip_path_arg: str | None,
                file_id_arg: str | None, default_id: str,
                cache_zip: Path) -> Path:
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
        f"See docstring at top for fixes (use_5brand / Drive API / manual zip)."
    )


def index_zip(zip_path: Path) -> dict[str, list[zipfile.ZipInfo]]:
    """
    Group ZipInfo entries by page-folder. A 'page folder' is the parent
    directory of any shot.png entry that has a sibling info.txt.

    Returns: { folder_id (relative path str) : [ZipInfo, ...] }
    Where folder_id = the basename of the page folder.
    """
    print(f"[index] {zip_path} ...")
    by_parent: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    has_shot, has_info = set(), set()

    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        for info in infos:
            if info.is_dir():
                continue
            posix = PurePosixPath(info.filename)
            parent = posix.parent.as_posix()
            name = posix.name.lower()
            by_parent[parent].append(info)
            if name == "shot.png":
                has_shot.add(parent)
            elif name == "info.txt":
                has_info.add(parent)

    page_parents = has_shot & has_info
    page_map = {parent.split("/")[-1] or parent: by_parent[parent]
                for parent in sorted(page_parents)}
    print(f"  -> {len(page_map)} page folders inside zip")
    return page_map


def sample_folders(folder_map: dict[str, list[zipfile.ZipInfo]],
                   n: int, seed: int) -> dict[str, list[zipfile.ZipInfo]]:
    keys = sorted(folder_map.keys())
    if n >= len(keys):
        return folder_map
    chosen = random.Random(seed).sample(keys, n)
    return {k: folder_map[k] for k in chosen}


def extract_folders(zip_path: Path, sampled: dict[str, list[zipfile.ZipInfo]],
                    dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for folder_id, infos in sampled.items():
            target_dir = dst_root / folder_id
            target_dir.mkdir(parents=True, exist_ok=True)
            for info in infos:
                fname = PurePosixPath(info.filename).name
                if not fname:
                    continue
                with z.open(info) as src, open(target_dir / fname, "wb") as dst:
                    dst.write(src.read())


def parse_info(info_txt: Path) -> dict:
    lines = info_txt.read_text(errors="ignore").strip().splitlines()
    out = {"url": lines[0].strip() if lines else ""}
    if len(lines) >= 2 and lines[1].strip():
        out["brand"] = lines[1].strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_phish",   type=int, default=500)
    ap.add_argument("--n_benign",  type=int, default=500)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--data_dir",  default="data")

    ap.add_argument("--use_5brand", action="store_true")
    ap.add_argument("--phish_id",   help="Override Drive ID for phishing zip")
    ap.add_argument("--benign_id",  help="Override Drive ID for benign zip")
    ap.add_argument("--phish_zip",  help="Path to a locally-uploaded phishing zip")
    ap.add_argument("--benign_zip", help="Path to a locally-uploaded benign zip")
    ap.add_argument("--delete_zips_after", action="store_true",
                    help="Remove the large source zips once sampling is done")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir  = data_dir / "_raw"
    phish_default = PHISH_5BRAND_ID if args.use_5brand else PHISH_30K_ID

    # 1. Acquire zips
    phish_zip  = acquire_zip("phish",  args.phish_zip,  args.phish_id,
                             phish_default, raw_dir / "phish.zip")
    benign_zip = acquire_zip("benign", args.benign_zip, args.benign_id,
                             BENIGN_30K_ID, raw_dir / "benign.zip")

    # 2. Index zips (no extraction, just metadata)
    phish_idx  = index_zip(phish_zip)
    benign_idx = index_zip(benign_zip)
    if not phish_idx or not benign_idx:
        sys.exit("[error] No page folders (shot.png + info.txt) found in zip.")

    # 3. Sample folder ids
    phish_sample  = sample_folders(phish_idx,  args.n_phish,  args.seed)
    benign_sample = sample_folders(benign_idx, args.n_benign, args.seed + 1)
    print(f"sampled: {len(phish_sample)} phish, {len(benign_sample)} benign")

    # 4. Extract only the sampled folders
    print("[extract] sampled phish folders ...")
    extract_folders(phish_zip,  phish_sample,  data_dir / "phish")
    print("[extract] sampled benign folders ...")
    extract_folders(benign_zip, benign_sample, data_dir / "benign")

    # 5. Build sample.json
    rows = []
    for folder_id in phish_sample:
        d = data_dir / "phish" / folder_id
        info_txt = d / "info.txt"
        rows.append({"folder": str(d), "label": "phish",
                     **(parse_info(info_txt) if info_txt.exists() else {})})
    for folder_id in benign_sample:
        d = data_dir / "benign" / folder_id
        info_txt = d / "info.txt"
        rows.append({"folder": str(d), "label": "benign",
                     **(parse_info(info_txt) if info_txt.exists() else {})})

    out_json = data_dir / "sample.json"
    out_json.write_text(json.dumps(rows, indent=2))
    n_phish      = sum(r["label"] == "phish"  for r in rows)
    n_benign     = sum(r["label"] == "benign" for r in rows)
    n_with_brand = sum(bool(r.get("brand")) for r in rows if r["label"] == "phish")
    print(f"\nWrote {out_json}: {n_phish} phish + {n_benign} benign")
    print(f"  phish rows with brand label: {n_with_brand}/{n_phish}")
    if n_with_brand == 0:
        print("[warn] no brand labels found in info.txt line 2 - "
              "the dataset format may differ; identification rate will be 0.")

    # 6. Optional cleanup
    if args.delete_zips_after:
        for z in (phish_zip, benign_zip):
            if z.exists() and z.is_relative_to(raw_dir):
                print(f"[cleanup] removing {z}")
                z.unlink()


if __name__ == "__main__":
    main()
