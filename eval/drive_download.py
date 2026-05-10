"""
Download a public Drive file using YOUR Colab Google session (Drive API).

Why: gdown hits per-FILE public quotas; the Drive API uses per-USER quotas
which almost never trip. This is the most reliable workaround.

IMPORTANT: `google.colab.auth.authenticate_user()` only works inside a
notebook cell - not in a `!python script.py` subprocess. So this module
is meant to be imported and called from a cell, not run via `!python`.

USAGE - paste these as one Colab cell:

    from google.colab import auth
    auth.authenticate_user()

    import sys; sys.path.insert(0, '/content/Phishpedia/eval')
    from drive_download import download
    download('12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g', 'data/_raw/phish.zip')
    download('1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_', 'data/_raw/benign.zip')

Then in the next cell:

    !python eval/prepare_data.py \\
        --phish_zip  data/_raw/phish.zip \\
        --benign_zip data/_raw/benign.zip \\
        --n_phish 500 --n_benign 500
"""
from __future__ import annotations

import os, subprocess, sys
from pathlib import Path


def _ensure_pkg() -> None:
    try:
        import googleapiclient  # noqa
        import google.auth      # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "google-api-python-client", "google-auth"])


def download(file_id: str, out_path: str | os.PathLike) -> None:
    """
    Download a Drive file via the authenticated Drive API.

    Pre-requisite: `from google.colab import auth; auth.authenticate_user()`
    must have been called in a notebook cell BEFORE this function runs.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} ({out.stat().st_size / 1e6:.1f} MB) already exists")
        return

    _ensure_pkg()
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    try:
        creds, _ = google.auth.default()
    except Exception as e:
        sys.exit(
            "[error] No Google credentials available.\n"
            "First run in a notebook cell:\n"
            "    from google.colab import auth\n"
            "    auth.authenticate_user()\n"
            f"(underlying: {e})"
        )

    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    request = service.files().get_media(fileId=file_id)

    print(f"[download] {file_id} -> {out}")
    with open(out, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  {int(status.progress() * 100)}%  "
                      f"({status.resumable_progress / 1e6:.1f} MB)")
    print(f"[done] {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(
        "This module must be imported from a Colab notebook cell - "
        "`!python eval/drive_download.py ...` won't work because\n"
        "google.colab.auth requires the IPython kernel context.\n\n"
        "Paste this in a cell instead:\n\n"
        "    from google.colab import auth\n"
        "    auth.authenticate_user()\n\n"
        "    import sys; sys.path.insert(0, '/content/Phishpedia/eval')\n"
        "    from drive_download import download\n"
        "    download('<FILE_ID>', '<OUT_PATH>')\n"
    )
