"""
Download a public Drive file using YOUR Colab Google session (Drive API).

Why: gdown hits per-FILE public quotas. The Drive API uses per-USER quotas,
which almost never trip. This is the most reliable workaround.

Usage in a Colab cell:
    !python eval/drive_download.py 1EJnx9oX9wQieF7UPQJeTVg850nZsuxTi data/_raw/phish.zip

You'll see an auth prompt the first time. Approve it.
"""
import sys
from pathlib import Path


def download(file_id: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Auth via Colab (works in Colab; fails outside it)
    try:
        from google.colab import auth          # type: ignore
        auth.authenticate_user()
    except ImportError:
        sys.exit("[error] This script requires Google Colab. "
                 "Outside Colab, see prepare_data.py --phish_zip / --benign_zip.")

    # Build Drive API client
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import google.auth
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "google-api-python-client"])
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import google.auth

    creds, _ = google.auth.default()
    service = build("drive", "v3", credentials=creds)

    request = service.files().get_media(fileId=file_id)
    with open(out, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  {int(status.progress() * 100)}%  ({status.resumable_progress / 1e6:.1f} MB)")
    print(f"[done] {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python drive_download.py <file_id> <out_path>")
    download(sys.argv[1], sys.argv[2])
