"""
Download a public Drive file using YOUR Colab Google session (Drive API).

Why: gdown hits per-FILE public quotas; the Drive API uses per-USER quotas
which almost never trip. This is the most reliable workaround.

This version downloads in PARALLEL CHUNKS via HTTP Range requests against
the authenticated alt=media endpoint, then stitches them together. On Colab
that's typically 4-8x faster than the sequential MediaIoBaseDownload path
because Drive throttles per-connection, not per-account.

IMPORTANT: `google.colab.auth.authenticate_user()` only works inside a
notebook cell - not in a `!python script.py` subprocess. So this module
is meant to be imported and called from a cell, not run via `!python`.

USAGE - paste these as one Colab cell:

    from google.colab import auth
    auth.authenticate_user()

    import sys; sys.path.insert(0, '/content/Phishpedia/eval')
    from drive_download import download_many
    download_many([
        ('12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g', 'data/_raw/phish.zip'),
        ('1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_', 'data/_raw/benign.zip'),
    ])

Single-file form is also available:
    from drive_download import download
    download('<file_id>', '<out_path>')

Tuning knobs (rarely needed):
    download(file_id, out_path, n_workers=8, chunk_mb=32)
    download_many(jobs, n_workers_per_file=4, chunk_mb=32)
"""
from __future__ import annotations

import os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _ensure_pkg() -> None:
    try:
        import google.auth                           # noqa
        from google.auth.transport.requests import AuthorizedSession  # noqa
        from googleapiclient.discovery import build  # noqa
        import requests                              # noqa
        import tqdm                                  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "google-api-python-client", "google-auth",
                               "requests", "tqdm"])


def _get_creds():
    try:
        import google.auth
        creds, _ = google.auth.default()
        return creds
    except Exception as e:
        sys.exit(
            "[error] No Google credentials available.\n"
            "First run in a notebook cell:\n"
            "    from google.colab import auth\n"
            "    auth.authenticate_user()\n"
            f"(underlying: {e})"
        )


def _get_metadata(creds, file_id: str) -> tuple[int, str]:
    """Return (total_size, name) for the Drive file."""
    from googleapiclient.discovery import build
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    meta = service.files().get(fileId=file_id, fields="size,name").execute()
    total = int(meta.get("size") or 0)
    name = meta.get("name") or file_id
    return total, name


# -------------------------- parallel chunked path --------------------------

def _download_chunk(session, url: str, start: int, end: int, out: Path,
                    bar, lock: threading.Lock, max_retries: int = 5) -> None:
    """Pull bytes[start..end] (inclusive) into the right offset of `out`."""
    headers = {"Range": f"bytes={start}-{end}"}
    last_err = None
    for attempt in range(max_retries):
        try:
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"unexpected {r.status_code} for "
                                       f"bytes={start}-{end}")
                # write at the chunk's offset under a lock (one fd shared)
                buf = bytearray()
                got = 0
                for piece in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    if piece:
                        buf.extend(piece)
                        got += len(piece)
                with lock:
                    with open(out, "r+b") as f:
                        f.seek(start)
                        f.write(buf)
                    bar.update(got)
                return
        except Exception as e:
            last_err = e
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"chunk {start}-{end} failed after "
                       f"{max_retries} attempts: {last_err}")


def _download_parallel(creds, file_id: str, out: Path, total: int, name: str,
                       n_workers: int, chunk_mb: int) -> None:
    from google.auth.transport.requests import AuthorizedSession
    from tqdm.auto import tqdm

    chunk = chunk_mb * 1024 * 1024
    ranges = [(s, min(s + chunk - 1, total - 1)) for s in range(0, total, chunk)]
    print(f"[parallel] {name}  total={total/1e9:.2f} GB  "
          f"{len(ranges)} chunks of {chunk_mb} MB  workers={n_workers}")

    # Pre-allocate the output file so seek+write hits a real offset.
    with open(out, "wb") as f:
        f.truncate(total)

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    session = AuthorizedSession(creds)
    lock = threading.Lock()

    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
              desc=name, leave=True) as bar:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_download_chunk, session, url, s, e, out,
                                 bar, lock) for (s, e) in ranges]
            for f in as_completed(futures):
                f.result()  # surface any exception

    actual = out.stat().st_size
    if actual != total:
        raise RuntimeError(f"size mismatch: expected {total}, got {actual}")


# -------------------------- sequential fallback --------------------------

def _download_sequential(creds, file_id: str, out: Path, total: int,
                         name: str) -> None:
    """Original MediaIoBaseDownload path, kept as fallback."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from tqdm.auto import tqdm

    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    request = service.files().get_media(fileId=file_id)
    print(f"[sequential] {name}  total={total/1e9:.2f} GB")
    with open(out, "wb") as f, tqdm(
        total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
        desc=name, leave=True,
    ) as bar:
        downloader = MediaIoBaseDownload(f, request, chunksize=64 * 1024 * 1024)
        done, prev = False, 0
        while not done:
            status, done = downloader.next_chunk()
            if status is not None:
                cur = status.resumable_progress
                bar.update(cur - prev)
                prev = cur
        if total and prev < total:
            bar.update(total - prev)


# --------------------------------- entry --------------------------------

def download(file_id: str, out_path: str | os.PathLike,
             n_workers: int = 8, chunk_mb: int = 32,
             parallel: bool = True) -> None:
    """
    Download a Drive file via the authenticated Drive API.

    Pre-requisite: `from google.colab import auth; auth.authenticate_user()`
    must have been called in a notebook cell BEFORE this function runs.

    Args:
        file_id   : Drive file ID.
        out_path  : where to write the bytes.
        n_workers : parallel HTTP range workers (default 8).
        chunk_mb  : size of each range request in MB (default 32).
        parallel  : if False, use the slow single-stream path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} ({out.stat().st_size / 1e6:.1f} MB) already exists")
        return

    _ensure_pkg()
    creds = _get_creds()
    total, name = _get_metadata(creds, file_id)

    # Files smaller than 2 chunks aren't worth parallelising.
    use_parallel = parallel and total > 2 * chunk_mb * 1024 * 1024

    if use_parallel:
        try:
            _download_parallel(creds, file_id, out, total, name,
                               n_workers, chunk_mb)
            print(f"[done] {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:
            print(f"[warn] parallel download failed ({type(e).__name__}: {e}); "
                  f"falling back to sequential")
            # Reset the file so the sequential path starts clean.
            try:
                out.unlink()
            except FileNotFoundError:
                pass

    _download_sequential(creds, file_id, out, total, name)
    print(f"[done] {out} ({out.stat().st_size / 1e6:.1f} MB)")


def download_many(jobs: list[tuple[str, str | os.PathLike]],
                  n_workers_per_file: int | None = None,
                  chunk_mb: int = 32, parallel: bool = True) -> None:
    """
    Download multiple Drive files concurrently.

    Args:
        jobs               : list of (file_id, out_path) tuples.
        n_workers_per_file : range workers per file. Defaults to
                             max(2, 8 // len(jobs)) so total concurrent
                             connections stay around 8 - matching the
                             single-file default.
        chunk_mb, parallel : forwarded to download() for each job.

    Example:
        download_many([
            ('12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g', 'data/_raw/phish.zip'),
            ('1yORUeSrF5vGcgxYrsCoqXcpOUHt-iHq_', 'data/_raw/benign.zip'),
        ])
    """
    if not jobs:
        return
    if n_workers_per_file is None:
        n_workers_per_file = max(2, 8 // len(jobs))
    print(f"[download_many] {len(jobs)} files, "
          f"{n_workers_per_file} workers each "
          f"(~{len(jobs) * n_workers_per_file} concurrent connections)")

    # Single ThreadPoolExecutor with one outer thread per file. Each call
    # to download() spins up its own inner pool of n_workers_per_file.
    errors: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {
            ex.submit(download, fid, out,
                      n_workers=n_workers_per_file,
                      chunk_mb=chunk_mb, parallel=parallel): (fid, out)
            for fid, out in jobs
        }
        for f in as_completed(futs):
            fid, out = futs[f]
            try:
                f.result()
            except BaseException as e:
                errors.append((str(out), e))
                print(f"[error] {out}: {type(e).__name__}: {e}")
    if errors:
        raise RuntimeError(f"{len(errors)} download(s) failed: "
                           + ", ".join(p for p, _ in errors))


if __name__ == "__main__":
    sys.exit(
        "This module must be imported from a Colab notebook cell - "
        "`!python eval/drive_download.py ...` won't work because\n"
        "google.colab.auth requires the IPython kernel context.\n\n"
        "Paste this in a cell instead:\n\n"
        "    from google.colab import auth\n"
        "    auth.authenticate_user()\n\n"
        "    import importlib.util\n"
        "    spec = importlib.util.spec_from_file_location("
        "'dd', '/content/Phishpedia/eval/drive_download.py')\n"
        "    dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)\n"
        "    dd.download('<FILE_ID>', '<OUT_PATH>')\n"
    )
