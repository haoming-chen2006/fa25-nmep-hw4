#!/usr/bin/env python3
"""Download and extract the Europarl fr-en.tgz without curl/wget.

Usage:
    python3 scripts/download_europarl.py [--url URL] [--out-dir data/nmt/europarl] [--keep]

By default it downloads https://www.statmt.org/europarl/v7/fr-en.tgz to
data/nmt/fr-en.tgz, extracts to data/nmt/europarl/, then removes the .tgz.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://www.statmt.org/europarl/v7/fr-en.tgz"


def download(url: str, dest: Path, chunk_size: int = 16 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
            total = resp.getheader("Content-Length")
            if total is not None:
                total = int(total)
            written = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    print(f"\r{written}/{total} bytes ({pct:.1f}%)", end="", flush=True)
            if total:
                print()
    except Exception as e:
        print(f"Download failed: {e}")
        raise


def extract(tgz: Path, out_dir: Path) -> None:
    print(f"Extracting {tgz} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(path=out_dir)
    except Exception as e:
        print(f"Extraction failed: {e}")
        raise


def check_files(out_dir: Path) -> int:
    # expected base filenames from the europarl tarball
    expected = ["europarl-v7.fr-en.en", "europarl-v7.fr-en.fr"]
    found = 0
    print("\nVerification:")
    for name in expected:
        p = out_dir / name
        if p.exists():
            found += 1
            size = p.stat().st_size
            print(f" - {name}: exists, size={size} bytes")
            try:
                with p.open("rb") as fh:
                    first = fh.readline(4096).decode(errors="replace").strip()
                print(f"   first line (trimmed): {first[:200]!r}")
            except Exception:
                pass
        else:
            print(f" - {name}: MISSING")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="URL to download")
    parser.add_argument("--out-dir", default="data/nmt/europarl", help="Directory to extract into")
    parser.add_argument("--tgz", default="data/nmt/fr-en.tgz", help="Where to save the downloaded tgz")
    parser.add_argument("--keep", action="store_true", help="Keep the downloaded .tgz file")
    args = parser.parse_args(argv)

    url = args.url
    tgz = Path(args.tgz)
    out_dir = Path(args.out_dir)

    try:
        download(url, tgz)
    except Exception:
        print("Error: download failed. You can try installing curl or wget, or retry later.")
        return 2

    try:
        extract(tgz, out_dir)
    except Exception:
        print("Error: extraction failed")
        return 3

    found = check_files(out_dir)
    if not args.keep:
        try:
            tgz.unlink()
        except Exception:
            pass

    if found == 2:
        print("Done: both files present.")
        return 0
    else:
        print("Done: some expected files are missing; check the archive contents in the output directory.")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
