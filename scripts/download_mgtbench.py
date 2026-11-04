#!/usr/bin/env python3
"""Download just the three original wide MGTBench CSVs, preserving provenance.

Default: pinned, public GitHub research mirror. Its bytes could not be compared
with the official Google Drive source in this restricted environment. Select
--source official to use that source with optional ``gdown==5.2.0`` installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MIRROR_REPO = "derek33125/AITextDetection"
MIRROR_COMMIT = "6f289c4cf4e3966e9442d104ea250bae43f3ff9e"
FOLDER_URL = "https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola"
FILENAMES = ["Essay_LLMs.csv", "WP_LLMs.csv", "Reuters_LLMs.csv"]
MIRROR_SHA256 = {
    "Essay_LLMs.csv": "f7ae324296b3d81b2134e873406f70a457c0383168aa7a9de83b70051ca41707",
    "WP_LLMs.csv": "ee787f528f2b5d4440cf9cd714ea4f81f9cf7b59086c37d90a376ba3db3a7278",
    "Reuters_LLMs.csv": "d3a31dd8fd3b04ec333ec29ad3a929618b5a88beaf2e8338b50594957a6fa32b",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(output: Path, source: str = "mirror") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in FILENAMES):
        raise FileExistsError(f"{output}: existing CSV detected; use a separate directory to preserve source files.")
    metadata = {"method": source, "official_url": FOLDER_URL, "official_bytes_verified": source == "official",
                "dataset_license": "Not established by this mirror; upstream source corpus terms apply. No raw data redistribution intended."}
    # Stage the entire download before making any input CSV visible.
    with tempfile.TemporaryDirectory(prefix="mgtbench-", dir=output) as temporary:
        temp = Path(temporary)
        if source == "mirror":
            print("Using a pinned third-party MGTBench mirror; equality with official Google Drive bytes is unverified.", file=sys.stderr)
            metadata.update({"mirror_repository": f"https://github.com/{MIRROR_REPO}",
                             "mirror_commit": MIRROR_COMMIT,
                             "mirror_claim": "README identifies Data_Generation as original data source; CSV schema matches official loader."})
            for name in FILENAMES:
                url = f"https://raw.githubusercontent.com/{MIRROR_REPO}/{MIRROR_COMMIT}/Data_Generation/MGTBench/{name}"
                with urllib.request.urlopen(url, timeout=120) as response, (temp / name).open("wb") as target:
                    while block := response.read(1024 * 1024):
                        target.write(block)
        else:
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("Official Drive download requires: python -m pip install gdown==5.2.0") from exc
            files = gdown.download_folder(url=FOLDER_URL, skip_download=True, quiet=True, use_cookies=False)
            by_name = {f.name: f.id for f in files or []}
            missing = set(FILENAMES) - set(by_name)
            if missing:
                raise RuntimeError(f"Official Drive listing is missing expected files: {sorted(missing)}")
            for name in FILENAMES:
                result = gdown.download(id=by_name[name], output=str(temp / name), quiet=False,
                                        use_cookies=False, verify=True)
                if result is None:
                    raise RuntimeError(f"Official download failed: {name}")
        import csv
        checksums = {}
        for name in FILENAMES:
            path = temp / name
            checksum = digest(path)
            if source == "mirror" and checksum != MIRROR_SHA256[name]:
                raise RuntimeError(f"{name}: SHA-256 mismatch against the pinned mirror. Do not use these bytes.")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not {"human", "ChatGPT-turbo", "Claude"}.issubset(reader.fieldnames or []):
                    raise RuntimeError(f"{name}: unexpected CSV schema (possibly an HTML denial page).")
                count = sum(1 for _ in reader)
                if count < 300:
                    raise RuntimeError(f"{name}: only {count} rows, not the expected original benchmark.")
            checksums[name] = {"sha256": checksum, "rows": count, "bytes": path.stat().st_size}
        metadata["files"] = checksums
        for name in FILENAMES:
            (temp / name).replace(output / name)
    (output / "SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", choices=["mirror", "official"], default="mirror")
    args = parser.parse_args()
    try:
        metadata = download(args.output, args.source)
    except (OSError, RuntimeError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print("Keep TLS/proxy verification enabled. For an explicit policy denial, authorize the source domain in environment settings "
              "(official: drive.google.com and drive.usercontent.google.com; mirror: raw.githubusercontent.com), "
              "or provide the three original CSV files locally.", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
