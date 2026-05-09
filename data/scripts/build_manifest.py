"""
Build a content-addressed manifest of the processed dataset.

Walks `data/processed/`, hashes every image with SHA256, and writes a single
JSON manifest with per-file digests, sizes, and a top-level dataset_hash
(hash of all sorted file digests). Two manifests with the same dataset_hash
describe the same data — a lightweight alternative to DVC for reproducibility.

Run:
    python data/scripts/build_manifest.py
    python data/scripts/build_manifest.py --output models/results/data_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _file_sha256(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(root: Path) -> dict:
    if not root.exists():
        raise FileNotFoundError(f"dataset not found: {root}")

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        files.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        })

    digests = sorted(f["sha256"] for f in files)
    dataset_hash = hashlib.sha256("\n".join(digests).encode()).hexdigest()

    return {
        "dataset_hash": dataset_hash,
        "root": str(root),
        "n_files": len(files),
        "total_bytes": sum(f["size_bytes"] for f in files),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="models/results/data_manifest.json")
    args = parser.parse_args()

    root = Path(args.data_dir)
    print(f"Building manifest for: {root}")

    try:
        manifest = build_manifest(root)
    except FileNotFoundError as e:
        print(f"  [ERR] {e}")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    print(f"  files:        {manifest['n_files']}")
    print(f"  total bytes:  {manifest['total_bytes']:,}")
    print(f"  dataset_hash: {manifest['dataset_hash']}")
    print(f"  written to:   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
