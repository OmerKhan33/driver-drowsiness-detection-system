"""
Dataset validation for Driver Drowsiness Detection.

Performs structural and quality checks on the processed dataset:
    - Required directory layout (train/val/test x alert/drowsy)
    - Per-image readability and minimum-resolution check
    - Class-balance ratio per split
    - Cross-split leakage check via filename hashing

Run:
    python data/scripts/validate_dataset.py
    python data/scripts/validate_dataset.py --data-dir data/processed --strict

Exit code 0 if all checks pass, 1 otherwise — suitable for CI gating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SPLITS = ("train", "val", "test")
CLASSES = ("alert", "drowsy")
MIN_DIMENSION = 32  # smallest acceptable side, in pixels
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _file_sha256(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_layout(root: Path) -> list[str]:
    errors = []
    if not root.exists():
        errors.append(f"data dir does not exist: {root}")
        return errors
    for split in SPLITS:
        for cls in CLASSES:
            d = root / split / cls
            if not d.is_dir():
                errors.append(f"missing directory: {d}")
    return errors


def check_images(root: Path) -> tuple[dict, list[str]]:
    """Walk every image, ensure it opens cleanly and meets the minimum size."""
    counts: dict = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    bad: list[str] = []
    for split in SPLITS:
        for cls in CLASSES:
            d = root / split / cls
            if not d.is_dir():
                continue
            for path in d.iterdir():
                if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
                    continue
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        w, h = im.size
                        if min(w, h) < MIN_DIMENSION:
                            bad.append(f"{path}: too small ({w}x{h})")
                            continue
                    counts[split][cls] += 1
                except (UnidentifiedImageError, OSError) as e:
                    bad.append(f"{path}: {e}")
    return counts, bad


def check_class_balance(counts: dict, max_skew: float = 3.0) -> list[str]:
    warnings: list[str] = []
    for split, by_cls in counts.items():
        n_alert = by_cls["alert"]
        n_drowsy = by_cls["drowsy"]
        if n_alert == 0 or n_drowsy == 0:
            warnings.append(f"{split}: empty class (alert={n_alert}, drowsy={n_drowsy})")
            continue
        ratio = max(n_alert, n_drowsy) / min(n_alert, n_drowsy)
        if ratio > max_skew:
            warnings.append(
                f"{split}: class skew {ratio:.2f}x (alert={n_alert}, drowsy={n_drowsy}), "
                f"threshold {max_skew}x"
            )
    return warnings


def check_split_leakage(root: Path, sample_limit: int = 2000) -> list[str]:
    """Hash files and flag any image present in more than one split."""
    seen: dict[str, str] = {}
    leaks: list[str] = []
    for split in SPLITS:
        for cls in CLASSES:
            d = root / split / cls
            if not d.is_dir():
                continue
            files = sorted(d.iterdir())[:sample_limit]
            for path in files:
                if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
                    continue
                try:
                    digest = _file_sha256(path)
                except OSError:
                    continue
                key = f"{split}/{cls}/{path.name}"
                if digest in seen and seen[digest].split("/")[0] != split:
                    leaks.append(f"leak: {key} matches {seen[digest]}")
                else:
                    seen[digest] = key
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed",
                        help="Root of processed dataset")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures")
    parser.add_argument("--report", default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    root = Path(args.data_dir)
    print(f"Validating: {root}\n")

    errors = check_layout(root)
    if errors:
        for e in errors:
            print(f"  [ERR] {e}")
        return 1

    counts, bad = check_images(root)
    print("Image counts per split/class:")
    for s in SPLITS:
        print(f"  {s:<5} alert={counts[s]['alert']:>5}  drowsy={counts[s]['drowsy']:>5}")
    print()

    if bad:
        print(f"  [ERR] {len(bad)} unreadable / undersized images:")
        for b in bad[:10]:
            print(f"    {b}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")

    warnings = check_class_balance(counts)
    if warnings:
        print("Class balance warnings:")
        for w in warnings:
            print(f"  [WARN] {w}")

    print("\nLeakage check (sampled SHA256)...")
    leaks = check_split_leakage(root)
    if leaks:
        print(f"  [ERR] {len(leaks)} duplicates across splits:")
        for ln in leaks[:5]:
            print(f"    {ln}")
    else:
        print("  no cross-split duplicates found")

    report = {
        "data_dir": str(root),
        "counts": counts,
        "bad_images": bad,
        "balance_warnings": warnings,
        "leakage": leaks,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved to: {args.report}")

    failed = bool(bad) or bool(leaks) or (args.strict and bool(warnings))
    print("\n" + ("  [OK] all checks passed" if not failed else "  [FAIL] validation failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
