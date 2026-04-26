"""
Dataset Preparation Script for Driver Drowsiness Detection.

Restructures the Kaggle drowsiness dataset into train/val/test splits
with alert and drowsy classes.

Dataset source: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

Raw structure:
    train/Open_Eyes/    → alert
    train/no_yawn/      → alert
    train/Closed_Eyes/  → drowsy
    train/Yawn/         → drowsy
    test/Open_Eyes/     → alert
    test/no_yawn/       → alert
    test/Closed_Eyes/   → drowsy
    test/Yawn/          → drowsy

Output structure:
    data/processed/train/alert/
    data/processed/train/drowsy/
    data/processed/val/alert/
    data/processed/val/drowsy/
    data/processed/test/alert/
    data/processed/test/drowsy/
"""

import random
import shutil
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).resolve().parent.parent / "raw" / "drowsiness-dataset"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Mapping from raw folder names to target class
ALERT_FOLDERS = ["Open_Eyes", "Open", "no_yawn"]
DROWSY_FOLDERS = ["Closed_Eyes", "Closed", "Yawn", "yawn"]


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Collect all images from raw directory and categorize into alert/drowsy.

    Args:
        raw_dir: Path to the raw dataset directory.

    Returns:
        Dictionary with 'alert' and 'drowsy' keys, each containing a list of image paths.

    Raises:
        FileNotFoundError: If raw_dir does not exist.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_dir}\n"
            f"Please download the dataset from Kaggle and extract it to: {raw_dir}\n"
            f"URL: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset"
        )

    images = {"alert": [], "drowsy": []}

    # Search through all subdirectories (train/ and test/ in the raw download)
    for split_dir in sorted(raw_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            folder_name = class_dir.name
            class_label = None

            if folder_name in ALERT_FOLDERS:
                class_label = "alert"
            elif folder_name in DROWSY_FOLDERS:
                class_label = "drowsy"
            else:
                print(f"  ⚠ Unknown folder '{folder_name}' in {split_dir.name}/, skipping")
                continue

            folder_images = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            images[class_label].extend(folder_images)
            print(f"  Found {len(folder_images):>5} images in {split_dir.name}/{folder_name}/ → {class_label}")

    return images


def split_data(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Split a list of image paths into train/val/test sets.

    Args:
        images: List of image file paths.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing split image paths.

    Raises:
        ValueError: If ratios don't sum to approximately 1.0.
    """
    total_ratio = train_ratio + val_ratio + (1.0 - train_ratio - val_ratio)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, "
            f"val={val_ratio}, test={1.0 - train_ratio - val_ratio}"
        )

    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def copy_images(
    file_list: list[Path],
    dest_dir: Path,
    class_name: str,
    split_name: str,
) -> int:
    """Copy images to the destination directory.

    Args:
        file_list: List of source image paths.
        dest_dir: Destination directory path.
        class_name: Class label ('alert' or 'drowsy').
        split_name: Split name ('train', 'val', or 'test').

    Returns:
        Number of images copied successfully.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for i, src_path in enumerate(file_list):
        # Create unique filename to avoid collisions
        new_name = f"{class_name}_{split_name}_{i:05d}{src_path.suffix.lower()}"
        dst_path = dest_dir / new_name

        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except (OSError, shutil.Error) as e:
            print(f"  ✗ Failed to copy {src_path.name}: {e}")

    return copied


def prepare_dataset() -> None:
    """Main function to prepare the drowsiness dataset.

    Reads images from the raw Kaggle download, splits them into
    train/val/test sets, and copies them to the processed directory.
    """
    print("=" * 60)
    print("  Driver Drowsiness Dataset Preparation")
    print("=" * 60)
    print(f"\n  Raw directory:    {RAW_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Split ratios:     train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"  Random seed:      {RANDOM_SEED}")
    print()

    # Step 1: Collect all images
    print("Step 1: Collecting images from raw directory...")
    images = collect_images(RAW_DIR)
    print(f"\n  Total ALERT images:  {len(images['alert'])}")
    print(f"  Total DROWSY images: {len(images['drowsy'])}")
    print(f"  Total images:        {len(images['alert']) + len(images['drowsy'])}")

    if len(images["alert"]) == 0 and len(images["drowsy"]) == 0:
        print("\n  ✗ No images found! Please check the raw directory structure.")
        return

    # Step 2: Split each class
    print("\nStep 2: Splitting dataset...")
    splits = {}
    for class_name in ["alert", "drowsy"]:
        splits[class_name] = split_data(
            images[class_name],
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=RANDOM_SEED,
        )

    # Step 3: Clear output directory if it exists
    if OUTPUT_DIR.exists():
        print(f"\nStep 3: Clearing existing output directory: {OUTPUT_DIR}")
        for split_name in ["train", "val", "test"]:
            for class_name in ["alert", "drowsy"]:
                target = OUTPUT_DIR / split_name / class_name
                if target.exists():
                    shutil.rmtree(target)

    # Step 4: Copy images
    print("\nStep 4: Copying images to processed directory...\n")
    summary = {}

    for split_name in ["train", "val", "test"]:
        summary[split_name] = {}
        for class_name in ["alert", "drowsy"]:
            dest = OUTPUT_DIR / split_name / class_name
            count = copy_images(
                splits[class_name][split_name],
                dest,
                class_name,
                split_name,
            )
            summary[split_name][class_name] = count
            print(f"  ✓ {split_name:>5}/{class_name:<6}: {count:>5} images")

    # Step 5: Print final summary
    print("\n" + "=" * 60)
    print("  FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"\n  {'Split':<8} {'Alert':>8} {'Drowsy':>8} {'Total':>8}")
    print(f"  {'-' * 36}")

    grand_total = 0
    for split_name in ["train", "val", "test"]:
        alert_count = summary[split_name]["alert"]
        drowsy_count = summary[split_name]["drowsy"]
        total = alert_count + drowsy_count
        grand_total += total
        print(f"  {split_name:<8} {alert_count:>8} {drowsy_count:>8} {total:>8}")

    print(f"  {'-' * 36}")
    total_alert = sum(summary[s]["alert"] for s in summary)
    total_drowsy = sum(summary[s]["drowsy"] for s in summary)
    print(f"  {'TOTAL':<8} {total_alert:>8} {total_drowsy:>8} {grand_total:>8}")
    print()
    print("  ✓ Dataset preparation complete!")
    print(f"  ✓ Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()
