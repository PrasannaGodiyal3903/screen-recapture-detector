#!/usr/bin/env python3
"""Train the screen recapture detector.

Expected dataset layout::

    dataset/
        real/        ← direct photographs
        screen/      ← photos of a screen showing a picture

Usage::

    python train.py --data dataset/
    python train.py --data dataset/ --estimators 300 --depth 10
    python train.py --data dataset/ --output models/my_model.joblib
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import ScreenDetector
from src.preprocessing import load_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_dataset(data_dir: Path) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Load images from real/ and screen/ sub-directories.

    Returns:
        (images, labels, paths) where label 0 = real, 1 = screen.
    """
    real_dir = data_dir / "real"
    screen_dir = data_dir / "screen"

    if not real_dir.is_dir():
        sys.exit(f"Error: directory not found — {real_dir}")
    if not screen_dir.is_dir():
        sys.exit(f"Error: directory not found — {screen_dir}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for label, folder in [(0, real_dir), (1, screen_dir)]:
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"Warning: no images found in {folder}", file=sys.stderr)

        for fpath in files:
            try:
                img = load_image(fpath)
                images.append(img)
                labels.append(label)
                paths.append(str(fpath))
            except (ValueError, FileNotFoundError) as exc:
                print(f"Skipping {fpath}: {exc}", file=sys.stderr)

    return images, labels, paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the screen recapture detector.")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Root directory containing real/ and screen/ sub-folders.",
    )
    parser.add_argument("--estimators", type=int, default=200, help="Number of trees.")
    parser.add_argument("--depth", type=int, default=8, help="Max tree depth.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output model path (default: models/screen_detector.joblib).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    print(f"Loading dataset from {data_dir} ...")
    images, labels, paths = collect_dataset(data_dir)

    n_real = labels.count(0)
    n_screen = labels.count(1)
    print(f"  Real images   : {n_real}")
    print(f"  Screen images : {n_screen}")
    print(f"  Total         : {len(images)}")

    if len(images) < 4:
        sys.exit("Error: need at least 4 images (2 per class) to train.")

    # ---- train ----
    print("\nExtracting features and training ...")
    detector = ScreenDetector(model_path=None)

    t0 = time.perf_counter()
    meta = detector.train(
        images, labels,
        n_estimators=args.estimators,
        max_depth=args.depth,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Features       : {meta['n_features']}")
    print(f"  OOB accuracy   : {meta['oob_score']:.4f}")

    # ---- save ----
    out_path = detector.save(args.output)
    print(f"\nModel saved to {out_path}")

    # ---- quick sanity check on training data ----
    print("\n--- Sanity check (predictions on training data) ---")
    correct = 0
    for img, label, fpath in zip(images, labels, paths):
        score = detector.predict(img)
        pred = 1 if score >= 0.5 else 0
        if pred == label:
            correct += 1
        status = "OK" if pred == label else "XX"
        print(f"  {status} {Path(fpath).name:>30s}  score={score:.3f}  true={'screen' if label else 'real'}")

    print(f"\nTraining accuracy: {correct}/{len(images)} = {correct / len(images):.2%}")


if __name__ == "__main__":
    main()
