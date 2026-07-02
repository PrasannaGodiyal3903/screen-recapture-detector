#!/usr/bin/env python3
"""One-line screen recapture predictor.

Usage::

    python predict.py image.jpg
    python predict.py image.jpg --model models/screen_detector.joblib
    python predict.py image.jpg --verbose

Output: a single float in [0, 1].
    0 = real photograph, 1 = photo of a screen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import ScreenDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether an image is a real photo or a screen recapture.",
    )
    parser.add_argument("image", type=str, help="Path to the image file.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a trained model file (default: models/screen_detector.joblib).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print additional details (latency, mode, label).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found — {image_path}", file=sys.stderr)
        sys.exit(1)

    detector = ScreenDetector(model_path=args.model)

    score, latency_ms = detector.predict_timed(image_path)

    if args.verbose:
        mode = "trained model" if detector.is_trained else "deterministic heuristic"
        label = "SCREEN" if score >= 0.5 else "REAL"
        print(f"Score : {score:.4f}")
        print(f"Label : {label}")
        print(f"Mode  : {mode}")
        print(f"Latency: {latency_ms:.1f} ms")
    else:
        print(f"{score:.2f}")


if __name__ == "__main__":
    main()
