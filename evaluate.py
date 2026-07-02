#!/usr/bin/env python3
"""Evaluate the trained detector with full classification metrics.

Reports: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
per-image predictions, and feature importances.

Usage::

    python evaluate.py --data dataset/
    python evaluate.py --data dataset/ --model models/screen_detector.joblib
    python evaluate.py --data dataset/ --split 0.3
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

from src.features.extractor import FeatureExtractor
from src.model import ScreenDetector
from src.preprocessing import load_image
from train import SUPPORTED_EXTENSIONS, collect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the screen recapture detector.")
    parser.add_argument("--data", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--model", type=str, default=None, help="Trained model path.")
    parser.add_argument(
        "--split", type=float, default=0.25,
        help="Fraction of data to hold out for evaluation (default: 0.25).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Evaluate on the full dataset (no train/test split — use only when model was trained on separate data).",
    )
    args = parser.parse_args()

    # ---- load data ----
    data_dir = Path(args.data)
    images, labels, paths = collect_dataset(data_dir)
    print(f"Loaded {len(images)} images  (real={labels.count(0)}, screen={labels.count(1)})")

    if len(images) < 4:
        sys.exit("Need at least 4 images to evaluate.")

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold

    if args.full:
        _evaluate_full(images, labels, paths, args.model)
    else:
        _evaluate_cv(images, labels, paths, args.model, args.split)


def _evaluate_full(
    images: list[np.ndarray],
    labels: list[int],
    paths: list[str],
    model_path: str | None,
) -> None:
    """Evaluate a pre-trained model on the full dataset."""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    )

    detector = ScreenDetector(model_path=model_path)
    if not detector.is_trained:
        print("Warning: no trained model found — using deterministic heuristic.", file=sys.stderr)

    y_true, y_scores, latencies = [], [], []
    for img, label, fpath in zip(images, labels, paths):
        score, ms = detector.predict_timed(img)
        y_true.append(label)
        y_scores.append(score)
        latencies.append(ms)

    _print_results(y_true, y_scores, latencies, paths)


def _evaluate_cv(
    images: list[np.ndarray],
    labels: list[int],
    paths: list[str],
    model_path: str | None,
    test_size: float,
) -> None:
    """Stratified k-fold cross-validation (default k derived from test_size)."""
    from sklearn.model_selection import StratifiedKFold

    k = max(2, int(round(1.0 / test_size)))
    print(f"\nRunning {k}-fold stratified cross-validation ...\n")

    y_all = np.array(labels)
    all_scores = np.zeros(len(images))
    all_latencies = np.zeros(len(images))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels), 1):
        train_images = [images[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        detector = ScreenDetector(model_path=None)
        detector.train(train_images, train_labels)

        for i in test_idx:
            score, ms = detector.predict_timed(images[i])
            all_scores[i] = score
            all_latencies[i] = ms

        fold_preds = (all_scores[test_idx] >= 0.5).astype(int)
        fold_acc = np.mean(fold_preds == y_all[test_idx])
        print(f"  Fold {fold}: accuracy = {fold_acc:.4f}")

    _print_results(labels, all_scores.tolist(), all_latencies.tolist(), paths)


def _print_results(
    y_true: list[int],
    y_scores: list[float],
    latencies: list[float],
    paths: list[str],
) -> None:
    """Pretty-print evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    )

    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    y_s = np.array(y_scores)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy_score(y_t, y_p):.4f}")
    try:
        print(f"ROC-AUC   : {roc_auc_score(y_t, y_s):.4f}")
    except ValueError:
        print("ROC-AUC   : N/A (single class in test set)")

    print(f"\n{classification_report(y_t, y_p, target_names=['real', 'screen'])}")

    cm = confusion_matrix(y_t, y_p)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              real  screen")
    print(f"  Actual real  {cm[0, 0]:4d}   {cm[0, 1]:4d}")
    print(f"       screen  {cm[1, 0]:4d}   {cm[1, 1]:4d}")

    print(f"\nLatency  : mean={np.mean(latencies):.1f} ms, "
          f"median={np.median(latencies):.1f} ms, "
          f"p95={np.percentile(latencies, 95):.1f} ms")

    # Per-image breakdown
    print("\n--- Per-image predictions ---")
    for path, true, score, pred in zip(paths, y_true, y_scores, y_pred):
        status = "OK" if true == pred else "XX"
        true_label = "screen" if true else "real"
        print(f"  {status} {Path(path).name:>30s}  score={score:.3f}  true={true_label}")


if __name__ == "__main__":
    main()
