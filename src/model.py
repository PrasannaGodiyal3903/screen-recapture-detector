"""Screen recapture detector with trained and deterministic prediction paths.

The detector wraps a scikit-learn pipeline (StandardScaler + classifier)
and falls back to a hand-tuned heuristic when no trained model is available.

Classifier strategy
-------------------
An ensemble of **Gradient Boosting** and **Random Forest** combined via
soft-voting.  GBM excels at learning complex feature interactions while
RF provides stability.  Both use aggressive regularisation suited to
small datasets (50–100 images per class).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.features.extractor import FeatureExtractor
from src.preprocessing import load_image

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "screen_detector.joblib"


class ScreenDetector:
    """End-to-end screen recapture detector.

    Usage::

        detector = ScreenDetector()              # uses default model path
        score = detector.predict("photo.jpg")    # 0.0 = real, 1.0 = screen
    """

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        self.extractor = FeatureExtractor()
        self._pipeline = None
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if self._model_path.exists():
            self._load(self._model_path)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, image_or_path: np.ndarray | str | Path) -> float:
        """Return the probability that the image is a screen recapture.

        Args:
            image_or_path: BGR numpy array **or** path to an image file.

        Returns:
            Confidence score in [0, 1].  0 → real, 1 → screen.
        """
        if isinstance(image_or_path, (str, Path)):
            image = load_image(image_or_path)
        else:
            image = image_or_path

        features = self.extractor.extract(image)

        if self._pipeline is not None:
            return self._predict_trained(features)
        return self._predict_deterministic(features)

    def predict_timed(self, image_or_path: np.ndarray | str | Path) -> tuple[float, float]:
        """Like :meth:`predict` but also returns wall-clock latency in ms."""
        t0 = time.perf_counter()
        score = self.predict(image_or_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return score, elapsed_ms

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(
        self,
        images: list[np.ndarray],
        labels: list[int],
        *,
        n_estimators: int = 300,
        max_depth: int = 6,
    ) -> dict:
        """Train the classifier on extracted features.

        Uses a soft-voting ensemble of Gradient Boosting + Random Forest
        for robustness on small datasets.

        Args:
            images: List of BGR images.
            labels: 0 = real, 1 = screen.
            n_estimators: Base number of estimators per sub-classifier.
            max_depth: Maximum tree depth for sub-classifiers.

        Returns:
            Dict with training metadata.
        """
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        print("  Extracting features ...")
        X = np.array([self.extractor.extract(img) for img in images])
        y = np.array(labels)

        gbm = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=min(max_depth, 4),
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=42,
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )

        ensemble = VotingClassifier(
            estimators=[("gbm", gbm), ("rf", rf)],
            voting="soft",
            weights=[1.2, 1.0],   # slight edge to GBM
        )

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ])

        print("  Fitting ensemble (GBM + RF) ...")
        self._pipeline.fit(X, y)

        # Retrieve OOB from the RF sub-estimator
        rf_fitted = self._pipeline.named_steps["clf"].named_estimators_["rf"]
        oob = rf_fitted.oob_score_ if hasattr(rf_fitted, "oob_score_") else None

        return {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "oob_score": round(oob, 4) if oob is not None else None,
        }

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Persist the trained pipeline to disk."""
        path = Path(path) if path else self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        return path

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _load(self, path: Path) -> None:
        self._pipeline = joblib.load(path)

    def _predict_trained(self, features: np.ndarray) -> float:
        proba = self._pipeline.predict_proba(features.reshape(1, -1))
        return float(proba[0, 1])

    def _predict_deterministic(self, features: np.ndarray) -> float:
        """Heuristic fallback when no trained model is available.

        Feature layout (v2 — 87 features):
            0–17  : frequency features  (18)
            18–37 : colour features     (20)
            38–55 : texture features    (18)
            56–60 : edge features       (5)
            61–64 : noise features      (4)
            65–76 : channel features    (12)  <- NEW
            77–86 : wavelet features    (10)  <- NEW
        """
        # --- inter-channel HF correlation (STRONGEST signal) ---
        hf_mean_corr_fine = (features[65] + features[66] + features[67]) / 3.0
        hf_mean_corr_coarse = (features[68] + features[69] + features[70]) / 3.0
        # Real photos: high correlation (~0.8–1.0)
        # Screen photos: lower correlation (~0.3–0.7)
        channel_score = _sigmoid(-8.0 * hf_mean_corr_fine + 5.0)

        # --- gradient periodicity ---
        period_h = features[75]
        period_v = features[76]
        period_score = _sigmoid(5.0 * max(period_h, period_v) - 1.5)

        # --- frequency signals ---
        high_freq_ratio = features[10]
        peak_count = features[12]
        peak_prominence = features[13]

        freq_score = _sigmoid(
            2.5 * peak_prominence + 0.3 * peak_count + 3.0 * high_freq_ratio - 2.0
        )

        # --- wavelet detail ratio ---
        dwt1_detail = features[83]  # dwt1_detail_ratio
        wavelet_score = _sigmoid(10.0 * dwt1_detail - 2.0)

        # --- noise ---
        noise_sigma = features[61]
        snr = features[63]
        noise_score = _sigmoid(0.5 * noise_sigma - 0.05 * snr - 1.0)

        # --- colour ---
        dynamic_range = features[33]
        color_score = _sigmoid(-3.0 * dynamic_range + 1.5)

        # Weighted combination — channel features get highest weight
        score = (
            0.35 * channel_score
            + 0.15 * period_score
            + 0.20 * freq_score
            + 0.10 * wavelet_score
            + 0.10 * noise_score
            + 0.10 * color_score
        )

        return float(np.clip(score, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
