"""Unified feature extractor that aggregates all feature modules.

Orchestrates preprocessing and calls each feature module in turn,
concatenating the results into a single fixed-length vector.

Total features: 83
    frequency : 18   (was 14, +4 new)
    color     : 20
    texture   : 18
    edge      :  5
    noise     :  4
    channel   : 12   (NEW — inter-channel HF analysis)
    wavelet   : 10   (NEW — Haar DWT energy)
"""

import numpy as np

from src.features import channel, color, edge, frequency, noise, texture, wavelet
from src.preprocessing import prepare_for_analysis

ALL_FEATURE_NAMES: list[str] = (
    frequency.FEATURE_NAMES
    + color.FEATURE_NAMES
    + texture.FEATURE_NAMES
    + edge.FEATURE_NAMES
    + noise.FEATURE_NAMES
    + channel.FEATURE_NAMES
    + wavelet.FEATURE_NAMES
)

FEATURE_DIM: int = len(ALL_FEATURE_NAMES)


class FeatureExtractor:
    """Stateless feature extractor — call :meth:`extract` on a BGR image."""

    @property
    def feature_names(self) -> list[str]:
        return list(ALL_FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        return FEATURE_DIM

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract the full feature vector from a BGR image.

        Args:
            image: BGR uint8 image of any size.

        Returns:
            1-D float64 array of length :pyattr:`n_features`.
        """
        bgr, gray, hsv = prepare_for_analysis(image)

        freq_feats = frequency.extract(gray)
        color_feats = color.extract(bgr, hsv)
        tex_feats = texture.extract(gray)
        edge_feats = edge.extract(gray)
        noise_feats = noise.extract(gray)

        # Channel features work on the ORIGINAL resolution to preserve
        # fine sub-pixel detail that down-sampling destroys.
        channel_feats = channel.extract(image)

        # Wavelet features on the standard-size grayscale
        wavelet_feats = wavelet.extract(gray)

        features = np.concatenate([
            freq_feats, color_feats, tex_feats, edge_feats, noise_feats,
            channel_feats, wavelet_feats,
        ])

        # Replace any NaN / Inf with 0 for robustness
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features
