"""Colour-space feature extraction.

Screen recaptures differ from real photos in several colour dimensions:
    - Limited display colour gamut compresses the chromaticity range.
    - Back-light colour temperature shifts the white-point.
    - Gamma mapping between display and camera alters tonal distribution.
    - Pixel sub-structure decorrelates high-frequency inter-channel signals.

Features extracted (20):
    RGB per-channel mean / std / skewness (9),
    HSV per-channel mean / std (6),
    dynamic range, colour-temperature ratio (2),
    cross-channel Pearson correlations R-G, R-B, G-B (3).
"""

import numpy as np

FEATURE_NAMES: list[str] = [
    "r_mean", "g_mean", "b_mean",
    "r_std", "g_std", "b_std",
    "r_skew", "g_skew", "b_skew",
    "h_mean", "s_mean", "v_mean",
    "h_std", "s_std", "v_std",
    "dynamic_range", "color_temp_ratio",
    "rg_corr", "rb_corr", "gb_corr",
]


def extract(bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """Return a 20-element colour feature vector.

    Args:
        bgr: 3-channel BGR uint8 image.
        hsv: Corresponding HSV uint8 image.
    """
    # Work in float32 for speed — double precision is unnecessary for stats
    bgr_f = bgr.astype(np.float32) / 255.0
    b, g, r = bgr_f[:, :, 0], bgr_f[:, :, 1], bgr_f[:, :, 2]

    hsv_f = hsv.astype(np.float32)
    h_ch = hsv_f[:, :, 0] / 180.0
    s_ch = hsv_f[:, :, 1] / 255.0
    v_ch = hsv_f[:, :, 2] / 255.0

    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()

    # Colour-temperature proxy: warm / cool ratio
    warm = r_mean + g_mean * 0.5
    cool = b_mean + g_mean * 0.5 + 1e-7
    color_temp_ratio = warm / cool

    features = np.array([
        r_mean, g_mean, b_mean,
        r.std(), g.std(), b.std(),
        _skewness(r), _skewness(g), _skewness(b),
        h_ch.mean(), s_ch.mean(), v_ch.mean(),
        h_ch.std(), s_ch.std(), v_ch.std(),
        float(v_ch.max() - v_ch.min()),
        color_temp_ratio,
        _fast_corr(r, g), _fast_corr(r, b), _fast_corr(g, b),
    ], dtype=np.float64)

    return features


def _skewness(arr: np.ndarray) -> float:
    """Compute skewness without scipy — 5× faster on large arrays."""
    m = arr.mean()
    s = arr.std()
    if s < 1e-10:
        return 0.0
    return float(((arr - m) ** 3).mean() / (s ** 3))


def _fast_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation via direct formula — avoids allocating a 2×N matrix."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    a_m = a_flat - a_flat.mean()
    b_m = b_flat - b_flat.mean()
    num = (a_m * b_m).sum()
    denom = np.sqrt((a_m ** 2).sum() * (b_m ** 2).sum()) + 1e-10
    r = float(num / denom)
    return r if np.isfinite(r) else 0.0
