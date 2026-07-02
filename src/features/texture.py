"""Texture feature extraction: LBP, GLCM, and Laplacian variance.

Screen recaptures introduce micro-texture artefacts from the display's
pixel grid and anti-aliasing that are absent in direct photographs.

Features extracted (18):
    LBP histogram (10 bins),
    Laplacian variance and mean (2),
    GLCM contrast, correlation, energy, homogeneity at two scales (4 × 1 = 4),
    LBP uniformity ratio, LBP entropy (2).
"""

import cv2
import numpy as np

_LBP_BINS: int = 10
_GLCM_LEVELS: int = 16

FEATURE_NAMES: list[str] = [
    *[f"lbp_bin_{i}" for i in range(_LBP_BINS)],
    "laplacian_var", "laplacian_mean",
    "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity",
    "lbp_uniformity", "lbp_entropy",
]


def extract(gray: np.ndarray) -> np.ndarray:
    """Return an 18-element texture feature vector.

    Args:
        gray: Single-channel uint8 image.
    """
    lbp_hist = _compute_lbp_histogram(gray)
    lap_var, lap_mean = _laplacian_stats(gray)
    glcm_feats = _compute_glcm_features(gray)
    lbp_uni = _lbp_uniformity(gray)
    lbp_ent = _histogram_entropy(lbp_hist)

    return np.concatenate([
        lbp_hist,
        [lap_var, lap_mean],
        glcm_feats,
        [lbp_uni, lbp_ent],
    ]).astype(np.float64)


# ---------- LBP ---------- #

def _compute_lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """Compute a normalised Local Binary Pattern histogram.

    Uses a 3×3 neighbourhood (radius-1, 8 neighbours).  Outputs a
    fixed-length histogram with *_LBP_BINS* bins over the 0–255 code range.
    """
    padded = np.pad(gray, 1, mode="edge").astype(np.int16)
    center = padded[1:-1, 1:-1]

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1),  (1, 0),  (1, -1), (0, -1)]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[1 + dy: 1 + dy + center.shape[0],
                           1 + dx: 1 + dx + center.shape[1]]
        lbp |= ((neighbour >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp, bins=_LBP_BINS, range=(0, 256))
    total = hist.sum() + 1e-10
    return hist.astype(np.float64) / total


def _lbp_uniformity(gray: np.ndarray) -> float:
    """Fraction of *uniform* LBP codes (≤ 2 bit-transitions)."""
    padded = np.pad(gray, 1, mode="edge").astype(np.int16)
    center = padded[1:-1, 1:-1]

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1),  (1, 0),  (1, -1), (0, -1)]

    bits = []
    for dy, dx in offsets:
        neighbour = padded[1 + dy: 1 + dy + center.shape[0],
                           1 + dx: 1 + dx + center.shape[1]]
        bits.append((neighbour >= center).astype(np.uint8))

    # Count 0→1 and 1→0 transitions in the circular bit-string
    transitions = np.zeros_like(center, dtype=np.int32)
    for i in range(8):
        transitions += np.abs(bits[i].astype(np.int32) - bits[(i + 1) % 8].astype(np.int32))

    uniform_mask = transitions <= 2
    return float(np.mean(uniform_mask))


def _histogram_entropy(hist: np.ndarray) -> float:
    """Shannon entropy of a normalised histogram."""
    h = hist[hist > 0]
    return float(-np.sum(h * np.log2(h + 1e-10)))


# ---------- Laplacian ---------- #

def _laplacian_stats(gray: np.ndarray) -> tuple[float, float]:
    """Variance and mean of the Laplacian — a sharpness proxy."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap)), float(np.mean(np.abs(lap)))


# ---------- GLCM ---------- #

def _compute_glcm_features(gray: np.ndarray) -> np.ndarray:
    """Grey-Level Co-occurrence Matrix features at distance=1.

    Quantises intensity to *_GLCM_LEVELS* to keep the matrix small and
    operates on a down-sampled 128×128 copy for speed.
    """
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    quantised = np.clip(
        (small.astype(np.float64) / 256.0 * _GLCM_LEVELS).astype(np.int32),
        0, _GLCM_LEVELS - 1,
    )

    glcm = np.zeros((_GLCM_LEVELS, _GLCM_LEVELS), dtype=np.float64)

    # Accumulate co-occurrences for horizontal and vertical adjacency
    for dy, dx in [(0, 1), (1, 0)]:
        src = quantised[:quantised.shape[0] - abs(dy), :quantised.shape[1] - abs(dx)]
        dst = quantised[abs(dy):, abs(dx):]
        np.add.at(glcm, (src.ravel(), dst.ravel()), 1)
        np.add.at(glcm, (dst.ravel(), src.ravel()), 1)  # symmetric

    glcm /= glcm.sum() + 1e-10

    I, J = np.ogrid[:_GLCM_LEVELS, :_GLCM_LEVELS]
    I_f = I.astype(np.float64)
    J_f = J.astype(np.float64)

    contrast = float(np.sum(glcm * (I_f - J_f) ** 2))
    energy = float(np.sum(glcm ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(I_f - J_f))))

    mu_i = float(np.sum(I_f * glcm.sum(axis=1, keepdims=True)))
    mu_j = float(np.sum(J_f * glcm.sum(axis=0, keepdims=True)))
    sigma_i = float(np.sqrt(np.sum((I_f - mu_i) ** 2 * glcm.sum(axis=1, keepdims=True))))
    sigma_j = float(np.sqrt(np.sum((J_f - mu_j) ** 2 * glcm.sum(axis=0, keepdims=True))))

    if sigma_i * sigma_j > 1e-10:
        correlation = float(np.sum(glcm * (I_f - mu_i) * (J_f - mu_j)) / (sigma_i * sigma_j))
    else:
        correlation = 0.0

    return np.array([contrast, correlation, energy, homogeneity], dtype=np.float64)
