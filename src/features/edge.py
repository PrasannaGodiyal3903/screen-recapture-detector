"""Edge and gradient feature extraction.

Screen recaptures alter edge characteristics because:
    - The display's anti-aliasing softens sharp transitions.
    - Pixel-grid overlay adds spurious high-frequency edges.
    - Re-photographing introduces slight defocus/motion blur.

Features extracted (5):
    Canny edge-pixel density,
    Sobel gradient magnitude mean / std,
    Gradient direction entropy,
    High-frequency edge ratio.
"""

import cv2
import numpy as np

FEATURE_NAMES: list[str] = [
    "edge_density",
    "grad_mag_mean",
    "grad_mag_std",
    "grad_dir_entropy",
    "hf_edge_ratio",
]


def extract(gray: np.ndarray) -> np.ndarray:
    """Return a 5-element edge/gradient feature vector.

    Args:
        gray: Single-channel uint8 image.
    """
    # ---- Canny edge density ----
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # ---- Sobel gradients ----
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag_mean = float(np.mean(mag))
    grad_mag_std = float(np.std(mag))

    # ---- gradient direction entropy ----
    direction = np.arctan2(gy, gx)  # range [-π, π]
    # Quantise into 36 bins (10° each)
    n_bins = 36
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(direction.ravel(), bins=bins)
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)
    grad_dir_entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10)))

    # ---- high-frequency edge ratio ----
    # Edges detected with tighter thresholds capture fine (possibly moiré) edges
    fine_edges = cv2.Canny(gray, 100, 200)
    fine_density = float(np.mean(fine_edges > 0))
    hf_edge_ratio = fine_density / (edge_density + 1e-10)

    return np.array(
        [edge_density, grad_mag_mean, grad_mag_std, grad_dir_entropy, hf_edge_ratio],
        dtype=np.float64,
    )
