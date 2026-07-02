"""Wavelet-domain feature extraction using Haar DWT.

The Discrete Wavelet Transform provides spatially-localised frequency
analysis superior to FFT for detecting screen recapture artefacts.
Screen pixel grids and moiré patterns deposit distinctive energy in
the detail sub-bands, particularly in the diagonal (HH) coefficients.

Implemented from scratch with Haar wavelets to avoid adding pywt
as a dependency.

Features extracted (10):
    Level-1 detail sub-band energies  LH, HL, HH  (3),
    Level-2 detail sub-band energies  LH, HL, HH  (3),
    Detail-to-approximation energy ratio at each level (2),
    Cross-level HH ratio  (1),
    Detail energy spread / std  (1).
"""

import cv2
import numpy as np

FEATURE_NAMES: list[str] = [
    "dwt1_lh_energy",
    "dwt1_hl_energy",
    "dwt1_hh_energy",
    "dwt2_lh_energy",
    "dwt2_hl_energy",
    "dwt2_hh_energy",
    "dwt1_detail_ratio",
    "dwt2_detail_ratio",
    "dwt_hh_cross_level",
    "dwt_energy_spread",
]


def extract(gray: np.ndarray) -> np.ndarray:
    """Return a 10-element wavelet feature vector.

    Args:
        gray: Single-channel uint8 image (already resized to standard size).
    """
    # Ensure even dimensions for the Haar split
    h, w = gray.shape
    img = gray[: h - h % 2, : w - w % 2].astype(np.float32)

    # Level 1
    ll1, lh1, hl1, hh1 = _haar_dwt2d(img)

    # Level 2 (on the approximation from level 1)
    h2, w2 = ll1.shape
    ll1_even = ll1[: h2 - h2 % 2, : w2 - w2 % 2]
    ll2, lh2, hl2, hh2 = _haar_dwt2d(ll1_even)

    # Sub-band energies (mean of squared coefficients)
    ll1_e = float(np.mean(ll1 ** 2)) + 1e-10
    lh1_e = float(np.mean(lh1 ** 2))
    hl1_e = float(np.mean(hl1 ** 2))
    hh1_e = float(np.mean(hh1 ** 2))

    ll2_e = float(np.mean(ll2 ** 2)) + 1e-10
    lh2_e = float(np.mean(lh2 ** 2))
    hl2_e = float(np.mean(hl2 ** 2))
    hh2_e = float(np.mean(hh2 ** 2))

    detail1 = lh1_e + hl1_e + hh1_e
    detail2 = lh2_e + hl2_e + hh2_e

    return np.array(
        [
            lh1_e, hl1_e, hh1_e,
            lh2_e, hl2_e, hh2_e,
            detail1 / ll1_e,
            detail2 / ll2_e,
            (hh1_e + 1e-10) / (hh2_e + 1e-10),
            float(np.std([lh1_e, hl1_e, hh1_e, lh2_e, hl2_e, hh2_e])),
        ],
        dtype=np.float64,
    )


def _haar_dwt2d(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-level 2-D Haar wavelet decomposition.

    Returns (LL, LH, HL, HH) sub-bands, each half the original size.
    """
    # Column-wise low / high pass
    L_cols = (img[:, 0::2] + img[:, 1::2]) * 0.5
    H_cols = (img[:, 0::2] - img[:, 1::2]) * 0.5

    # Row-wise low / high pass on L columns → LL, LH
    LL = (L_cols[0::2, :] + L_cols[1::2, :]) * 0.5
    LH = (L_cols[0::2, :] - L_cols[1::2, :]) * 0.5

    # Row-wise low / high pass on H columns → HL, HH
    HL = (H_cols[0::2, :] + H_cols[1::2, :]) * 0.5
    HH = (H_cols[0::2, :] - H_cols[1::2, :]) * 0.5

    return LL, LH, HL, HH
