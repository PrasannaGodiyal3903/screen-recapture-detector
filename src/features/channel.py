"""Inter-channel high-frequency correlation analysis.

THE single strongest signal for screen recapture detection.

When a camera photographs a screen, the display's RGB sub-pixels are
spatially separated (RGB stripe, PenTile, etc.).  This means the
high-frequency content in each colour channel is DIFFERENT — unlike
a real photograph where all channels share the same edges and textures.

The decorrelation between high-passed colour channels is measurable
even on high-DPI "retina" displays where moiré is invisible to the eye.

Features extracted (12):
    High-pass inter-channel correlations at two scales (6),
    Mean absolute inter-channel HF differences (3),
    HF channel energy ratio (1),
    Horizontal/vertical gradient periodicity peaks (2).
"""

import cv2
import numpy as np

FEATURE_NAMES: list[str] = [
    "hf_rg_corr_fine",
    "hf_rb_corr_fine",
    "hf_gb_corr_fine",
    "hf_rg_corr_coarse",
    "hf_rb_corr_coarse",
    "hf_gb_corr_coarse",
    "hf_rg_diff",
    "hf_rb_diff",
    "hf_gb_diff",
    "hf_energy_ratio",
    "grad_periodicity_h",
    "grad_periodicity_v",
]

# Two scales: fine captures sub-pixel structure, coarse captures moiré
_FINE_KERNEL: int = 3
_COARSE_KERNEL: int = 9


def extract(bgr_original: np.ndarray) -> np.ndarray:
    """Extract inter-channel HF features from original-resolution image.

    Works on a centre crop of the original to preserve fine detail that
    down-sampling would destroy.

    Args:
        bgr_original: BGR image at original camera resolution.
    """
    crop = _centre_crop(bgr_original, 512)
    b, g, r = (crop[:, :, i].astype(np.float32) for i in range(3))

    # Fine-scale high-pass (captures sub-pixel structure)
    r_hp_f = _highpass(r, _FINE_KERNEL)
    g_hp_f = _highpass(g, _FINE_KERNEL)
    b_hp_f = _highpass(b, _FINE_KERNEL)

    rg_fine = _fast_corr(r_hp_f, g_hp_f)
    rb_fine = _fast_corr(r_hp_f, b_hp_f)
    gb_fine = _fast_corr(g_hp_f, b_hp_f)

    # Coarse-scale high-pass (captures moiré interference)
    r_hp_c = _highpass(r, _COARSE_KERNEL)
    g_hp_c = _highpass(g, _COARSE_KERNEL)
    b_hp_c = _highpass(b, _COARSE_KERNEL)

    rg_coarse = _fast_corr(r_hp_c, g_hp_c)
    rb_coarse = _fast_corr(r_hp_c, b_hp_c)
    gb_coarse = _fast_corr(g_hp_c, b_hp_c)

    # Mean absolute inter-channel HF differences
    rg_diff = float(np.mean(np.abs(r_hp_f - g_hp_f)))
    rb_diff = float(np.mean(np.abs(r_hp_f - b_hp_f)))
    gb_diff = float(np.mean(np.abs(g_hp_f - b_hp_f)))

    # HF energy ratio: max / min across channels
    energies = [
        float(np.mean(r_hp_f ** 2)) + 1e-10,
        float(np.mean(g_hp_f ** 2)) + 1e-10,
        float(np.mean(b_hp_f ** 2)) + 1e-10,
    ]
    energy_ratio = max(energies) / min(energies)

    # Gradient periodicity — periodic patterns in gradient autocorrelation
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    period_h = _gradient_periodicity(gray_crop, axis=1)
    period_v = _gradient_periodicity(gray_crop, axis=0)

    return np.array([
        rg_fine, rb_fine, gb_fine,
        rg_coarse, rb_coarse, gb_coarse,
        rg_diff, rb_diff, gb_diff,
        energy_ratio,
        period_h, period_v,
    ], dtype=np.float64)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _centre_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Extract a square centre crop at original resolution."""
    h, w = image.shape[:2]
    crop_sz = min(h, w, size)
    cy, cx = h // 2, w // 2
    half = crop_sz // 2
    return image[cy - half: cy + half, cx - half: cx + half]


def _highpass(channel: np.ndarray, ksize: int) -> np.ndarray:
    """Subtract Gaussian-blurred version to isolate high frequencies."""
    blurred = cv2.GaussianBlur(channel, (ksize, ksize), 0)
    return channel - blurred


def _fast_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two arrays."""
    a_f = a.ravel()
    b_f = b.ravel()
    a_m = a_f - a_f.mean()
    b_m = b_f - b_f.mean()
    num = float(np.dot(a_m, b_m))
    denom = float(np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))) + 1e-10
    r = num / denom
    return r if np.isfinite(r) else 0.0


def _gradient_periodicity(gray: np.ndarray, axis: int) -> float:
    """Detect periodic patterns via autocorrelation of the gradient profile.

    Screen pixel grids produce regular gradients.  The autocorrelation
    of the averaged gradient profile reveals secondary peaks whose
    magnitude indicates periodicity strength.

    Args:
        gray: Grayscale uint8 crop.
        axis: 0 for vertical periodicity, 1 for horizontal.
    """
    if axis == 1:
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    else:
        grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Average across the orthogonal axis to get a 1-D profile
    profile = np.mean(grad, axis=0 if axis == 1 else 1).astype(np.float64)
    profile -= profile.mean()

    n = len(profile)
    if n < 20:
        return 0.0

    # Normalised autocorrelation via FFT (fast)
    fft_p = np.fft.rfft(profile, n=2 * n)
    acf = np.fft.irfft(fft_p * np.conj(fft_p))[:n]
    acf_norm = acf / (acf[0] + 1e-10)

    # Find the maximum secondary peak (skip lag < 3 to avoid self-correlation)
    search = acf_norm[3: min(n // 2, 80)]
    if len(search) == 0:
        return 0.0

    return float(np.max(search))
