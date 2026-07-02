"""Noise-level feature extraction.

Screen recaptures compound two independent noise sources — the
display's rendering noise and the camera's sensor noise — which
produces a measurably different noise profile from a single real
capture.

Features extracted (4):
    Estimated noise σ (median-filter residual method),
    Noise spatial uniformity (std of local noise patches),
    Signal-to-noise ratio estimate,
    Noise spectral ratio (high-freq noise / total noise).
"""

import cv2
import numpy as np

FEATURE_NAMES: list[str] = [
    "noise_sigma",
    "noise_uniformity",
    "snr_estimate",
    "noise_spectral_ratio",
]


def extract(gray: np.ndarray) -> np.ndarray:
    """Return a 4-element noise feature vector.

    Args:
        gray: Single-channel uint8 image.
    """
    img = gray.astype(np.float32)

    # ---- global noise estimate via median-filter residual ----
    denoised = cv2.medianBlur(gray, 5).astype(np.float32)
    residual = img - denoised
    noise_sigma = float(np.std(residual))

    # ---- noise spatial uniformity ----
    ph, pw = gray.shape[0] // 4, gray.shape[1] // 4
    patches = residual[:4 * ph, :4 * pw].reshape(4, ph, 4, pw).transpose(0, 2, 1, 3)
    patch_stds = patches.reshape(16, -1).std(axis=1)
    noise_uniformity = float(patch_stds.std())

    # ---- SNR estimate ----
    signal_power = float(np.var(denoised))
    noise_power = noise_sigma ** 2 + 1e-10
    snr_estimate = float(10.0 * np.log10(signal_power / noise_power + 1e-10))

    # ---- noise spectral ratio (using rfft2 for speed) ----
    noise_fft = np.fft.rfft2(residual)
    noise_mag_sq = np.real(noise_fft * np.conj(noise_fft))
    h, fw = noise_mag_sq.shape
    # In rfft2 output, higher column indices = higher x-frequencies
    # and rows near h/2 = higher y-frequencies
    freq_y = np.fft.fftfreq(h).astype(np.float32)
    freq_x = np.arange(fw, dtype=np.float32) / (2 * fw)
    FX, FY = np.meshgrid(freq_x, freq_y)
    radius_norm = np.sqrt(FX ** 2 + FY ** 2)
    high_mask = radius_norm > 0.25  # top 50% of frequency range
    total_noise_energy = float(noise_mag_sq.sum()) + 1e-10
    high_noise_energy = float(noise_mag_sq[high_mask].sum())
    noise_spectral_ratio = high_noise_energy / total_noise_energy

    return np.array(
        [noise_sigma, noise_uniformity, snr_estimate, noise_spectral_ratio],
        dtype=np.float64,
    )
