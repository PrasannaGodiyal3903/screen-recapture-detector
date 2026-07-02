"""Frequency-domain feature extraction via FFT analysis.

Screen recaptures exhibit moiré patterns caused by aliasing between the
display's pixel grid and the camera sensor grid.  These periodic
interference patterns produce distinctive peaks in the 2-D Fourier
spectrum that are absent in real-world photographs.

Features extracted (18):
    - Normalised energy in 8 concentric frequency bands  (8)
    - Spectral entropy and flatness  (2)
    - High-frequency and mid-frequency energy ratios  (2)
    - Moiré peak count and mean prominence  (2)
    - Spectral centroid, rolloff, kurtosis, band contrast  (4)
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

N_BANDS: int = 8
N_RADIAL_BINS: int = 64

FEATURE_NAMES: list[str] = [
    *[f"fft_band_{i}" for i in range(N_BANDS)],
    "spectral_entropy",
    "spectral_flatness",
    "high_freq_ratio",
    "mid_freq_ratio",
    "peak_count",
    "peak_mean_prominence",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_kurtosis",
    "band_contrast",
]


def extract(gray: np.ndarray) -> np.ndarray:
    """Return an 18-element feature vector from the frequency domain.

    Args:
        gray: Single-channel uint8 image (already resized to standard size).
    """
    h, w = gray.shape
    img = gray.astype(np.float32)

    # Hanning window suppresses spectral leakage at image boundaries
    window = np.outer(
        np.hanning(h).astype(np.float32),
        np.hanning(w).astype(np.float32),
    )
    windowed = img * window

    fft = np.fft.rfft2(windowed)
    magnitude = np.abs(fft)
    log_mag = np.log1p(magnitude)

    # Build radius map for the rfft2 output (h × w//2+1)
    fh, fw = log_mag.shape
    freq_y = np.fft.fftshift(np.fft.fftfreq(h)) * h
    freq_x = np.arange(fw).astype(np.float32)
    FX, FY = np.meshgrid(freq_x, freq_y)
    radius = np.sqrt(FX ** 2 + FY ** 2)
    max_radius = float(min(h, w) // 2)

    # ---- band energies (vectorised via digitize) ----
    band_edges = np.linspace(0, max_radius, N_BANDS + 1)
    bin_idx = np.clip(np.digitize(radius.ravel(), band_edges) - 1, 0, N_BANDS - 1)
    flat_mag = log_mag.ravel()

    band_energies = np.zeros(N_BANDS, dtype=np.float64)
    band_counts = np.zeros(N_BANDS, dtype=np.float64)
    np.add.at(band_energies, bin_idx, flat_mag)
    np.add.at(band_counts, bin_idx, 1)
    band_counts[band_counts == 0] = 1
    band_energies /= band_counts

    total = band_energies.sum() + 1e-10
    norm_energies = band_energies / total

    # ---- spectral entropy ----
    probs = norm_energies / (norm_energies.sum() + 1e-10)
    spectral_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # ---- spectral flatness ----
    geo_mean = float(np.exp(np.mean(np.log(band_energies + 1e-10))))
    arith_mean = float(np.mean(band_energies)) + 1e-10
    spectral_flatness = geo_mean / arith_mean

    # ---- frequency-band ratios ----
    low_energy = band_energies[:3].sum() + 1e-10
    high_energy = band_energies[5:].sum()
    high_freq_ratio = float(high_energy / low_energy)
    mid_freq_ratio = float(band_energies[3:5].sum() / total)

    # ---- moiré peak detection ----
    radial_profile = _build_radial_profile(log_mag, radius, max_radius)
    peak_count, peak_prominence = _detect_spectral_peaks(radial_profile)

    # ---- NEW: spectral centroid ----
    band_centres = (band_edges[:-1] + band_edges[1:]) / 2.0
    spectral_centroid = float(np.sum(band_centres * band_energies) / total)

    # ---- NEW: spectral rolloff (frequency below which 85% of energy) ----
    cumulative = np.cumsum(band_energies)
    threshold_85 = 0.85 * cumulative[-1]
    rolloff_idx = int(np.searchsorted(cumulative, threshold_85))
    spectral_rolloff = float(rolloff_idx) / N_BANDS

    # ---- NEW: spectral kurtosis (peakedness of the radial profile) ----
    rp = radial_profile
    rp_mean = rp.mean()
    rp_std = rp.std() + 1e-10
    spectral_kurtosis = float(np.mean(((rp - rp_mean) / rp_std) ** 4)) - 3.0

    # ---- NEW: band contrast (mean absolute diff of adjacent bands) ----
    band_contrast = float(np.mean(np.abs(np.diff(norm_energies))))

    return np.array(
        [*norm_energies, spectral_entropy, spectral_flatness,
         high_freq_ratio, mid_freq_ratio, peak_count, peak_prominence,
         spectral_centroid, spectral_rolloff, spectral_kurtosis, band_contrast],
        dtype=np.float64,
    )


def _build_radial_profile(
    log_mag: np.ndarray, radius: np.ndarray, max_radius: float,
) -> np.ndarray:
    """Build a mean-energy radial profile using vectorised binning."""
    bin_edges = np.linspace(0, max_radius, N_RADIAL_BINS + 1)
    flat_r = radius.ravel()
    flat_m = log_mag.ravel()
    idx = np.clip(np.digitize(flat_r, bin_edges) - 1, 0, N_RADIAL_BINS - 1)

    radial_sum = np.zeros(N_RADIAL_BINS, dtype=np.float64)
    radial_cnt = np.zeros(N_RADIAL_BINS, dtype=np.float64)
    np.add.at(radial_sum, idx, flat_m)
    np.add.at(radial_cnt, idx, 1)
    radial_cnt[radial_cnt == 0] = 1
    return radial_sum / radial_cnt


def _detect_spectral_peaks(
    radial_profile: np.ndarray,
) -> Tuple[float, float]:
    """Count prominent peaks above a smoothed baseline."""
    baseline = uniform_filter1d(radial_profile, size=11)
    residual = radial_profile - baseline
    threshold = np.std(residual) * 2.0

    peaks = residual > threshold
    transitions = np.diff(peaks.astype(np.int8))
    n_peaks = float(np.sum(transitions == 1))
    mean_prominence = float(np.mean(residual[peaks])) if np.any(peaks) else 0.0

    return n_peaks, mean_prominence
