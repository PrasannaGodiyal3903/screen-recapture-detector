# Spot the Fake Photo — Screen Recapture Detector

A fast, lightweight, explainable detector that tells real photographs apart
from **photos of a screen** (recaptures).  Built with classical computer-vision
features and a GBM + Random Forest ensemble — no GPU required, runs in
~340 ms per image on a laptop CPU.

**4-fold CV accuracy: 94.53% | ROC-AUC: 0.9858 | 201 images**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train on your own data
python train.py --data dataset/

# 3. Predict
python predict.py image.jpg            # prints score: 0.00–1.00
python predict.py image.jpg --verbose   # adds label, latency, mode
```

**Score interpretation:** `0 = real photo`, `1 = screen recapture`.

---

## Project Structure

```
screen-detect/
├── predict.py              ← one-line CLI predictor
├── train.py                ← train from dataset/real + dataset/screen
├── evaluate.py             ← precision, recall, F1, ROC-AUC, confusion matrix
├── requirements.txt
├── report.md               ← methodology, analysis, and discussion
│
├── src/
│   ├── preprocessing.py    ← image loading, resizing, colour-space conversion
│   ├── model.py            ← ScreenDetector (GBM+RF ensemble + deterministic)
│   └── features/
│       ├── extractor.py    ← unified feature pipeline (87 features)
│       ├── frequency.py    ← FFT / moiré / spectral analysis   (18 features)
│       ├── color.py        ← RGB + HSV statistics & correlations (20 features)
│       ├── texture.py      ← LBP, GLCM, Laplacian variance      (18 features)
│       ├── edge.py         ← Canny density, Sobel gradients       (5 features)
│       ├── noise.py        ← noise σ, uniformity, SNR             (4 features)
│       ├── channel.py      ← inter-channel HF correlation        (12 features)
│       └── wavelet.py      ← Haar DWT energy features             (10 features)
│
├── models/                 ← saved .joblib model (after training)
└── dataset/
    ├── real/               ← put real photos here
    └── screen/             ← put screen recaptures here
```

---

## How It Works

### Feature Engineering (the core idea)

Instead of training a heavy CNN, we extract **87 hand-crafted features** that
capture the physics of screen recapture:

| Feature Group | # | What it measures |
|---|---|---|
| **Frequency (FFT)** | 18 | Moiré pattern peaks, spectral energy distribution, entropy, flatness, centroid, rolloff, kurtosis |
| **Colour** | 20 | RGB & HSV channel statistics, dynamic range, cross-channel correlation, colour-temperature proxy |
| **Texture** | 18 | Local Binary Patterns, GLCM (contrast, correlation, energy, homogeneity), Laplacian variance |
| **Edge** | 5 | Canny edge density, gradient magnitude stats, gradient direction entropy |
| **Noise** | 4 | Noise σ, spatial uniformity, SNR, noise spectral ratio |
| **Channel** | 12 | Inter-channel high-frequency correlation at fine/coarse scales, HF differences, gradient periodicity |
| **Wavelet** | 10 | Haar DWT sub-band energies (LH, HL, HH at 2 levels), detail-to-approximation ratios |

### Why these features work

When you photograph a screen, the image acquires artefacts that **no
real-world photo has**:

1. **Sub-pixel decorrelation** *(strongest signal)* — screen RGB sub-pixels are
   spatially separated, causing the high-frequency content in each colour
   channel to be **different**.  In real photos, R, G, B channels share the
   same edges and show ~0.9 HF correlation; screen recaptures drop to ~0.4–0.7.
2. **Moiré patterns** — aliasing between the display's pixel grid and the
   camera sensor creates periodic interference visible in the FFT.
3. **Wavelet energy anomalies** — the screen pixel grid deposits distinctive
   energy in the diagonal (HH) detail sub-bands of the wavelet transform.
4. **Compounded noise** — the image carries both display rendering noise and
   sensor capture noise, increasing overall σ and changing the spectral
   profile.
5. **Gamut compression** — screens reproduce a limited colour gamut, reducing
   dynamic range and altering cross-channel correlation.
6. **Pixel micro-texture** — the display's sub-pixel structure (RGB stripes or
   PenTile) injects a regular high-frequency pattern captured by LBP and
   GLCM.

### Classifier

A **soft-voting ensemble** of Gradient Boosting (300 trees, lr=0.05, subsample=0.8)
and Random Forest (300 trees, balanced class weights).  GBM captures complex
feature interactions while RF provides stability.  If no trained model is
present, `predict.py` falls back to a deterministic heuristic that uses
hand-tuned weights on the strongest features.

---

## Results

Evaluated on 201 images (99 real + 102 screen) with 4-fold stratified
cross-validation:

| Metric | Value |
|---|---|
| **Accuracy** | **94.53%** |
| **ROC-AUC** | **0.9858** |
| Precision (real) | 97% |
| Recall (screen) | 97% |
| F1-score | 0.95 |
| Latency (median) | 339 ms |

---

## Dataset Preparation

1. Take **~50+ photos of real objects** with your phone.
2. Take **~50+ photos of a screen/printout showing pictures**.  Vary:
   - Screen types (phone, laptop, monitor, tablet)
   - Angles, distances, lighting conditions
   - Content variety (faces, text, objects, landscapes)
3. Place them in `dataset/real/` and `dataset/screen/`.

> The more variety, the more robust the model.

---

## Training

```bash
python train.py --data dataset/

# Customise the classifier
python train.py --data dataset/ --estimators 300 --depth 10

# Save to a custom path
python train.py --data dataset/ --output models/v2.joblib
```

The script prints:
- Dataset size breakdown
- OOB (out-of-bag) accuracy
- Per-image sanity check on training data

---

## Evaluation

```bash
# Stratified k-fold cross-validation (default: 4-fold from 25% holdout)
python evaluate.py --data dataset/

# Use a specific model on the full dataset
python evaluate.py --data dataset/ --model models/screen_detector.joblib --full
```

Reports:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix
- Per-image predictions
- Latency statistics (mean, median, p95)

---

## Measuring Latency

```bash
python predict.py image.jpg --verbose
# Output includes: Latency: 339.2 ms
```

For benchmarking over multiple images:

```python
import time
from src.model import ScreenDetector
from src.preprocessing import load_image

detector = ScreenDetector()
image = load_image("test.jpg")

times = []
for _ in range(100):
    t0 = time.perf_counter()
    detector.predict(image)
    times.append((time.perf_counter() - t0) * 1000)

import numpy as np
print(f"Mean: {np.mean(times):.1f} ms")
print(f"Median: {np.median(times):.1f} ms")
print(f"P95: {np.percentile(times, 95):.1f} ms")
```

---

## Requirements

- Python 3.11+
- numpy >= 1.24
- opencv-python-headless >= 4.8
- scikit-learn >= 1.3
- scipy >= 1.11

Install: `pip install -r requirements.txt`

---
