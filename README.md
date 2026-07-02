# Spot the Fake Photo — Screen Recapture Detector

A fast, lightweight, explainable detector that tells real photographs apart
from **photos of a screen** (recaptures).  Built with classical computer-vision
features and a small Random Forest classifier — no GPU required, runs in
~300–800 ms per image on a laptop CPU.

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
│   ├── model.py            ← ScreenDetector (trained + deterministic paths)
│   └── features/
│       ├── extractor.py    ← unified feature pipeline (61 features)
│       ├── frequency.py    ← FFT / moiré / spectral analysis   (14 features)
│       ├── color.py        ← RGB + HSV statistics & correlations (20 features)
│       ├── texture.py      ← LBP, GLCM, Laplacian variance      (18 features)
│       ├── edge.py         ← Canny density, Sobel gradients       (5 features)
│       └── noise.py        ← noise σ, uniformity, SNR             (4 features)
│
├── models/                 ← saved .joblib model (after training)
└── dataset/
    ├── real/               ← put real photos here
    └── screen/             ← put screen recaptures here
```

---

## How It Works

### Feature Engineering (the core idea)

Instead of training a heavy CNN, we extract **61 hand-crafted features** that
capture the physics of screen recapture:

| Feature Group | # | What it measures |
|---|---|---|
| **Frequency (FFT)** | 14 | Moiré pattern peaks, spectral energy distribution, spectral entropy / flatness |
| **Colour** | 20 | RGB & HSV channel statistics, dynamic range, cross-channel correlation, colour-temperature proxy |
| **Texture** | 18 | Local Binary Patterns, GLCM (contrast, correlation, energy, homogeneity), Laplacian variance |
| **Edge** | 5 | Canny edge density, gradient magnitude stats, gradient direction entropy |
| **Noise** | 4 | Noise σ, spatial uniformity, SNR, noise spectral ratio |

### Why these features work

When you photograph a screen, the image acquires artefacts that **no
real-world photo has**:

1. **Moiré patterns** — aliasing between the display's pixel grid and the
   camera sensor creates periodic interference visible in the FFT.
2. **Compounded noise** — the image carries both display rendering noise and
   sensor capture noise, increasing overall σ and changing the spectral
   profile.
3. **Gamut compression** — screens reproduce a limited colour gamut, reducing
   dynamic range and altering cross-channel correlation.
4. **Pixel micro-texture** — the display's sub-pixel structure (RGB stripes or
   PenTile) injects a regular high-frequency pattern captured by LBP and
   GLCM.

### Classifier

A **Random Forest** (200 trees, max depth 8, balanced class weights) trained on
the extracted features.  If no trained model is present, `predict.py` falls
back to a deterministic heuristic that uses hand-tuned weights on the strongest
features.

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
# Output includes: Latency: 42.3 ms
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
- numpy ≥ 1.24
- opencv-python-headless ≥ 4.8
- scikit-learn ≥ 1.3
- scipy ≥ 1.11

Install: `pip install -r requirements.txt`

---
