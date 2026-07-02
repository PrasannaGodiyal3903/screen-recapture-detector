"""Image loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

STANDARD_SIZE: Tuple[int, int] = (512, 512)


def load_image(path: str | Path) -> np.ndarray:
    """Load image from disk as BGR numpy array.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be decoded by OpenCV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(
    image: np.ndarray, size: Tuple[int, int] = STANDARD_SIZE
) -> np.ndarray:
    """Resize image to a fixed size for consistent feature extraction."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def prepare_for_analysis(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize image and produce BGR, grayscale, and HSV representations.

    Returns:
        Tuple of (resized_bgr, resized_gray, resized_hsv).
    """
    resized = resize_image(image)
    gray = to_grayscale(resized)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    return resized, gray, hsv
