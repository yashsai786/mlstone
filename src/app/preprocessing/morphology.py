import cv2
import numpy as np
from typing import Tuple


class BilateralFilterService:
    """
    Service for applying bilateral filtering to preserve edges while suppressing texture noise.
    """

    def filter(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0
    ) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


class MorphologyGradientService:
    """
    Service for performing advanced morphology operations (morphological gradient, closing, and opening).
    """

    def compute_gradient(
        self,
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5)
    ) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    def apply_close_open(
        self,
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (41, 41)
    ) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return opened
