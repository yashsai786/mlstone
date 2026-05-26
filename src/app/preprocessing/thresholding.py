import cv2
import numpy as np


class ThresholdingService:
    """
    Service for applying OTSU binarization thresholding on grayscale gradient maps.
    """

    def apply_otsu(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
