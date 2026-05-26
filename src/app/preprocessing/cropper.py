import cv2
import numpy as np
from typing import Tuple


class SafeCropper:
    """
    Service for applying safe rectangular slab cropping based on localization lock status
    and minimum dimension bounds checking.
    """

    def crop(
        self,
        image: np.ndarray,
        bbox_coords: Tuple[int, int, int, int],
        is_locked: bool
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
        """
        Crops the slab region with dynamic insets and performs dimensions fallback validation.
        Returns:
            (cropped_image, crop_coordinates_box, is_fallback_applied)
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        orig_h, orig_w = image.shape[:2]
        x, y, w, h = bbox_coords

        is_fallback = False

        if not is_locked:
            # Low confidence / no slab locked: Apply 7% fallback inset crop of entire image
            pad_w = int(orig_w * 0.07)
            pad_h = int(orig_h * 0.07)
            y1, y2 = pad_h, orig_h - pad_h
            x1, x2 = pad_w, orig_w - pad_w
            is_fallback = True
        else:
            # Locked slab: Apply safe 5% inset crop of detected region
            pad_w = int(w * 0.05)
            pad_h = int(h * 0.05)
            y1 = max(0, y + pad_h)
            y2 = min(orig_h, y + h - pad_h)
            x1 = max(0, x + pad_w)
            x2 = min(orig_w, x + w - pad_w)

        cropped = image[y1:y2, x1:x2]

        # Minimum dimensions validation: crop height/width must be at least 30% of original
        if (cropped.shape[0] < (orig_h * 0.3)) or (cropped.shape[1] < (orig_w * 0.3)):
            y1, y2 = int(orig_h * 0.05), int(orig_h * 0.95)
            x1, x2 = int(orig_w * 0.05), int(orig_w * 0.95)
            cropped = image[y1:y2, x1:x2]
            is_fallback = True

        cropped_box = (x1, y1, x2 - x1, y2 - y1)
        return cropped.copy(), cropped_box, is_fallback
