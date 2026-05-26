import cv2
import numpy as np
from typing import List, Tuple, Optional


class ContourExtractionService:
    """
    Service for extracting external contours from binary maps.
    """

    def extract_contours(self, binary_map: np.ndarray) -> List[np.ndarray]:
        if binary_map is None or binary_map.size == 0:
            raise ValueError("Input binary map is empty or invalid.")
        contours, _ = cv2.findContours(
            binary_map,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)


class RectangleScoringService:
    """
    Service for scoring contour candidates according to slab-specific geometric attributes:
    - Area coverage
    - Center proximity
    - Rectangularity (contour area / bounding box area)
    - Solidity (contour area / convex hull area)
    """

    def score_contour(
        self,
        contour: np.ndarray,
        img_w: int,
        img_h: int
    ) -> Tuple[float, bool]:
        """
        Calculates a composite score and determines if the contour qualifies as a locked slab.
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        total_area = img_w * img_h

        if total_area == 0 or img_w == 0:
            return 0.0, False

        cnt_center_x = x + (w / 2)
        cnt_center_y = y + (h / 2)
        img_center_x = img_w / 2
        img_center_y = img_h / 2

        distance = np.sqrt(
            (cnt_center_x - img_center_x)**2 +
            (cnt_center_y - img_center_y)**2
        )

        # Check the experimentally validated threshold lock criteria
        is_locked = bool((area > (total_area * 0.20)) and (distance < (img_w * 0.30)))

        # Calculate modular features for multidimensional scoring
        area_coverage = area / total_area
        center_proximity = max(0.0, 1.0 - (distance / (img_w * 0.5)))
        
        c_area = cv2.contourArea(contour)
        rectangularity = c_area / area if area > 0 else 0.0
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = c_area / hull_area if hull_area > 0 else 0.0

        # Composite score calculation
        composite_score = (
            0.4 * area_coverage +
            0.3 * center_proximity +
            0.15 * rectangularity +
            0.15 * solidity
        )

        return float(composite_score), is_locked
