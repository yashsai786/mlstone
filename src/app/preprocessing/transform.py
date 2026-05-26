import cv2
import numpy as np
from typing import Tuple


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 box points in the sequence:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum: top-left has min sum, bottom-right has max sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference: top-right has min diff, bottom-left has max diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_perspective_transform(image: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs perspective transform to unwarp/straighten a rotated quad.
    Automatically horizontally aligns wide stone slabs.
    Returns:
        (warped_image, transformation_matrix, destination_points)
    """
    ordered_pts = order_points(pts)
    (tl, tr, br, bl) = ordered_pts

    # Compute width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Safe dimensions fallback
    if max_width <= 0 or max_height <= 0:
        max_width, max_height = 100, 100

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # Auto-rotation correction: if slab height is larger than width, orient it horizontally
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped, M, dst
