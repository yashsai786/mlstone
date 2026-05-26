import pytest
import numpy as np
from src.app.preprocessing.transform import order_points, four_point_perspective_transform


def test_order_points():
    # Ordered coordinates: top-left, top-right, bottom-right, bottom-left
    pts = np.array([
        [300, 300], # bottom-right
        [100, 300], # bottom-left
        [300, 100], # top-right
        [100, 100]  # top-left
    ], dtype="float32")
    
    ordered = order_points(pts)
    
    # Assert top-left is [100, 100]
    assert np.array_equal(ordered[0], [100, 100])
    # Assert top-right is [300, 100]
    assert np.array_equal(ordered[1], [300, 100])
    # Assert bottom-right is [300, 300]
    assert np.array_equal(ordered[2], [300, 300])
    # Assert bottom-left is [100, 300]
    assert np.array_equal(ordered[3], [100, 300])


def test_four_point_perspective_transform():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Define a clean rectangle
    pts = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ], dtype="float32")
    
    warped, M, dst = four_point_perspective_transform(img, pts)
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0
    assert M.shape == (3, 3)
    assert dst.shape == (4, 2)
