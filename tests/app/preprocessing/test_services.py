import pytest
import numpy as np
import cv2
from src.app.preprocessing.morphology import BilateralFilterService, MorphologyGradientService
from src.app.preprocessing.thresholding import ThresholdingService
from src.app.preprocessing.contour_scorer import ContourExtractionService, RectangleScoringService
from src.app.preprocessing.cropper import SafeCropper


def test_bilateral_filter_service():
    service = BilateralFilterService()
    img = np.zeros((100, 100), dtype=np.uint8)
    filtered = service.filter(img)
    assert filtered.shape == img.shape


def test_morphology_gradient_service():
    service = MorphologyGradientService()
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    
    grad = service.compute_gradient(img)
    closed_opened = service.apply_close_open(img)
    
    assert grad.shape == img.shape
    assert closed_opened.shape == img.shape


def test_thresholding_service():
    service = ThresholdingService()
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (50, 100), 50, -1)
    cv2.rectangle(img, (50, 0), (100, 100), 200, -1)
    
    thresh = service.apply_otsu(img)
    assert thresh.shape == img.shape


def test_contour_extraction_service():
    service = ContourExtractionService()
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    
    contours = service.extract_contours(img)
    assert len(contours) > 0


def test_rectangle_scoring_service():
    service = RectangleScoringService()
    c = np.array([[[100, 100]], [[300, 100]], [[300, 300]], [[100, 300]]], dtype=np.int32)
    score, is_locked = service.score_contour(c, 400, 400)
    assert score > 0.0
    assert is_locked is True


def test_safe_cropper():
    cropper = SafeCropper()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Locked crop
    cropped, bbox, is_fallback = cropper.crop(img, (100, 100, 200, 200), is_locked=True)
    assert is_fallback is False
    assert cropped.shape[0] == 180 # 200 - 2 * pad (5% of 200 = 10)
    
    # Unlocked crop -> fallback inset of 7%
    cropped_fallback, bbox_fallback, is_fallback_true = cropper.crop(img, (100, 100, 200, 200), is_locked=False)
    assert is_fallback_true is True
    assert cropped_fallback.shape[0] == 344 # 400 - 2 * pad (7% of 400 = 28)
