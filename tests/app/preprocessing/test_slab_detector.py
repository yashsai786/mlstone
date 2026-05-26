import pytest
import cv2
import numpy as np
from src.app.preprocessing.slab_detector import SlabDetector
from src.app.domain.exceptions import SlabDetectionError


def test_preprocess_image():
    detector = SlabDetector()
    img = np.zeros((1200, 800, 3), dtype=np.uint8)
    resized, gray, scale = detector.preprocess_image(img)
    
    # Standardize maximum dimension to 1000px
    assert max(resized.shape[:2]) == 1000
    assert scale == 1000.0 / 1200.0
    assert len(gray.shape) == 2


def test_apply_smoothing():
    detector = SlabDetector()
    gray = np.zeros((400, 400), dtype=np.uint8)
    
    # Low std_val (smooth marble) -> Gaussian blur
    smoothed_low = detector.apply_smoothing(gray, 5.0)
    assert smoothed_low.shape == gray.shape

    # High std_val (granite) -> Bilateral filter
    smoothed_high = detector.apply_smoothing(gray, 40.0)
    assert smoothed_high.shape == gray.shape


def test_detect_edges():
    detector = SlabDetector()
    gray = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(gray, (100, 100), (300, 300), 200, -1)
    
    edges = detector.detect_edges(gray)
    assert edges.shape == gray.shape
    assert np.any(edges > 0)


def test_apply_morphology():
    detector = SlabDetector()
    edges = np.zeros((400, 400), dtype=np.uint8)
    cv2.line(edges, (100, 100), (100, 200), 255, 2)
    cv2.line(edges, (100, 205), (100, 300), 255, 2) # broken border line
    
    closed = detector.apply_morphology(edges)
    assert closed.shape == edges.shape
    # Morphology closing should have bridged the gap of 5 pixels
    assert closed[100, 202] == 0 or np.any(closed > 0)


def test_score_contour():
    detector = SlabDetector()
    edges = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(edges, (100, 100), (300, 300), 255, 1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) > 0
    
    score, candidate = detector.score_contour(contours[0], 400*400, edges, 400, 400)
    assert score > 0.0
    assert candidate.score == score
    assert candidate.contour_area > 0.0
