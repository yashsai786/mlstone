import pytest
import numpy as np
import cv2
from src.app.preprocessing.pipeline import OpenCVPipeline
from src.app.preprocessing.slab_detector import SlabDetector
from src.app.domain.exceptions import SlabDetectionError
from src.app.domain.slab_detection import SlabDetectionResult


# 1. perfect centered slab
def test_perfect_centered_slab():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0
    bbox = regions[0].bounding_box
    assert bbox.width > 150
    assert bbox.height > 150


# 2. rotated slab
def test_rotated_slab():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
    
    # Rotate 15 degrees
    matrix = cv2.getRotationMatrix2D((200, 200), 15, 1.0)
    rotated = cv2.warpAffine(img, matrix, (400, 400))
    
    regions = pipeline.detect_slabs(rotated)
    assert len(regions) > 0
    # Core requirement: do NOT perform perspective warp or rotation normalization
    assert regions[0].is_rotated is False
    assert regions[0].rotation_angle == 0.0


# 3. low contrast slab
def test_low_contrast_slab():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Extremely dark slab on dark background
    cv2.rectangle(img, (100, 100), (300, 300), (25, 25, 25), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 4. noisy warehouse background
def test_noisy_warehouse_background():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Background noise lines
    for i in range(0, 400, 20):
        cv2.line(img, (0, i), (400, i), (50, 50, 50), 1)
    # Centered slab
    cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 5. multiple slabs
def test_multiple_slabs():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 100), (170, 300), (200, 200, 200), -1)
    cv2.rectangle(img, (230, 100), (350, 300), (210, 210, 210), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 6. tiny contours
def test_tiny_contours():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw tiny boxes (noise)
    cv2.rectangle(img, (10, 10), (15, 15), (200, 200, 200), -1)
    cv2.rectangle(img, (380, 380), (385, 385), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    # Should fallback crop since no large slab is present
    assert len(regions) > 0
    assert regions[0].confidence == 0.10


# 7. no slab present
def test_no_slab_present():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0
    assert regions[0].confidence == 0.10


# 8. slab touching borders
def test_slab_touching_borders():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (250, 400), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 9. straps crossing slab
def test_straps_crossing_slab():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
    # Draw dark vertical strap cutting through slab
    cv2.rectangle(img, (180, 0), (200, 400), (10, 10, 10), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 10. partial slab visibility
def test_partial_slab_visibility():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Slab cut off at the edge
    cv2.rectangle(img, (200, 100), (400, 300), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 11. dark slab on dark background
def test_dark_slab_on_dark_background():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (15, 15, 15), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 12. bright slab on bright background
def test_bright_slab_on_bright_background():
    pipeline = OpenCVPipeline()
    img = np.ones((400, 400, 3), dtype=np.uint8) * 230
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    
    regions = pipeline.detect_slabs(img)
    assert len(regions) > 0


# 13. fallback crop execution
def test_fallback_crop_execution():
    detector = SlabDetector()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    res = detector.detect(img)
    assert res.crop_metadata.is_fallback is True
    assert res.crop_metadata.warning == "Fallback crop executed"


# 14. invalid image bytes
def test_invalid_image_bytes():
    from src.app.infrastructure.image_decoder import ImageDecoder
    decoder = ImageDecoder()
    with pytest.raises(Exception):
        decoder.decode(b"corrupted_bytes_that_cannot_be_decoded")


# 15. malformed image URL
@pytest.mark.anyio
async def test_malformed_image_url():
    from src.app.infrastructure.downloader import HTTPImageDownloader
    downloader = HTTPImageDownloader()
    with pytest.raises(Exception):
        await downloader.download("http://invalid_malformed_domain_nonexistent/image.jpg")


# 16. contour scoring ranking
def test_contour_scoring_ranking():
    from src.app.preprocessing.contour_scorer import RectangleScoringService
    scorer = RectangleScoringService()
    
    # Perfect centered contour
    c1 = np.array([[[100, 100]], [[300, 100]], [[300, 300]], [[100, 300]]], dtype=np.int32)
    score1, locked1 = scorer.score_contour(c1, 400, 400)
    
    # Small off-center contour
    c2 = np.array([[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]], dtype=np.int32)
    score2, locked2 = scorer.score_contour(c2, 400, 400)
    
    assert score1 > score2


# 17. crop boundary safety
def test_crop_boundary_safety():
    from src.app.preprocessing.cropper import SafeCropper
    cropper = SafeCropper()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Bounding box exceeding image bounds
    cropped, bbox, fallback = cropper.crop(img, (-50, -50, 500, 500), is_locked=True)
    assert cropped.shape[0] <= 400
    assert cropped.shape[1] <= 400


# 18. crop minimum dimension validation
def test_crop_minimum_dimension_validation():
    from src.app.preprocessing.cropper import SafeCropper
    cropper = SafeCropper()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Super tiny bounding box (10x10) -> Should trigger 5% inset validation fallback
    cropped, bbox, fallback = cropper.crop(img, (100, 100, 10, 10), is_locked=True)
    assert fallback is True
    # The output crop must be the safe 5% to 95% bounding inset
    assert cropped.shape[0] == 360 # 400 * 0.90


# 19. morphology pipeline correctness
def test_morphology_pipeline_correctness():
    from src.app.preprocessing.morphology import MorphologyGradientService
    service = MorphologyGradientService()
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    
    grad = service.compute_gradient(img)
    assert grad is not None
    # Gradient of flat region is zero, edge lines should be white
    assert np.any(grad > 0)


# 20. threshold generation correctness
def test_threshold_generation_correctness():
    from src.app.preprocessing.thresholding import ThresholdingService
    service = ThresholdingService()
    img = np.zeros((100, 100), dtype=np.uint8)
    # Create two strong peaks (bimodality for Otsu)
    cv2.rectangle(img, (0, 0), (50, 100), 50, -1)
    cv2.rectangle(img, (50, 0), (100, 100), 200, -1)
    
    thresh = service.apply_otsu(img)
    assert thresh is not None
    assert np.all(np.logical_or(thresh == 0, thresh == 255))


# Crop & Background tests

def test_crop_slab():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    cropped = pipeline.crop_slab(img, regions[0])
    assert cropped is not None
    assert cropped.shape[0] > 0


def test_remove_background():
    pipeline = OpenCVPipeline()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
    
    regions = pipeline.detect_slabs(img)
    rgba = pipeline.remove_background(img, regions[0])
    assert rgba.shape[2] == 4
    assert np.all(rgba[:, :, 3] == 255)


def test_detect_slabs_invalid_input():
    pipeline = OpenCVPipeline()
    with pytest.raises(SlabDetectionError):
        pipeline.detect_slabs(None) # type: ignore
