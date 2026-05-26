import pytest
from src.app.domain.models import BoundingBox, SlabRegion, StoneImage
from src.app.domain.exceptions import (
    StoneColorAppException,
    DownloadError,
    InvalidImageError,
    SlabDetectionError,
    MessagingError,
    ConfigurationError,
)


def test_bounding_box_properties():
    # Test valid BoundingBox
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    assert bbox.area == 5000
    assert bbox.aspect_ratio == 2.0
    assert bbox.to_tuple() == (10, 20, 100, 50)

    # Test BoundingBox with zero height (edge case)
    bbox_zero = BoundingBox(x=0, y=0, width=10, height=0)
    assert bbox_zero.area == 0
    assert bbox_zero.aspect_ratio == 0.0


def test_slab_region_initialization():
    bbox = BoundingBox(x=5, y=5, width=80, height=80)
    contour = [(5, 5), (85, 5), (85, 85), (5, 85)]
    polygon_points = [(5, 5), (85, 5), (85, 85), (5, 85)]
    
    region = SlabRegion(
        bounding_box=bbox,
        contour=contour,
        confidence=0.95,
        is_rotated=True,
        rotation_angle=12.5,
        polygon_points=polygon_points
    )

    assert region.bounding_box == bbox
    assert region.contour == contour
    assert region.confidence == 0.95
    assert region.is_rotated is True
    assert region.rotation_angle == 12.5
    assert region.polygon_points == polygon_points


def test_stone_image_properties():
    img = StoneImage(
        original_url="https://example.com/stone.jpg",
        width=1920,
        height=1080
    )

    assert img.original_url == "https://example.com/stone.jpg"
    assert img.width == 1920
    assert img.height == 1080
    assert img.raw_path is None
    assert img.cropped_path is None
    assert img.processed is False
    assert len(img.regions) == 0

    # Simulate raw path set
    img.raw_path = "/tmp/raw.jpg"
    assert img.processed is False

    # Simulate processing completion
    img.cropped_path = "/tmp/cropped.png"
    assert img.processed is True


def test_custom_exceptions():
    # Test inheritance
    assert issubclass(DownloadError, StoneColorAppException)
    assert issubclass(InvalidImageError, StoneColorAppException)
    assert issubclass(SlabDetectionError, StoneColorAppException)
    assert issubclass(MessagingError, StoneColorAppException)
    assert issubclass(ConfigurationError, StoneColorAppException)

    # Test instantiation
    exc = DownloadError("Failed to fetch image.")
    assert exc.message == "Failed to fetch image."
    assert str(exc) == "Failed to fetch image."
