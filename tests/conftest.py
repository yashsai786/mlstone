import pytest
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
from src.app.infrastructure.config import AppConfig


@pytest.fixture
def temp_storage_dir():
    """Provides a clean temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_storage_dir):
    """Provides an AppConfig pointed to the temporary storage directory."""
    return AppConfig(
        storage_base_dir=temp_storage_dir,
        min_slab_area_ratio=0.05,
        debug_mode=True
    )


@pytest.fixture
def valid_slab_image() -> np.ndarray:
    """Generates a synthetic image containing a single clear rectangular slab on a dark background."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw background noise
    cv2.randn(img, (20, 20, 20), (5, 5, 5))
    
    # Draw a clean slab (250x200) centered
    # Use marble-like texture: draw some lines inside
    slab_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(slab_mask, (75, 100), (325, 300), 255, -1)
    
    # Fill slab region with light granite color
    slab_texture = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.randn(slab_texture, (200, 180, 160), (15, 15, 15))
    
    # Draw some "veins"
    cv2.line(slab_texture, (100, 120), (300, 280), (120, 100, 90), 2)
    cv2.line(slab_texture, (250, 110), (150, 290), (140, 120, 110), 1)

    # Copy texture inside the slab mask
    img[slab_mask == 255] = slab_texture[slab_mask == 255]
    return img


@pytest.fixture
def slab_with_straps_image(valid_slab_image) -> np.ndarray:
    """Generates a synthetic slab image with dark vertical straps cutting through the slab."""
    img = valid_slab_image.copy()
    # Draw two dark vertical straps (width 15 pixels)
    cv2.rectangle(img, (120, 0), (135, 400), (30, 30, 30), -1)
    cv2.rectangle(img, (270, 0), (285, 400), (30, 30, 30), -1)
    return img


@pytest.fixture
def slab_with_holders_image(valid_slab_image) -> np.ndarray:
    """Generates a synthetic slab image with metallic holders clamping the edges."""
    img = valid_slab_image.copy()
    # Draw dark metallic blocks holding bottom and top edges
    # Top edge holder
    cv2.rectangle(img, (190, 80), (210, 110), (80, 80, 80), -1)
    # Bottom edge holder
    cv2.rectangle(img, (190, 290), (210, 320), (80, 80, 80), -1)
    return img


@pytest.fixture
def slab_with_hand_image(valid_slab_image) -> np.ndarray:
    """Generates a synthetic slab image with a flesh-colored hand overlapping the side."""
    img = valid_slab_image.copy()
    # Draw a flesh-colored polygon / ellipses representing a hand on the right side
    cv2.ellipse(img, (330, 200), (40, 20), -30, 0, 360, (180, 200, 240), -1) # BGR flesh tone
    cv2.ellipse(img, (315, 190), (30, 8), -15, 0, 360, (180, 200, 240), -1)
    cv2.ellipse(img, (315, 210), (30, 8), -45, 0, 360, (180, 200, 240), -1)
    return img


@pytest.fixture
def low_contrast_image() -> np.ndarray:
    """Generates a solid dark grey image with extremely low variance."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.randn(img, (30, 30, 30), (1, 1, 1))
    return img


@pytest.fixture
def empty_image() -> np.ndarray:
    """Generates an empty black image."""
    return np.zeros((400, 400, 3), dtype=np.uint8)


@pytest.fixture
def multiple_slabs_image() -> np.ndarray:
    """Generates an image containing two separate slabs."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Slab 1
    cv2.rectangle(img, (50, 50), (170, 350), (180, 180, 180), -1)
    # Slab 2
    cv2.rectangle(img, (230, 50), (350, 350), (200, 200, 200), -1)
    return img


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Bytes that represent corrupted image files."""
    return b"NOT_A_VALID_IMAGE_HEADER_1234567890_CORRUPTED_DATA_!!!"
