import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import cv2

from src.app.application.use_cases import DownloadImageUseCase, ExtractSlabUseCase, CleanupBackgroundUseCase
from src.app.domain.models import StoneImage, SlabRegion, BoundingBox
from src.app.domain.exceptions import DownloadError, InvalidImageError, SlabDetectionError


# --- DownloadImageUseCase Tests ---

@pytest.mark.asyncio
async def test_download_image_use_case_success(valid_slab_image):
    # Encode valid_slab_image into JPEG bytes
    success, img_bytes = cv2.imencode(".jpg", valid_slab_image)
    assert success
    img_bytes = img_bytes.tobytes()

    mock_downloader = MagicMock()
    mock_downloader.download = AsyncMock(return_value=img_bytes)

    mock_storage = MagicMock()
    mock_storage.save_raw = MagicMock(return_value="/storage/raw/file.jpg")

    use_case = DownloadImageUseCase(downloader=mock_downloader, storage=mock_storage)
    
    stone_image = await use_case.execute("https://example.com/stone_slab.jpg")

    assert isinstance(stone_image, StoneImage)
    assert stone_image.original_url == "https://example.com/stone_slab.jpg"
    assert stone_image.raw_path == "/storage/raw/file.jpg"
    assert stone_image.width == valid_slab_image.shape[1]
    assert stone_image.height == valid_slab_image.shape[0]
    
    mock_downloader.download.assert_called_once_with("https://example.com/stone_slab.jpg")
    mock_storage.save_raw.assert_called_once_with("stone_slab.jpg", img_bytes)


@pytest.mark.asyncio
async def test_download_image_use_case_corrupted_data(corrupted_image_bytes):
    mock_downloader = MagicMock()
    mock_downloader.download = AsyncMock(return_value=corrupted_image_bytes)

    mock_storage = MagicMock()
    
    use_case = DownloadImageUseCase(downloader=mock_downloader, storage=mock_storage)

    with pytest.raises(InvalidImageError) as exc_info:
        await use_case.execute("https://example.com/corrupted.jpg")
    
    assert "corrupted" in str(exc_info.value)
    mock_storage.save_raw.assert_not_called()


# --- ExtractSlabUseCase Tests ---

def test_extract_slab_use_case_success(valid_slab_image):
    stone_image = StoneImage(
        original_url="https://example.com/stone.jpg",
        width=400,
        height=400,
        raw_path="/storage/raw/stone.jpg"
    )

    bbox = BoundingBox(x=75, y=100, width=250, height=200)
    mock_region = SlabRegion(
        bounding_box=bbox,
        contour=[(75, 100), (325, 100), (325, 300), (75, 300)],
        confidence=0.98
    )

    mock_detector = MagicMock()
    mock_detector.detect_slabs = MagicMock(return_value=[mock_region])

    mock_storage = MagicMock()

    use_case = ExtractSlabUseCase(detector=mock_detector, storage=mock_storage)

    # Patch cv2.imread to avoid loading real file
    with patch("cv2.imread", return_value=valid_slab_image):
        region = use_case.execute(stone_image, debug_mode=True)

    assert region == mock_region
    assert stone_image.regions == [mock_region]
    mock_detector.detect_slabs.assert_called_once()
    mock_storage.save_debug.assert_called_once()  # Called because debug_mode=True


def test_extract_slab_use_case_no_slabs_detected(valid_slab_image):
    stone_image = StoneImage(
        original_url="https://example.com/stone.jpg",
        width=400,
        height=400,
        raw_path="/storage/raw/stone.jpg"
    )

    mock_detector = MagicMock()
    mock_detector.detect_slabs = MagicMock(return_value=[])  # Empty regions list

    mock_storage = MagicMock()

    use_case = ExtractSlabUseCase(detector=mock_detector, storage=mock_storage)

    with patch("cv2.imread", return_value=valid_slab_image):
        with pytest.raises(SlabDetectionError) as exc_info:
            use_case.execute(stone_image)

    assert "No stone slabs detected" in str(exc_info.value)


# --- CleanupBackgroundUseCase Tests ---

def test_cleanup_background_use_case_success(valid_slab_image):
    stone_image = StoneImage(
        original_url="https://example.com/stone.jpg",
        width=400,
        height=400,
        raw_path="/storage/raw/stone.jpg"
    )

    bbox = BoundingBox(x=75, y=100, width=250, height=200)
    mock_region = SlabRegion(
        bounding_box=bbox,
        contour=[(75, 100), (325, 100), (325, 300), (75, 300)],
        confidence=0.98
    )

    cleaned_img = np.zeros((200, 250, 4), dtype=np.uint8) # RGBA

    mock_detector = MagicMock()
    mock_detector.remove_background = MagicMock(return_value=cleaned_img)

    mock_storage = MagicMock()
    mock_storage.save_cropped = MagicMock(return_value="/storage/cropped/stone.png")

    use_case = CleanupBackgroundUseCase(detector=mock_detector, storage=mock_storage)

    with patch("cv2.imread", return_value=valid_slab_image):
        cropped_path = use_case.execute(stone_image, mock_region)

    assert cropped_path == "/storage/cropped/stone.png"
    assert stone_image.cropped_path == "/storage/cropped/stone.png"
    mock_detector.remove_background.assert_called_once_with(valid_slab_image, mock_region)
    mock_storage.save_cropped.assert_called_once_with("stone.jpg", cleaned_img)
