import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import cv2

from src.app.domain.models import BoundingBox, SlabRegion
from src.app.dataset.domain.models import DatasetItem, IngestionResult
from src.app.dataset.application.use_cases import IngestDatasetUseCase, ProgressMetrics
from src.app.infrastructure.config import AppConfig


def test_normalize_image_preserving_aspect_ratio():
    # Setup dependencies
    mock_reader = MagicMock()
    mock_writer = MagicMock()
    mock_downloader = MagicMock()
    mock_detector = MagicMock()
    config = AppConfig(min_slab_area_ratio=0.05)

    use_case = IngestDatasetUseCase(
        config=config,
        reader=mock_reader,
        writer=mock_writer,
        downloader_service=mock_downloader,
        detector=mock_detector
    )

    # 1. Landscape image (600x300) -> resized to target max dimension of 200 -> (200, 100)
    landscape = np.zeros((300, 600, 3), dtype=np.uint8)
    resized_l = use_case.normalize_image(landscape, 200)
    assert resized_l.shape[1] == 200  # width
    assert resized_l.shape[0] == 100  # height

    # 2. Portrait image (400x800) -> resized to target max dimension of 100 -> (50, 100)
    portrait = np.zeros((800, 400, 3), dtype=np.uint8)
    resized_p = use_case.normalize_image(portrait, 100)
    assert resized_p.shape[1] == 50   # width
    assert resized_p.shape[0] == 100  # height


@pytest.mark.asyncio
async def test_ingest_single_item_success(valid_slab_image):
    success, img_bytes = cv2.imencode(".jpg", valid_slab_image)
    img_bytes = img_bytes.tobytes()

    # Mock config
    config = AppConfig(min_slab_area_ratio=0.05)
    config.output_image_size = 200

    # Mock ports and services
    mock_reader = MagicMock()
    mock_writer = MagicMock()
    mock_writer.save_processed_image = MagicMock(return_value="/processed/beige/file.png")
    
    mock_downloader = MagicMock()
    mock_downloader.download = AsyncMock(return_value=img_bytes)

    # Mock Detector
    bbox = BoundingBox(x=75, y=100, width=250, height=200)
    mock_region = SlabRegion(
        bounding_box=bbox,
        contour=[(75, 100), (325, 100), (325, 300), (75, 300)],
        confidence=0.95
    )
    mock_detector = MagicMock()
    mock_detector.detect_slabs = MagicMock(return_value=[mock_region])
    # remove_background returns the cropped RGBA image
    cropped_dummy = np.zeros((200, 250, 4), dtype=np.uint8)
    mock_detector.remove_background = MagicMock(return_value=cropped_dummy)

    use_case = IngestDatasetUseCase(
        config=config,
        reader=mock_reader,
        writer=mock_writer,
        downloader_service=mock_downloader,
        detector=mock_detector
    )

    item = DatasetItem(
        url="http://example.com/beige_slab.jpg",
        color_class="beige",
        source_file="beige.txt"
    )

    semaphore = asyncio.Semaphore(1)
    metrics = ProgressMetrics()

    result = await use_case.ingest_single_item(item, semaphore, metrics)

    # Assertions
    assert result.crop_success is True
    assert result.local_path == "/processed/beige/file.png"
    assert result.failure_reason is None
    assert metrics.success_count == 1
    assert metrics.failed_download == 0
    assert metrics.failed_crop == 0

    mock_downloader.download.assert_called_once_with(item.url)
    mock_detector.detect_slabs.assert_called_once()
    mock_detector.remove_background.assert_called_once()
    mock_writer.save_processed_image.assert_called_once()
    mock_writer.write_metadata_row.assert_called_once_with(result)


@pytest.mark.asyncio
async def test_ingest_single_item_crop_failure(valid_slab_image):
    # Encode BGR image
    success, img_bytes = cv2.imencode(".jpg", valid_slab_image)
    img_bytes = img_bytes.tobytes()

    config = AppConfig(min_slab_area_ratio=0.05)

    mock_reader = MagicMock()
    mock_writer = MagicMock()
    
    mock_downloader = MagicMock()
    mock_downloader.download = AsyncMock(return_value=img_bytes)

    # Mock Detector to return NO regions (causes Crop/Detection Failure)
    mock_detector = MagicMock()
    mock_detector.detect_slabs = MagicMock(return_value=[])

    use_case = IngestDatasetUseCase(
        config=config,
        reader=mock_reader,
        writer=mock_writer,
        downloader_service=mock_downloader,
        detector=mock_detector
    )

    item = DatasetItem(
        url="http://example.com/no_slab.jpg",
        color_class="black",
        source_file="black.txt"
    )

    semaphore = asyncio.Semaphore(1)
    metrics = ProgressMetrics()

    result = await use_case.ingest_single_item(item, semaphore, metrics)

    # Assert grace degradation: pipeline continues, reports crop failure
    assert result.crop_success is False
    assert result.local_path is None
    assert "No stone slabs detected" in result.failure_reason
    assert metrics.success_count == 0
    assert metrics.failed_crop == 1
    
    # Verify failure reports were logged to disk
    mock_writer.save_failed_metadata.assert_called_once_with(item, result.failure_reason, img_bytes)
    mock_writer.write_metadata_row.assert_called_once_with(result)


@pytest.mark.asyncio
async def test_full_pipeline_orchestration_and_resumption():
    config = AppConfig(min_slab_area_ratio=0.05)
    config.dataset_concurrency = 2
    config.output_image_size = 200

    # 1. Mock Reader loaded 3 items
    item1 = DatasetItem(url="http://example.com/item1.jpg", color_class="beige", source_file="beige.txt")
    item2 = DatasetItem(url="http://example.com/item2.jpg", color_class="beige", source_file="beige.txt")
    item3 = DatasetItem(url="http://example.com/item3.jpg", color_class="black", source_file="black.txt")
    
    mock_reader = MagicMock()
    mock_reader.read_dataset = MagicMock(return_value=[item1, item2, item3])

    # 2. Mock Writer get_processed_urls reports:
    # item1 was already successfully ingested -> Skip
    # item2 failed previously -> scheduled only if resuming
    # item3 never attempted -> scheduled
    mock_writer = MagicMock()
    mock_writer.get_processed_urls = MagicMock(return_value={
        "http://example.com/item1.jpg": True,
        "http://example.com/item2.jpg": False
    })

    mock_downloader = MagicMock()
    mock_detector = MagicMock()

    use_case = IngestDatasetUseCase(
        config=config,
        reader=mock_reader,
        writer=mock_writer,
        downloader_service=mock_downloader,
        detector=mock_detector
    )

    # --- Scenario A: Run without resume_failed ---
    metrics_a = await use_case.execute(resume_failed=False, dry_run=True)
    assert metrics_a.total_found == 3
    # item1 (success) skipped, item2 (previous fail) skipped without resume
    assert metrics_a.skipped_completed == 2
    assert metrics_a.to_process == 1  # Only item3

    # --- Scenario B: Run with resume_failed ---
    metrics_b = await use_case.execute(resume_failed=True, dry_run=True)
    assert metrics_b.total_found == 3
    # item1 (success) skipped, item2 retried!
    assert metrics_b.skipped_completed == 1
    assert metrics_b.to_process == 2  # item2 and item3
