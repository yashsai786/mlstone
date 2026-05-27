import csv
import pytest
from pathlib import Path
import numpy as np
import cv2

from src.app.dataset.domain.models import DatasetItem, IngestionResult
from src.app.dataset.infrastructure.adapters import LocalDatasetReader, LocalDatasetWriter


def test_dataset_reader_loading_and_deduplication(temp_storage_dir):
    # Setup a dummy dataset structure in the temp directory
    dataset_dir = temp_storage_dir / "dataset"
    beige_dir = dataset_dir / "Beige"
    black_dir = dataset_dir / "Black"
    
    beige_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create beige.txt with some valid, empty, invalid, and duplicate URLs
    beige_txt = beige_dir / "beige.txt"
    with open(beige_txt, "w", encoding="utf-8") as f:
        f.write("https://example.com/beige1.jpg\n")
        f.write("   \n")  # Empty line
        f.write("ftp://invalid-url-format\n")  # Invalid format
        f.write("https://example.com/beige1.jpg\n")  # Internal duplicate
        f.write("https://example.com/shared.jpg\n")  # Shared URL

    # 2. Create black.txt with some URLs including a duplicate across files
    black_txt = black_dir / "black.txt"
    with open(black_txt, "w", encoding="utf-8") as f:
        f.write("https://example.com/black1.jpg\n")
        f.write("https://example.com/shared.jpg\n")  # Global duplicate

    reader = LocalDatasetReader()

    # Test full read
    items = reader.read_dataset(str(dataset_dir))
    
    # Expected loaded items: beige1 (beige), shared (beige), black1 (black)
    # Total count = 3
    assert len(items) == 3
    
    # Assert correct auto-detected categories
    assert items[0].color_class == "beige"
    assert items[0].url == "https://example.com/beige1.jpg"
    assert items[1].color_class == "beige"
    assert items[1].url == "https://example.com/shared.jpg"
    assert items[2].color_class == "black"
    assert items[2].url == "https://example.com/black1.jpg"

    # Test single-color filter read
    items_beige = reader.read_dataset(str(dataset_dir), single_color="beige")
    assert len(items_beige) == 2
    assert all(i.color_class == "beige" for i in items_beige)


def test_dataset_writer_metadata_and_images(temp_storage_dir):
    processed_dir = temp_storage_dir / "processed"
    failed_dir = temp_storage_dir / "failed"
    metadata_csv = temp_storage_dir / "metadata.csv"

    writer = LocalDatasetWriter(
        processed_base_dir=processed_dir,
        failed_base_dir=failed_dir,
        metadata_path=metadata_csv
    )

    # 1. Test saving processed image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    local_path = writer.save_processed_image("beige", "item1.jpg", dummy_img)
    
    # Path should end with png (since we force PNG formatting for cropped alphas)
    assert local_path.endswith("item1.png")
    assert Path(local_path).exists()
    assert (processed_dir / "beige").exists()

    # 2. Test saving failed metadata
    item = DatasetItem(
        url="https://example.com/fail.jpg",
        color_class="black",
        source_file="dataset/Black/black.txt"
    )
    writer.save_failed_metadata(item, "Timeout Error", b"Partial raw data")
    
    # Check that error report exists
    failed_files = list(failed_dir.glob("*_error.txt"))
    assert len(failed_files) == 1
    with open(failed_files[0], "r", encoding="utf-8") as f:
        content = f.read()
        assert "Timeout Error" in content
        assert "fail.jpg" in content

    # Check that raw data was saved
    raw_files = list(failed_dir.glob("*_raw.jpg"))
    assert len(raw_files) == 1
    assert raw_files[0].read_bytes() == b"Partial raw data"

    # 3. Test writing metadata rows & resumption scan
    res1 = IngestionResult(
        image_id="img1",
        source_url="https://example.com/beige1.jpg",
        color_class="beige",
        local_path=local_path,
        crop_success=True,
        failure_reason=None
    )
    res2 = IngestionResult(
        image_id="img2",
        source_url="https://example.com/fail.jpg",
        color_class="black",
        local_path=None,
        crop_success=False,
        failure_reason="Timeout Error"
    )

    writer.write_metadata_row(res1)
    writer.write_metadata_row(res2)

    # Test reading back attempted URLs (resumption scanning)
    attempted = writer.get_processed_urls()
    assert len(attempted) == 2
    assert attempted["https://example.com/beige1.jpg"] is True  # Success
    assert attempted["https://example.com/fail.jpg"] is False   # Failure
