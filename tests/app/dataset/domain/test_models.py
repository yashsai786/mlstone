from datetime import datetime
from src.app.dataset.domain.models import DatasetItem, IngestionResult


def test_dataset_item_deterministic_filename():
    # 1. Normal JPG URL
    item = DatasetItem(
        url="https://example.com/mg-carrara/P231533/Cover/cover.jpg",
        color_class="beige",
        source_file="dataset/Beige/beige.txt"
    )
    filename = item.get_deterministic_filename()
    assert filename.endswith(".jpg")
    # Verify deterministic (running twice gives same name)
    assert filename == item.get_deterministic_filename()

    # 2. WebP URL
    item_webp = DatasetItem(
        url="https://iblocky.work/mondial-granit/Bundle/cover_v177.webp",
        color_class="black",
        source_file="dataset/Black/black.txt"
    )
    assert item_webp.get_deterministic_filename().endswith(".webp")

    # 3. Non-image extension URL fallback to PNG (to preserve crop alpha channel)
    item_query = DatasetItem(
        url="https://iblocky.work/mondial-granit/img?id=123",
        color_class="brown",
        source_file="dataset/Brown/brown.txt"
    )
    assert item_query.get_deterministic_filename().endswith(".png")


def test_ingestion_result_creation():
    result = IngestionResult(
        image_id="test_image_id",
        source_url="http://example.com/img.jpg",
        color_class="grey",
        local_path="/storage/processed/grey/test_image_id.png",
        crop_success=True,
        failure_reason=None
    )

    assert result.image_id == "test_image_id"
    assert result.source_url == "http://example.com/img.jpg"
    assert result.crop_success is True
    assert result.failure_reason is None
    assert isinstance(result.timestamp, datetime)
