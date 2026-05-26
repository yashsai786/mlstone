import pytest
import os
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

from src.app.infrastructure.downloader import HTTPImageDownloader
from src.app.infrastructure.persistence import LocalFileStorage
from src.app.infrastructure.config import AppConfig, get_config
from src.app.domain.exceptions import DownloadError, InvalidImageError, ConfigurationError


class MockAsyncContextManager:
    """Helper class to mock async context managers in httpx client calls."""
    def __init__(self, return_value, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect

    async def __aenter__(self):
        if self.side_effect:
            raise self.side_effect
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# --- Configuration Tests ---

def test_app_config_creation(temp_storage_dir):
    config = AppConfig(storage_base_dir=temp_storage_dir)
    assert config.storage_base_dir == temp_storage_dir
    assert config.raw_dir == temp_storage_dir / "raw"
    assert config.cropped_dir == temp_storage_dir / "cropped"
    assert config.debug_dir == temp_storage_dir / "debug"
    
    # Assert directories are automatically created
    assert config.raw_dir.exists()
    assert config.cropped_dir.exists()
    assert config.debug_dir.exists()


def test_get_config():
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2  # Singleton check


# --- File Storage Tests ---

def test_local_file_storage_save_and_load(temp_storage_dir, valid_slab_image):
    raw_dir = temp_storage_dir / "raw"
    cropped_dir = temp_storage_dir / "cropped"
    debug_dir = temp_storage_dir / "debug"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    storage = LocalFileStorage(raw_dir=raw_dir, cropped_dir=cropped_dir, debug_dir=debug_dir)

    # 1. Save raw
    raw_bytes = b"fake_raw_data"
    raw_path = storage.save_raw("test.jpg", raw_bytes)
    assert os.path.exists(raw_path)
    with open(raw_path, "rb") as f:
        assert f.read() == raw_bytes

    # 2. Save cropped
    cropped_path = storage.save_cropped("test.jpg", valid_slab_image)
    assert os.path.exists(cropped_path)
    assert cropped_path.endswith(".png")  # Cropped should force png
    
    # Load and verify
    loaded_img = storage.load_image(cropped_path)
    assert loaded_img is not None
    assert loaded_img.shape == valid_slab_image.shape

    # 3. Save debug
    debug_path = storage.save_debug("test_debug.jpg", valid_slab_image)
    assert os.path.exists(debug_path)

    # 4. Load non-existent file
    with pytest.raises(FileNotFoundError):
        storage.load_image("/nonexistent/file.png")

    # 5. Load invalid file content (corrupted)
    corrupt_path = raw_dir / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"not_an_image")
    with pytest.raises(InvalidImageError):
        storage.load_image(str(corrupt_path))


# --- Image Downloader Tests ---

@pytest.mark.asyncio
async def test_downloader_invalid_url():
    downloader = HTTPImageDownloader()
    with pytest.raises(DownloadError) as exc_info:
        await downloader.download("invalid_url_without_scheme")
    assert "Invalid URL format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_downloader_non_image_content_type():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html", "content-length": "100"}
    
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))
    client_manager = MockAsyncContextManager(mock_client)

    downloader = HTTPImageDownloader()
    
    with patch("httpx.AsyncClient", return_value=client_manager):
        with pytest.raises(InvalidImageError) as exc_info:
            await downloader.download("https://example.com/not-an-image.html")
        assert "Content-Type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_downloader_exceeds_size_limit():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/jpeg", "content-length": "20000000"}  # 20MB
    
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))
    client_manager = MockAsyncContextManager(mock_client)

    downloader = HTTPImageDownloader(max_size_bytes=1024 * 1024)  # 1MB limit
    
    with patch("httpx.AsyncClient", return_value=client_manager):
        with pytest.raises(DownloadError) as exc_info:
            await downloader.download("https://example.com/huge.jpg")
        assert "Image size exceeds maximum limit" in str(exc_info.value)


@pytest.mark.asyncio
async def test_downloader_http_error():
    mock_client = MagicMock()
    client_manager = MockAsyncContextManager(None, side_effect=httpx.HTTPError("Connection failed"))

    downloader = HTTPImageDownloader()
    
    with patch("httpx.AsyncClient", return_value=client_manager):
        with pytest.raises(DownloadError) as exc_info:
            await downloader.download("https://example.com/broken.jpg")
        assert "HTTP error downloading image" in str(exc_info.value)


@pytest.mark.asyncio
async def test_downloader_success():
    fake_img_bytes = b"fake_jpeg_bytes"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/jpeg", "content-length": str(len(fake_img_bytes))}
    mock_response.aread = AsyncMock(return_value=fake_img_bytes)
    
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))
    client_manager = MockAsyncContextManager(mock_client)

    downloader = HTTPImageDownloader()
    
    with patch("httpx.AsyncClient", return_value=client_manager):
        data = await downloader.download("https://example.com/valid.jpg")
        assert data == fake_img_bytes
