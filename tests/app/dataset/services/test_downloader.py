import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
import cv2
import numpy as np

from src.app.dataset.services.downloader import DownloaderService
from src.app.domain.exceptions import DownloadError, InvalidImageError


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


@pytest.mark.asyncio
async def test_downloader_success(valid_slab_image):
    # Encode synthetic valid slab image to JPEG bytes
    success, img_bytes = cv2.imencode(".jpg", valid_slab_image)
    assert success
    img_bytes = img_bytes.tobytes()

    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.aread = AsyncMock(return_value=img_bytes)

    # Mock client and its stream context manager
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))
    
    # Mock client context manager
    client_manager = MockAsyncContextManager(mock_client)

    service = DownloaderService(timeout_seconds=2, retry_count=1)

    with patch("httpx.AsyncClient", return_value=client_manager):
        data = await service.download("http://example.com/stone.jpg")
        assert data == img_bytes


@pytest.mark.asyncio
async def test_downloader_reject_non_image():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.aread = AsyncMock(return_value=b"<html>Page</html>")

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))
    client_manager = MockAsyncContextManager(mock_client)

    service = DownloaderService(timeout_seconds=2, retry_count=1)

    with patch("httpx.AsyncClient", return_value=client_manager):
        with pytest.raises(InvalidImageError) as exc_info:
            await service.download("http://example.com/index.html")
        assert "Rejecting non-image Content-Type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_downloader_reject_corrupted_image(corrupted_image_bytes):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.aread = AsyncMock(return_value=corrupted_image_bytes)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=MockAsyncContextManager(mock_response))
    client_manager = MockAsyncContextManager(mock_client)

    service = DownloaderService(timeout_seconds=2, retry_count=1)

    with patch("httpx.AsyncClient", return_value=client_manager):
        with pytest.raises(InvalidImageError) as exc_info:
            await service.download("http://example.com/corrupt.jpg")
        assert "corrupted" in str(exc_info.value)


@pytest.mark.asyncio
async def test_downloader_retry_on_http_error():
    # Setup HTTPError on client creation
    client_manager = MockAsyncContextManager(None, side_effect=httpx.HTTPError("Connection failed"))

    service = DownloaderService(timeout_seconds=1, retry_count=2)

    with patch("httpx.AsyncClient", return_value=client_manager):
        # Patch sleep to avoid testing delay
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            with pytest.raises(DownloadError) as exc_info:
                await service.download("http://example.com/error.jpg")
            
            assert "attempts" in str(exc_info.value)
            # Retries exhausted
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)
