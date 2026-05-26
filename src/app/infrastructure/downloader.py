import httpx
from urllib.parse import urlparse
from src.app.application.ports import ImageDownloaderPort
from src.app.domain.exceptions import DownloadError, InvalidImageError
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class HTTPImageDownloader(ImageDownloaderPort):
    """
    Production-grade HTTP Image Downloader using httpx.
    """
    def __init__(self, timeout_seconds: int = 10, max_size_bytes: int = 10 * 1024 * 1024):
        self.timeout_seconds = timeout_seconds
        self.max_size_bytes = max_size_bytes

    async def download(self, url: str) -> bytes:
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error("Invalid URL format", extra={"url": url})
            raise DownloadError(f"Invalid URL format: '{url}'")

        try:
            logger.info("Downloading image starting...", extra={"url": url})
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                # Use stream to check content length and type before loading the whole body into memory
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        logger.error(
                            "Failed to download image: Non-200 status",
                            extra={"url": url, "status_code": response.status_code}
                        )
                        raise DownloadError(f"HTTP download failed with status {response.status_code}")

                    # Check Content-Type
                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        logger.error(
                            "Failed to download image: Invalid content type",
                            extra={"url": url, "content_type": content_type}
                        )
                        raise InvalidImageError(f"URL did not point to an image. Content-Type: '{content_type}'")

                    # Check Content-Length if available
                    content_length_str = response.headers.get("content-length")
                    if content_length_str:
                        try:
                            content_length = int(content_length_str)
                            if content_length > self.max_size_bytes:
                                logger.error(
                                    "Failed to download image: File exceeds size limit",
                                    extra={"url": url, "size": content_length, "limit": self.max_size_bytes}
                                )
                                raise DownloadError(f"Image size exceeds maximum limit of {self.max_size_bytes} bytes")
                        except ValueError:
                            pass

                    # Read body
                    data = await response.aread()
                    
                    if len(data) > self.max_size_bytes:
                        logger.error(
                            "Failed to download image: Content body exceeds size limit",
                            extra={"url": url, "size": len(data), "limit": self.max_size_bytes}
                        )
                        raise DownloadError(f"Downloaded image size exceeds maximum limit of {self.max_size_bytes} bytes")

                    if not data:
                        logger.error("Downloaded empty image data", extra={"url": url})
                        raise InvalidImageError("Downloaded image data is empty.")

                    logger.info("Image downloaded successfully", extra={"url": url, "bytes": len(data)})
                    return data

        except httpx.HTTPError as e:
            logger.error("HTTP error downloading image", extra={"url": url, "error": str(e)})
            raise DownloadError(f"HTTP error downloading image: {e}")
        except Exception as e:
            if isinstance(e, (DownloadError, InvalidImageError)):
                raise e
            logger.error("Unexpected error during image download", extra={"url": url, "error": str(e)})
            raise DownloadError(f"Unexpected error during download: {e}")
