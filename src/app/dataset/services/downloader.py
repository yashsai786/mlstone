import asyncio
import httpx
import cv2
import numpy as np
from typing import List, Optional
from src.app.domain.exceptions import DownloadError, InvalidImageError
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DownloaderService:
    """
    Asynchronous robust image downloader supporting exponential backoff,
    strict timeouts, content-type checks, corrupt-image validation,
    and streaming optimization.
    """
    def __init__(
        self,
        timeout_seconds: int = 10,
        retry_count: int = 3,
        allowed_mime_types: Optional[List[str]] = None
    ):
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.allowed_mime_types = allowed_mime_types or [
            "image/jpeg", "image/png", "image/webp", "image/jpg"
        ]
        # Standard browser User-Agent to bypass simple anti-scraping blocks
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    async def download(self, url: str) -> bytes:
        """
        Downloads raw image bytes asynchronously with exponential backoff retries.
        """
        headers = {"User-Agent": self.user_agent}
        attempt = 0
        backoff_factor = 1.0

        while True:
            try:
                logger.info("Initiating download attempt...", extra={"url": url, "attempt": attempt + 1})
                async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                    async with client.stream("GET", url, headers=headers) as response:
                        if response.status_code != 200:
                            raise DownloadError(f"HTTP error status: {response.status_code}")

                        # Content-Type Validation
                        content_type = response.headers.get("content-type", "").lower()
                        if not any(mime in content_type for mime in self.allowed_mime_types):
                            logger.error(
                                "Invalid Content-Type rejected",
                                extra={"url": url, "content_type": content_type}
                            )
                            raise InvalidImageError(
                                f"Rejecting non-image Content-Type: '{content_type}'"
                            )

                        # Load body stream
                        data = await response.aread()
                        if not data:
                            raise InvalidImageError("Downloaded empty image body.")

                        # Integrity & Corruption Check
                        nparr = np.frombuffer(data, np.uint8)
                        decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        if decoded is None or decoded.size == 0:
                            logger.error("Downloaded image data is corrupted or undecodable", extra={"url": url})
                            raise InvalidImageError("Downloaded image data is corrupted.")

                        logger.info("Image downloaded and verified successfully", extra={"url": url, "bytes_count": len(data)})
                        return data

            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                attempt += 1
                if attempt > self.retry_count:
                    logger.error(
                        "Download failed completely. Retries exhausted.",
                        extra={"url": url, "retries": self.retry_count, "error": str(e)}
                    )
                    raise DownloadError(f"Download failed after {self.retry_count} attempts: {e}")

                # Exponential backoff delay calculation
                sleep_duration = backoff_factor * (2 ** (attempt - 1))
                logger.warning(
                    "Temporary download failure. Retrying...",
                    extra={"url": url, "attempt": attempt, "sleep": sleep_duration, "error": str(e)}
                )
                await asyncio.sleep(sleep_duration)

            except Exception as e:
                # Do not retry on domain-specific validation issues
                if isinstance(e, (DownloadError, InvalidImageError)):
                    raise e
                
                attempt += 1
                if attempt > self.retry_count:
                    raise DownloadError(f"Unexpected error during download: {e}")

                sleep_duration = backoff_factor * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_duration)
