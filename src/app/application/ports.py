from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np
from src.app.domain.models import SlabRegion, StoneImage


class ImageDownloaderPort(ABC):
    """Port for downloading images from a URL."""
    @abstractmethod
    async def download(self, url: str) -> bytes:
        """
        Downloads an image as bytes.
        Raises DownloadError or InvalidImageError on failure.
        """
        pass


class FileStoragePort(ABC):
    """Port for saving and retrieving image files on the filesystem."""
    @abstractmethod
    def save_raw(self, filename: str, data: bytes) -> str:
        """Saves raw image bytes and returns the file path."""
        pass

    @abstractmethod
    def save_cropped(self, filename: str, image: np.ndarray) -> str:
        """Saves cropped slab image and returns the file path."""
        pass

    @abstractmethod
    def save_debug(self, filename: str, image: np.ndarray) -> str:
        """Saves intermediate debug images and returns the file path."""
        pass

    @abstractmethod
    def save_pipeline_debug(self, request_id: str, step_name: str, image: np.ndarray) -> str:
        """
        Saves intermediate pipeline debug images in debug_outputs/<request_id>/<step_name>.jpg.
        Returns the saved file path.
        """
        pass


class SlabDetectorPort(ABC):
    """Port for stone slab detection and cropping."""
    @abstractmethod
    def detect_slabs(self, image: np.ndarray) -> List[SlabRegion]:
        """
        Analyzes the image and returns a list of detected SlabRegion objects.
        """
        pass

    @abstractmethod
    def crop_slab(self, image: np.ndarray, region: SlabRegion) -> np.ndarray:
        """
        Crops and warps (if rotated) the slab from the original image.
        """
        pass

    @abstractmethod
    def remove_background(self, image: np.ndarray, region: SlabRegion) -> np.ndarray:
        """
        Removes background outside the slab boundaries.
        Returns RGBA image where background is transparent.
        """
        pass
