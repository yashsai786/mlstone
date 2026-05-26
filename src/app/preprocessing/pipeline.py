import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.app.domain.models import SlabRegion
from src.app.application.ports import SlabDetectorPort
from src.app.preprocessing.slab_detector import SlabDetector
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class OpenCVPipeline(SlabDetectorPort):
    """
    Orchestration adapter delegating stone slab preprocessing to the newly
    morphological-gradient based SlabDetector implementation.
    """

    def __init__(self, min_slab_area_ratio: float = 0.02):
        self.detector = SlabDetector()
        
    @property
    def debug_images(self) -> Dict[str, np.ndarray]:
        return self.detector.debug_images

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def detect_edges(self, image: np.ndarray, low_thresh: int = 30, high_thresh: int = 150) -> np.ndarray:
        return cv2.Canny(image, low_thresh, high_thresh)

    def extract_contours(self, edge_map: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def detect_slabs(self, image: np.ndarray) -> List[SlabRegion]:
        """
        Detects stone slabs using the morphology-gradient localization algorithm.
        """
        return self.detector.detect_slabs(image)

    def crop_slab(self, image: np.ndarray, region: SlabRegion) -> np.ndarray:
        """
        Crops stone slab safely without skewing or perspective transform.
        """
        return self.detector.crop_slab(image, region)

    def remove_background(self, image: np.ndarray, region: SlabRegion) -> np.ndarray:
        """
        Returns solid rectangular crop inside an opaque alpha channel.
        """
        return self.detector.remove_background(image, region)
