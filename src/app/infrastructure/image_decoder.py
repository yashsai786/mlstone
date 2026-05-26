import cv2
import numpy as np
from src.app.domain.exceptions import InvalidImageError


class ImageDecoder:
    """
    Infrastructure service for decoding raw image bytes safely into numpy arrays.
    """

    def decode(self, data: bytes) -> np.ndarray:
        if not data:
            raise InvalidImageError("Cannot decode empty image bytes.")
        try:
            image_bytes = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise InvalidImageError("OpenCV failed to decode the provided image bytes.")
            return img
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise e
            raise InvalidImageError(f"Error decoding image bytes: {e}")
