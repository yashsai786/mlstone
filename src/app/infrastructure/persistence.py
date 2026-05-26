import os
import uuid
import cv2
import numpy as np
from pathlib import Path
from src.app.application.ports import FileStoragePort
from src.app.domain.exceptions import InvalidImageError
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LocalFileStorage(FileStoragePort):
    """
    Local filesystem storage implementation for raw, cropped, and debug images.
    """
    def __init__(self, raw_dir: Path, cropped_dir: Path, debug_dir: Path, debug_outputs_dir: Path = None):
        self.raw_dir = Path(raw_dir)
        self.cropped_dir = Path(cropped_dir)
        self.debug_dir = Path(debug_dir)
        self.debug_outputs_dir = Path(debug_outputs_dir) if debug_outputs_dir else Path(debug_dir).parent / "debug_outputs"

    def save_pipeline_debug(self, request_id: str, step_name: str, image: np.ndarray) -> str:
        req_dir = self.debug_outputs_dir / request_id
        try:
            req_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure proper extension
            if not step_name.lower().endswith((".jpg", ".jpeg", ".png")):
                step_name = f"{step_name}.jpg"
                
            file_path = req_dir / step_name
            success = cv2.imwrite(str(file_path), image)
            if not success:
                raise IOError(f"cv2.imwrite returned False for path {file_path}")
            logger.info("Saved pipeline debug image successfully", extra={"path": str(file_path)})
            return str(file_path)
        except Exception as e:
            logger.error("Failed to save pipeline debug image", extra={"request_id": request_id, "step_name": step_name, "error": str(e)})
            raise IOError(f"Failed to save pipeline debug file: {e}")

    def save_raw(self, filename: str, data: bytes) -> str:
        unique_name = f"{uuid.uuid4()}_{filename}"
        file_path = self.raw_dir / unique_name
        try:
            with open(file_path, "wb") as f:
                f.write(data)
            logger.info("Saved raw image successfully", extra={"path": str(file_path)})
            return str(file_path)
        except Exception as e:
            logger.error("Failed to save raw image to disk", extra={"filename": filename, "error": str(e)})
            raise IOError(f"Failed to save raw file: {e}")

    def save_cropped(self, filename: str, image: np.ndarray) -> str:
        unique_name = f"{uuid.uuid4()}_{filename}"
        # Force saving cropped image as .png to preserve alpha channel
        if not unique_name.lower().endswith(".png"):
            # Replace extension or append .png
            path_obj = Path(unique_name)
            unique_name = f"{path_obj.stem}.png"

        file_path = self.cropped_dir / unique_name
        try:
            # cv2.imwrite returns False if failed
            success = cv2.imwrite(str(file_path), image)
            if not success:
                raise IOError(f"cv2.imwrite returned False for path {file_path}")
            logger.info("Saved cropped image successfully", extra={"path": str(file_path)})
            return str(file_path)
        except Exception as e:
            logger.error("Failed to save cropped image to disk", extra={"filename": filename, "error": str(e)})
            raise IOError(f"Failed to save cropped file: {e}")

    def save_debug(self, filename: str, image: np.ndarray) -> str:
        unique_name = f"{uuid.uuid4()}_{filename}"
        file_path = self.debug_dir / unique_name
        try:
            success = cv2.imwrite(str(file_path), image)
            if not success:
                raise IOError(f"cv2.imwrite returned False for path {file_path}")
            logger.info("Saved debug image successfully", extra={"path": str(file_path)})
            return str(file_path)
        except Exception as e:
            logger.error("Failed to save debug image to disk", extra={"filename": filename, "error": str(e)})
            raise IOError(f"Failed to save debug file: {e}")

    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """
        Loads an image from the filesystem into a numpy array.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: '{file_path}'")
        
        # Using cv2.imread
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise InvalidImageError(f"Failed to decode image at path '{file_path}' or image is empty.")
        return image
