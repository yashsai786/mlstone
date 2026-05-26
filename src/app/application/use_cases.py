import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from src.app.domain.models import StoneImage, SlabRegion
from src.app.domain.exceptions import SlabDetectionError, InvalidImageError
from src.app.application.ports import ImageDownloaderPort, FileStoragePort, SlabDetectorPort
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DownloadImageUseCase:
    """
    Use case to download an image from a URL, validate its format/corruptness, and save it raw.
    """
    def __init__(self, downloader: ImageDownloaderPort, storage: FileStoragePort):
        self.downloader = downloader
        self.storage = storage

    async def execute(self, url: str, request_id: Optional[str] = None) -> StoneImage:
        # Download raw bytes
        image_bytes = await self.downloader.download(url)

        # Convert bytes to numpy array to validate
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if image is None or image.size == 0:
            logger.error("Failed to decode downloaded image", extra={"url": url})
            raise InvalidImageError("Downloaded image is corrupted or in an unsupported format.")

        h, w = image.shape[:2]

        # Extract filename from URL or default to slab.jpg
        parsed_name = Path(url).name
        if not parsed_name or "." not in parsed_name:
            parsed_name = "slab.jpg"
        
        raw_path = self.storage.save_raw(parsed_name, image_bytes)

        return StoneImage(
            original_url=url,
            width=w,
            height=h,
            raw_path=raw_path
        )


class ExtractSlabUseCase:
    """
    Use case to analyze the stone image, detect the main slab, and crop it.
    """
    def __init__(self, detector: SlabDetectorPort, storage: FileStoragePort):
        self.detector = detector
        self.storage = storage

    def execute(self, stone_image: StoneImage, debug_mode: bool = False, request_id: Optional[str] = None) -> SlabRegion:
        if not stone_image.raw_path:
            raise SlabDetectionError("Raw image path is missing on StoneImage.")

        # Load image from disk
        image = cv2.imread(stone_image.raw_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise InvalidImageError(f"Could not load raw image from path '{stone_image.raw_path}'")

        # Detect slabs
        regions = self.detector.detect_slabs(image)
        if not regions:
            logger.error("No stone slabs detected in the image", extra={"path": stone_image.raw_path})
            raise SlabDetectionError("No stone slabs detected in the image. Try adjusting parameters or lighting.")

        # Prioritize the highest confidence / largest slab (first element)
        main_slab = regions[0]
        stone_image.regions = regions

        # Save intermediate pipeline debug images if request_id is present (mandatory for CV debugging)
        if request_id:
            try:
                # 1. Save original image
                self.storage.save_pipeline_debug(request_id, "original.jpg", image)
                
                # 2. Save all intermediate pipeline steps
                if hasattr(self.detector, "debug_images"):
                    debug_imgs = getattr(self.detector, "debug_images")
                    for step_name, img in debug_imgs.items():
                        self.storage.save_pipeline_debug(request_id, f"{step_name}.jpg", img)
            except Exception as e:
                logger.warning("Failed to save intermediate pipeline debug outputs", extra={"request_id": request_id, "error": str(e)})

        # Optional debug visualization
        if debug_mode:
            self._save_debug_visualization(image, regions, stone_image.raw_path)

        return main_slab

    def _save_debug_visualization(self, image: np.ndarray, regions: list, raw_path: str):
        try:
            debug_img = image.copy()
            for i, r in enumerate(regions):
                color = (0, 255, 0) if i == 0 else (0, 0, 255) # Green for selected, red for others
                
                # Draw outer contour
                contour_arr = np.array(r.contour, dtype=np.int32).reshape((-1, 1, 2))
                cv2.drawContours(debug_img, [contour_arr], -1, color, 3)

                # Draw bounding box
                bx, by, bw, bh = r.bounding_box.to_tuple()
                cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                
                # Text annotation
                label = f"Slab {i+1}: Conf {r.confidence:.2f}"
                cv2.putText(debug_img, label, (bx, max(30, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            filename = Path(raw_path).name
            self.storage.save_debug(f"debug_{filename}", debug_img)
        except Exception as e:
            logger.warning("Failed to save debug visualization", extra={"error": str(e)})


class CleanupBackgroundUseCase:
    """
    Use case to remove straps, holders, and background outside the slab boundaries.
    Applies the mask and crops/unwarps the slab, saving it to disk.
    """
    def __init__(self, detector: SlabDetectorPort, storage: FileStoragePort):
        self.detector = detector
        self.storage = storage

    def execute(self, stone_image: StoneImage, slab_region: SlabRegion) -> str:
        if not stone_image.raw_path:
            raise SlabDetectionError("Raw image path is missing on StoneImage.")

        # Load original image
        image = cv2.imread(stone_image.raw_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise InvalidImageError(f"Could not load raw image from path '{stone_image.raw_path}'")

        # Cleanup background (generates RGBA image where outside of contour is transparent)
        cleaned_image = self.detector.remove_background(image, slab_region)

        # Save cropped and cleaned image
        filename = Path(stone_image.raw_path).name
        cropped_path = self.storage.save_cropped(filename, cleaned_image)

        stone_image.cropped_path = cropped_path
        return cropped_path
