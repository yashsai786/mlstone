from typing import Optional
from src.app.application.ports import ImageDownloaderPort, FileStoragePort
from src.app.preprocessing.slab_detector import SlabDetector
from src.app.infrastructure.image_decoder import ImageDecoder
from src.app.domain.slab_detection import SlabDetectionResult
from src.app.domain.exceptions import SlabDetectionError


class SlabExtractionUseCase:
    """
    Application orchestrator for stone slab extraction.
    Downloads, decodes, preprocesses, localizes, crops, and persists stone slabs from a URL.
    Contains NO direct OpenCV / NumPy logic inside the use case itself.
    """

    def __init__(
        self,
        downloader: ImageDownloaderPort,
        decoder: ImageDecoder,
        detector: SlabDetector,
        storage: FileStoragePort
    ):
        self.downloader = downloader
        self.decoder = decoder
        self.detector = detector
        self.storage = storage

    async def execute(self, image_url: str, request_id: str) -> SlabDetectionResult:
        """
        Orchestrates downloading, decoding, morphology localization, cropping, and debug/crop persistence.
        """
        if not image_url:
            raise SlabDetectionError("Image URL must not be empty.")

        # 1. Download image bytes
        raw_bytes = await self.downloader.download(image_url)

        # 2. Decode image safely
        image = self.decoder.decode(raw_bytes)

        # 3. Save raw image
        raw_filename = f"{request_id}_original.jpg"
        self.storage.save_raw(raw_filename, raw_bytes)

        # 4. Run preprocessing pipeline & localize slab
        result = self.detector.detect(image)

        # 5. Crop slab region
        # Reconstruct SlabRegion wrapper to invoke port-compliant crop method
        from src.app.domain.models import SlabRegion
        dummy_region = SlabRegion(
            bounding_box=None, # type: ignore
            contour=result.contour,
            confidence=result.confidence.value,
            is_rotated=False,
            rotation_angle=0.0,
            polygon_points=result.contour
        )
        cropped_img = self.detector.crop_slab(image, dummy_region)

        # 6. Persist cropped slab
        cropped_filename = f"{request_id}_cropped.jpg"
        self.storage.save_cropped(cropped_filename, cropped_img)

        # 7. Persist pipeline debug images (original.jpg, gray.jpg, blurred.jpg, edges.jpg, morphology.jpg, contours.jpg)
        for step, debug_img in self.detector.debug_images.items():
            self.storage.save_pipeline_debug(request_id, step, debug_img)

        return result
