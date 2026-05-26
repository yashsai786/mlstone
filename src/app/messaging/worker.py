import asyncio
import sys
import uuid
from src.app.infrastructure.config import get_config
from src.app.infrastructure.logging import setup_logging, get_logger
from src.app.infrastructure.downloader import HTTPImageDownloader
from src.app.infrastructure.persistence import LocalFileStorage
from src.app.preprocessing.pipeline import OpenCVPipeline
from src.app.application.use_cases import DownloadImageUseCase, ExtractSlabUseCase, CleanupBackgroundUseCase
from src.app.domain.exceptions import StoneColorAppException
from src.app.messaging.connection import ZMQWorker

logger = get_logger(__name__)


class ProcessingWorker:
    """
    Worker orchestrator that coordinates dependencies and processes preprocessing requests.
    """
    def __init__(self):
        self.config = get_config()
        
        # Setup infrastructure adapters
        self.downloader = HTTPImageDownloader(
            timeout_seconds=self.config.download_timeout_seconds,
            max_size_bytes=self.config.max_download_size_bytes
        )
        self.storage = LocalFileStorage(
            raw_dir=self.config.raw_dir,
            cropped_dir=self.config.cropped_dir,
            debug_dir=self.config.debug_dir,
            debug_outputs_dir=self.config.debug_outputs_dir
        )
        self.detector = OpenCVPipeline(
            min_slab_area_ratio=self.config.min_slab_area_ratio
        )

        # Setup application use cases
        self.download_use_case = DownloadImageUseCase(self.downloader, self.storage)
        self.extract_use_case = ExtractSlabUseCase(self.detector, self.storage)
        self.cleanup_use_case = CleanupBackgroundUseCase(self.detector, self.storage)

        self.zmq_worker = ZMQWorker(
            broker_url=self.config.zmq_broker_url,
            request_handler=self.handle_request
        )

    def handle_request(self, payload: dict) -> dict:
        """
        Synchronous handler running in thread pool.
        Executes the processing pipeline for a single image URL.
        """
        url = payload.get("image_url")
        if not url:
            return {"success": False, "error": "Missing 'image_url' in request payload."}

        request_id = payload.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info("Processing request received in worker", extra={"url": url, "request_id": request_id})

        try:
            # 1. Download and validate image (run async function in thread loop)
            # Create a clean loop for this worker thread to handle async downloader
            loop = asyncio.new_event_loop()
            try:
                stone_image = loop.run_until_complete(self.download_use_case.execute(url, request_id))
            finally:
                loop.close()

            # 2. Detect and extract slab region
            slab_region = self.extract_use_case.execute(
                stone_image=stone_image,
                debug_mode=self.config.debug_mode,
                request_id=request_id
            )

            # 3. Cleanup background (remove straps, holders, etc.) and crop
            cropped_path = self.cleanup_use_case.execute(stone_image, slab_region)

            logger.info("Successfully processed stone image", extra={
                "url": url,
                "request_id": request_id,
                "cropped_path": cropped_path,
                "width": stone_image.width,
                "height": stone_image.height
            })

            return {
                "success": True,
                "cropped_image_path": cropped_path,
                "request_id": request_id,
                "metadata": {
                    "width": stone_image.width,
                    "height": stone_image.height,
                    "confidence": float(f"{slab_region.confidence:.3f}"),
                    "is_rotated": slab_region.is_rotated,
                    "rotation_angle": float(f"{slab_region.rotation_angle:.2f}"),
                    "rotation_corrected": True,
                    "detected_angle": float(f"{slab_region.rotation_angle:.2f}"),
                    "contour_area_ratio": float(f"{slab_region.contour_area_ratio:.3f}")
                }
            }

        except StoneColorAppException as e:
            logger.error("Application error during slab extraction", extra={"url": url, "request_id": request_id, "error": str(e)})
            return {"success": False, "error": str(e), "request_id": request_id}
        except Exception as e:
            logger.error("Unhandled error during slab extraction", extra={"url": url, "request_id": request_id, "error": str(e)})
            return {"success": False, "error": f"Internal server error: {e}", "request_id": request_id}

    async def start(self):
        await self.zmq_worker.start()

    def stop(self):
        self.zmq_worker.stop()


if __name__ == "__main__":
    setup_logging()
    worker = ProcessingWorker()
    
    logger.info("Starting Stone Slab Preprocessing Worker...")
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user.")
    finally:
        worker.stop()
        logger.info("Worker shutdown complete.")
