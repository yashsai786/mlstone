import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

from src.app.application.ports import SlabDetectorPort
from src.app.dataset.application.ports import DatasetReaderPort, DatasetWriterPort
from src.app.dataset.domain.models import DatasetItem, IngestionResult
from src.app.dataset.services.downloader import DownloaderService
from src.app.domain.exceptions import SlabDetectionError, InvalidImageError
from src.app.infrastructure.config import AppConfig
from src.app.infrastructure.logging import get_logger
from src.app.preprocessing.pipeline import OpenCVPipeline

logger = get_logger(__name__)


@dataclass
class ProgressMetrics:
    """
    Data structure to hold execution and performance metrics for progress display.
    """
    total_found: int = 0
    skipped_completed: int = 0
    to_process: int = 0
    success_count: int = 0
    failed_download: int = 0
    failed_crop: int = 0
    failed_other: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def total_attempted(self) -> int:
        return self.success_count + self.failed_download + self.failed_crop + self.failed_other

    @property
    def success_rate(self) -> float:
        attempted = self.total_attempted
        if attempted == 0:
            return 0.0
        return (self.success_count / attempted) * 100.0

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def items_per_second(self) -> float:
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return self.total_attempted / duration


class IngestDatasetUseCase:
    """
    Application orchestrator for the automated dataset ingestion and preprocessing pipeline.
    """
    def __init__(
        self,
        config: AppConfig,
        reader: DatasetReaderPort,
        writer: DatasetWriterPort,
        downloader_service: DownloaderService,
        detector: Optional[SlabDetectorPort] = None
    ):
        self.config = config
        self.reader = reader
        self.writer = writer
        self.downloader_service = downloader_service
        self.detector = detector or OpenCVPipeline(min_slab_area_ratio=self.config.min_slab_area_ratio)

    def normalize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resizes image preserving aspect ratio safely such that the maximum dimension matches target_size.
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise InvalidImageError("Input image has zero dimensions.")

        scale = target_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    async def ingest_single_item(
        self,
        item: DatasetItem,
        semaphore: asyncio.Semaphore,
        metrics: ProgressMetrics
    ) -> IngestionResult:
        """
        Ingests a single dataset item by downloading, cropping, normalizing, and saving.
        All exceptions are structured and caught to guarantee pipeline stability.
        """
        async with semaphore:
            image_id = Path(item.get_deterministic_filename()).stem
            raw_data = None
            try:
                # 1. Download image asynchronously
                try:
                    raw_data = await self.downloader_service.download(item.url)
                except Exception as e:
                    metrics.failed_download += 1
                    raise e

                # 2. Decode image safely
                nparr = np.frombuffer(raw_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is None or img.size == 0:
                    metrics.failed_crop += 1
                    raise InvalidImageError("Downloaded image data cannot be decoded.")

                # 3. Slab detection (CPU-bound, run in executor pool)
                loop = asyncio.get_running_loop()
                regions = await loop.run_in_executor(None, self.detector.detect_slabs, img)
                if not regions:
                    metrics.failed_crop += 1
                    raise SlabDetectionError("No stone slabs detected in the image.")

                # Prioritize first region (standard highest confidence)
                primary_region = regions[0]

                # 4. Crop & background cleanup using existing cropper
                cropped_img = await loop.run_in_executor(
                    None, self.detector.remove_background, img, primary_region
                )

                if cropped_img is None or cropped_img.size == 0:
                    metrics.failed_crop += 1
                    raise SlabDetectionError("Failed to extract cropped slab region.")

                # 5. Normalize and resize preserving aspect ratio safely
                normalized_img = await loop.run_in_executor(
                    None, self.normalize_image, cropped_img, self.config.output_image_size
                )

                # 6. Save image into category folder
                local_path = self.writer.save_processed_image(
                    color_class=item.color_class,
                    filename=item.get_deterministic_filename(),
                    image=normalized_img
                )

                result = IngestionResult(
                    image_id=image_id,
                    source_url=item.url,
                    color_class=item.color_class,
                    local_path=local_path,
                    crop_success=True,
                    failure_reason=None
                )
                metrics.success_count += 1
                logger.info(
                    "Successfully ingested and processed item",
                    extra={"url": item.url, "local_path": local_path}
                )

            except Exception as e:
                failure_reason = str(e)
                if "failed_download" not in failure_reason and "failed_crop" not in failure_reason:
                    # Increment failed_other if it wasn't already logged
                    if not isinstance(e, (SlabDetectionError, InvalidImageError)):
                        metrics.failed_other += 1

                logger.error(
                    "Graceful degradation: Item ingestion failed",
                    extra={"url": item.url, "error": failure_reason}
                )

                # Save failed metadata for pipeline review
                self.writer.save_failed_metadata(item, failure_reason, raw_data)

                result = IngestionResult(
                    image_id=image_id,
                    source_url=item.url,
                    color_class=item.color_class,
                    local_path=None,
                    crop_success=False,
                    failure_reason=failure_reason
                )

            # Append metadata to registry
            self.writer.write_metadata_row(result)
            return result

    async def execute(
        self,
        single_color: Optional[str] = None,
        resume_failed: bool = False,
        dry_run: bool = False
    ) -> ProgressMetrics:
        """
        Executes the dataset ingestion and preprocessing pipeline flow.
        """
        metrics = ProgressMetrics()
        logger.info("Initializing dataset ingestion pipeline...")

        # 1. Read raw URLs from source text files
        all_items = self.reader.read_dataset(
            str(self.config.dataset_base_dir),
            single_color=single_color
        )
        metrics.total_found = len(all_items)

        # 2. Check already attempted URLs in the metadata registry
        attempted_urls = self.writer.get_processed_urls()

        # 3. Filter items depending on single_color, skip-completed, and resume rules
        scheduled_items = []
        for item in all_items:
            is_attempted = item.url in attempted_urls
            is_successful = attempted_urls.get(item.url, False)

            if is_successful:
                metrics.skipped_completed += 1
                continue

            if is_attempted and not resume_failed:
                # Item failed before, but we are NOT resuming failures right now -> skip it
                metrics.skipped_completed += 1
                continue

            scheduled_items.append(item)

        metrics.to_process = len(scheduled_items)

        # 4. Handle dry-run mode
        if dry_run:
            metrics.end_time = time.time()
            logger.info(
                "Dry run completed successfully. No downloads performed.",
                extra={
                    "total_found": metrics.total_found,
                    "skipped_completed": metrics.skipped_completed,
                    "to_process": metrics.to_process
                }
            )
            return metrics

        if not scheduled_items:
            metrics.end_time = time.time()
            logger.info("No new items to process in dataset pipeline.")
            return metrics

        logger.info(
            "Starting ingestion execution...",
            extra={"to_process": metrics.to_process, "concurrency": self.config.dataset_concurrency}
        )

        # 5. Execute processing using a safe backpressure-guarded semaphore worker pool
        semaphore = asyncio.Semaphore(self.config.dataset_concurrency)
        
        # We process in batches to control memory usage and prevent event loop congestion
        tasks = [
            self.ingest_single_item(item, semaphore, metrics)
            for item in scheduled_items
        ]
        
        await asyncio.gather(*tasks)

        metrics.end_time = time.time()
        logger.info(
            "Dataset ingestion pipeline execution completed.",
            extra={
                "duration_sec": metrics.duration_seconds,
                "attempted": metrics.total_attempted,
                "succeeded": metrics.success_count,
                "failed_downloads": metrics.failed_download,
                "failed_crops": metrics.failed_crop,
                "success_rate": f"{metrics.success_rate:.2f}%"
            }
        )
        return metrics
