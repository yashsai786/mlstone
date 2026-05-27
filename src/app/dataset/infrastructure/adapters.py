import csv
import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
from src.app.dataset.application.ports import DatasetReaderPort, DatasetWriterPort
from src.app.dataset.domain.models import DatasetItem, IngestionResult
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LocalDatasetReader(DatasetReaderPort):
    """
    Local filesystem reader that recursively scans folder-structured URL txt files.
    """
    def __init__(self):
        # A simple, robust regex to check that the URL starts with http:// or https://
        self.url_regex = re.compile(
            r'^(?:http)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

    def read_dataset(self, base_path: str, single_color: Optional[str] = None) -> List[DatasetItem]:
        base_dir = Path(base_path)
        if not base_dir.exists():
            logger.warning("Dataset base directory does not exist", extra={"path": base_path})
            return []

        items = []
        seen_urls = set()

        # Traverse directories recursively
        for path in sorted(base_dir.rglob("*.txt")):
            # Category name is extracted from the immediate parent folder name
            folder_name = path.parent.name
            color_class = folder_name.lower().strip()

            # Filter single color if requested
            if single_color and color_class != single_color.lower().strip():
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        url = line.strip()
                        if not url:
                            continue

                        # Validate URL format before processing
                        if not self.url_regex.match(url):
                            logger.warning(
                                "Invalid URL structure skipped",
                                extra={"url": url, "file": path.name}
                            )
                            continue

                        # Deduplicate globally
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                        items.append(DatasetItem(
                            url=url,
                            color_class=color_class,
                            source_file=str(path)
                        ))
            except Exception as e:
                logger.error(
                    "Error reading dataset URL file",
                    extra={"file": str(path), "error": str(e)}
                )

        logger.info(
            "Dataset reading completed successfully",
            extra={"total_loaded": len(items), "single_color_filter": single_color}
        )
        return items


class LocalDatasetWriter(DatasetWriterPort):
    """
    Local filesystem writer managing processed dataset category folders,
    the metadata CSV registry, and the failed metadata storage directory.
    """
    def __init__(self, processed_base_dir: Path, failed_base_dir: Path, metadata_path: Path):
        self.processed_base_dir = Path(processed_base_dir)
        self.failed_base_dir = Path(failed_base_dir)
        self.metadata_path = Path(metadata_path)

        # Initialize directories
        self.processed_base_dir.mkdir(parents=True, exist_ok=True)
        self.failed_base_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata headers if it's a new CSV registry file
        if not self.metadata_path.exists():
            self._write_headers()

    def _write_headers(self):
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_id",
                    "source_url",
                    "color_class",
                    "local_path",
                    "crop_success",
                    "failure_reason",
                    "timestamp"
                ])
            logger.info("Initialized new metadata CSV file successfully", extra={"path": str(self.metadata_path)})
        except Exception as e:
            logger.error("Failed to initialize metadata.csv headers", extra={"error": str(e)})

    def save_processed_image(self, color_class: str, filename: str, image: np.ndarray) -> str:
        color_dir = self.processed_base_dir / color_class.lower()
        color_dir.mkdir(parents=True, exist_ok=True)

        # Always save as PNG to support high-quality Alpha composites
        if not filename.endswith(".png"):
            filename = f"{Path(filename).stem}.png"

        file_path = color_dir / filename
        success = cv2.imwrite(str(file_path), image)
        if not success:
            logger.error("cv2.imwrite returned False during save", extra={"path": str(file_path)})
            raise IOError(f"Failed to write cropped image to '{file_path}'")
        return str(file_path)

    def save_failed_metadata(self, item: DatasetItem, reason: str, raw_data: Optional[bytes] = None) -> None:
        url_hash = hashlib.sha256(item.url.encode("utf-8")).hexdigest()
        
        # Save failure details to error text file
        log_path = self.failed_base_dir / f"{url_hash}_error.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"URL: {item.url}\n")
                f.write(f"Category: {item.color_class}\n")
                f.write(f"Source file: {item.source_file}\n")
                f.write(f"Failure Reason: {reason}\n")
            
            # Save raw bytes if available
            if raw_data:
                ext = Path(item.url).suffix.lower()
                if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                    ext = ".jpg"
                raw_path = self.failed_base_dir / f"{url_hash}_raw{ext}"
                with open(raw_path, "wb") as f:
                    f.write(raw_data)
        except Exception as e:
            logger.error(
                "Failed to save failure metadata on disk",
                extra={"url": item.url, "error": str(e)}
            )

    def write_metadata_row(self, result: IngestionResult) -> None:
        try:
            with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.image_id,
                    result.source_url,
                    result.color_class,
                    result.local_path or "",
                    str(result.crop_success).upper(),
                    result.failure_reason or "",
                    result.timestamp.isoformat()
                ])
        except Exception as e:
            logger.error("Failed to append ingestion metadata row", extra={"error": str(e)})

    def get_processed_urls(self) -> Dict[str, bool]:
        processed = {}
        if not self.metadata_path.exists():
            return processed
        try:
            with open(self.metadata_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = row.get("source_url")
                    success = row.get("crop_success", "").upper() == "TRUE"
                    if url:
                        processed[url] = success
        except Exception as e:
            logger.error("Failed to read metadata CSV registry file", extra={"error": str(e)})
        return processed
