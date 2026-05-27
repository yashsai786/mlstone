import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetItem:
    """
    Pure domain model representing an item in the dataset to be ingested.
    Contains zero framework or infrastructure dependencies.
    """
    url: str
    color_class: str
    source_file: str

    def get_deterministic_filename(self) -> str:
        """
        Generates a deterministic filename by hashing the source URL.
        Preserves original extension suffix if it's a known image format,
        otherwise defaults to .png to preserve high quality cropped alpha channel outputs.
        """
        ext = Path(self.url).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = ".png"
        
        # Calculate SHA-256 hash of URL for deterministic filename
        url_hash = hashlib.sha256(self.url.encode("utf-8")).hexdigest()
        return f"{url_hash}{ext}"


@dataclass(frozen=True)
class IngestionResult:
    """
    Pure domain model representing the result of an ingestion operation.
    """
    image_id: str
    source_url: str
    color_class: str
    local_path: Optional[str]
    crop_success: bool
    failure_reason: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
