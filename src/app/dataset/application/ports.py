from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from src.app.dataset.domain.models import DatasetItem, IngestionResult


class DatasetReaderPort(ABC):
    """
    Port defining interface for reading raw dataset source files.
    """
    @abstractmethod
    def read_dataset(self, base_path: str, single_color: Optional[str] = None) -> List[DatasetItem]:
        """
        Reads, category-detects, validates and globally deduplicates dataset URLs.
        """
        pass


class DatasetWriterPort(ABC):
    """
    Port defining interface for storing processed ML-ready datasets and tracking metadata.
    """
    @abstractmethod
    def save_processed_image(self, color_class: str, filename: str, image: np.ndarray) -> str:
        """
        Saves a normalized processed image to the correct category subfolder.
        """
        pass

    @abstractmethod
    def save_failed_metadata(self, item: DatasetItem, reason: str, raw_data: Optional[bytes] = None) -> None:
        """
        Saves details and optionally raw bytes of failed ingestion pipeline items.
        """
        pass

    @abstractmethod
    def write_metadata_row(self, result: IngestionResult) -> None:
        """
        Appends an ingestion result row to metadata.csv.
        """
        pass

    @abstractmethod
    def get_processed_urls(self) -> Dict[str, bool]:
        """
        Returns a mapping of already attempted URLs to their success status.
        Allows for resuming and avoiding duplicate processing.
        """
        pass
