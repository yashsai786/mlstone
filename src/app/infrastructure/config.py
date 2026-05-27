import os
from pathlib import Path
from dataclasses import dataclass, field
from src.app.domain.exceptions import ConfigurationError


@dataclass
class AppConfig:
    # API Settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Messaging (ZeroMQ) Settings
    zmq_broker_url: str = field(default_factory=lambda: os.getenv("ZMQ_BROKER_URL", "tcp://127.0.0.1:5555"))
    zmq_response_timeout_ms: int = field(default_factory=lambda: int(os.getenv("ZMQ_RESPONSE_TIMEOUT_MS", "15000")))

    # Storage Settings
    storage_base_dir: Path = field(
        default_factory=lambda: Path(os.getenv("STORAGE_BASE_DIR", "./storage"))
    )

    # Image Downloader Settings
    max_download_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_DOWNLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10MB
    )
    download_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "10"))
    )

    # Preprocessing Settings
    min_slab_area_ratio: float = field(
        default_factory=lambda: float(os.getenv("MIN_SLAB_AREA_RATIO", "0.05"))  # At least 5% of the image
    )
    debug_mode: bool = field(
        default_factory=lambda: os.getenv("DEBUG_MODE", "True").lower() in ("true", "1", "yes")
    )

    # Dataset Ingestion Settings
    dataset_base_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATASET_BASE_DIR", "./dataset"))
    )
    processed_dataset_dir: Path = field(
        default_factory=lambda: Path(os.getenv("PROCESSED_DATASET_DIR", "./processed_dataset"))
    )
    dataset_concurrency: int = field(
        default_factory=lambda: int(os.getenv("DATASET_CONCURRENCY", "4"))
    )
    output_image_size: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_IMAGE_SIZE", "224"))
    )
    download_retry_count: int = field(
        default_factory=lambda: int(os.getenv("DOWNLOAD_RETRY_COUNT", "3"))
    )

    # ML Bounded Context Settings
    ml_model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ML_MODEL_DIR", "./models"))
    )
    ml_reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ML_REPORTS_DIR", "./reports"))
    )
    ml_batch_size: int = field(
        default_factory=lambda: int(os.getenv("ML_BATCH_SIZE", "32"))
    )
    ml_epochs: int = field(
        default_factory=lambda: int(os.getenv("ML_EPOCHS", "10"))
    )
    ml_learning_rate: float = field(
        default_factory=lambda: float(os.getenv("ML_LEARNING_RATE", "0.001"))
    )
    ml_dropout: float = field(
        default_factory=lambda: float(os.getenv("ML_DROPOUT", "0.3"))
    )
    ml_freeze_epochs: int = field(
        default_factory=lambda: int(os.getenv("ML_FREEZE_EPOCHS", "2"))
    )
    ml_early_stopping_patience: int = field(
        default_factory=lambda: int(os.getenv("ML_EARLY_STOPPING_PATIENCE", "3"))
    )

    def __post_init__(self):
        # Create directories if they do not exist
        try:
            self.storage_base_dir.mkdir(parents=True, exist_ok=True)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.cropped_dir.mkdir(parents=True, exist_ok=True)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_outputs_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_base_dir.mkdir(parents=True, exist_ok=True)
            self.processed_dataset_dir.mkdir(parents=True, exist_ok=True)
            self.failed_dir.mkdir(parents=True, exist_ok=True)
            self.ml_model_dir.mkdir(parents=True, exist_ok=True)
            self.ml_reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigurationError(f"Failed to create storage directories: {e}")

    @property
    def raw_dir(self) -> Path:
        return self.storage_base_dir / "raw"

    @property
    def cropped_dir(self) -> Path:
        return self.storage_base_dir / "cropped"

    @property
    def debug_dir(self) -> Path:
        return self.storage_base_dir / "debug"

    @property
    def debug_outputs_dir(self) -> Path:
        return self.storage_base_dir / "debug_outputs"

    @property
    def failed_dir(self) -> Path:
        return self.dataset_base_dir / "failed"

    @property
    def metadata_csv_path(self) -> Path:
        return self.dataset_base_dir / "metadata.csv"


# Global configuration instance
_config = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
