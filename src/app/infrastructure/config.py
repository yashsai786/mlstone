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

    def __post_init__(self):
        # Create directories if they do not exist
        try:
            self.storage_base_dir.mkdir(parents=True, exist_ok=True)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.cropped_dir.mkdir(parents=True, exist_ok=True)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_outputs_dir.mkdir(parents=True, exist_ok=True)
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


# Global configuration instance
_config = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
