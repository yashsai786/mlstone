from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height


@dataclass(frozen=True)
class DetectionConfidence:
    value: float

    def __post_init__(self):
        object.__setattr__(self, 'value', max(0.0, min(1.0, float(self.value))))

    def is_low(self) -> bool:
        return self.value < 0.35


@dataclass(frozen=True)
class CropMetadata:
    original_width: int
    original_height: int
    cropped_width: int
    cropped_height: int
    is_fallback: bool
    confidence: float
    warning: Optional[str] = None
