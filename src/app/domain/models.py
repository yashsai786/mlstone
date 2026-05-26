from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


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

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True)
class DetectionConfidence:
    value: float

    def __post_init__(self):
        # Clip confidence between 0.0 and 1.0 to guarantee safety
        object.__setattr__(self, 'value', max(0.0, min(1.0, float(self.value))))

    def is_low(self) -> bool:
        return self.value < 0.35

    def is_high(self) -> bool:
        return self.value >= 0.70


@dataclass(frozen=True)
class RectangleCandidate:
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle: float
    score: float
    contour_area: float


@dataclass(frozen=True)
class PerspectiveTransformData:
    source_points: List[Tuple[float, float]]
    destination_points: List[Tuple[float, float]]
    transform_matrix: List[List[float]]


@dataclass
class SlabRegion:
    bounding_box: BoundingBox
    contour: List[Tuple[int, int]]
    confidence: float
    is_rotated: bool = False
    rotation_angle: float = 0.0
    polygon_points: List[Tuple[int, int]] = field(default_factory=list)
    contour_area_ratio: float = 0.0


@dataclass(frozen=True)
class SlabDetectionResult:
    bounding_box: BoundingBox
    contour: List[Tuple[int, int]]
    confidence: DetectionConfidence
    is_rotated: bool
    rotation_angle: float
    polygon_points: List[Tuple[int, int]]
    contour_area_ratio: float
    candidate: RectangleCandidate
    transform_data: Optional[PerspectiveTransformData] = None
    warning: Optional[str] = None


@dataclass
class StoneImage:
    original_url: str
    width: int
    height: int
    raw_path: Optional[str] = None
    cropped_path: Optional[str] = None
    regions: List[SlabRegion] = field(default_factory=list)

    @property
    def processed(self) -> bool:
        return self.cropped_path is not None
