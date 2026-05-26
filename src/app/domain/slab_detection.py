from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.app.domain.value_objects import BoundingBox, DetectionConfidence, CropMetadata


@dataclass(frozen=True)
class SlabDetectionResult:
    bounding_box: BoundingBox
    confidence: DetectionConfidence
    crop_metadata: CropMetadata
    contour: List[Tuple[int, int]]
