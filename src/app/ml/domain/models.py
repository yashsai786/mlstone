from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StoneClassification:
    """
    Pure domain model representing the result of stone color inference.
    Contains no dependencies on PyTorch, PIL, or any deep learning libraries.
    """
    predicted_class: str
    confidence: float
    top_k: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: Optional[float] = None
    device_used: str = "cpu"


@dataclass(frozen=True)
class EvaluationReport:
    """
    Pure domain model representing the overall metrics after model validation/testing.
    """
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass(frozen=True)
class TrainingMetadata:
    """
    Pure domain model containing checkpoint and run-level configurations,
    metrics, and class indexing for model serialization traceability.
    """
    best_accuracy: float
    best_loss: float
    epochs_trained: int
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    model_version: str = "1.0.0"
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass(frozen=True)
class PredictionRequest:
    """
    Pure domain model representing the input request structure.
    """
    image_url: str


@dataclass(frozen=True)
class PredictionCandidate:
    """
    Pure domain model representing a single prediction probability candidate.
    """
    class_name: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """
    Pure domain model representing the comprehensive prediction output.
    """
    predicted_color: str
    confidence: float
    top_predictions: List[PredictionCandidate]
    processing_time_ms: float
    model_version: Optional[str] = None
