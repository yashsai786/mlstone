from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.app.ml.domain.models import EvaluationReport, TrainingMetadata, StoneClassification


class ModelStoragePort(ABC):
    """
    Port for serializing and loading trained models, weights, and configurations.
    """
    @abstractmethod
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        metadata: TrainingMetadata,
        path: str
    ) -> None:
        """Saves a model checkpoint atomically."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Tuple[Dict[str, Any], TrainingMetadata]:
        """Loads a model checkpoint from disk."""
        pass


class EvaluationReporterPort(ABC):
    """
    Port for generating human-readable and machine-parseable metrics and reports.
    """
    @abstractmethod
    def generate_report(
        self,
        y_true: List[int],
        y_pred: List[int],
        probs: np.ndarray,
        classes: List[str]
    ) -> EvaluationReport:
        """Generates a complete metrics report including class mappings and matrix."""
        pass

    @abstractmethod
    def save_report(self, report: EvaluationReport, path: str) -> str:
        """Saves the evaluation report to disk."""
        pass
