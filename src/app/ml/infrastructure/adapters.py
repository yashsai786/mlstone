import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.app.ml.application.ports import ModelStoragePort, EvaluationReporterPort
from src.app.ml.domain.models import EvaluationReport, TrainingMetadata
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class PyTorchModelStorage(ModelStoragePort):
    """
    Adapter implementing ModelStoragePort using standard PyTorch save/load functions.
    Bundles state_dict, metadata, and mapping registry into a single .pt artifact.
    """
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        metadata: TrainingMetadata,
        path: str
    ) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Package state weights and descriptive models dynamically
        checkpoint = {
            "model_state_dict": model_state,
            "metadata": {
                "best_accuracy": metadata.best_accuracy,
                "best_loss": metadata.best_loss,
                "epochs_trained": metadata.epochs_trained,
                "class_to_idx": metadata.class_to_idx,
                "idx_to_class": metadata.idx_to_class,
                "model_version": metadata.model_version,
                "trained_at": metadata.trained_at
            }
        }
        
        # Atomic write
        temp_path = save_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, save_path)
        
        logger.info(
            "Model checkpoint saved successfully",
            extra={"path": str(save_path), "best_accuracy": f"{metadata.best_accuracy:.4f}"}
        )

    def load_checkpoint(self, path: str) -> Tuple[Dict[str, Any], TrainingMetadata]:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"No model checkpoint found at '{path}'")

        # Map to CPU by default to keep load operations lightweight
        checkpoint = torch.load(load_path, map_location="cpu")
        
        state_dict = checkpoint["model_state_dict"]
        raw_meta = checkpoint["metadata"]
        
        # Reconstruct pure domain model
        metadata = TrainingMetadata(
            best_accuracy=raw_meta["best_accuracy"],
            best_loss=raw_meta["best_loss"],
            epochs_trained=raw_meta["epochs_trained"],
            class_to_idx=raw_meta["class_to_idx"],
            # Convert keys back to integers since JSON saves them as strings
            idx_to_class={int(k): v for k, v in raw_meta["idx_to_class"].items()},
            model_version=raw_meta.get("model_version", "1.0.0"),
            trained_at=raw_meta.get("trained_at", "")
        )
        
        logger.info(
            "Model checkpoint loaded successfully",
            extra={"path": str(load_path), "best_accuracy": f"{metadata.best_accuracy:.4f}"}
        )
        return state_dict, metadata


class ScikitEvaluationReporter(EvaluationReporterPort):
    """
    Adapter implementing EvaluationReporterPort using scikit-learn metrics.
    Saves metrics in human-readable and JSON format.
    """
    def generate_report(
        self,
        y_true: List[int],
        y_pred: List[int],
        probs: np.ndarray,
        classes: List[str]
    ) -> EvaluationReport:
        # Calculate standard metrics
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        
        # Scikit classification report dictionary representation
        report_dict = classification_report(
            y_true, y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0
        )
        
        macro_f1 = float(report_dict["macro avg"]["f1-score"])
        weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

        # Construct raw confusion matrix
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

        # Format per-class reports cleanly
        per_class = {}
        for c in classes:
            if c in report_dict:
                per_class[c] = {
                    "precision": float(report_dict[c]["precision"]),
                    "recall": float(report_dict[c]["recall"]),
                    "f1-score": float(report_dict[c]["f1-score"]),
                    "support": float(report_dict[c]["support"])
                }

        return EvaluationReport(
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            per_class_metrics=per_class,
            confusion_matrix=matrix,
            class_labels=classes
        )

    def save_report(self, report: EvaluationReport, path: str) -> str:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Construct output dictionary
        payload = {
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "weighted_f1": report.weighted_f1,
            "generated_at": report.generated_at,
            "class_labels": report.class_labels,
            "per_class_metrics": report.per_class_metrics,
            "confusion_matrix": report.confusion_matrix
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        logger.info("Evaluation metrics report saved to disk", extra={"path": str(report_path)})
        return str(report_path)
