import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.app.ml.application.ports import EvaluationReporterPort
from src.app.ml.domain.models import EvaluationReport
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class StoneModelEvaluator:
    """
    Supervises the model evaluation flow.
    Runs validation datasets, extracts predictions, and writes complete reports.
    """
    def __init__(
        self,
        model: nn.Module,
        reporter: EvaluationReporterPort,
        idx_to_class: Dict[int, str],
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.reporter = reporter
        self.idx_to_class = idx_to_class
        self.device = torch.device(device)

    def evaluate(
        self,
        dataloader: DataLoader,
        report_path: Optional[str] = None
    ) -> Tuple[EvaluationReport, List[Dict[str, Any]]]:
        """
        Executes prediction iteration over validation/testing dataloader.
        Returns the constructed EvaluationReport along with a detailed list of failure items.
        """
        self.model.eval()
        
        y_true = []
        y_pred = []
        y_probs = []
        failures = []
        
        classes = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]

        with torch.no_grad():
            # Keep track of paths if available in the dataset class (for failure analysis)
            dataset = dataloader.dataset
            image_paths = getattr(dataset, "image_paths", None)
            
            for idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                # Compute probabilities using standard Softmax activation
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)

                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(predicted.cpu().numpy().tolist())
                y_probs.extend(probs.cpu().numpy().tolist())

                # Collect structural misclassifications for visual failure analysis
                for j in range(images.size(0)):
                    global_idx = idx * dataloader.batch_size + j
                    true_idx = targets[j].item()
                    pred_idx = predicted[j].item()
                    
                    if true_idx != pred_idx:
                        img_path = str(image_paths[global_idx]) if image_paths else "unknown"
                        failures.append({
                            "image_path": img_path,
                            "true_label": self.idx_to_class[true_idx],
                            "predicted_label": self.idx_to_class[pred_idx],
                            "confidence": float(probs[j][pred_idx].item())
                        })

        # Compile reports using SciKit Adapter
        report = self.reporter.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            probs=np.array(y_probs),
            classes=classes
        )

        # Save report to disk if requested
        if report_path:
            self.reporter.save_report(report, report_path)
            
            # Save failure analysis list to a sibling JSON file
            failures_path = Path(report_path).parent / f"{Path(report_path).stem}_failures.json"
            try:
                with open(failures_path, "w", encoding="utf-8") as f:
                    json.dump(failures, f, indent=4)
                logger.info("Failure analysis report saved to disk", extra={"path": str(failures_path)})
            except Exception as e:
                logger.error("Failed to write failure analysis list", extra={"error": str(e)})

        logger.info(
            "Evaluation complete",
            extra={
                "accuracy": f"{report.accuracy * 100:.2f}%",
                "macro_f1": f"{report.macro_f1:.4f}",
                "failures_count": len(failures)
            }
        )
        return report, failures
