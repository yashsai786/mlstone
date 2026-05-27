from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from unittest.mock import MagicMock

from src.app.ml.evaluation.evaluator import StoneModelEvaluator
from src.app.ml.domain.models import EvaluationReport


def test_evaluator_predictions_and_failures(temp_storage_dir):
    report_path = temp_storage_dir / "val_report.json"
    
    # 2 classes, linear classifier
    model = nn.Linear(4, 2)
    # Set weights deterministically so predictions are predictable
    with torch.no_grad():
        model.weight.fill_(0)
        # Class 0 bias is 1, Class 1 bias is 0 -> Predicts Class 0 for everything
        model.bias[0] = 1.0
        model.bias[1] = 0.0

    mock_report = EvaluationReport(
        accuracy=0.5,
        macro_f1=0.5,
        weighted_f1=0.5,
        per_class_metrics={},
        confusion_matrix=[],
        class_labels=["beige", "black"]
    )
    mock_reporter = MagicMock()
    mock_reporter.generate_report = MagicMock(return_value=mock_report)
    
    idx_to_class = {0: "beige", 1: "black"}

    evaluator = StoneModelEvaluator(
        model=model,
        reporter=mock_reporter,
        idx_to_class=idx_to_class,
        device="cpu"
    )

    # 1. Setup mock loaders (total 4 samples)
    images = torch.randn(4, 4)
    targets = torch.tensor([0, 0, 1, 1])  # 2 correct (c0), 2 incorrect (c1)
    
    dataset = TensorDataset(images, targets)
    # Inject absolute paths helper to verify failures registry
    dataset.image_paths = [
        Path("dataset/beige/1.png"), Path("dataset/beige/2.png"),
        Path("dataset/black/3.png"), Path("dataset/black/4.png")
    ]
    
    loader = DataLoader(dataset, batch_size=2)

    # 2. Run evaluate
    report, failures = evaluator.evaluate(loader, report_path=str(report_path))

    # Assertions
    # 2 predictions were wrong (Class 1 target got predicted Class 0)
    assert len(failures) == 2
    assert failures[0]["image_path"] == "dataset/black/3.png"
    assert failures[0]["true_label"] == "black"
    assert failures[0]["predicted_label"] == "beige"

    # Verify that mock reporter generated the classification report
    mock_reporter.generate_report.assert_called_once()
    mock_reporter.save_report.assert_called_once()
