import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.app.ml.domain.models import TrainingMetadata
from src.app.ml.infrastructure.adapters import PyTorchModelStorage, ScikitEvaluationReporter


def test_pytorch_model_storage(temp_storage_dir):
    checkpoint_path = temp_storage_dir / "test_model.pt"
    storage = PyTorchModelStorage()

    # Create dummy state dict
    dummy_model = nn.Linear(5, 2)
    state_dict = dummy_model.state_dict()

    metadata = TrainingMetadata(
        best_accuracy=0.945,
        best_loss=0.15,
        epochs_trained=5,
        class_to_idx={"beige": 0, "black": 1},
        idx_to_class={0: "beige", 1: "black"}
    )

    # 1. Save
    storage.save_checkpoint(state_dict, metadata, str(checkpoint_path))
    assert checkpoint_path.exists()

    # 2. Load
    loaded_state, loaded_meta = storage.load_checkpoint(str(checkpoint_path))
    
    assert loaded_meta.best_accuracy == 0.945
    assert loaded_meta.epochs_trained == 5
    assert loaded_meta.class_to_idx["black"] == 1
    assert loaded_meta.idx_to_class[0] == "beige"
    
    # Assert weights are matchable
    assert torch.equal(loaded_state["weight"], state_dict["weight"])


def test_scikit_evaluation_reporter(temp_storage_dir):
    report_path = temp_storage_dir / "report.json"
    reporter = ScikitEvaluationReporter()

    # Dummy inputs: 6 items, 3 classes
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]  # Acc = 4/6 = 0.6667
    
    classes = ["beige", "black", "grey"]
    dummy_probs = np.zeros((6, 3))

    # 1. Generate Report
    report = reporter.generate_report(y_true, y_pred, dummy_probs, classes)

    assert abs(report.accuracy - 0.6667) < 1e-3
    assert len(report.class_labels) == 3
    assert len(report.confusion_matrix) == 3
    assert "beige" in report.per_class_metrics
    assert report.per_class_metrics["black"]["precision"] == 0.6666666666666666  # 2 of 3 predictions are black

    # 2. Save Report
    saved_path = reporter.save_report(report, str(report_path))
    assert Path(saved_path).exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        assert abs(payload["accuracy"] - 0.6667) < 1e-3
        assert payload["class_labels"] == classes
