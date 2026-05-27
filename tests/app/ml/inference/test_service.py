from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.app.ml.inference.service import StoneColorInferenceService
from src.app.ml.domain.models import TrainingMetadata


@pytest.fixture
def mock_inference_checkpoint(temp_storage_dir) -> str:
    """
    Creates a valid, lightweight PyTorch model checkpoint.
    """
    checkpoint_path = temp_storage_dir / "test_stone_model.pt"
    
    # Sequential model matching test structure to align dictionary keys
    dummy_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 10)
    )
    state_dict = dummy_model.state_dict()
    
    checkpoint = {
        "model_state_dict": state_dict,
        "metadata": {
            "best_accuracy": 0.95,
            "best_loss": 0.1,
            "epochs_trained": 5,
            "class_to_idx": {f"color_{i}": i for i in range(10)},
            "idx_to_class": {str(i): f"color_{i}" for i in range(10)},
            "model_version": "1.0.0",
            "trained_at": "2026-05-27T00:00:00"
        }
    }
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def test_inference_service_initialization(mock_inference_checkpoint):
    # Patch model constructor to return a simple mock linear layer instead of EfficientNet-B0
    # to avoid pulling real backbone weights in unit tests
    dummy_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 10)
    )
    
    with patch("src.app.ml.inference.service.get_efficientnet_model", return_value=dummy_model):
        service = StoneColorInferenceService(
            model_path=mock_inference_checkpoint,
            device="cpu"
        )
        
        assert len(service.class_to_idx) == 10
        assert service.idx_to_class[2] == "color_2"
        assert service.model is dummy_model


def test_inference_service_predict_numpy_array(mock_inference_checkpoint):
    # 224x224 BGR image
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    dummy_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 10)
    )
    # Set bias of color_3 to 10.0 to force color_3 prediction
    with torch.no_grad():
        dummy_model[1].bias.fill_(0)
        dummy_model[1].bias[3] = 10.0

    with patch("src.app.ml.inference.service.get_efficientnet_model", return_value=dummy_model):
        service = StoneColorInferenceService(
            model_path=mock_inference_checkpoint,
            device="cpu"
        )
        
        classification = service.predict(dummy_img, top_k=2)
        
        assert classification.predicted_class == "color_3"
        assert classification.confidence > 0.8
        assert len(classification.top_k) == 2
        assert "color_3" in classification.top_k
        assert classification.device_used == "cpu"
        assert classification.inference_time_ms > 0.0
