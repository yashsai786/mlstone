import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.app.ml.application.use_cases import TrainModelUseCase, EvaluateModelUseCase
from src.app.ml.domain.models import TrainingMetadata, EvaluationReport
from src.app.infrastructure.config import AppConfig


@pytest.fixture
def mock_app_config(temp_storage_dir) -> AppConfig:
    config = AppConfig()
    # Override directories to point to our test temp storage
    config.ml_model_dir = temp_storage_dir / "models"
    config.ml_reports_dir = temp_storage_dir / "reports"
    config.processed_dataset_dir = temp_storage_dir / "processed_dataset"
    
    config.ml_model_dir.mkdir(parents=True, exist_ok=True)
    config.ml_reports_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    return config


def test_train_model_use_case_orchestration(mock_app_config):
    # 1. Setup mocks for loaders, model constructor, and trainer
    mock_loader = MagicMock()
    mock_class_to_idx = {"beige": 0, "grey": 1}
    mock_idx_to_class = {0: "beige", 1: "grey"}
    
    mock_get_loaders = MagicMock(return_value=(
        mock_loader, mock_loader, None, mock_class_to_idx, mock_idx_to_class
    ))
    
    mock_metadata = TrainingMetadata(
        best_accuracy=0.92,
        best_loss=0.18,
        epochs_trained=4,
        class_to_idx=mock_class_to_idx,
        idx_to_class=mock_idx_to_class
    )
    
    mock_trainer = MagicMock()
    mock_trainer.fit = MagicMock(return_value=mock_metadata)
    
    # 2. Patch infrastructure dependencies to verify isolated use-case orchestrations
    with patch("src.app.ml.application.use_cases.get_stratified_loaders", mock_get_loaders), \
         patch("src.app.ml.application.use_cases.get_efficientnet_model", return_value=MagicMock()), \
         patch("src.app.ml.application.use_cases.StoneModelTrainer", return_value=mock_trainer):
         
         use_case = TrainModelUseCase(config=mock_app_config)
         metadata = use_case.execute(epochs=2, batch_size=4, device="cpu")
         
         assert metadata.best_accuracy == 0.92
         assert metadata.epochs_trained == 4
         
         # Assert trainer was fit with 2 epochs
         mock_trainer.fit.assert_called_once()
         call_kwargs = mock_trainer.fit.call_args[1]
         assert call_kwargs["epochs"] == 2
         assert call_kwargs["learning_rate"] == mock_app_config.ml_learning_rate


def test_evaluate_model_use_case_orchestration(mock_app_config):
    mock_class_to_idx = {"beige": 0, "grey": 1}
    mock_idx_to_class = {0: "beige", 1: "grey"}
    
    mock_metadata = TrainingMetadata(
        best_accuracy=0.92,
        best_loss=0.18,
        epochs_trained=4,
        class_to_idx=mock_class_to_idx,
        idx_to_class=mock_idx_to_class
    )
    
    # Mock storage loader
    mock_storage = MagicMock()
    mock_storage.load_checkpoint = MagicMock(return_value=({"weight": None}, mock_metadata))
    
    # Mock data loader
    mock_loader = MagicMock()
    mock_get_loaders = MagicMock(return_value=(
        None, mock_loader, None, mock_class_to_idx, mock_idx_to_class
    ))
    
    # Mock evaluator
    mock_report = EvaluationReport(
        accuracy=0.95,
        macro_f1=0.94,
        weighted_f1=0.95,
        per_class_metrics={},
        confusion_matrix=[],
        class_labels=["beige", "grey"]
    )
    
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = MagicMock(return_value=(mock_report, []))

    # Patch dependencies
    with patch("src.app.ml.application.use_cases.get_stratified_loaders", mock_get_loaders), \
         patch("src.app.ml.application.use_cases.get_efficientnet_model", return_value=MagicMock()), \
         patch("src.app.ml.application.use_cases.StoneModelEvaluator", return_value=mock_evaluator):
         
         use_case = EvaluateModelUseCase(config=mock_app_config, storage=mock_storage)
         report, save_path = use_case.execute(device="cpu")
         
         assert report.accuracy == 0.95
         assert report.macro_f1 == 0.94
         assert Path(save_path).name == "evaluation_report.json"
         
         # Assert storage and evaluator were triggered correctly
         mock_storage.load_checkpoint.assert_called_once()
         mock_evaluator.evaluate.assert_called_once()
