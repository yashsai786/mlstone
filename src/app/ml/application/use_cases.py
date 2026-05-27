from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.app.ml.application.ports import ModelStoragePort, EvaluationReporterPort
from src.app.ml.domain.models import EvaluationReport, TrainingMetadata
from src.app.ml.infrastructure.dataset import get_stratified_loaders
from src.app.ml.infrastructure.transforms import get_transforms
from src.app.ml.infrastructure.model import get_efficientnet_model
from src.app.ml.infrastructure.adapters import PyTorchModelStorage, ScikitEvaluationReporter
from src.app.ml.training.trainer import StoneModelTrainer, set_seed
from src.app.ml.evaluation.evaluator import StoneModelEvaluator
from src.app.infrastructure.config import AppConfig
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class TrainModelUseCase:
    """
    Application orchestrator for the complete model training workflow.
    Decoupled from CLI concerns, enabling easy scheduling or microservice triggers.
    """
    def __init__(
        self,
        config: AppConfig,
        storage: Optional[ModelStoragePort] = None
    ):
        self.config = config
        self.storage = storage or PyTorchModelStorage()

    def execute(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        device: str = "cpu",
        seed: int = 42
    ) -> TrainingMetadata:
        # Enforce deterministic training execution seeds
        set_seed(seed)

        # Allow dynamic parameter overrides via execution arguments
        run_epochs = epochs or self.config.ml_epochs
        run_batch_size = batch_size or self.config.ml_batch_size
        run_lr = learning_rate or self.config.ml_learning_rate
        
        logger.info(
            "Executing TrainModelUseCase...",
            extra={
                "epochs": run_epochs,
                "batch_size": run_batch_size,
                "learning_rate": run_lr,
                "device": device
            }
        )

        # 1. Retrieve training and validation transforms
        train_transform, val_transform = get_transforms(
            image_size=self.config.output_image_size,
            augment=True
        )

        # 2. Automatically discover classes and split dataloaders
        train_loader, val_loader, _, class_to_idx, idx_to_class = get_stratified_loaders(
            dataset_dir=str(self.config.processed_dataset_dir),
            batch_size=run_batch_size,
            val_split=0.2,
            seed=seed,
            train_transform=train_transform,
            val_transform=val_transform
        )

        num_classes = len(class_to_idx)
        
        # 3. Instantiate model with pretrained ImageNet backbone
        model = get_efficientnet_model(
            num_classes=num_classes,
            dropout=self.config.ml_dropout,
            pretrained=True
        )

        # 4. Orchestrate the training execution
        trainer = StoneModelTrainer(
            model=model,
            storage=self.storage,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            device=device,
            checkpoint_path=str(self.config.ml_model_dir / "stone_color_model.pt"),
            early_stopping_patience=self.config.ml_early_stopping_patience
        )

        # Fit model
        metadata = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=run_epochs,
            learning_rate=run_lr,
            freeze_epochs=self.config.ml_freeze_epochs
        )

        return metadata


class EvaluateModelUseCase:
    """
    Application orchestrator for running evaluations and export reports.
    """
    def __init__(
        self,
        config: AppConfig,
        storage: Optional[ModelStoragePort] = None,
        reporter: Optional[EvaluationReporterPort] = None
    ):
        self.config = config
        self.storage = storage or PyTorchModelStorage()
        self.reporter = reporter or ScikitEvaluationReporter()

    def execute(
        self,
        device: str = "cpu",
        seed: int = 42
    ) -> Tuple[EvaluationReport, str]:
        logger.info("Executing EvaluateModelUseCase...", extra={"device": device})
        
        # 1. Load trained checkpoint metadata
        model_path = str(self.config.ml_model_dir / "stone_color_model.pt")
        state_dict, metadata = self.storage.load_checkpoint(model_path)
        
        num_classes = len(metadata.class_to_idx)

        # 2. Reconstruct model backbone
        model = get_efficientnet_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state_dict)

        # 3. Load validation dataset splits
        _, val_transform = get_transforms(image_size=self.config.output_image_size, augment=False)
        
        _, val_loader, _, _, _ = get_stratified_loaders(
            dataset_dir=str(self.config.processed_dataset_dir),
            batch_size=self.config.ml_batch_size,
            val_split=0.2,
            seed=seed,
            train_transform=None,
            val_transform=val_transform
        )

        # 4. Orchestrate evaluation and failures logging
        evaluator = StoneModelEvaluator(
            model=model,
            reporter=self.reporter,
            idx_to_class=metadata.idx_to_class,
            device=device
        )

        report_save_path = str(self.config.ml_reports_dir / "evaluation_report.json")
        
        report, _ = evaluator.evaluate(
            dataloader=val_loader,
            report_path=report_save_path
        )

        return report, report_save_path
