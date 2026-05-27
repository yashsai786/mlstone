from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.app.application.ports import ImageDownloaderPort, SlabDetectorPort
from src.app.ml.application.ports import ModelStoragePort, EvaluationReporterPort, InferencePort
from src.app.ml.domain.models import (
    EvaluationReport, TrainingMetadata, PredictionRequest,
    PredictionCandidate, PredictionResult
)
from src.app.domain.exceptions import (
    DownloadError, InvalidImageError, SlabDetectionError, InferenceError
)
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


class PredictStoneColorUseCase:
    """
    Application orchestrator for the complete stone color prediction API workflow.
    Validates inputs, downloads images, processes/crops slabs, and runs ML inference.
    """
    def __init__(
        self,
        downloader: ImageDownloaderPort,
        slab_detector: SlabDetectorPort,
        inference_service: InferencePort,
        model_version: str = "1.0.0"
    ):
        self.downloader = downloader
        self.slab_detector = slab_detector
        self.inference_service = inference_service
        self.model_version = model_version

    async def execute(self, request: PredictionRequest) -> PredictionResult:
        import time
        import cv2
        import numpy as np

        start_time = time.time()

        # 1. Validate URL Input
        url = request.image_url.strip()
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            raise DownloadError("Invalid image URL format provided.")

        # 2. Download Image bytes asynchronously
        try:
            image_bytes = await self.downloader.download(url)
        except (DownloadError, InvalidImageError) as e:
            # Propagate custom domain exceptions directly
            raise e
        except Exception as e:
            raise DownloadError(f"Unexpected image download failure: {e}")

        # 3. Decode image bytes to OpenCV numpy array
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise InvalidImageError("Downloaded image could not be decoded.")
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise e
            raise InvalidImageError(f"Failed to parse downloaded image bytes: {e}")

        # 4. Preprocess / Crop Slab Region
        try:
            regions = self.slab_detector.detect_slabs(img)
            if not regions:
                raise SlabDetectionError("No stone slabs detected in the image.")
            
            # Select best matching region
            best_region = max(regions, key=lambda r: r.confidence)
            
            cropped_slab = self.slab_detector.crop_slab(img, best_region)
            if cropped_slab is None or cropped_slab.size == 0:
                raise SlabDetectionError("Failed to crop stone slab region from the image.")
        except SlabDetectionError as e:
            raise e
        except Exception as e:
            raise SlabDetectionError(f"Slab preprocessing execution failed: {e}")

        # 5. Run ML Inference classification
        try:
            classification = self.inference_service.predict(cropped_slab)
        except Exception as e:
            raise InferenceError(f"Model color classification inference execution failed: {e}")

        # 6. Structure prediction candidates
        top_candidates = []
        for name, confidence in classification.top_k.items():
            top_candidates.append(
                PredictionCandidate(class_name=name, confidence=confidence)
            )

        # Ensure top predictions are strictly ordered by descending confidence
        top_candidates.sort(key=lambda c: c.confidence, reverse=True)

        processing_duration_ms = (time.time() - start_time) * 1000.0

        return PredictionResult(
            predicted_color=classification.predicted_class,
            confidence=classification.confidence,
            top_predictions=top_candidates,
            processing_time_ms=processing_duration_ms,
            model_version=self.model_version
        )
