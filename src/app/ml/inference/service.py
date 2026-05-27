import time
from pathlib import Path
from typing import Dict, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from src.app.ml.domain.models import StoneClassification
from src.app.ml.application.ports import InferencePort
from src.app.ml.infrastructure.adapters import PyTorchModelStorage
from src.app.ml.infrastructure.model import get_efficientnet_model
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class StoneColorInferenceService(InferencePort):
    """
    Supervises CPU/GPU standalone color inference services.
    Render deployment friendly; does not require the raw training folder structure.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model_path = model_path
        self.storage = PyTorchModelStorage()
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Initialize transformation pipeline
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the combined PyTorch model checkpoint.
        Auto-configures the classifier architecture based on discovered output metadata size.
        """
        logger.info("Initializing inference service...", extra={"path": self.model_path, "device": str(self.device)})
        
        # Load weights and training metadata
        state_dict, metadata = self.storage.load_checkpoint(self.model_path)
        
        self.class_to_idx = metadata.class_to_idx
        self.idx_to_class = metadata.idx_to_class
        num_classes = len(self.class_to_idx)
        
        # Standardize backbone reconstruction using discovered class counts
        self.model = get_efficientnet_model(
            num_classes=num_classes,
            pretrained=False  # Do not need ImageNet downloads since we overwrite weights immediately
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            "Inference service model loaded and verified.",
            extra={"num_classes": num_classes, "device": str(self.device)}
        )

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        top_k: int = 3
    ) -> StoneClassification:
        """
        Executes prediction on stone slab image inputs.
        Supports file paths, raw image bytes, and OpenCV numpy arrays.
        """
        start_time = time.time()
        
        # 1. Standardize image loader BGR/RGB layouts
        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, bytes):
                import io
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                # Check if it's OpenCV BGR layout
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                else:
                    rgb_arr = image_input
                img = Image.fromarray(rgb_arr)
            else:
                raise ValueError("Unsupported input format for prediction.")
        except Exception as e:
            logger.error("Failed to load/convert image input for inference", extra={"error": str(e)})
            raise ValueError(f"Failed to decode image input: {e}")

        # 2. Execute transforms and infer
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)

        # 3. Aggregate Top-K confidence metrics
        k = min(top_k, len(self.idx_to_class))
        top_probs, top_indices = probs.topk(k)

        top_k_results = {}
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            top_k_results[self.idx_to_class[idx]] = float(p)

        # Top confidence prediction
        best_idx = top_indices[0].item()
        predicted_class = self.idx_to_class[best_idx]
        confidence = float(top_probs[0].item())

        inference_duration_ms = (time.time() - start_time) * 1000.0

        logger.info(
            "Inference execution completed",
            extra={
                "predicted": predicted_class,
                "confidence": f"{confidence * 100:.2f}%",
                "duration_ms": f"{inference_duration_ms:.1f}ms"
            }
        )

        return StoneClassification(
            predicted_class=predicted_class,
            confidence=confidence,
            top_k=top_k_results,
            inference_time_ms=inference_duration_ms,
            device_used=str(self.device)
        )
