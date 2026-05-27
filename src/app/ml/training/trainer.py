import random
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.app.ml.application.ports import ModelStoragePort
from src.app.ml.domain.models import TrainingMetadata
from src.app.ml.infrastructure.dataset import StoneColorDataset
from src.app.ml.infrastructure.model import freeze_backbone, unfreeze_backbone
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Sets reproducible random seeds across Python, NumPy, and PyTorch frameworks.
    """
    logger.info("Enforcing deterministic execution seeds...", extra={"seed": seed})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_class_weights(dataloader: DataLoader, num_classes: int) -> torch.Tensor:
    """
    Computes normalized class weights based on the inverse class frequency.
    Ensures minority classes receive a higher gradient weight contribution.
    """
    labels = []
    # Dataloader has lazy dataset under it
    dataset = dataloader.dataset
    if hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        # Fallback to scan dataset items sequentially
        for _, label in dataloader.dataset:
            labels.append(int(label))
            
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    weights = []
    for count in counts:
        if count > 0:
            # Weighted Loss: Total / (Classes * Count)
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        else:
            weights.append(1.0)
            
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(
        "Computed class balancing weights successfully",
        extra={"class_counts": counts.tolist(), "weights": weight_tensor.tolist()}
    )
    return weight_tensor


def calculate_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 2) -> float:
    """
    Computes validation Top-K accuracy count.
    """
    with torch.no_grad():
        maxk = min(k, outputs.size(1))
        if maxk <= 0:
            return 0.0
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        return float(correct_k.item())


class StoneModelTrainer:
    """
    Supervised model training coordinator.
    Implements Warm-Up Head training followed by Backbone fine-tuning.
    """
    def __init__(
        self,
        model: nn.Module,
        storage: ModelStoragePort,
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
        device: str = "cpu",
        checkpoint_path: str = "models/stone_color_model.pt",
        early_stopping_patience: int = 3,
        top_k: int = 2
    ):
        self.model = model.to(device)
        self.storage = storage
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.early_stopping_patience = early_stopping_patience
        self.top_k = top_k

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float, float]:
        """
        Runs one full epoch over the training loader.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        correct_k = 0.0

        for images, targets in train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion on unstable transfers
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correct_k += calculate_topk_accuracy(outputs, targets, k=self.top_k)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        epoch_topk = correct_k / total_samples
        return epoch_loss, epoch_acc, epoch_topk

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, float]:
        """
        Runs validation inference over the validation split.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        correct_k = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_samples += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                correct_k += calculate_topk_accuracy(outputs, targets, k=self.top_k)

        val_loss = total_loss / total_samples
        val_acc = correct / total_samples
        val_topk = correct_k / total_samples
        return val_loss, val_acc, val_topk

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001,
        freeze_epochs: int = 2
    ) -> TrainingMetadata:
        """
        Coordinates full transfer learning procedure.
        Frozens features for freeze_epochs, unfreezes backbone for remaining.
        """
        logger.info(
            "Starting model training...",
            extra={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "freeze_epochs": freeze_epochs,
                "device": str(self.device)
            }
        )

        # 1. Compute imbalance weights and compile standard loss
        num_classes = len(self.class_to_idx)
        class_weights = calculate_class_weights(train_loader, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 2. Setup initial frozen backbone state
        if freeze_epochs > 0:
            freeze_backbone(self.model)
            # Only optimize parameters that require gradients (the classifier head)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate
            )
        else:
            unfreeze_backbone(self.model)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_val_acc = 0.0
        best_val_loss = float("inf")
        epochs_no_improvement = 0
        backbone_unfrozen = (freeze_epochs <= 0)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Switch training mode from frozen warmup to complete unfreeze
            if not backbone_unfrozen and epoch > freeze_epochs:
                logger.info("Transitioning epoch: unfreezing backbone parameters for fine-tuning...")
                unfreeze_backbone(self.model)
                # Lower learning rate slightly for feature extraction adjustments
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate * 0.1)
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
                backbone_unfrozen = True

            # Train and Validate
            train_loss, train_acc, train_topk = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_topk = self.validate(val_loader, criterion)
            
            # Step scheduler based on validation accuracy
            scheduler.step(val_acc)

            duration = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch}/{epochs} Completed in {duration:.1f}s",
                extra={
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc * 100:.2f}%",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc * 100:.2f}%",
                    "val_topk": f"{val_topk * 100:.2f}%"
                }
            )

            # Checkpoint evaluation
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_no_improvement = 0
                
                # Reconstruct TrainingMetadata
                metadata = TrainingMetadata(
                    best_accuracy=best_val_acc,
                    best_loss=best_val_loss,
                    epochs_trained=epoch,
                    class_to_idx=self.class_to_idx,
                    idx_to_class=self.idx_to_class
                )
                
                # Save best state dict
                self.storage.save_checkpoint(
                    model_state=self.model.state_dict(),
                    metadata=metadata,
                    path=self.checkpoint_path
                )
            else:
                epochs_no_improvement += 1

            # Early Stopping evaluation
            if epochs_no_improvement >= self.early_stopping_patience:
                logger.warning(
                    f"Early stopping triggered. Validation accuracy did not improve for {self.early_stopping_patience} epochs.",
                    extra={"stopped_epoch": epoch, "best_val_acc": f"{best_val_acc * 100:.2f}%"}
                )
                break

        # Reconstruct final metadata response
        final_metadata = TrainingMetadata(
            best_accuracy=best_val_acc,
            best_loss=best_val_loss,
            epochs_trained=epochs,
            class_to_idx=self.class_to_idx,
            idx_to_class=self.idx_to_class
        )
        return final_metadata
