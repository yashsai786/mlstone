import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


class StoneColorDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for stone slab classification.
    Decoupled from filesystem scan procedures to keep indexing pure and testable.
    """
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Standard lazy loading using PIL to guarantee torchvision compatibility
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                
                if self.transform:
                    img_tensor = self.transform(img_rgb)
                else:
                    import torchvision.transforms as T
                    img_tensor = T.ToTensor()(img_rgb)
                    
                return img_tensor, label
        except Exception as e:
            logger.warning(
                "Failed to read image at index during runtime, returning black canvas fallback",
                extra={"path": str(img_path), "error": str(e)}
            )
            # Safe training fallback: return black canvas
            fallback = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform:
                return self.transform(fallback), label
            import torchvision.transforms as T
            return T.ToTensor()(fallback), label


def discover_and_validate_dataset(
    dataset_dir: str
) -> Tuple[List[Path], List[int], Dict[str, int], Dict[int, str]]:
    """
    Auto-discovers, validates, and indexes processed stone color dataset subfolders.
    Deterministic class-to-index mapping is guaranteed by alphabetical sorting.
    Skips corrupted or empty image files gracefully.
    """
    base_path = Path(dataset_dir)
    if not base_path.exists():
        logger.error("Base processed dataset directory does not exist", extra={"path": dataset_dir})
        return [], [], {}, {}

    # Discover and sort color categories alphabetically
    color_classes = sorted([
        d.name for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ])
    
    class_to_idx = {name: idx for idx, name in enumerate(color_classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    valid_paths = []
    valid_labels = []

    for color in color_classes:
        color_dir = base_path / color
        label_idx = class_to_idx[color]

        # Scan for standard image extensions
        for img_file in sorted(color_dir.glob("*.*")):
            if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
                continue

            # Verify image structural integrity before putting in the loaders
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Cheap integrity verification without full decode
                valid_paths.append(img_file)
                valid_labels.append(label_idx)
            except Exception as e:
                logger.warning(
                    "Image verification failed. Skipping corrupted file.",
                    extra={"path": str(img_file), "error": str(e)}
                )

    logger.info(
        "Auto-discovered and validated stone dataset classes",
        extra={
            "classes": color_classes,
            "total_verified": len(valid_paths),
            "class_counts": {
                c: len(list((base_path / c).glob("*.*"))) for c in color_classes
            }
        }
    )
    
    return valid_paths, valid_labels, class_to_idx, idx_to_class


def get_stratified_loaders(
    dataset_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.0,
    seed: int = 42,
    train_transform=None,
    val_transform=None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, int], Dict[int, str]]:
    """
    Generates PyTorch DataLoaders with stratified splits preserving category balances.
    Handles minority classes cleanly down to few-shot training.
    """
    paths, labels, class_to_idx, idx_to_class = discover_and_validate_dataset(dataset_dir)
    if not paths:
        raise ValueError(f"No valid images found in the dataset directory: '{dataset_dir}'")

    # If test split is requested
    if test_split > 0.0:
        # First split into train_val and test
        paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
            paths, labels,
            test_size=test_split,
            random_state=seed,
            stratify=labels
        )
        
        # Then split train_val into train and val
        adjusted_val_size = val_split / (1.0 - test_split)
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths_train_val, labels_train_val,
            test_size=adjusted_val_size,
            random_state=seed,
            stratify=labels_train_val
        )
    else:
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths, labels,
            test_size=val_split,
            random_state=seed,
            stratify=labels
        )
        paths_test, labels_test = [], []

    # Instantiate distinct datasets with specific transforms
    train_dataset = StoneColorDataset(paths_train, labels_train, transform=train_transform)
    val_dataset = StoneColorDataset(paths_val, labels_val, transform=val_transform)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Zero workers prevents context leakage in small CPU systems
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = None
    if test_split > 0.0:
        test_dataset = StoneColorDataset(paths_test, labels_test, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    logger.info(
        "Successfully created train/val/test splits",
        extra={
            "train_size": len(paths_train),
            "val_size": len(paths_val),
            "test_size": len(paths_test)
        }
    )

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class
