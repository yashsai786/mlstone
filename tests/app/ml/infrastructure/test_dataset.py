import os
from pathlib import Path
import numpy as np
import cv2
import pytest
from torch.utils.data import DataLoader

from src.app.ml.infrastructure.dataset import (
    StoneColorDataset,
    discover_and_validate_dataset,
    get_stratified_loaders
)


@pytest.fixture
def mock_processed_dataset(temp_storage_dir) -> Path:
    """
    Creates a temporary structure mimicking processed_dataset/
    populated with small valid and corrupted images.
    """
    base_dir = temp_storage_dir / "processed_dataset"
    beige_dir = base_dir / "beige"
    black_dir = base_dir / "black"
    
    beige_dir.mkdir(parents=True, exist_ok=True)
    black_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write valid images in beige folder
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(5):
        cv2.imwrite(str(beige_dir / f"beige_{i}.png"), dummy_img)

    # 2. Write valid images in black folder
    for i in range(5):
        cv2.imwrite(str(black_dir / f"black_{i}.png"), dummy_img)

    # 3. Write one corrupted/empty image in black folder
    with open(black_dir / "black_corrupt.png", "w") as f:
        f.write("not_an_image_data")

    # 4. Write a non-image file that should be ignored completely
    with open(beige_dir / "info.txt", "w") as f:
        f.write("text info")

    return base_dir


def test_discover_and_validate_dataset(mock_processed_dataset):
    paths, labels, class_to_idx, idx_to_class = discover_and_validate_dataset(
        str(mock_processed_dataset)
    )

    # Sorted alphabetically: beige (0) then black (1)
    assert class_to_idx == {"beige": 0, "black": 1}
    assert idx_to_class == {0: "beige", 1: "black"}

    # Expected: 5 beige + 5 black = 10 valid images (black_corrupt and info.txt must be skipped)
    assert len(paths) == 10
    assert len(labels) == 10
    
    # 5 labels must be 0, 5 labels must be 1
    assert labels.count(0) == 5
    assert labels.count(1) == 5


def test_stone_color_dataset_fallback(mock_processed_dataset):
    # Retrieve files
    paths = [mock_processed_dataset / "black" / "black_corrupt.png"]
    labels = [1]

    dataset = StoneColorDataset(paths, labels)
    assert len(dataset) == 1

    # Fetching the corrupt index should fallback to a safe black image tensor without throwing
    img_tensor, label = dataset[0]
    assert label == 1
    assert img_tensor.shape == (3, 224, 224)


def test_get_stratified_loaders(mock_processed_dataset):
    train_loader, val_loader, _, class_to_idx, idx_to_class = get_stratified_loaders(
        dataset_dir=str(mock_processed_dataset),
        batch_size=2,
        val_split=0.2,
        seed=10
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Total 10 valid samples, val_split=0.2 means 8 train, 2 validation
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2

    # Verify both classes are preserved in validation (stratification check)
    val_labels = val_loader.dataset.labels
    assert val_labels.count(0) == 1
    assert val_labels.count(1) == 1
