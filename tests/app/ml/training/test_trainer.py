import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from unittest.mock import MagicMock

from src.app.ml.training.trainer import (
    calculate_class_weights,
    calculate_topk_accuracy,
    StoneModelTrainer,
    set_seed
)


def test_seed_determinism():
    set_seed(123)
    val1 = torch.randn(2, 2)
    
    set_seed(123)
    val2 = torch.randn(2, 2)
    
    assert torch.equal(val1, val2)


def test_calculate_topk_accuracy():
    # 3 classes, batch size 2
    outputs = torch.tensor([
        [0.1, 0.6, 0.3],  # Preds: 1, 2, 0
        [0.8, 0.1, 0.1]   # Preds: 0, 1, 2
    ])
    
    # Target values
    targets1 = torch.tensor([1, 0])  # Both correct in top-1
    targets2 = torch.tensor([2, 0])  # Only index 1 correct in top-1, index 0 is correct in top-2
    
    assert calculate_topk_accuracy(outputs, targets1, k=1) == 2.0
    assert calculate_topk_accuracy(outputs, targets2, k=1) == 1.0
    assert calculate_topk_accuracy(outputs, targets2, k=2) == 2.0  # Top-2 brings it to 100%


class MockDataset(list):
    def __init__(self, data, labels):
        super().__init__(data)
        self.labels = labels


def test_calculate_class_weights():
    # 2 classes, total 6 samples (class 0: 2 items, class 1: 4 items)
    data = [
        (torch.zeros(3, 10, 10), 0),
        (torch.zeros(3, 10, 10), 0),
        (torch.zeros(3, 10, 10), 1),
        (torch.zeros(3, 10, 10), 1),
        (torch.zeros(3, 10, 10), 1),
        (torch.zeros(3, 10, 10), 1)
    ]
    labels = [0, 0, 1, 1, 1, 1]
    dummy_dataset = MockDataset(data, labels)
    
    loader = DataLoader(dummy_dataset, batch_size=2)
    weights = calculate_class_weights(loader, num_classes=2)
    
    # Expected: Class 0 weight = 6 / (2 * 2) = 1.5
    #           Class 1 weight = 6 / (2 * 4) = 0.75
    assert weights[0].item() == 1.5
    assert weights[1].item() == 0.75


def test_trainer_early_stopping():
    model = nn.Linear(3, 2)
    mock_storage = MagicMock()

    trainer = StoneModelTrainer(
        model=model,
        storage=mock_storage,
        class_to_idx={"c0": 0, "c1": 1},
        idx_to_class={0: "c0", 1: "c1"},
        device="cpu",
        early_stopping_patience=2
    )

    # 1. Setup mock loaders (total 4 samples)
    images = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(images, targets)
    # Inject mock labels list
    dataset.labels = [0, 1, 0, 1]
    loader = DataLoader(dataset, batch_size=2)

    # 2. Trigger training run. Since optimizer uses backpropagation,
    # we can run fit immediately for 5 epochs. Since gradients on small linear layers are unstable,
    # early stopping will trigger quickly if loss/acc does not keep rising
    trainer.fit(
        train_loader=loader,
        val_loader=loader,
        epochs=5,
        learning_rate=0.1,
        freeze_epochs=0
    )
    
    # Verify that model saved checkpoints at least once (when validation accuracy improved)
    assert mock_storage.save_checkpoint.called
