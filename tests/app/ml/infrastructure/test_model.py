import torch
import torch.nn as nn
from src.app.ml.infrastructure.model import get_efficientnet_model, freeze_backbone, unfreeze_backbone


def test_get_efficientnet_model_architecture():
    # Construct model for 5 classes
    model = get_efficientnet_model(num_classes=5, dropout=0.4, pretrained=False)
    
    assert isinstance(model, nn.Module)
    
    # Assert classifier layer layout matches custom requirements
    classifier = model.classifier
    assert isinstance(classifier, nn.Sequential)
    assert isinstance(classifier[0], nn.Dropout)
    assert classifier[0].p == 0.4
    assert isinstance(classifier[1], nn.Linear)
    assert classifier[1].out_features == 5


def test_freeze_and_unfreeze_backbone():
    model = get_efficientnet_model(num_classes=3, pretrained=False)
    
    # 1. Freeze backbone features
    freeze_backbone(model)
    
    # Verify backbone parameters requires_grad is False, classifier requires_grad is True
    for name, param in model.named_parameters():
        if "classifier" not in name:
            assert param.requires_grad is False
        else:
            assert param.requires_grad is True

    # 2. Unfreeze backbone features
    unfreeze_backbone(model)
    
    # Verify all parameters require gradients
    for param in model.parameters():
        assert param.requires_grad is True
