import torch
import torch.nn as nn
import torchvision.models as models
from src.app.infrastructure.logging import get_logger

logger = get_logger(__name__)


def get_efficientnet_model(
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = True
) -> nn.Module:
    """
    Builds an EfficientNet-B0 model initialized with ImageNet weights
    and configures a custom classifier head targeting our dynamic categories.
    """
    logger.info("Initializing EfficientNet-B0 model weights...")
    
    if pretrained:
        # Load weights using current torchvision conventions
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            logger.info("EfficientNet-B0 ImageNet v1 pre-trained weights loaded successfully.")
        except ImportError:
            # Fallback for older torchvision library environments
            model = models.efficientnet_b0(pretrained=True)
            logger.info("Fallback: Pre-trained weights loaded using legacy flag.")
    else:
        model = models.efficientnet_b0(pretrained=False)
        logger.info("Model initialized with random weights (no pre-training).")

    # Replace classifier head for stone color target classification
    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )
    
    logger.info(
        "Classifier head constructed successfully",
        extra={"in_features": in_features, "num_classes": num_classes, "dropout": dropout}
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all feature extraction layers of EfficientNet-B0
    to allow initial head warm-up training without backpropagation distorting pre-trained features.
    """
    logger.info("Freezing EfficientNet backbone feature extractor layers...")
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreezes all layers including backbone layers to enable complete fine-tuning.
    """
    logger.info("Unfreezing entire model for complete fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
