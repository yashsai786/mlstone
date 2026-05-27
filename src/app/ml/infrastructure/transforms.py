import torchvision.transforms as T
from typing import Tuple, List, Optional


def get_transforms(
    image_size: int = 224,
    augment: bool = True,
    rotation_degrees: float = 15.0,
    brightness_jitter: float = 0.2,
    contrast_jitter: float = 0.2
) -> Tuple[T.Compose, T.Compose]:
    """
    Creates torchvision training and validation transformation pipelines.
    
    Training: Heavy augmentations that preserve key stone texture characteristics
    while avoiding severe distortion that destroys mineral signatures.
    
    Validation: Minimal deterministic resize and ImageNet normalizations.
    """
    # Standard ImageNet normalization coefficients
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        train_list = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),  # Additional rotation symmetry for slabs
            T.RandomRotation(degrees=rotation_degrees),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            ),
            T.ColorJitter(
                brightness=brightness_jitter,
                contrast=contrast_jitter,
                saturation=0.1,
                hue=0.02
            ),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            normalize
        ]
    else:
        train_list = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize
        ]

    val_list = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize
    ]

    return T.Compose(train_list), T.Compose(val_list)
