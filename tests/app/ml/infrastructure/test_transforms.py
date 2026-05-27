from PIL import Image
import torchvision.transforms as T
from src.app.ml.infrastructure.transforms import get_transforms


def test_transforms_generation():
    # 1. Generate transforms
    train_tf, val_tf = get_transforms(image_size=128, augment=True)
    
    assert isinstance(train_tf, T.Compose)
    assert isinstance(val_tf, T.Compose)
    
    # 2. Assert augmentation transformations list size differences
    assert len(train_tf.transforms) > len(val_tf.transforms)
    
    # 3. Test execution on dummy PIL Image
    dummy_img = Image.new("RGB", (300, 300), (255, 0, 0))
    
    train_tensor = train_tf(dummy_img)
    val_tensor = val_tf(dummy_img)
    
    assert train_tensor.shape == (3, 128, 128)
    assert val_tensor.shape == (3, 128, 128)


def test_transforms_generation_no_augment():
    # Test generation when augment option is False
    train_tf, val_tf = get_transforms(image_size=224, augment=False)
    
    assert len(train_tf.transforms) == len(val_tf.transforms)
    
    dummy_img = Image.new("RGB", (100, 100), (0, 255, 0))
    train_tensor = train_tf(dummy_img)
    
    assert train_tensor.shape == (3, 224, 224)
