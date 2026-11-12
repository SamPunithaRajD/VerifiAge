"""Shared image preprocessing for training and inference. Must stay identical between the two."""

from torchvision import transforms

# Dataset images are natively 128x128 (see data audit). Training at 224 would upsample with no
# real detail gain and is ~6x slower on CPU (measured), so we train/infer at native resolution.
IMAGE_SIZE = 128

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training: light augmentation on top of the base pipeline
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation / test / inference: no randomness
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_and_preprocess(image_path):
    """Used by inference: load an image file and apply the eval transform."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image)
    return tensor.unsqueeze(0)
