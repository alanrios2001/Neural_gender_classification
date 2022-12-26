import torch
from torch.utils.data import DataLoader

from torchvision import transforms, datasets


def load(train_dir: str, test_dir: str, batch_size: int, device: torch.device):

    train_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.TrivialAugmentWide(num_magnitude_bins=32),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    class_names = train_data.classes

    loaded_train = DataLoader(
        batch_size=batch_size,
        dataset=train_data,
        shuffle=True,
        pin_memory=True
    )

    loaded_test = DataLoader(
        batch_size=batch_size,
        dataset=test_data,
        shuffle=True,
        pin_memory=True
    )

    return loaded_train, loaded_test, class_names
