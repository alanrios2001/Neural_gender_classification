import torch
from torch.utils.data import DataLoader

from torchvision import transforms, datasets


def load(train_dir: str, test_dir: str, batch_size: int):

    train_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.TrivialAugmentWide(num_magnitude_bins=32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(torch.device("cuda")))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(torch.device("cuda")))
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
    )

    loaded_test = DataLoader(
        batch_size=batch_size,
        dataset=test_data,
        shuffle=True,
    )

    return loaded_train, loaded_test, class_names
