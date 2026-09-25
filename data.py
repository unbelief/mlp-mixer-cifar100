import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_loaders(data_dir="data", batch_size=64, num_workers=0,
                  val_ratio=0.1, augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=generator
    )
    val_set.dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=test_transform
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, test_set.classes
