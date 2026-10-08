"""CIFAR-10 transforms, split, and DataLoader construction."""

import torch
from config import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA_ROOT,
    EVALUATION_BATCH_SIZE,
    SEED,
    TRAIN_BATCH_SIZE,
    VALIDATION_SIZE,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def build_transforms():
    """Return train-only augmentation and deterministic evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    return train_transform, evaluation_transform


def get_data_loaders(
    train_batch_size=TRAIN_BATCH_SIZE,
    evaluation_batch_size=EVALUATION_BATCH_SIZE,
):
    """Build fixed train/validation split and CIFAR-10 test loader."""
    train_transform, evaluation_transform = build_transforms()

    # Two views of the training set let train samples use augmentation while
    # validation samples from the same fixed split remain deterministic.
    full_train_augmented = CIFAR10(
        DATA_ROOT,
        train=True,
        transform=train_transform,
        download=True,
    )
    full_train_clean = CIFAR10(
        DATA_ROOT,
        train=True,
        transform=evaluation_transform,
        download=True,
    )
    test_set = CIFAR10(
        DATA_ROOT,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    training_size = len(full_train_augmented) - VALIDATION_SIZE
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_train_augmented), generator=generator).tolist()

    train_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    train_set = Subset(full_train_augmented, train_indices)
    validation_set = Subset(full_train_clean, validation_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=evaluation_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=evaluation_batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader
