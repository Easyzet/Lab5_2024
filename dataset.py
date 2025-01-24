# dataset.py
from torchvision import datasets
import torch

def load_cifar10():
    # Загрузка обучающей и тестовой выборки
    train_set = datasets.CIFAR10(
        root="data",
        train=True,
        download=True
    )

    test_set = datasets.CIFAR10(
        root="data",
        train=False,
        download=True
    )

    return train_set, test_set

def extract_tensors(dataset):
    x = torch.tensor(dataset.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dataset.targets, dtype=torch.int8)
    return x, y