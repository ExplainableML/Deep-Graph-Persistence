import torch
import mnist
import numpy as np
import multiprocessing

from torch import Tensor
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Iterable
from containers import EvaluationData
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split


registered_datasets = {}
data_shapes = {}


def register_dataset(
    name: str, input_size: Union[int, Iterable[int]], num_classes: int
):
    def decorator(dataset_loader):
        if name in registered_datasets:
            raise ValueError(f"Dataset {name} already registered")

        registered_datasets[name] = dataset_loader
        data_shapes[name] = (input_size, num_classes)
        return dataset_loader

    return decorator


@register_dataset("mnist", input_size=784, num_classes=10)
def _get_mnist_data():
    # Load MNIST Train Data
    train_images = mnist.train_images()
    train_images = train_images / 255.0
    train_labels = mnist.train_labels()

    # Load MNIST Test Data
    test_images = mnist.test_images()
    test_images = test_images / 255.0
    test_labels = mnist.test_labels()

    return train_images, train_labels, test_images, test_labels


@register_dataset("fashion-mnist", input_size=784, num_classes=10)
def _get_fashion_mnist_datat():
    # Load Fashion MNIST Data
    train_images = FashionMNIST("./training_datasets", train=True, download=True).data
    train_labels = FashionMNIST(
        "./training_datasets", train=True, download=True
    ).targets

    test_images = FashionMNIST("./training_datasets", train=False, download=True).data
    test_labels = FashionMNIST(
        "./training_datasets", train=False, download=True
    ).targets

    # Convert to numpy
    train_images = train_images.numpy() / 255.0
    train_labels = train_labels.numpy()
    test_images = test_images.numpy() / 255.0
    test_labels = test_labels.numpy()

    return train_images, train_labels, test_images, test_labels


@register_dataset("cifar10", input_size=3 * 32 * 32, num_classes=10)
def _get_cifar10_data():
    # Load CIFAR10 Data
    train_images = CIFAR10("./training_datasets", train=True, download=True).data
    train_labels = CIFAR10("./training_datasets", train=True, download=True).targets

    test_images = CIFAR10("./training_datasets", train=False, download=True).data
    test_labels = CIFAR10("./training_datasets", train=False, download=True).targets

    # Convert labels to numpy
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Move channels to first dimension
    train_images = np.moveaxis(train_images, -1, 1) / 255.0
    test_images = np.moveaxis(test_images, -1, 1) / 255.0

    return train_images, train_labels, test_images, test_labels


def generate_dataloaders(
    batch_size: int,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Optional[Tensor] = None,
    y_val: Optional[Tensor] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    # Create PyTorch Dataset
    train_dataset = TensorDataset(x_train, y_train)
    if x_val is not None and y_val is not None:
        val_dataset = TensorDataset(x_val, y_val)
    else:
        val_dataset = None

    # Create PyTorch DataLoader
    num_workers = min(multiprocessing.cpu_count(), 6)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def prepare_dataloader(
    train_images: np.ndarray, train_labels: np.ndarray, batch_size: int
):
    # Split into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )

    # Convert to tensors
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_images = torch.from_numpy(val_images).float()
    val_labels = torch.from_numpy(val_labels).long()

    return generate_dataloaders(
        batch_size=batch_size,
        x_train=train_images,
        y_train=train_labels,
        x_val=val_images,
        y_val=val_labels,
    )


def load_dataset_for_training(name: str, batch_size: int):
    if name not in registered_datasets:
        raise ValueError(f"Unknown dataset: {name}")

    train_images, train_labels, _, _ = registered_datasets[name]()
    train_dataloader, val_dataloader = prepare_dataloader(
        train_images, train_labels, batch_size
    )

    return train_dataloader, val_dataloader


def get_data_shape(name: str):
    if name not in data_shapes:
        raise ValueError(f"Unknown dataset: {name}")

    return data_shapes[name]


def load_dataset(
    dataset_name: str, num_train_samples: int, num_test_samples: int
) -> EvaluationData:
    if dataset_name in registered_datasets:
        x_train, y_train, x_test, y_test = registered_datasets[dataset_name]()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Sample `num_train_samples` images from MNIST train data
    x_train, _, y_train, _ = train_test_split(
        x_train,
        y_train,
        train_size=num_train_samples,
        stratify=y_train,
    )

    # Sample `num_test_samples` images from MNIST test data
    x_test, _, y_test, _ = train_test_split(
        x_test,
        y_test,
        train_size=num_test_samples,
        stratify=y_test,
    )

    return EvaluationData(
        x_train=x_train,
        y_train=y_train,
        x_clean=x_test,
        y_clean=y_test,
        x_test=x_test,
        y_test=y_test,
    )
