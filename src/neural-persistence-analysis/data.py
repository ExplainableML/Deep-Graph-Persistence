import multiprocessing

from torchvision.datasets import MNIST
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    input_size = None
    num_classes = None

    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = min(multiprocessing.cpu_count(), 8)

        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        validation_dataloader = DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        test_dataloader = self.test_dataloader()

        return [validation_dataloader, test_dataloader]

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class MNISTDataModule(DataModule):
    input_size = 28 * 28
    num_classes = 10

    def setup(self, stage: str = None):
        self.test_data = MNIST(
            self.data_dir, train=False, download=True, transform=ToTensor()
        )
        mnist_full = MNIST(
            self.data_dir, train=True, download=True, transform=ToTensor()
        )
        self.train_data, self.validation_data = random_split(mnist_full, [55000, 5000])


class FashionMNISTDataModule(DataModule):
    input_size = 28 * 28
    num_classes = 10

    def setup(self, stage: str = None):
        self.test_data = FashionMNIST(
            self.data_dir, train=False, download=True, transform=ToTensor()
        )
        fashion_mnist_full = FashionMNIST(
            self.data_dir, train=True, download=True, transform=ToTensor()
        )
        self.train_data, self.validation_data = random_split(
            fashion_mnist_full, [55000, 5000]
        )


class EMNISTDataModule(DataModule):
    input_size = 28 * 28
    num_classes = 47

    def setup(self, stage: str = None):
        self.test_data = EMNIST(
            self.data_dir,
            split="balanced",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        emnist_full_balanced = EMNIST(
            self.data_dir,
            split="balanced",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        train_size = int(round(len(emnist_full_balanced) * 0.9))
        validation_size = len(emnist_full_balanced) - train_size
        self.train_data, self.validation_data = random_split(
            emnist_full_balanced, [train_size, validation_size]
        )
