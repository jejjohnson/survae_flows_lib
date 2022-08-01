from os import path
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
from pathlib import Path
from .utils import get_data_path
from survae.datasets.tabular.uci_datamodule import UCIDataModule


class PowerDataModule(UCIDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    url = "https://zenodo.org/record/1161203/files/data.tar.gz?download=1"
    folder = "uci_maf"
    download_file = "data.tar.gz"
    raw_folder = "uci_maf/data/power"
    raw_file = "data.npy"
    num_features = 6
    num_train = 1_659_917
    num_valid = 184_435
    num_test = 204_928

    def __init__(
        self,
        data_dir: str = get_data_path(),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        download_data: bool = True,
    ):
        super().__init__(data_dir=data_dir, download_data=download_data)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        def load_data():
            file = self.raw_data_path
            return np.load(file)

        def load_data_split_with_noise():
            rng = np.random.RandomState(42)

            data = load_data()
            rng.shuffle(data)
            N = data.shape[0]

            data = np.delete(data, 3, axis=1)
            data = np.delete(data, 1, axis=1)
            ############################
            # Add noise
            ############################
            # global_intensity_noise = 0.1*rng.rand(N, 1)
            voltage_noise = 0.01 * rng.rand(N, 1)
            # grp_noise = 0.001*rng.rand(N, 1)
            gap_noise = 0.001 * rng.rand(N, 1)
            sm_noise = rng.rand(N, 3)
            time_noise = np.zeros((N, 1))
            # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
            # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
            noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
            data += noise

            N_test = int(0.1 * data.shape[0])
            data_test = data[-N_test:]
            data = data[0:-N_test]
            N_validate = int(0.1 * data.shape[0])
            data_validate = data[-N_validate:]
            data_train = data[0:-N_validate]

            return data_train, data_validate, data_test

        def load_data_normalised():
            data_train, data_validate, data_test = load_data_split_with_noise()
            data = np.vstack((data_train, data_validate))
            mu = data.mean(axis=0)
            s = data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

            return data_train, data_validate, data_test

        if self.data_train is None or self.data_val is None or self.data_test is None:
            data_train, data_val, data_test = load_data_normalised()

            self.data_train = data_train  # Dataset(data_train)
            self.data_val = data_val  # Dataset(data_val)
            self.data_test = data_test  # Dataset(data_test)

            assert data_train.shape == (self.num_train, self.num_features)
            assert data_val.shape == (self.num_valid, self.num_features)
            assert data_test.shape == (self.num_test, self.num_features)

        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`."""

    def train_dataloader(self):
        return DataLoader(
            dataset=torch.FloatTensor(self.data_train),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=torch.FloatTensor(self.data_val),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=torch.FloatTensor(self.data_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
