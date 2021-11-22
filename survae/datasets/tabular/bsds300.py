from os import path
from typing import Optional, Tuple
import h5py
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from survae.data.tabular.utils import get_data_path
import os
from survae.data.tabular.uci_datamodule import UCIDataModule


class BSDS300DataModule(UCIDataModule):
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
    raw_folder = "uci_maf/data/BSDS300"
    raw_file = "BSDS300.hdf5"
    # data statistics
    num_features = 63
    num_train = 1_000_000
    num_valid = 50_000
    num_test = 250_000

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
        def load_data(split):
            # Taken from https://github.com/bayesiains/nsf/blob/master/data/bsds300.py
            file = h5py.File(self.raw_data_path, "r")
            return np.array(file[split]).astype(np.float32)

        if self.data_train is None or self.data_val is None or self.data_test is None:
            data_train, data_val, data_test = (
                load_data("train"),
                load_data("validation"),
                load_data("test"),
            )

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
