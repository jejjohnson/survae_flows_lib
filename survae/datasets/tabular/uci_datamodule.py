from os import path
from typing import Optional, Tuple
import tarfile
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from pathlib import Path
from survae.datasets.tabular.utils import get_data_path
import os


class UCIDataModule(LightningDataModule):
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
    uci_folder = "uci_maf"
    uci_file = "data.tar.gz"
    raw_folder = None
    raw_file = None

    def __init__(
        self,
        data_dir: str = get_data_path(),
        download_data: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.download_data = download_data

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        if not self._check_download():
            if self.download_data:
                self.download()
            else:
                raise RuntimeError(
                    "Dataset not found." + " You can use download=True to download it"
                )

        if not self._check_raw():
            self.extract()

    @property
    def raw_uci_data_path(self):
        return os.path.join(self.data_dir, self.uci_folder, self.uci_file)

    @property
    def raw_data_path(self):
        return os.path.join(self.data_dir, self.raw_folder, self.raw_file)

    def _check_download(self):
        return os.path.exists(self.raw_uci_data_path)

    def download(self):
        """Download the data if it doesn't exist in parent_folder already."""
        from six.moves import urllib

        if not os.path.exists(os.path.join(self.data_dir, self.uci_folder)):
            os.makedirs(os.path.join(self.data_dir, self.uci_folder))

        print("Downloading", self.uci_file)
        urllib.request.urlretrieve(self.url, self.raw_uci_data_path)
        print("Completed")

    def _check_raw(self):
        return os.path.exists(self.raw_data_path)

    def extract(self):

        print("Extracting data...")
        tar = tarfile.open(self.raw_uci_data_path)
        tar.extractall(os.path.join(self.data_dir, self.uci_folder))
        tar.close()
        print("Completed!")
