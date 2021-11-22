from os import path
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import get_data_path
import os
from collections import Counter
from survae.data.tabular.uci_datamodule import UCIDataModule


class HEPMASSDataModule(UCIDataModule):
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
    raw_folder = "uci_maf/data/hepmass"
    raw_file = ""
    num_features = 21
    num_train = 315_123
    num_valid = 35_013
    num_test = 174_987

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

        # self.dims is returned when you call datamodule.size()
        # self.dims is returned when you call datamodule.size()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        def load_data(path):

            data_train = pd.read_csv(
                filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False
            )
            data_test = pd.read_csv(
                filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False
            )

            return data_train, data_test

        def load_data_no_discrete(path):
            """Loads the positive class examples from the first 10% of the dataset."""
            data_train, data_test = load_data(path)

            # Gets rid of any background noise examples i.e. class label 0.
            data_train = data_train[data_train[data_train.columns[0]] == 1]
            data_train = data_train.drop(data_train.columns[0], axis=1)
            data_test = data_test[data_test[data_test.columns[0]] == 1]
            data_test = data_test.drop(data_test.columns[0], axis=1)
            # Because the data_ set is messed up!
            data_test = data_test.drop(data_test.columns[-1], axis=1)

            return data_train, data_test

        def load_data_no_discrete_normalised(path):

            data_train, data_test = load_data_no_discrete(path)
            mu = data_train.mean()
            s = data_train.std()
            data_train = (data_train - mu) / s
            data_test = (data_test - mu) / s

            return data_train, data_test

        def load_data_no_discrete_normalised_as_array(path):

            data_train, data_test = load_data_no_discrete_normalised(path)
            data_train, data_test = data_train.values, data_test.values

            i = 0
            # Remove any features that have too many re-occurring real values.
            features_to_remove = []
            for feature in data_train.T:
                c = Counter(feature)
                max_count = np.array([v for k, v in sorted(c.items())])[0]
                if max_count > 5:
                    features_to_remove.append(i)
                i += 1
            data_train = data_train[
                :,
                np.array(
                    [
                        i
                        for i in range(data_train.shape[1])
                        if i not in features_to_remove
                    ]
                ),
            ]
            data_test = data_test[
                :,
                np.array(
                    [
                        i
                        for i in range(data_test.shape[1])
                        if i not in features_to_remove
                    ]
                ),
            ]

            N = data_train.shape[0]
            N_validate = int(N * 0.1)
            data_validate = data_train[-N_validate:]
            data_train = data_train[0:-N_validate]

            return data_train, data_validate, data_test

        if self.data_train is None or self.data_val is None or self.data_test is None:
            data_train, data_val, data_test = load_data_no_discrete_normalised_as_array(
                self.raw_data_path
            )

            self.data_train = data_train  # Dataset(data_train)
            self.data_val = data_val  # Dataset(data_val)
            self.data_test = data_test  # Dataset(data_test)

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
