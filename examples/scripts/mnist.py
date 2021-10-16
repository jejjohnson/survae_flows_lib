import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))


import numpy as np

# Data
from torch.utils.data import DataLoader, Dataset

# PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam

# pytorch lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

# Plot
import matplotlib.pyplot as plt
import corner

# logging
import wandb


CONFIG = {"dataset": "mnist"}
## Initialize wandb logger
wandb_logger = WandbLogger(
    project="gfs4ml",
    entity="ipl_uv",
)
wandb_logger.experiment.config.update(CONFIG)

# %%
from pl_bolts.datasets import MNIST
from pl_bolts.datamodules import MNISTDataModule
from torchvision.transforms import ToTensor, Compose
from survae.transforms.preprocess import Quantize

# %%
# Define transformations

num_bits = 8
trans_train = [ToTensor(), Quantize(num_bits)]
trans_test = [ToTensor(), Quantize(num_bits)]

# get datasets
ds_train = MNIST(
    train=True,
    transform=Compose(trans_train),
    root="/datadrive/eman/survae_flows_lib/data",
)
ds_test = MNIST(
    train=False,
    transform=Compose(trans_train),
    root="/datadrive/eman/survae_flows_lib/data",
)


# create dataloaders
batch_size = 32
train_loader = DataLoader(
    ds_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
)
test_loader = DataLoader(
    ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
)


# ====================
# MODEL
# ====================
from survae.flows import Flow
from survae.distributions import StandardNormal
from survae.transforms.bijections.elementwise_nonlinear import (
    GaussianMixtureCDF,
    InverseGaussCDF,
)
from survae.transforms.bijections.conv1x1 import Conv1x1Householder
from survae.transforms.bijections.linear_orthogonal import FastHouseholder
from survae.transforms.surjections.dequantization_uniform import UniformDequantization
from survae.transforms.bijections.reshape import Reshape
from survae.transforms.bijections.squeeze import Squeeze2d
from survae.transforms.surjections.slice import Slice

n_channels, height, width = 1, 28, 28
total_dims = np.prod((n_channels, height, width))


# transforms
num_mixtures = 8

transforms = [
    UniformDequantization(num_bits=8),
    GaussianMixtureCDF((1, 28, 28), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(1, 10),
    GaussianMixtureCDF((1, 28, 28), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(1, 10),
    GaussianMixtureCDF((1, 28, 28), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(1, 10),
    # (1,28,28) -> (4,14,14)
    Squeeze2d(),
    GaussianMixtureCDF((4, 14, 14), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 10),
    GaussianMixtureCDF((4, 14, 14), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 10),
    GaussianMixtureCDF((4, 14, 14), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 10),
    GaussianMixtureCDF((4, 14, 14), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 10),
    GaussianMixtureCDF((4, 14, 14), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 10),
    # (4,14,14) -> (16,7,7)
    Squeeze2d(),
    GaussianMixtureCDF((16, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(16, 20),
    GaussianMixtureCDF((16, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(16, 20),
    GaussianMixtureCDF((16, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(16, 20),
    GaussianMixtureCDF((16, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(16, 20),
    GaussianMixtureCDF((16, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(16, 20),
    # (16,7,7) -> (8,7,7)
    Slice(StandardNormal((8, 7, 7)), num_keep=8),
    GaussianMixtureCDF((8, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(8, 20),
    GaussianMixtureCDF((8, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(8, 20),
    GaussianMixtureCDF((8, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(8, 20),
    GaussianMixtureCDF((8, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(8, 20),
    GaussianMixtureCDF((8, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(8, 20),
    GaussianMixtureCDF((8, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(8, 20),
    # (8,7,7) -> (4,7,7)
    Slice(StandardNormal((4, 7, 7)), num_keep=4),
    GaussianMixtureCDF((4, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 4),
    GaussianMixtureCDF((4, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 4),
    GaussianMixtureCDF((4, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 4),
    GaussianMixtureCDF((4, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 4),
    GaussianMixtureCDF((4, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 4),
    GaussianMixtureCDF((4, 7, 7), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(4, 4),
    # (4,7,7) -> (2,7,7)
    Slice(StandardNormal((2, 7, 7)), num_keep=2),
    # (2,7,7) -> (98,)
    Reshape((2, 7, 7), (np.prod((2, 7, 7)),)),
    GaussianMixtureCDF((98,), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    FastHouseholder(98, 20),
    GaussianMixtureCDF((98,), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    FastHouseholder(98, 20),
    GaussianMixtureCDF((98,), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    FastHouseholder(98, 20),
    GaussianMixtureCDF((98,), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    FastHouseholder(98, 20),
    GaussianMixtureCDF((98,), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    FastHouseholder(98, 20),
]

# base distribution
base_dist = StandardNormal((np.prod((2, 7, 7)),))

# flow model
model = Flow(base_dist=base_dist, transforms=transforms)

### TEST

test_x = ds_test.data[:500].unsqueeze(1)
loss = model.log_prob(test_x)


from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
import numpy as np


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tv_F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


show(make_grid(test_x[:16] / 255))


z, ldj = model.forward_transform(test_x)

x_approx = model.inverse_transform(z[:32])


show(make_grid(x_approx[:32] / 255))


fig = corner.corner(z.detach().numpy()[:, :10])

# DEMO LATENT SPACE
samples = model.sample(32).detach()

show(make_grid(samples[:32] / 255))

### Trainer

# %%
import pytorch_lightning as pl
import math


class LearnerImage(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, _ = batch

        # loss function
        loss = -self.model.log_prob(x).sum() / (math.log(2) * x.numel())
        self.log("train_nll", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_dataloader(self):
        return train_loader


# initialize trainer
learn = LearnerImage(model)


# %%
n_epochs = 50

# initialize trainer
trainer = pl.Trainer(min_epochs=1, max_epochs=n_epochs, gpus="1", logger=wandb_logger)

# train model
trainer.fit(learn)


z_latent, ldj = model.forward_transform(test_x)


# %%
fig = corner.corner(z_latent.detach().numpy()[:, :10])

## Sampling

samples = model.sample(32).detach()


# %%
show(make_grid(samples[:32] / 255))
