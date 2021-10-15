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

# pytorch lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

# Optim
from torch.optim import Adam

# Plot
import matplotlib.pyplot as plt
import corner

# logging
import wandb


# %%
from pl_bolts.datasets import CIFAR10
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision.transforms import ToTensor, Compose
from survae.transforms.preprocess import Quantize


num_bits = 8
trans_train = [ToTensor(), Quantize(num_bits)]
trans_test = [ToTensor(), Quantize(num_bits)]

# get datasets
ds_train = CIFAR10(
    train=True,
    transform=Compose(trans_train),
    data_dir="/datadrive/eman/survae_flows_lib/data",
)
ds_test = CIFAR10(
    train=False,
    transform=Compose(trans_test),
    data_dir="/datadrive/eman/survae_flows_lib/data",
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
from survae.transforms.surjections.dequantization_uniform import UniformDequantization
from survae.transforms.bijections.reshape import Reshape
from survae.transforms.bijections.squeeze import Squeeze2d
from survae.transforms.surjections.slice import Slice

n_channels, height, width = 3, 32, 32
total_dims = np.prod((n_channels, height, width))


# transforms
num_mixtures = 8

transforms = [
    UniformDequantization(num_bits=8),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    GaussianMixtureCDF((3, 32, 32), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(3, 3),
    # (3,32,32) -> (12,16,16)
    Squeeze2d(),
    # (12,16,16) -> (6,16,16)
    Slice(StandardNormal((6, 16, 16)), num_keep=6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    GaussianMixtureCDF((6, 16, 16), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(6, 6),
    # (6,16,16) -> (24,8,8)
    Squeeze2d(),
    # (24,8,8) -> (12,8,8)
    Slice(StandardNormal((12, 8, 8)), num_keep=12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    GaussianMixtureCDF((12, 8, 8), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(12, 12),
    # (12,8,8) -> (48,4,4)
    Squeeze2d(),
    # (48,4,4) -> (24,4,4)
    Slice(StandardNormal((24, 4, 4)), num_keep=24),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    GaussianMixtureCDF((24, 4, 4), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(24, 12),
    # (24,4,4) -> (96,2,2)
    Squeeze2d(),
    # (96,2,2) -> (48,2,2)
    Slice(StandardNormal((48, 2, 2)), num_keep=48),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    GaussianMixtureCDF((48, 2, 2), num_mixtures=num_mixtures),
    InverseGaussCDF(),
    Conv1x1Householder(48, 12),
    # (48,2,2) -> (192,)
    Reshape((48, 2, 2), (np.prod((48, 2, 2)),)),
]

# base distribution
base_dist = StandardNormal((np.prod((48, 2, 2)),))

# flow model
model = Flow(base_dist=base_dist, transforms=transforms)


### TEST

from einops import rearrange

test_x = ds_test.data[:500]

test_x = rearrange(test_x, "B (C H W) -> B C H W", C=3, H=32, W=32)

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


# =====================
# TRAINER
# =====================

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def train_dataloader(self):
        return train_loader


# initialize trainer
learn = LearnerImage(model)


n_epochs = 50

# initialize trainer
trainer = pl.Trainer(min_epochs=1, max_epochs=n_epochs, gpus="0", logger=wandb_logger)

# train model
trainer.fit(learn)

# LATENT SPACE


z_latent, ldj = model.forward_transform(test_x)


fig = corner.corner(z_latent.detach().numpy()[:, :10])

# SAMPLING

samples = model.sample(32).detach()

show(make_grid(samples[:32] / 255))
