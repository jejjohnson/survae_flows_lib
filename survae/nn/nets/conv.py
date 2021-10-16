"""
Code from:
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers.conv import GatedConv, ConcatELU
from survae.nn.layers.norm import LayerNormChannels


class GatedConvNet(nn.Module):
    def __init__(
        self, c_in: int, c_hidden: int = 32, c_out: int = -1, num_layers: int = 3
    ):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden), LayerNormChannels(c_hidden)]
        layers += [
            ConcatELU(),
            nn.Conv2d(2 * c_hidden, c_out, kernel_size=3, padding=1),
        ]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)
