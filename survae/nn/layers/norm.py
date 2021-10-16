"""
Code from:
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html

"""
import torch.nn as nn


class LayerNormChannels(nn.Module):
    def __init__(self, c_in):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
