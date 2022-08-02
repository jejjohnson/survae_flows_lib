from functools import reduce
from operator import mul
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections import Bijection
from survae.transforms.bijections.functional.householder import (
    householder_matrix,
)
from survae.transforms.bijections.functional.householder import householder_matrix_fast
from torch.nn.utils import parametrizations




class Conv1x1Householder(Bijection):
    """
    Linear bijection y=Rx with an orthogonal parameterization via the Householder transformation. This
    restricts the form of the rotation matrix, R, to be orthogonal which enables a very cheap Jacobian
    calculation, i.e. it's zero.

    Costs:
        forward = O(BD^2)
        inverse = O(BD^2 + D^3)
        ldj = O(1)
    where:
        B = batch size
        D = number of features

    Args:
        num_features: int, Number of features in the input and output.
        orthogonal_init: bool, if True initialize weights to be a random orthogonal matrix (default=True).
    """
    def __init__(
            self,
            num_channels: int,
            num_reflections: int = 2,
            fixed: bool = False,
            fast: bool = True,
            loop: bool = False,
            stride: int = 2
    ):
        super().__init__()
        self.num_channels = num_channels
        self.fixed = fixed
        self.loop = loop
        self.fast = fast
        self.stride = stride

        if fixed:
            vs = torch.randn(num_channels, num_channels)
            vs = vs.transpose(-1, -2)
            self.weight = nn.Parameter(vs)
            nn.init.orthogonal_(self.weight)
        else:
            vs = torch.eye(num_channels, num_reflections)
            vs += torch.randn_like(vs) * 0.1
            vs = vs.transpose(-1, -2)
            self.register_parameter("weight", nn.Parameter(vs))

    @property
    def R(self):
        if self.fast:
            return householder_matrix_fast(self.weight, self.stride)
        else:
            return householder_matrix(self.weight, self.loop)

    def _conv(self, weight, v):

        # Get tensor dimensions
        _, channel, *features = v.shape
        n_feature_dims = len(features)

        # expand weight matrix
        fill = (1,) * n_feature_dims
        weight = weight.view(channel, channel, *fill)

        if n_feature_dims == 1:
            return F.conv1d(v, weight)
        elif n_feature_dims == 2:
            return F.conv2d(v, weight)
        elif n_feature_dims == 3:
            return F.conv3d(v, weight)
        else:
            raise ValueError(f"Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d")

    def forward(self, x):

        Q = self.weight if self.fixed else self.R

        z = self._conv(Q, x)

        ldj = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        return z, ldj

    def inverse(self, z):

        Q = self.weight if self.fixed else self.R

        x = self._conv(Q.t(), z)

        return x


class Conv1x1Orthogonal(Bijection):
    """
    Linear bijection y=Rx with an orthogonal parameterization via the Householder transformation. This
    restricts the form of the rotation matrix, R, to be orthogonal which enables a very cheap Jacobian
    calculation, i.e. it's zero.

    Costs:
        forward = O(BD^2)
        inverse = O(BD^2 + D^3)
        ldj = O(1)
    where:
        B = batch size
        D = number of features

    Args:
        num_features: int, Number of features in the input and output.
        orthogonal_init: bool, if True initialize weights to be a random orthogonal matrix (default=True).
    """
    def __init__(self, num_features, orthogonal_init=True, norm="householder"):
        super(Conv1x1Orthogonal, self).__init__()
        self.num_features = num_features
        self.orthogonal_init = orthogonal_init

        layer = nn.Linear(num_features, num_features, bias=False)

        if self.orthogonal_init:
            nn.init.orthogonal_(layer.weight)
        else:
            nn.init.xavier_normal_(layer.weight)

        if norm in ["matrix_exp", "householder", "cayley"]:
            layer = parametrizations.orthogonal(
                layer, name="weight", orthogonal_map=norm,
            )
        elif norm == "spectral":
            layer = parametrizations.spectral_norm(
                layer, name="weight"
            )
        else:
            raise ValueError(f"Unrecognized orthogonal parameterization: {norm}")

        self.layer = layer

    @property
    def weight(self):
        return self.layer.weight

    @property
    def weight_inv(self):
        return self.layer.weight.t()

    def _conv(self, weight, v):

        # Get tensor dimensions
        _, channel, *features = v.shape
        n_feature_dims = len(features)

        # expand weight matrix
        fill = (1,) * n_feature_dims
        weight = weight.view(channel, channel, *fill)

        if n_feature_dims == 1:
            return F.conv1d(v, weight)
        elif n_feature_dims == 2:
            return F.conv2d(v, weight)
        elif n_feature_dims == 3:
            return F.conv3d(v, weight)
        else:
            raise ValueError(f"Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d")

    def forward(self, x):

        z = self._conv(self.weight, x)

        ldj = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        return z, ldj

    def inverse(self, z):

        x = self._conv(self.weight.t(), z)

        return x

# class Conv1x1Householder(Conv1x1):
#     """
#     Invertible 1x1 Convolution [1].
#     The weight matrix is initialized as a random rotation matrix
#     as described in Section 3.2 of [1].
#
#     Args:
#         num_channels (int): Number of channels in the input and output.
#         orthogonal_init (bool): If True, initialize weights to be a random orthogonal matrix (default=True).
#         slogdet_cpu (bool): If True, compute slogdet on cpu (default=True).
#
#     Note:
#         torch.slogdet appears to run faster on CPU than on GPU.
#         slogdet_cpu is thus set to True by default.
#
#     References:
#         [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
#             Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
#     """
#
#     def __init__(
#         self,
#         num_channels: int,
#         num_householder: int = 2,
#         fixed: bool = False,
#     ):
#         super().__init__()
#
#         # init close to identity
#         weight = torch.eye(num_channels, num_householder)
#         weight += torch.randn_like(weight) * 0.1
#         weight = weight.transpose(-1, -2)
#
#         self.weight = nn.Parameter(weight)
#
#         nn.init.orthogonal_(self.weight)
#
#         if fixed:
#             self.weight = householder_matrix_fast(self.weight)
#             self.weight = nn.Parameter(self.weight, requires_grad=False)
#             self.register_parameter("weight", self.weight)
#
#     def _conv(self, weight, v):
#
#         # Get tensor dimensions
#         _, channel, *features = v.shape
#         n_feature_dims = len(features)
#
#         # expand weight matrix
#         fill = (1,) * n_feature_dims
#         weight = weight.view(channel, channel, *fill)
#
#         if n_feature_dims == 1:
#             return F.conv1d(v, weight)
#         elif n_feature_dims == 2:
#             return F.conv2d(v, weight)
#         elif n_feature_dims == 3:
#             return F.conv3d(v, weight)
#         else:
#             raise ValueError(f"Got {n_feature_dims}d tensor, expected 1d, 2d, or 3d")
#
#     def _logdet(self, x_shape):
#         n_batches, _, *dims = x_shape
#
#         ldj_per_pixel = torch.zeros_like(self.weight)
#
#         ldj = ldj_per_pixel * reduce(mul, dims)
#
#         return ldj.expand([n_batches]).to(self.weight.device)
#
#     def forward(self, x):
#
#         # construct householder matrix
#         Q = householder_matrix(self.weight)
#
#         z = self._conv(Q, x)
#         # ldj = self._logdet(x.shape)
#         return z, 0
#
#     def inverse(self, z):
#         # construct householder matrix
#         Q_inv = householder_matrix(self.weight).t()
#
#         x = self._conv(Q_inv, z)
#
#         return x