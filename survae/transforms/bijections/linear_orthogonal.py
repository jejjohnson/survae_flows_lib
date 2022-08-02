"""
Taken from:
* https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/modules/orthogonal.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from survae.transforms.bijections import Bijection
from survae.transforms.bijections.functional.householder import householder_matrix_fast, householder_matrix


class LinearOrthogonal(Bijection):
    """
    Linear bijection y=Rx.
    where R is an orthogonal matrix

    Costs:
        forward = O(BD^2)
        inverse = O(BD^2)
        ldj = O(1)
    where:
        B = batch size
        D = number of features

    Args:
        num_features: int, Number of features in the input and output.
        orthogonal_init: bool, if True initialize weights to be a random orthogonal matrix (default=True).
        norm: str, the orthogonal parameterization ("matrix_exp", "householder", "cayley").
    """

    def __init__(self, num_features, orthogonal_init=True, norm="householder"):
        super(LinearOrthogonal, self).__init__()
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
        else:
            raise ValueError(f"Unrecognized orthogonal parameterization: {norm}")

        self.layer = layer

    @property
    def weight(self):
        return self.layer.weight

    @property
    def weight_inv(self):
        return self.layer.weight.t()

    def forward(self, x):
        z = F.linear(x, self.weight)
        ldj = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return z, ldj

    def inverse(self, z):
        x = F.linear(z, self.weight_inv)
        return x


class LinearHouseholder(Bijection):
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
            num_features: int,
            num_reflections: int = 2,
            fixed: bool = False,
            fast: bool = True,
            loop: bool = False,
            stride: int = 2
    ):
        super().__init__()
        self.num_features = num_features
        self.fixed = fixed
        self.loop = loop
        self.fast = fast
        self.stride = stride

        if fixed:
            vs = torch.randn(num_features, num_features)
            vs = vs.transpose(-1, -2)
            self.weight = nn.Parameter(vs)
            nn.init.orthogonal_(self.weight)
        else:
            vs = torch.eye(num_features, num_reflections)
            vs += torch.randn_like(vs) * 0.1
            vs = vs.transpose(-1, -2)
            self.register_parameter("weight", nn.Parameter(vs))

    @property
    def R(self):
        if self.fast:
            return householder_matrix_fast(self.weight, self.stride)
        else:
            return householder_matrix(self.weight, self.loop)

    def forward(self, x):

        if self.fixed:
            z = F.linear(x, self.weight)
        else:
            z = F.linear(x, self.R)

        ldj = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        return z, ldj

    def inverse(self, z):

        if self.fixed:
            x = F.linear(z, self.weight.t())
        else:
            x = F.linear(z, self.R.t())

        return x





# class FastHouseholder(Bijection):
#     """
#     Linear bijection y=Rx with an orthogonal parameterization via the Householder transformation. This
#     restricts the form of the rotation matrix, R, to be orthogonal which enables a very cheap Jacobian
#     calculation, i.e. it's zero.
#
#     Costs:
#         forward = O(BD^2)
#         inverse = O(BD^2 + D^3)
#         ldj = O(1)
#     where:
#         B = batch size
#         D = number of features
#
#     Args:
#         num_features: int, Number of features in the input and output.
#         orthogonal_init: bool, if True initialize weights to be a random orthogonal matrix (default=True).
#     """
#
#     def __init__(
#         self,
#         num_features: int,
#         num_householder: int = 2,
#         fixed: bool = False,
#     ):
#         super().__init__()
#
#         # init close to identity
#         weight = torch.eye(num_features, num_householder)
#         weight += torch.randn_like(weight) * 0.1
#         weight = weight.transpose(-1, -2)
#         print(weight.shape)
#
#         self.weight = nn.Parameter(weight)
#
#         # nn.init.orthogonal_(self.weight)
#
#         if fixed:
#             self.weight = fast_householder_matrix(self.weight)
#             self.weight = nn.Parameter(self.weight, requires_grad=False)
#             self.register_parameter("weight", self.weight)
#
#     def forward(self, x):
#
#         W = fast_householder_matrix(self.weight)
#         z = x.mm(W)
#
#         return z, 0
#
#     def inverse(self, z):
#         W_inv = fast_householder_matrix(self.weight).t()
#         x = z.mm(W_inv)
#
#         return x
