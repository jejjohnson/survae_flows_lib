import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections import Bijection
from nflows.transforms import HouseholderSequence
from survae.transforms.bijections.functional.householder import fast_householder_matrix


class LinearHouseholder(Bijection):
    """
    Linear bijection y=Wx.

    Costs:
        forward = O(BD^2)
        inverse = O(BD^2 + D^3)
        ldj = O(D^3)
    where:
        B = batch size
        D = number of features

    Args:
        num_features: int, Number of features in the input and output.
        orthogonal_init: bool, if True initialize weights to be a random orthogonal matrix (default=True).
        bias: bool, if True a bias is included (default=False).
    """

    def __init__(self, num_features: int, num_householder: int = 2):
        super(LinearHouseholder, self).__init__()
        self.num_features = num_features
        self.nflows_layer = HouseholderSequence(
            features=num_features, num_transforms=num_householder
        )

    def forward(self, x):

        z, ldj = self.nflows_layer.forward(x)

        return z, ldj

    def inverse(self, z):
        x, _ = self.nflows_layer.inverse(z)
        return x


class FastHouseholder(Bijection):
    def __init__(
        self,
        num_features: int,
        num_householder: int = 2,
        fixed: bool = False,
    ):
        super().__init__()

        # init close to identity
        weight = torch.eye(num_features, num_householder)
        weight += torch.randn_like(weight) * 0.1
        weight = weight.transpose(-1, -2)

        self.weight = nn.Parameter(weight)

        nn.init.orthogonal_(self.weight)

        if fixed:
            self.weight = fast_householder_matrix(self.weight)
            self.weight = nn.Parameter(self.weight, requires_grad=False)
            self.register_parameter("weight", self.weight)

    def forward(self, x):

        W = fast_householder_matrix(self.weight)
        z = x.mm(W)

        return z, 0

    def inverse(self, z):
        W_inv = fast_householder_matrix(self.weight).t()
        x = z.mm(W_inv)

        return x
