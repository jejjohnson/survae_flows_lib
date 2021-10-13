from typing import Tuple, Optional
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
from survae.transforms.bijections.functional.kernel.logistic import (
    logistic_kernel_transform,
)
from survae.utils import sum_except_batch
from survae.transforms.bijections import Bijection
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional.mixtures import (
    gaussian_mixture_transform,
    logistic_mixture_transform,
)


from nflows.transforms import PiecewiseRationalQuadraticCDF
from nflows.transforms.splines.rational_quadratic import (
    DEFAULT_MIN_BIN_HEIGHT,
    DEFAULT_MIN_BIN_WIDTH,
    DEFAULT_MIN_DERIVATIVE,
)


class LeakyReLU(Bijection):
    def __init__(self, negative_slope=1e-2):
        super(LeakyReLU, self).__init__()
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.log(torch.as_tensor(self.negative_slope))

    def forward(self, x):
        z = F.leaky_relu(x, negative_slope=self.negative_slope)
        mask = x < 0
        ldj = self.log_negative_slope * mask.float()
        ldj = sum_except_batch(ldj)
        return z, ldj

    def inverse(self, z):
        x = F.leaky_relu(z, negative_slope=(1 / self.negative_slope))
        return x


class SneakyReLU(Bijection):
    """
    SneakyReLU as proposed in [1].

    [1] Finzi et al. 2019, Invertible Convolutional Networks
        https://invertibleworkshop.github.io/accepted_papers/pdfs/INNF_2019_paper_26.pdf
    """

    def __init__(self, negative_slope=0.1):
        super(SneakyReLU, self).__init__()
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        self.negative_slope = negative_slope
        self.alpha = (1 - negative_slope) / (1 + negative_slope)

    def forward(self, x):
        z = (x + self.alpha * (torch.sqrt(1 + x ** 2) - 1)) / (1 + self.alpha)
        ldj = torch.log(1 + self.alpha * x / torch.sqrt(1 + x ** 2)) - math.log(
            1 + self.alpha
        )
        ldj = sum_except_batch(ldj)
        return z, ldj

    def inverse(self, z):
        b = (1 + self.alpha) * z + self.alpha
        sqrt = torch.sqrt(
            self.alpha ** 2 + (self.alpha ** 2) * (b ** 2) - self.alpha ** 4
        )
        x = (sqrt - b) / (self.alpha ** 2 - 1)
        return x


class Tanh(Bijection):
    def forward(self, x):
        # The "+ 1e-45" bit is for numerical stability. Otherwise the ldj will be -inf where any element of x is around
        # 6.0 or greater, since torch.tanh() returns 1.0 around that point. This way it maxes out around -103.2789.
        z = torch.tanh(x)
        ldj = torch.log(1 - z ** 2 + 1e-45)
        ldj = sum_except_batch(ldj)
        return z, ldj

    def inverse(self, z):
        assert torch.min(z) >= -1 and torch.max(z) <= 1, "z must be in [-1,1]"
        x = 0.5 * torch.log((1 + z) / (1 - z))
        return x


class Sigmoid(Bijection):
    def __init__(self, temperature=1, eps=0.0):
        super(Sigmoid, self).__init__()
        self.eps = eps
        self.register_buffer("temperature", torch.Tensor([temperature]))

    def forward(self, x):
        x = self.temperature * x
        z = torch.sigmoid(x)
        ldj = sum_except_batch(
            torch.log(self.temperature) - F.softplus(-x) - F.softplus(x)
        )
        return z, ldj

    def inverse(self, z):
        assert torch.min(z) >= 0 and torch.max(z) <= 1, "input must be in [0,1]"
        z = torch.clamp(z, self.eps, 1 - self.eps)
        x = (1 / self.temperature) * (torch.log(z) - torch.log1p(-z))
        return x


class Logit(Sigmoid):
    def __init__(self, temperature=1, eps=1e-6, catch_error: bool = False):
        super(Logit, self).__init__()
        self.eps = eps
        self.catch_error = catch_error
        self.register_buffer("temperature", torch.Tensor([temperature]))

    def forward(self, x):
        if self.catch_error:
            assert torch.min(x) >= 0 and torch.max(x) <= 1, "x must be in [0,1]"
        x = torch.clamp(x, self.eps, 1 - self.eps)

        z = (1 / self.temperature) * (torch.log(x) - torch.log1p(-x))
        ldj = -sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * z)
            - F.softplus(self.temperature * z)
        )
        return z, ldj

    def inverse(self, z):
        z = self.temperature * z
        x = torch.sigmoid(z)
        return x


class InverseGaussCDF(Bijection):
    def __init__(self, eps: float = 1e-6, catch_error: bool = False):
        super(InverseGaussCDF, self).__init__()
        self.eps = eps
        self.catch_error = catch_error
        self.base_dist = dist.Normal(loc=0, scale=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.catch_error:
            assert torch.min(x) >= 0 and torch.max(x) <= 1, "x must be in [0,1]"

        x = torch.clamp(x, self.eps, 1 - self.eps)

        # forward transformation
        z = self.base_dist.icdf(x)

        # log determinant jacobian
        ldj = -self.base_dist.log_prob(z)

        ldj = sum_except_batch(ldj)

        return z, ldj

    def inverse(self, z: torch.Tensor) -> torch.Tensor:

        x = self.base_dist.cdf(z)

        return x


class Softplus(Bijection):
    def __init__(self, eps=1e-7):
        super(Softplus, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        z = softplus(x) = log(1+exp(z))
        ldj = log(dsoftplus(x)/dx) = log(1/(1+exp(-x))) = log(sigmoid(x))
        """
        z = F.softplus(x)
        ldj = sum_except_batch(F.logsigmoid(x))
        return z, ldj

    def inverse(self, z):
        """x = softplus_inv(z) = log(exp(z)-1) = z + log(1-exp(-z))"""
        zc = z.clamp(self.eps)
        return z + torch.log1p(-torch.exp(-zc))


class SoftplusInverse(Bijection):
    def __init__(self, eps=1e-7):
        super(SoftplusInverse, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
        z = softplus_inv(x) = log(exp(x)-1) = x + log(1-exp(-x))
        ldj = log(dsoftplus_inv(x)/dx)
            = log(exp(x)/(exp(x)-1))
            = log(1/(1-exp(-x)))
            = -log(1-exp(-x))
        """
        xc = x.clamp(self.eps)
        z = xc + torch.log1p(-torch.exp(-xc))
        ldj = -sum_except_batch(torch.log1p(-torch.exp(-xc)))
        return z, ldj

    def inverse(self, z):
        """x = softplus(z) = log(1+exp(z))"""
        return F.softplus(z)


class GaussianMixtureCDF(Bijection):
    def __init__(
        self,
        input_shape: Tuple[int],
        num_mixtures: int = 8,
        eps: float = 1e-10,
        max_iters: int = 100,
    ):
        super(GaussianMixtureCDF, self).__init__()
        param_shapes = input_shape + (num_mixtures,)
        self.num_mixtures = num_mixtures
        self.max_iters = max_iters
        self.eps = eps
        self.means = nn.Parameter(torch.randn(param_shapes), requires_grad=True)
        self.log_scales = nn.Parameter(torch.zeros(param_shapes), requires_grad=True)
        self.logit_weights = nn.Parameter(torch.zeros(param_shapes), requires_grad=True)

    def _elementwise(self, inputs, inverse):

        x = gaussian_mixture_transform(
            inputs=inputs,
            logit_weights=self.logit_weights,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=inverse,
        )

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def forward(self, x):
        return self._elementwise(x, inverse=False)

    def inverse(self, z):
        return self._elementwise(z, inverse=True)


class LogisticMixtureCDF(GaussianMixtureCDF):
    def __init__(
        self,
        input_shape: Tuple[int],
        num_mixtures: int = 8,
        eps: float = 1e-10,
        max_iters: int = 100,
    ):
        super(LogisticMixtureCDF, self).__init__(
            input_shape=input_shape,
            num_mixtures=num_mixtures,
            eps=eps,
            max_iters=max_iters,
        )

    def _elementwise(self, inputs, inverse):

        x = logistic_mixture_transform(
            inputs=inputs,
            logit_weights=self.logit_weights,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=inverse,
        )

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj


class KernelLogisticCDF(GaussianMixtureCDF):
    def __init__(
        self,
        input_shape: Tuple[int],
        num_mixtures: int = 12,
        eps: float = 1e-10,
        max_iters: int = 100,
    ):
        super(GaussianMixtureCDF, self).__init__()
        param_shapes = input_shape + (num_mixtures,)
        self.num_mixtures = num_mixtures
        self.max_iters = max_iters
        self.eps = eps
        self.means = nn.Parameter(torch.randn(param_shapes), requires_grad=True)
        self.log_scales = nn.Parameter(torch.zeros(param_shapes), requires_grad=True)

    def _elementwise(self, inputs, inverse):

        x = logistic_kernel_transform(
            inputs=inputs,
            means=self.means,
            log_scales=self.log_scales,
            eps=self.eps,
            max_iters=self.max_iters,
            inverse=inverse,
        )

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def forward(self, x):
        return self._elementwise(x, inverse=False)

    def inverse(self, z):
        return self._elementwise(z, inverse=True)


class RQSplineCDF(Bijection):
    def __init__(
        self,
        shape: Tuple,
        num_bins: int = 10,
        tails: Optional[str] = "linear",
        tail_bound: float = 5.0,
        identity_init: bool = False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        super(RQSplineCDF, self).__init__()

        self.nflows_transform = PiecewiseRationalQuadraticCDF(
            shape=shape,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            identity_init=identity_init,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    def forward(self, x):
        z, ldj = self.nflows_transform.forward(x)

        return z, ldj

    def inverse(self, z):

        x, _ = self.nflows_transform.inverse(z)

        return x
