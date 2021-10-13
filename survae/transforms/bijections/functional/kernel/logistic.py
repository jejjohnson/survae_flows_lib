import torch
import torch.nn.functional as F
from survae.transforms.bijections.functional.iterative_inversion import (
    bisection_inverse,
)
from survae.transforms.bijections.functional.mixtures.utils_logistic import (
    logistic_log_cdf,
    logistic_log_pdf,
)


def logistic_kernel_transform(
    inputs, means, log_scales, eps=1e-10, max_iters=100, inverse=False
):
    """
    Univariate mixture of logistics transform.

    Args:
        inputs: torch.Tensor, shape (shape,)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        eps: float, tolerance for bisection |f(x) - z_est| < eps
        max_iters: int, maximum iterations for bisection
        inverse: bool, if True, return inverse
    """

    def kernel_cdf(x):
        return logistic_kernel_log_cdf(x.unsqueeze(-1), means, log_scales).exp()

    def kernel_pdf(x):
        return logistic_kernel_log_pdf(x.unsqueeze(-1), means, log_scales).exp()

    if inverse:
        max_scales = torch.sum(torch.exp(log_scales), dim=-1, keepdim=True)
        init_lower, _ = (means - 20 * max_scales).min(dim=-1)
        init_upper, _ = (means + 20 * max_scales).max(dim=-1)
        return bisection_inverse(
            fn=lambda x: kernel_cdf(x),
            z=inputs,
            init_x=torch.zeros_like(inputs),
            init_lower=init_lower,
            init_upper=init_upper,
            eps=eps,
            max_iters=max_iters,
        )
    else:
        z = kernel_cdf(inputs)
        ldj = kernel_pdf(inputs)
        return z, ldj


import numpy as np


def logistic_kernel_log_cdf(x, means, log_scales):
    n_datapoints = means.shape[-1]
    log_cdfs = -F.softplus(-(x - means) / log_scales.exp()) - np.log(n_datapoints)
    log_cdf = torch.logsumexp(log_cdfs, dim=-1)
    return log_cdf


def logistic_kernel_log_pdf(x, means, log_scales):
    n_datapoints = means.shape[-1]
    log_pdfs = (
        -((x - means) / log_scales.exp())
        - log_scales
        - 2 * F.softplus(-(x - means) / log_scales.exp())
        - np.log(n_datapoints)
    )
    log_pdf = torch.logsumexp(log_pdfs, dim=-1)
    return log_pdf
