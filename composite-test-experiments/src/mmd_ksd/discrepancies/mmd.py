from typing import Literal, Sequence

import jax.numpy as jnp
from chex import assert_equal_shape, assert_shape
from jax import Array
from jax.numpy import ndarray
from jax.scipy import stats

from mmd_ksd.discrepancies.kernels import GaussianKernel, Kernel, gram, zero_diagonal
from mmd_ksd.extra_types import Scalar


def mmd_u_stat(kernel: Kernel, xs: ndarray, ys: ndarray) -> ndarray:
    m = xs.shape[0]
    n = ys.shape[0]
    K_xx = gram(kernel, xs, xs)
    K_xx = zero_diagonal(K_xx)
    K_yy = gram(kernel, ys, ys)
    K_yy = zero_diagonal(K_yy)
    K_xy = gram(kernel, xs, ys)
    term1 = K_xx.sum() / (m * (m - 1))
    term2 = -2 * K_xy.sum() / (n * m)
    term3 = K_yy.sum() / (n * (n - 1))
    return term1 + term2 + term3


def agg_mmd_u_stat(kernels: Sequence[Kernel], xs: ndarray, ys: ndarray) -> ndarray:
    return jnp.max(jnp.array([mmd_u_stat(k, xs, ys) for k in kernels]))


def mmd_v_stat(
    kernel: Kernel, xs: ndarray, ys: ndarray, include_yy: bool = True
) -> ndarray:
    """Computes the v-statistic estimate of MMD(xs,ys).

    :param include_yy: When False, does not compute or include the K_yy term. This may
                       be useful when doing optimisation, where the gradient wrt the
                       model parameters does not depend on the K_yy.
    """
    K_xx = gram(kernel, xs, xs)
    K_xy = gram(kernel, xs, ys)
    partial_mmd = K_xx.mean() - 2 * K_xy.mean()
    if include_yy:
        K_yy = gram(kernel, ys, ys)
        return partial_mmd + K_yy.mean()
    else:
        return partial_mmd


def mmd_h_gram(kernel: Kernel, xs: Array, ys: Array) -> Array:
    assert_equal_shape([xs, ys])
    zs = jnp.stack([xs, ys], axis=1)
    return gram(lambda x, y: mmd_h(kernel, x, y), zs, zs)


def mmd_h(k: Kernel, z1: Array, z2: Array) -> Array:
    assert_shape(z1, (2, None))
    assert_shape(z2, (2, None))
    x1, y1 = z1
    x2, y2 = z2
    return k(x1, x2) + k(y1, y2) - k(x1, y2) - k(x2, y1)


def weighted_mmd_v_stat(
    kernel: Kernel, xs: ndarray, ys: ndarray, ws: ndarray, include_yy: bool = True
) -> ndarray:
    K_xx = gram(kernel, xs, xs)
    K_xy = gram(kernel, xs, ys)
    term1 = (ws.transpose() @ K_xx) @ ws
    term2 = -2 * (ws.T @ K_xy).mean()
    if include_yy:
        K_yy = gram(kernel, ys, ys)
        return term1 + term2 + K_yy.mean()
    else:
        return term1 + term2


def compute_optimal_weights(
    us: ndarray,
    u_space_kernel: GaussianKernel,
    u_distribution: Literal["uniform", "gaussian"],
) -> ndarray:
    """Computes the optimal weights when the latent variables are uniform on [0,1]."""
    if not isinstance(u_space_kernel, GaussianKernel):
        raise NotImplementedError(
            "Optimal weight computation only implemented for Gaussian kernels"
        )

    if u_distribution == "uniform":
        kme = _embed_uniform_under_gaussian(us, u_space_kernel.l)
    elif u_distribution == "gaussian":
        kme = _embed_gaussian_under_gaussian(us, u_space_kernel.l)
    else:
        raise NotImplementedError

    m = us.shape[0]
    delta = 1e-7
    C = gram(u_space_kernel, us, us) + delta * jnp.identity(m)
    return jnp.linalg.solve(C, kme)


def _embed_uniform_under_gaussian(us: ndarray, l: Scalar) -> ndarray:
    """Computes the KME of a uniform distribution under a Gaussian kernel.

    :param us: samples from the uniform distribution on [0,1]
    """
    cdf_diff = stats.norm.cdf(1, loc=us, scale=l) - stats.norm.cdf(0, loc=us, scale=l)
    z = jnp.sqrt(2 * jnp.pi) * l * cdf_diff
    return jnp.prod(z, axis=1)


def _embed_gaussian_under_gaussian(us: ndarray, l: Scalar) -> ndarray:
    """Computes the KME of a standard Gaussian distribution under a Gaussian kernel.

    :param us: samples from the  Gaussian distribution
    """
    dim = us.shape[1]
    sigma = 1.0
    return (l**2 / (l**2 + sigma**2)) ** (dim / 2) * jnp.exp(
        -jnp.linalg.norm(us, axis=1) ** 2 / (2 * (l**2 + sigma**2))
    )
