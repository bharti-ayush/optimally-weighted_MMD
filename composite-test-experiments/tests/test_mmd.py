import jax.numpy as jnp
import jax.random
import numpy as np
from jax import Array, grad
from jax.random import PRNGKey, uniform
from scipy import stats
from scipy.spatial.distance import cdist

from mmd_ksd.discrepancies import kernels
from mmd_ksd.discrepancies.kernels import GaussianKernel
from mmd_ksd.discrepancies.mmd import (
    compute_optimal_weights,
    mmd_h_gram,
    mmd_v_stat,
    weighted_mmd_v_stat,
)
from mmd_ksd.distributions import Gaussian


def test__mmd_v_stat__param_gradient_equal_regardless_yy_included():
    rng = PRNGKey(seed=9088999)
    rng1, rng2 = jax.random.split(rng, num=2)
    ys = Gaussian(loc=0.0, scale=0.0).sample(rng1, n=10)
    kernel = GaussianKernel(l=0.5)

    def sample(mean: Array) -> Array:
        return Gaussian(loc=mean, scale=1.0).sample(rng2, n=5)

    def loss_with_yy(mean: Array) -> Array:
        return mmd_v_stat(kernel, sample(mean), ys, include_yy=True)

    def loss_without_yy(mean: Array) -> Array:
        return mmd_v_stat(kernel, sample(mean), ys, include_yy=False)

    g_with_yy = grad(loss_with_yy)(1.0)
    g_without_yy = grad(loss_without_yy)(1.0)

    assert jnp.allclose(g_without_yy, g_with_yy)


def test__weighted_mmd_v_stat__param_gradient_equal_regardless_yy_included():
    rng = PRNGKey(seed=9088999)
    rng1, rng2, rng3 = jax.random.split(rng, num=3)
    ys = Gaussian(loc=0.0, scale=0.0).sample(rng1, n=10)
    ws = uniform(rng3, shape=(5,))
    kernel = GaussianKernel(l=0.5)

    def sample(mean: Array) -> Array:
        return Gaussian(loc=mean, scale=1.0).sample(rng2, n=5)

    def loss_with_yy(mean: Array) -> Array:
        return weighted_mmd_v_stat(kernel, sample(mean), ys, ws, include_yy=True)

    def loss_without_yy(mean: Array) -> Array:
        return weighted_mmd_v_stat(kernel, sample(mean), ys, ws, include_yy=False)

    g_with_yy = grad(loss_with_yy)(1.0)
    g_without_yy = grad(loss_without_yy)(1.0)

    assert jnp.allclose(g_without_yy, g_with_yy)


def test__mmd_h_gram__has_same_mean_as_mmd_v_stat():
    rng = PRNGKey(seed=4234234)
    rng, rng_input = jax.random.split(rng)
    xs = uniform(rng_input, shape=(100, 1), minval=-10, maxval=10)
    rng, rng_input = jax.random.split(rng)
    ys = uniform(rng_input, shape=(100, 1), minval=-10, maxval=10)
    kernel = kernels.GaussianKernel(l=1.0)

    actual = mmd_h_gram(kernel, xs, ys).mean()
    expected = mmd_v_stat(kernel, xs, ys)

    assert jnp.allclose(actual, expected)


def test__weighted_mmd_v_stat__returns_same_as_reference_implementation():
    rng = PRNGKey(seed=4234234)
    rng, rng_input = jax.random.split(rng)
    xs = uniform(rng_input, shape=(100, 10), minval=0.0, maxval=1.0)
    rng, rng_input = jax.random.split(rng)
    ys = uniform(rng_input, shape=(100, 10), minval=0.0, maxval=1.0)
    rng, rng_input = jax.random.split(rng)
    ws = uniform(rng_input, shape=(100,), minval=0.0, maxval=1.0)

    l = 0.5
    mmd = weighted_mmd_v_stat(kernels.GaussianKernel(l), xs, ys, ws)
    expected = _reference_mmd_weighted(xs, ys, ws, l)

    assert jnp.allclose(mmd, expected)


def _reference_mmd_weighted(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, lengthscale: float
) -> np.ndarray:
    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()
        w = np.array(w, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    xy = np.concatenate((x, y), axis=0)

    K = _kernel_matrix(xy, xy, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m : (m + n), m : (m + n)]
    kxy = K[0:m, m : (m + n)]

    # first sum
    sum1 = np.matmul(np.matmul(w.transpose(), kxx), w)

    # second sum
    sum2 = np.sum(np.matmul(w.transpose(), kxy))

    # third sum
    sum3 = (1 / n**2) * np.sum(kyy)

    return sum1 - (2 / (n)) * sum2 + sum3


def _kernel_matrix(x, y, l):
    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    return np.exp(-(1 / (2 * l**2)) * cdist(x, y, "sqeuclidean"))


def test__compute_optimal_weights__returns_same_as_reference_implementation():
    rng = PRNGKey(seed=4234234)
    rng, rng_input = jax.random.split(rng)
    us = uniform(rng_input, shape=(100, 10), minval=0.0, maxval=1.0)

    kernel = kernels.GaussianKernel(kernels.median_heuristic(us))
    w = compute_optimal_weights(us, kernel, u_distribution="uniform")
    expected_w = _reference_compute_optimal_weights(us)

    assert jnp.allclose(w, expected_w, rtol=0.01)


def _reference_compute_optimal_weights(us: np.ndarray) -> np.ndarray:
    l = _median_heuristic(us)
    dim = us.shape[1]
    z = np.zeros(shape=us.shape)
    for i in range(dim):
        z[:, i] = (
            np.sqrt(2 * np.pi)
            * l
            * (
                stats.norm.cdf(1, loc=us[:, i], scale=l)
                - stats.norm.cdf(0, loc=us[:, i], scale=l)
            )
        )
    if dim > 1:
        z = np.prod(z, axis=1)

    m = us.shape[0]
    delta = 1e-8
    C = _kernel_matrix(us, us, l) + delta * np.identity(m)
    C_inv = np.linalg.inv(C)
    return np.matmul(C_inv, z)


def _median_heuristic(y):
    a = cdist(y, y, "sqeuclidean")
    return np.sqrt(np.median(a / 2))
