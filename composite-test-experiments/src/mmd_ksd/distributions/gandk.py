"""Contains the G and K model and the associated gradient-based optimizer config."""
from abc import ABC
from dataclasses import dataclass
from typing import NamedTuple, Protocol, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random
import optax
from jax import Array
from jax.lax import stop_gradient
from jax.random import KeyArray, normal

from mmd_ksd.discrepancies.kernels import GaussianKernel, median_heuristic
from mmd_ksd.discrepancies.mmd import (
    compute_optimal_weights,
    mmd_v_stat,
    weighted_mmd_v_stat,
)
from mmd_ksd.distributions import SampleableDist
from mmd_ksd.distributions.distributions import Hyperparams
from mmd_ksd.extra_types import Scalar
from mmd_ksd.optimizers import random_restart_optimizer


@dataclass(frozen=True)
class _GAndKParams(ABC):
    """Chex dataclasses break MyPy, so have a normal dataclass as the base class."""

    A: Scalar = 3.0
    B: Scalar = 1.0
    g: Scalar = 0.1
    k: Scalar = 0.1
    rho: Scalar = 0.1


@chex.dataclass(frozen=True)
class GAndKParams(_GAndKParams):
    pass


class GAndKHyps(NamedTuple):
    dim: int = 5


@dataclass(frozen=True)
class GAndK(SampleableDist[GAndKParams, Tuple[Array, Array]]):
    params: GAndKParams = GAndKParams()
    hyperparams: GAndKHyps = GAndKHyps()

    def get_params(self) -> GAndKParams:
        return self.params

    def get_hyperparams(self) -> Hyperparams:
        return self.hyperparams

    @staticmethod
    def sample_with_params(
        rng: KeyArray, params: GAndKParams, hyps: Hyperparams, n: int
    ) -> Tuple[Array, Array]:
        assert isinstance(hyps, GAndKHyps)
        p = params
        c = 0.8
        dim = hyps.dim

        cov = jnp.eye(dim) + p.rho * jnp.eye(dim, k=1) + p.rho * jnp.eye(dim, k=-1)
        L = jnp.linalg.cholesky(cov)

        z_standard = normal(rng, shape=(n, dim))

        z = (L @ z_standard.T).T
        x = (
            p.A
            + p.B
            * (1 + c * ((1 - jnp.exp(-p.g * z)) / (1 + jnp.exp(-p.g * z))))
            * ((1 + z**2) ** p.k)
            * z
        )

        return x, z_standard

    @staticmethod
    def get_prior_range() -> tuple[GAndKParams, GAndKParams]:
        lower = GAndKParams(A=0.001, B=0.001, g=0.001, k=0.001, rho=0.001)
        upper = GAndKParams(A=5.0, B=5.0, g=1.0, k=1.0, rho=1.0)
        return lower, upper

    @staticmethod
    def sample_initial_params(rng: KeyArray) -> GAndKParams:
        rngs = jax.random.split(rng, num=5)
        l, u = GAndK.get_prior_range()
        # We do not learn k and keep it fixed to the true value. Thus, do not specify it
        # here so it's initialized to the true value.
        return GAndKParams(
            A=jax.random.uniform(rngs[0], minval=l.A, maxval=u.A),
            B=jax.random.uniform(rngs[1], minval=l.B, maxval=u.B),
            g=jax.random.uniform(rngs[2], minval=l.g, maxval=u.g),
            rho=jax.random.uniform(rngs[4], minval=l.rho, maxval=u.rho),
        )


class Loss(Protocol):
    @staticmethod
    def __call__(
        rng: KeyArray, ys: Array, params: GAndKParams, hyps: GAndKHyps, m: int
    ) -> Array:
        pass


def run_opt(
    rng: KeyArray, loss: Loss, ys: Array, hyps: GAndKHyps, m: int
) -> GAndKParams:
    params_to_fix = GAndKParams(A=False, B=False, g=False, k=True, rho=False)
    return random_restart_optimizer(
        rng,
        optax.adam(learning_rate=0.04),
        loss=lambda r, p: loss(r, ys, p, hyps, m),
        sample_theta_init=lambda rng: GAndK.sample_initial_params(rng),
        iterations=200,
        n_initial_locations=50,
        n_optimized_locations=10,
        params_to_fix=params_to_fix,
    )


def vstat_loss(
    rng: KeyArray, ys: Array, params: GAndKParams, hyps: GAndKHyps, m: int
) -> Array:
    xs, _ = GAndK.sample_with_params(rng, params, hyps, m)
    return mmd_v_stat(_get_kernel(xs, ys), xs, ys)


def ow_loss(
    rng: KeyArray, ys: Array, params: GAndKParams, hyps: GAndKHyps, m: int
) -> Array:
    xs, us = GAndK.sample_with_params(rng, params, hyps, m)
    u_kernel = GaussianKernel(l=median_heuristic(us))
    ws = stop_gradient(compute_optimal_weights(us, u_kernel, u_distribution="gaussian"))
    return weighted_mmd_v_stat(_get_kernel(xs, ys), xs, ys, ws)


def _get_kernel(xs: Array, ys: Array):
    return GaussianKernel(l=median_heuristic(xs, ys))
