from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, NamedTuple, Optional, Protocol, TypeVar

import jax.numpy as jnp
import jax.random
from jax import Array, grad
from jax.random import KeyArray
from jax.scipy.stats import norm

from mmd_ksd.extra_types import Scalar
from mmd_ksd.to_scalar import to_scalar

R = TypeVar("R")
P = TypeVar("P")
Hyperparams = Optional[NamedTuple]


class Distribution(ABC, Generic[P]):
    @abstractmethod
    def get_params(self) -> P:
        pass

    def get_hyperparams(self) -> Hyperparams:
        return None


class SampleableDist(Distribution[P], ABC, Generic[P, R]):
    def sample(self, rng: KeyArray, n: int) -> R:
        return self.sample_with_params(
            rng, self.get_params(), self.get_hyperparams(), n
        )

    @staticmethod
    @abstractmethod
    def sample_with_params(rng: KeyArray, params: P, hyps: Hyperparams, n: int) -> R:
        pass


class UnnormalizedDist(Distribution[P], ABC, Generic[P]):
    def score(self, x: Array) -> Array:
        return self.score_with_params(self.get_params(), self.get_hyperparams(), x)

    @staticmethod
    @abstractmethod
    def score_with_params(params: P, hyps: Hyperparams, x: Array) -> Array:
        pass

    def unnorm_log_prob(self, x: Array) -> Array:
        return self.unnorm_log_prob_with_params(
            self.get_params(), self.get_hyperparams(), x
        )

    @classmethod
    @abstractmethod
    def unnorm_log_prob_with_params(
        cls, params: P, hyps: Hyperparams, x: Array
    ) -> Array:
        pass


class NormalizedDist(UnnormalizedDist[P], ABC, Generic[P]):
    def log_prob(self, x: Array) -> Array:
        return self.log_prob_with_params(self.get_params(), self.get_hyperparams(), x)

    @staticmethod
    @abstractmethod
    def log_prob_with_params(params: P, hyps: Hyperparams, x: Array) -> Array:
        pass

    @classmethod
    def unnorm_log_prob_with_params(
        cls, params: P, hyps: Hyperparams, x: Array
    ) -> Array:
        return cls.log_prob_with_params(params, hyps, x)


class ExpFamilyDist(Protocol):
    @staticmethod
    @abstractmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        pass

    @staticmethod
    @abstractmethod
    def sufficient_statistic(x: Array) -> Array:
        pass

    @staticmethod
    @abstractmethod
    def b(x: Array) -> Array:
        # TODO: Use proper name for b.
        pass


class SampleableAndNormalizedDist(
    SampleableDist[P, R], NormalizedDist[P], ABC, Generic[P, R]
):
    pass


@dataclass(frozen=True)
class Gaussian(SampleableAndNormalizedDist[Array, Array], ExpFamilyDist):
    loc: Scalar
    scale: Scalar

    @staticmethod
    def sample_with_params(
        rng: KeyArray, params: Array, hyps: Hyperparams, n: int
    ) -> Array:
        assert hyps is None
        loc, scale = params[0], params[1]
        return loc + scale * jax.random.normal(rng, shape=(n, 1))

    @staticmethod
    def score_with_params(params: Array, hyps: Hyperparams, x: Array) -> Array:
        assert hyps is None
        return grad(to_scalar(norm.logpdf), argnums=0)(
            x.reshape(x.shape[0]), params[0], params[1]
        )

    @staticmethod
    def log_prob_with_params(params: Array, hyps: Hyperparams, x: Array) -> Array:
        assert hyps is None
        return norm.logpdf(x, params[0], params[1])

    def get_params(self) -> Array:
        return jnp.array([self.loc, self.scale])

    @staticmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        mean = -0.5 * eta_val[0] / eta_val[1]
        std = jnp.sqrt(1 / (-2 * eta_val[1]))
        return jnp.array([mean, std])

    @staticmethod
    def sufficient_statistic(x: Array) -> Array:
        return jnp.concatenate([x, x**2], axis=0)

    @staticmethod
    def b(x: Array) -> Array:
        # Note that b does not depend on x, thus db/dx = 0. As, for the moment, we are
        # just using this to estimate parameters, we can return 0 to simplify things.
        # TODO: Properly implement this.
        return jnp.zeros(shape=())
