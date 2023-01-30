from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from jax import vmap
from jax.numpy import ndarray

from mmd_ksd.extra_types import Scalar


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: ndarray, x2: ndarray) -> ndarray:
        pass


@dataclass(frozen=True)
class GaussianKernel(Kernel):
    l: Scalar

    def __call__(self, x1: ndarray, x2: ndarray) -> ndarray:
        return jnp.exp(-((x1 - x2) ** 2).sum() / (2 * self.l**2))


class SumKernel(Kernel):
    def __init__(self, kernels: Sequence[Kernel]) -> None:
        self.kernels = tuple(kernels)

    def __call__(self, x1: ndarray, x2: ndarray) -> ndarray:
        return jnp.array([k(x1, x2) for k in self.kernels]).sum()

    def __hash__(self) -> int:
        return hash(self.kernels)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SumKernel):
            return False
        if len(self.kernels) != len(o.kernels):
            return False
        return all([k1 == k2 for (k1, k2) in zip(self.kernels, o.kernels)])


def gram(
    kernel: Callable[[ndarray, ndarray], ndarray], xs1: ndarray, xs2: ndarray
) -> ndarray:
    return vmap(lambda x: vmap(lambda y: kernel(x, y))(xs2))(xs1)


def zero_diagonal(x: ndarray) -> ndarray:
    return x.at[jnp.diag_indices(x.shape[0])].set(0)


def median_heuristic(x1: ndarray, x2: Optional[ndarray] = None) -> ndarray:
    if x2 is None:
        xs = x1
    else:
        xs = jnp.concatenate([x1, x2])

    distances = vmap(lambda xa: vmap(lambda xb: ((xa - xb) ** 2).sum() / 2)(xs))(xs)
    return jnp.sqrt(jnp.median(distances))
