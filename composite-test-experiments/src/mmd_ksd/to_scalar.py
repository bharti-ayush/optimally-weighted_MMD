from functools import wraps
from typing import Callable, ParamSpec

from jax.numpy import ndarray

P = ParamSpec("P")


def to_scalar(f: Callable[P, ndarray]) -> Callable[P, ndarray]:
    @wraps(f)
    def f2(*args: P.args, **kwargs: P.kwargs) -> ndarray:
        return f(*args, **kwargs).reshape(())

    return f2
