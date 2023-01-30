"""Copies of functions from jax.random that also return the underlying uniform RVs.

TODO: Need to sort out licensing before making this repository public.
"""
import warnings
from functools import partial
from typing import Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np
from jax import core, lax
from jax import numpy as jnp
from jax._src import dtypes, prng
from jax._src.api import jit
from jax._src.numpy.lax_numpy import _arraylike
from jax._src.typing import Array, DTypeLike
from jax.config import config
from jax.core import NamedShape
from jax.numpy import ndarray
from jax.random import KeyArray, default_prng_impl, split, uniform

Shape = Sequence[int]
DTypeLikeInt = DTypeLike
DTypeLikeFloat = DTypeLike


def truncated_normal_with_uniforms(
    key: KeyArray,
    lower: ndarray,
    upper: ndarray,
    shape: Optional[Union[Shape, NamedShape]] = None,
    dtype: DTypeLikeFloat = dtypes.float_,
) -> tuple[Array, Array]:
    """Sample truncated standard normal random values with given shape and dtype.

    Args:
      key: a PRNG key used as the random key.
      lower: a float or array of floats representing the lower bound for
        truncation. Must be broadcast-compatible with ``upper``.
      upper: a float or array of floats representing the  upper bound for
        truncation. Must be broadcast-compatible with ``lower``.
      shape: optional, a tuple of nonnegative integers specifying the result
        shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
        default (None) produces a result shape by broadcasting ``lower`` and
        ``upper``.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).

    Returns:
      (x, u), where x is sampled from the specified truncated normal, and u is sampled
      from a uniform distribution on [0,1]. Both x and u have the specified shape and
      dtype.
    """
    key, _ = _check_prng_key(key)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            f"dtype argument to `truncated_normal` must be a float "
            f"dtype, got {dtype}"
        )
    dtype = dtypes.canonicalize_dtype(dtype)
    if shape is not None:
        shape = core.as_named_shape(shape)
    return _truncated_normal(key, lower, upper, shape, dtype)  # type: ignore


@partial(jit, static_argnums=(3, 4), inline=True)
def _truncated_normal(key, lower, upper, shape, dtype) -> tuple[Array, Array]:
    if shape is None:
        shape = lax.broadcast_shapes(np.shape(lower), np.shape(upper))
    else:
        _check_shape("truncated_normal", shape, np.shape(lower), np.shape(upper))

    sqrt2 = np.array(np.sqrt(2), dtype)
    lower = lax.convert_element_type(lower, dtype)
    upper = lax.convert_element_type(upper, dtype)
    a = lax.erf(lower / sqrt2)
    b = lax.erf(upper / sqrt2)
    if not jnp.issubdtype(dtype, np.floating):
        raise TypeError("truncated_normal only accepts floating point dtypes.")
    u = uniform(key, shape, dtype)
    u_scaled = (u * (b - a) + a).astype(dtype)
    out = sqrt2 * lax.erf_inv(u_scaled)
    # Clamp the value to the open interval (lower, upper) to make sure that
    # rounding (or if we chose `a` for `u`) doesn't push us outside of the range.
    xs = jnp.clip(
        out,
        lax.nextafter(lax.stop_gradient(lower), np.array(np.inf, dtype=dtype)),
        lax.nextafter(lax.stop_gradient(upper), np.array(-np.inf, dtype=dtype)),
    )
    return xs, u


def _check_prng_key(key):
    # TODO(frostig): remove once we always enable_custom_prng
    if isinstance(key, prng.PRNGKeyArray):
        return key, False
    elif _arraylike(key):
        if config.jax_enable_custom_prng:
            warnings.warn(
                "Raw arrays as random keys to jax.random functions are deprecated. "
                "Assuming valid threefry2x32 key for now.",
                FutureWarning,
            )
        return prng.random_wrap(key, impl=default_prng_impl()), True
    else:
        raise TypeError(f"unexpected PRNG key type {type(key)}")


def _check_shape(name: str, shape: Union[Shape, NamedShape], *param_shapes) -> None:
    shape = core.as_named_shape(shape)

    if param_shapes:
        shape_ = lax.broadcast_shapes(shape.positional, *param_shapes)
        if shape.positional != shape_:
            msg = (
                "{} parameter shapes must be broadcast-compatible with shape "
                "argument, and the result of broadcasting the shapes must equal "
                "the shape argument, but got result {} for shape argument {}."
            )
            raise ValueError(msg.format(name, shape_, shape))


def normal_with_uniforms(
    key: KeyArray,
    shape: Union[Shape, NamedShape] = (),
    dtype: DTypeLikeFloat = dtypes.float_,
) -> tuple[Array, Array]:
    """Sample standard normal random values with given shape and float dtype.

    Args:
      key: a PRNG key used as the random key.
      shape: optional, a tuple of nonnegative integers representing the result
        shape. Default ().
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).

    Returns:
      (x, u), where x is sampled from the specified normal, and u is sampled from a
      uniform distribution on [0,1]. Both x and u have the specified shape and dtype.
    """
    key, _ = _check_prng_key(key)
    if not dtypes.issubdtype(dtype, np.inexact):
        raise ValueError(
            f"dtype argument to `normal` must be a float or complex dtype, "
            f"got {dtype}"
        )
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.as_named_shape(shape)
    return _normal(key, shape, dtype)


@partial(jit, static_argnums=(1, 2), inline=True)
def _normal(key, shape, dtype) -> tuple[Array, Array]:
    if dtypes.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError
    else:
        return _normal_real(key, shape, dtype)


@partial(jit, static_argnums=(1, 2), inline=True)
def _normal_real(key, shape, dtype) -> tuple[Array, Array]:
    _check_shape("normal", shape)
    lo = np.nextafter(np.array(-1.0, dtype), np.array(0.0, dtype), dtype=dtype)
    hi = np.array(1.0, dtype)
    u = uniform(key, shape, dtype)
    u_scaled = (u * (hi - lo) + lo).astype(dtype)
    return lax.mul(np.array(np.sqrt(2), dtype), lax.erf_inv(u_scaled)), u
