"""Contains utilities for caching experiment results to disk."""
from pathlib import Path
from typing import Callable, TypeVar, cast

import jax.numpy as jnp
from numpy.typing import ArrayLike

A = TypeVar("A")


def load_or_run(
    f: Callable[[], A], name: str, results_dir: Path = Path("results")
) -> A:
    if not results_dir.is_dir():
        results_dir.mkdir()
    results_file = results_dir / f"{name}.npy"

    if not results_file.exists():
        print(f"Running {name}")
        results = f()
        jnp.save(str(results_file), cast(ArrayLike, results), allow_pickle=True)
    else:
        print(f"Loading {name}")
        loaded = jnp.load(results_file, allow_pickle=True)
        if loaded.shape == ():
            results = cast(A, loaded.item())
        else:
            results = cast(A, loaded)

    return results
