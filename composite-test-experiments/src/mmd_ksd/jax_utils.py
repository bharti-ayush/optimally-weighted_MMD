from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, cast

import jax.numpy as jnp
from jax import Array, tree_util, vmap
from jax.random import KeyArray
from numpy.typing import ArrayLike
from tqdm import tqdm

T = TypeVar("T")


def tree_concatenate(trees: Iterable[T]) -> T:
    leaves, treedefs = zip(*[tree_util.tree_flatten(tree) for tree in trees])
    grouped_leaves = zip(*leaves)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return cast(T, treedefs[0].unflatten(result_leaves))


def batch_vmap(
    f: Callable[[KeyArray], T], rngs: KeyArray, batch_size: int, progress: bool = False
) -> T:
    n_batches = int(ceil(rngs.shape[0] / batch_size))

    batch_results: list[T] = []
    if progress:
        iterator = tqdm(range(n_batches))
    else:
        iterator = range(n_batches)
    for batch_i in iterator:
        batch_rngs = rngs[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_results.append(vmap(f)(batch_rngs))

    return tree_concatenate(batch_results)


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
