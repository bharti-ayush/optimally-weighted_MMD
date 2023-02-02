from pathlib import Path

import jax.numpy as jnp

from mmd_ksd.results_cache import load_or_run


def test__load_or_run__file_exists__loads(tmpdir):
    f1 = lambda: jnp.array(2)
    load_or_run(f1, "configname", results_dir=Path(tmpdir))

    def f2() -> int:
        raise ValueError

    result = load_or_run(f2, "configname", results_dir=Path(tmpdir))

    assert result == 2


def test__load_or_run__file_does_not_exist__runs(tmpdir):
    f1 = lambda: jnp.array(2)
    result = load_or_run(f1, "configname", results_dir=Path(tmpdir))
    assert result == 2


def test__load_or_run__loads_pickled_item__returns_item_not_array(tmpdir):
    f1 = lambda: {"a": jnp.array(1), "b": jnp.array(2)}
    load_or_run(f1, "configname", results_dir=Path(tmpdir))

    def f2() -> int:
        raise ValueError

    result = load_or_run(f2, "configname", results_dir=Path(tmpdir))

    assert isinstance(result, dict)
    assert jnp.allclose(result["a"], 1)
    assert jnp.allclose(result["b"], 2)
