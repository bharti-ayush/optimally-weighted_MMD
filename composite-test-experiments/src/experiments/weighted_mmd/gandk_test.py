from dataclasses import asdict
from functools import partial
from time import time
from typing import cast

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.lax import map
from jax.random import KeyArray, PRNGKey, split
from tap import Tap

from mmd_ksd.distributions.gandk import (
    GAndK,
    GAndKHyps,
    GAndKParams,
    Loss,
    ow_loss,
    run_opt,
    vstat_loss,
)
from mmd_ksd.jax_utils import tree_concatenate
from mmd_ksd.results_cache import load_or_run

SEED = 81238410
M = 100
N = 500
ALT_K = 0.5
HYPS = GAndKHyps(dim=5)
N_BOOTSTRAP_SAMPLES = 200
ALPHA = 0.05

jitted_run_opt = jit(run_opt, static_argnames=("loss", "hyps", "m"))
vstat_jitted = cast(Loss, jit(vstat_loss, static_argnames=("hyps", "m")))
ow_jitted = cast(Loss, jit(ow_loss, static_argnames=("hyps", "m")))


class Args(Tap):
    repeats: tuple[int, int] = (0, 100)
    batch_size: int = 50


def main() -> None:
    args = Args().parse_args()
    repeats = list(range(args.repeats[0], args.repeats[1]))
    results: list[str] = []
    results.extend(_compute_power(ALT_K, repeats, args.batch_size))
    results.extend(_compute_power(float(GAndKParams().k), repeats, args.batch_size))
    for r in results:
        print(r)


def _compute_power(k: float, repeats: list[int], batch_size: int) -> list[str]:
    print(f"Running k={k:.2f}")

    ow_results = []
    vstat_results = []
    for i in repeats:
        start = time()
        rng = PRNGKey(seed=SEED + i)
        rng, rng_input = split(rng)
        ow_results.append(_run_test(rng_input, ow_jitted, "ow", k, i, batch_size))
        rng, rng_input = split(rng)
        vstat_results.append(
            _run_test(rng_input, vstat_jitted, "vstat", k, i, batch_size)
        )
        print(f"{time() - start:1f}s per repeat")

    return [
        f"k={k:.2f}",
        f"ow {sum(ow_results)}/{len(ow_results)} "
        f"= {sum(ow_results) / len(ow_results):.3f}",
        f"vstat {sum(vstat_results)}/{len(vstat_results)} "
        f"= {sum(vstat_results) / len(vstat_results):.3f}",
    ]


def _run_test(
    rng: KeyArray, loss: Loss, loss_name: str, k: float, i: int, batch_size: int
) -> Array:
    name = (
        f"gandk_test_{loss_name}_M{M}_N{N}_dim{HYPS.dim}_bs{N_BOOTSTRAP_SAMPLES}"
        f"_k{k:.3f}_i{i}"
    )

    n_batches = N_BOOTSTRAP_SAMPLES // batch_size

    rng, rng_input = split(rng)
    ys = load_or_run(lambda: _sample(rng_input, GAndKParams(k=k), N), f"{name}_ys")

    rng, rng_input = split(rng)
    theta_hat_dict = load_or_run(
        lambda: asdict(jitted_run_opt(rng_input, loss, ys, HYPS, M)),
        f"{name}_theta_hat",
    )
    theta_hat = GAndKParams(**theta_hat_dict)

    def bootstrap_sample(rng: KeyArray) -> Array:
        rng1, rng2, rng3 = split(rng, num=3)
        b_ys = _sample(rng1, theta_hat, N)
        b_theta_hat = jitted_run_opt(rng2, loss, b_ys, HYPS, M)
        return loss(rng3, b_ys, b_theta_hat, HYPS, M)

    def run_batch(rng: KeyArray) -> Array:
        rng_inputs = split(rng, num=batch_size)
        return vmap(bootstrap_sample)(rng_inputs)

    rng, *rng_inputs = split(rng, num=n_batches + 1)
    sample_bs_deltas = lambda: tree_concatenate(map(run_batch, jnp.array(rng_inputs)))
    b_deltas = load_or_run(sample_bs_deltas, f"{name}_bs_deltas")

    critical_value = jnp.quantile(b_deltas, 1 - ALPHA)
    rng, rng_input = split(rng)
    delta = load_or_run(
        lambda: loss(rng_input, ys, theta_hat, HYPS, M), f"{name}_delta"
    )
    reject = delta > critical_value
    return reject


@partial(jit, static_argnames=("num"))
def _sample(rng: KeyArray, params: GAndKParams, num: int) -> Array:
    xs, _ = GAndK.sample_with_params(rng, params, HYPS, num)
    return xs


if __name__ == "__main__":
    main()
