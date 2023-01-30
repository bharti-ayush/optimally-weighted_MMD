from dataclasses import asdict
from functools import partial

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from jax import Array, jit
from jax.random import KeyArray, PRNGKey

from mmd_ksd import plotting
from mmd_ksd.distributions.gandk import (
    GAndK,
    GAndKHyps,
    GAndKParams,
    Loss,
    ow_loss,
    run_opt,
    vstat_loss,
)
from mmd_ksd.jax_utils import batch_vmap, load_or_run

M = 50
N = 1000
HYPS = GAndKHyps(dim=5)
PARAMS_TO_PLOT = ["A", "B", "g", "rho"]
PARAM_MAP = {
    "A": "$\\theta_1$",
    "B": "$\\theta_2$",
    "g": "$\\theta_3$",
    "rho": "$\\theta_5$",
}
PARAM_RANGES = {
    "A": (2.6, 3.1),
    "B": (0.9, 1.1),
    "g": (-0.2, 0.5),
    "rho": (-0.05, 0.25),
}

jitted_run_opt = jit(run_opt, static_argnames=("loss", "hyps", "m"))


def main() -> None:
    rng = PRNGKey(seed=43242)

    default_params = asdict(GAndKParams())
    plotting.configure_matplotlib()
    fig, axes_array = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(plotting.ONE_COL_WIDTH, plotting.ONE_COL_WIDTH * 0.75),
    )
    axes = {
        param_name: ax for param_name, ax in zip(PARAMS_TO_PLOT, axes_array.flatten())
    }

    rng, rng_input = jax.random.split(rng)
    ow_params = _compute_and_plot_estimates(rng_input, ow_loss, label="weighted")
    rng, rng_input = jax.random.split(rng)
    vstat_params = _compute_and_plot_estimates(
        rng_input, vstat_loss, label="unweighted"
    )

    for param_name, ax in axes.items():
        xmin, xmax = PARAM_RANGES[param_name]
        ow_inliers, ow_outliers = _get_outliers(ow_params[param_name], xmin, xmax)
        vstat_inliers, vstat_outliers = _get_outliers(
            vstat_params[param_name], xmin, xmax
        )
        print(
            f"{param_name} outliers: ow={len(ow_outliers)} "
            f"vstat={len(vstat_outliers)} {vstat_outliers}"
        )
        _, bins = jnp.histogram(jnp.concatenate([ow_inliers, vstat_inliers]), bins=30)
        hist_params = {"bins": bins, "alpha": 0.2}
        ax.hist(vstat_inliers, label="V-stat", **hist_params)
        ax.hist(ow_inliers, label="OW (ours)", **hist_params)

        ax.text(
            0.05,
            0.8,
            f"$\\leftrightarrow {len(vstat_outliers)}$",
            transform=ax.transAxes,
        )

        ax.axvline(default_params[param_name], color="black")
        ax.set_xlabel(PARAM_MAP.get(param_name) or param_name, labelpad=-9.0)
        ax.set_xticks(PARAM_RANGES[param_name])
        ax.set_yticks([])
        ax.set_xlim(xmin, xmax)

    # axes_array.flatten()[0].legend(**plotting.squashed_legend_params)
    plotting.save_fig("gandk_min_dist_estimates", w_pad=0.3, h_pad=0.5)
    plt.close()


def _get_outliers(p: Array, xmin: float, xmax: float) -> tuple[Array, Array]:
    inlier_indices = (xmin <= p) & (p <= xmax)
    return p[inlier_indices], p[~inlier_indices]


def _compute_and_plot_estimates(
    rng: KeyArray,
    loss: Loss,
    label: str,
) -> dict[str, Array]:
    def _estimate(rng: KeyArray):
        rng, rng_input = jax.random.split(rng)
        ys, _ = _sample(rng_input, GAndKParams(), N)

        rng, rng_input = jax.random.split(rng)
        return asdict(jitted_run_opt(rng_input, loss, ys, HYPS, M))

    return load_or_run(
        lambda: batch_vmap(
            _estimate, jax.random.split(rng, num=100), batch_size=50, progress=True
        ),
        name=f"gandk_opt_{label}_m{M}_n{N}_d{HYPS.dim}",
    )


@partial(jit, static_argnames=("num"))
def _sample(rng: KeyArray, params: GAndKParams, num: int):
    return GAndK.sample_with_params(rng, params, hyps=HYPS, n=num)


if __name__ == "__main__":
    main()
