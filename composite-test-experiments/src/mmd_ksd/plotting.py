import matplotlib.pyplot as plt

# These dimensions are based on the ICML layout.
ONE_COL_WIDTH = 3.25
TWO_COL_WIDTH = 6.75

squashed_legend_params = {
    "handlelength": 1.0,
    "handletextpad": 0.5,
    "labelspacing": 0.3,
    "borderaxespad": 0.2,
    "borderpad": 0.25,
    "columnspacing": 0.7,
}
squashed_label_params = {"labelpad": 1.5}


def configure_matplotlib() -> None:
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")
    plt.rc("font", family="serif", size=10)
    plt.rc("figure", figsize=(3.25, 3.25))


def save_fig(name: str, **kwargs) -> None:
    plt.tight_layout(**kwargs)
    plt.savefig(f"plots/{name}.png", dpi=200)
    plt.savefig(f"plots/{name}.pdf", bbox_inches="tight", transparent=True)
