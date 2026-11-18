"""
Plot the recovery initial condition and save solution_storage/initial_wound.pdf.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def u_init(x):
    x = np.asarray(x, dtype=float)
    return 1.0 - 0.75 * np.exp(-(0.1 * x)**6)


def plot_initial_wound(L=30.0, output_dir=None, show=True):
    """Plot on [-L,L] and return (fig, ax). By default save the PDF in ./solution_storage."""
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L must be a finite positive domain half-length.")
    x = np.linspace(-L, L, 2001)
    n0 = u_init(x)
    blue = "#2166ac"
    style = {
        "font.family": "serif", "font.size": 12,
        "mathtext.fontset": "stix", "text.usetex": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
        ax.fill_between(x, n0, 1.0, color=blue, alpha=0.10, linewidth=0)
        ax.plot(x, n0, color=blue, lw=1.8,
                label=r"$n_0(x)=1-0.75e^{-(0.1x)^6}$")
        ax.axhline(1, color="0.4", ls="--", lw=1,
                   label=r"Healthy density $n=1$")
        ax.plot(0, u_init(0), "o", color=blue, ms=5)
        ax.vlines(0, 0.25, 0.465, color=blue, lw=0.9)
        ax.text(0, 0.485, r"$n_0(0)=0.25$",
                ha="center", va="bottom", fontsize=11)
        ax.set(xlim=(-L, L), ylim=(0.18, 1.08),
               xlabel=r"$x$", ylabel=r"Cell density $n_0(x)$",
               yticks=[0.25, 0.50, 0.75, 1.00])
        ax.set_title("Initial wound profile", loc="left", fontsize=14, pad=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        #ax.grid(alpha=0.15, lw=0.5)
        ax.legend(loc="lower left", frameon=False, fontsize=11,
                  handlelength=2, labelspacing=0.35)

        folder = Path.cwd() / "solution_storage" if output_dir is None else Path(output_dir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        pdf_path = folder / "initial_wound.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {pdf_path.resolve()}")

        if show:
            plt.show()
            
    return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot the initial wound profile.")
    parser.add_argument("--output", metavar="DIR", help="PDF directory (default: ./solution_storage)")
    parser.add_argument("--no-show", action="store_true", help="Save the PDF without opening a window.")
    args = parser.parse_args()
    plot_initial_wound(output_dir=args.output, show=not args.no_show)
