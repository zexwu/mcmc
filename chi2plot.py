import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


parent_path = Path(__file__).parent.parent
plt.style.use((parent_path / "zexwu.mplstyle").resolve())

tex = {
    "chi2": [r"$\chi^2$", ".1f"],
    "t0": [r"$t_0$", ".1f"],
    "u0": [r"$u_0$", ".3f"],
    "tE": [r"$t_{\rm E}$", ".1f"],
    "teff": [r"$t_{\rm eff}$", ".1f"],
    "pi1": [r"$\pi_{\rm EN}$", ".3f"],
    "pi2": [r"$\pi_{\rm EE}$", ".3f"],
    "pi_perp": [r"$\pi_{\rm E\!\perp}$", ".3f"],
    "pi_para": [r"$\pi_{\rm E\!\parallel}$", ".3f"],
    "fs": [r"$f_{\rm S}$", ".1f"],
    "fb": [r"$f_{\rm B}$", ".1f"],
    "fb/fs": [r"$f_{\rm B}/f_{\rm S}$", ".1f"],
    "M_L_parallel/mu_rel": [r"$M_{\parallel}$", ".1f"],
    "M_L/mu_rel": [r"$M$", ".1f"],
    "thetaE": [r"$\theta_{\rm E}$", ".1f"],
    "D_L": [r"$D_{\rm L}$", ".1f"],
    "tE/piE": [r"$t_{\rm E}/\pi_{\rm E}/2000$", ".1f"],
    "tE/piE_para": [r"$t_{\rm E}/\pi_{\rm E\!\parallel}/2000$", ".1f"],
    "A_ref": [r"$A_{\rm ref}$", ".2f"]
}


def weighted_quantile(values, q, weights=None):
    """
    Weighted percentile.

    Parameters
    ----------
    values : array-like
    q : array-like in [0, 100]
    weights : array-like or None

    Returns
    -------
    ndarray
    """
    values = np.asarray(values)
    q = np.asarray(q) / 100.0

    if weights is None:
        return np.percentile(values, q * 100)

    weights = np.asarray(weights)
    sorter = np.argsort(values)

    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    return np.interp(q, cdf, values)


def _prepare_sample(table, parameters, nsigma, delta):
    cols = ["chi2"] + parameters
    arr = np.column_stack([table[c] for c in cols])

    chi2_min = arr[:, 0].min()
    dchi2 = arr[:, 0] - chi2_min

    mask = dchi2 < (nsigma**2) * delta
    arr = arr[mask]
    arr[:, 0] = dchi2[mask]

    return arr, chi2_min


def _layers(arr, nsigma, delta):
    """
    Precompute layer masks once.
    Avoid repeated filtering.
    """
    dchi2 = arr[:, 0]
    layers = []

    for i in range(nsigma):
        thresh = (nsigma - i) ** 2 * delta
        layers.append(arr[dchi2 < thresh])

    return layers


def _diag_stats(values, weights=None, mode="weighted"):
    if mode == "percentile":
        return np.percentile(values, [16, 50, 84])
    if mode == "mid":
        return np.array([values.min(), values.mean(), values.max()])
    return weighted_quantile(values, [16, 50, 84], weights)


def plot_chi2(
    table,
    parameters,
    *,
    nsigma=4,
    delta=1.0,
    bins=30,
    stat="percentile",
    colors=("red", "gold", "limegreen", "blue"),
    tex=tex,
    colorbar=True,
    figsize=None,
    filename=None,
):
    """
    Fast chi2 corner plot.

    Parameters
    ----------
    table : Astropy table or dict-like
    parameters : list[str]
    """

    n = len(parameters)
    figsize = figsize or (2.5 * n, 2.5 * n)

    fig, axs = plt.subplots(n, n, figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    arr, chi2_min = _prepare_sample(table, parameters, nsigma, delta)
    layers = _layers(arr, nsigma, delta)

    base = layers[0]
    weights = np.exp(-base[:, 0] / (2 * delta))
    weights /= weights

    # --- Diagonal ---
    for i, p in enumerate(parameters):
        ax = axs[i, i]
        vals = base[:, i + 1]

        ax.hist(
            vals,
            bins=bins,
            density=True,
            weights=weights,
            histtype="step",
            color="k",
            zorder=100
        )
        ax.set_yticks([])

        lo, med, hi = _diag_stats(vals, weights, stat)

        ax.axvline(med, color=colors[0])
        ax.axvline(lo, ls="--", color=colors[0])
        ax.axvline(hi, ls="--", color=colors[0])

        pfmt = tex.get(p, p)
        label, fmt = pfmt, ".4f"
        if isinstance(pfmt, list):
            label, fmt = pfmt
        ax.set_title(f"{label}\n${med:{fmt}}_{{{lo-med:{fmt}}}}^{{+{hi-med:{fmt}}}}$")

        plt.setp(axs[-1][i].xaxis.get_majorticklabels(), rotation=45)
        plt.setp(axs[i][0].yaxis.get_majorticklabels(), rotation=45)


    # --- Off-diagonal ---
    for i in range(n):
        for j in range(i):
            ax = axs[i, j]

            for k, layer in enumerate(layers):
                ax.plot(
                    layer[:, j + 1],
                    layer[:, i + 1],
                    ".",
                    markersize=1,
                    color=colors[nsigma - 1 - k],
                    rasterized=True,
                )
                # ax.scatter(
                #     layer[:, j + 1],
                #     layer[:, i + 1],
                #     s=1,
                #     c=colors[nsigma - 1 - k],
                #     rasterized=True,
                # )

            axs[j, i].set_visible(False)

    # --- Axis cleanup ---
    for ax in axs.flat:
        ax.label_outer()
    # --- Colorbar ---
    if colorbar:
        cmap = mpl.colors.ListedColormap(colors[:nsigma])
        bounds = delta * np.arange(nsigma + 1) ** 2
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axs,
            location="right"
        )
    else:
        ax = axs[0, -1]
        ax.set_visible(True)
        ax.set_axis_off()
        for i in np.arange(nsigma):
            ax.scatter([], [], c=colors[nsigma - 1 - i], label=rf"$\Delta \chi^2 < {i**2}$")
        legends = ax.legend(loc="center", frameon=False)
        # hide markers and change the color of the text to match the marker color
        for legobj in legends.legendHandles:
            legobj.set_visible(False)
        for text, color in zip(legends.get_texts(), colors[:nsigma]):
            text.set_color(color)

    fig.suptitle(rf"$\chi^2_\min$ = {chi2_min:.3f}", y=0.95)

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")

    return fig, axs
