"""Command-line interface for the package."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _print_summary(res) -> None:
    summary = res.summary()
    rows = [("chi2_best", f"{summary['chi2_best']['value']:.3f}")]
    for name in res.param_names:
        q = summary[name]
        median = q["median"]
        upper = max(0.0, q["p84"] - median)
        lower = max(0.0, median - q["p16"])
        if upper or lower:
            scale = max(upper, lower)
            prec = 6 if scale == 0 else min(max(0, 1 - math.floor(math.log10(scale))), 8)
            fmt = f"{{:.{prec}f}}"
            value = f"{fmt.format(median)} +{fmt.format(upper)} -{fmt.format(lower)}"
        else:
            value = f"{median:.6g}"
        rows.append((name, value))

    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)
    for key, value in rows:
        print(f"{key:<{key_width}} = {value:>{value_width}}")


def _load_chain(path: str):
    """Load a metadata-prefixed CSV chain file into a named record array."""
    import numpy as np

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.startswith("#"):
                names = line.strip().split(",")
                break
        else:
            raise ValueError(f"no header found in {path}")

    arr = np.loadtxt(path, delimiter=",", comments="#", skiprows=i + 1)
    return names, np.rec.fromarrays(arr.T, names=",".join(names))


def cmd_fit(args) -> int:
    from .fit import fit

    print(f"Saved: {fit(args.config)}")
    return 0


def cmd_run(args) -> int:
    from .sampler import mcmc

    _print_summary(mcmc(args.config))
    return 0


def cmd_lc(args) -> int:
    from .lc import plot_lightcurve

    plot_lightcurve(args.config)
    print("Saved: lc.png")
    return 0


def cmd_chi2(args) -> int:
    from .chi2plot import plot_chi2

    names, tab = _load_chain(args.chain)
    params = args.names or [n for n in names if n != "chi2"]
    out = f"{Path(args.chain).with_suffix('')}_chi2.png"
    plot_chi2(tab, parameters=params, colorbar=False, filename=out)
    print(f"Saved: {out}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="microlens-mcmc",
        description="Fit microlensing lightcurves with emcee",
    )
    subparsers = parser.add_subparsers(required=True)

    fit_parser = subparsers.add_parser("fit", help="Pre-fit with outlier rejection and write cleaned TOML")
    fit_parser.add_argument("config", help="Path to conf.toml")
    fit_parser.set_defaults(func=cmd_fit)

    run_parser = subparsers.add_parser("run", help="Run MCMC from TOML config")
    run_parser.add_argument("config", help="Path to conf.toml")
    run_parser.set_defaults(func=cmd_run)

    lc_parser = subparsers.add_parser("lc", help="Plot lightcurve from TOML config")
    lc_parser.add_argument("config", help="Path to conf.toml")
    lc_parser.set_defaults(func=cmd_lc)

    chi2 = subparsers.add_parser("chi2", help="Plot chi2 surface from chain.csv")
    chi2.add_argument("chain", help="Path to chain.csv")
    chi2.add_argument("--names", nargs="*", help="parameter names to plot")
    chi2.set_defaults(func=cmd_chi2)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
