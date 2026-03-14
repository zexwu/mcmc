from __future__ import annotations

import argparse
from pathlib import Path


def _fmt_pm(m: float, p16: float, p84: float) -> str:
    du = max(0.0, p84 - m)
    dl = max(0.0, m - p16)
    if not (du or dl):
        return f"{m:.6g}"

    import math

    u = max(du, dl)
    prec = 6 if u == 0 else min(max(0, 1 - math.floor(math.log10(u))), 8)
    fmt = f"{{:.{prec}f}}"
    return f"{fmt.format(m)} +{fmt.format(du)} -{fmt.format(dl)}"


def _print_summary(res) -> None:
    s = res.summary()
    rows = [("chi2_best", f"{s['chi2_best']['value']:.3f}")]
    rows += [
        (name, _fmt_pm(q["median"], q["p16"], q["p84"]))
        for name in res.param_names
        for q in [s[name]]
    ]
    wk = max(len(k) for k, _ in rows)
    wv = max(len(v) for _, v in rows)
    for k, v in rows:
        print(f"{k:<{wk}} = {v:>{wv}}")


def _load_chain(path: str):
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


def cmd_run(args) -> int:
    from .sampler import fit

    _print_summary(fit(args.config))
    return 0


def cmd_lc(args) -> int:
    from .lc import plot_lightcurve

    print(f"Saved: {plot_lightcurve(args.config)}")
    return 0


def cmd_chi2(args) -> int:
    from .chi2plot import chi2plot

    names, tab = _load_chain(args.chain)
    params = args.names or [n for n in names if n != "chi2"]
    out = f"{Path(args.chain).with_suffix('')}_chi2.png"
    chi2plot(tab, parameters=params, colorbar=False, filename=out)
    print(f"Saved: {out}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="microlens-mcmc",
        description="Fit microlensing lightcurves with emcee",
    )
    sp = p.add_subparsers(required=True)

    run = sp.add_parser("run", help="Run MCMC from TOML config")
    run.add_argument("config", help="Path to conf.toml")
    run.set_defaults(func=cmd_run)

    lc = sp.add_parser("lc", help="Plot lightcurve from TOML config")
    lc.add_argument("config", help="Path to conf.toml")
    lc.set_defaults(func=cmd_lc)

    chi2 = sp.add_parser("chi2", help="Plot chi2 surface from chain.csv")
    chi2.add_argument("chain", help="Path to chain.csv")
    chi2.add_argument("--names", nargs="*", help="parameter names to plot")
    chi2.set_defaults(func=cmd_chi2)

    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
