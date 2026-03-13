from __future__ import annotations

import argparse

from .sampler import fit
from .lc import plot_lightcurve


def _fmt_pm(m: float, p16: float, p84: float) -> str:
    du = max(0.0, p84 - m)
    dl = max(0.0, m - p16)
    if not (du > 0 or dl > 0):
        return f"{m:.6g}"
    # choose precision from the larger uncertainty (2 sig figs)
    u = max(du, dl) if max(du, dl) > 0 else 0.0
    if u == 0:
        prec = 6
    else:
        import math
        exp = math.floor(math.log10(u))
        # decimals to keep 2 significant digits of u
        decimals = max(0, 1 - exp)
        # clamp to a sane range
        decimals = min(max(decimals, 0), 8)
        prec = decimals
    fmt = f"{{:.{prec}f}}"
    m_s = fmt.format(m)
    du_s = fmt.format(du)
    dl_s = fmt.format(dl)
    return f"{m_s} +{du_s} -{dl_s}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="microlens-mcmc", description="Fit microlensing lightcurves with emcee")
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="Run MCMC from TOML config")
    r.add_argument("config", help="Path to conf.toml")

    r = sp.add_parser("plot", help="Plot lightcurve from TOML config")
    r.add_argument("config", help="Path to conf.toml")

    args = p.parse_args(argv)
    if args.cmd == "run":
        res = fit(args.config)
        s = res.summary()
        # build aligned lines
        lines = []
        lines.append(("chi2_best", f"{s['chi2_best']['value']:.3f}"))
        for name in res.param_names:
            q = s[name]
            lines.append((name, _fmt_pm(q['median'], q['p16'], q['p84'])))
        w_key = max(len(k) for k, _ in lines)
        w_val = max(len(v) for _, v in lines)
        for key, val in lines:
            print(f"{key:<{w_key}} = {val:>{w_val}}")
        return 0

    if args.cmd == "plot":
        out = plot_lightcurve(args.config)
        import matplotlib.pyplot as plt
        plt.show()
        print(f"Saved: {out}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
