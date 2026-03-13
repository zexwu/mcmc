"""Light curve plotting.

Reads the project TOML config, the sampler outputs (best params and chain),
reconstructs per-dataset flux scaling (fs, fb), converts to magnitudes on a
common reference scale, and produces a 3‑panel figure:

- E: entire data span (data + model)
- A: zoom around peak (data + model)
- B: residuals (data − model) in magnitudes

Usage:
    python -m ulens_mcmc.lc path/to/config.toml
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import toml
import numpy as np

import matplotlib.pyplot as plt

from .io import load_photometry
from .likelihood import solve_blending_chi2
from .models import PSPL, PSBL

parent_path = Path(__file__).parent.parent
plt.style.use((parent_path / "zexwu.mplstyle").resolve())

_COLORS = {
    "KMTC": ("tab:red", 4),
    "KMTA": ("tab:green", 2),
    "KMTS": ("tab:blue", 3),
    "OGLE": ("black", 5),
    "MOA": ("gray", 1),
}

_COLORS_LIST = [
        "black",
        "r",
        "blue",
        "lime",
        "magenta",
        "gray",
        "olivedrab",
        "darkslategray",
        "r",
        "lime",
        "gold",
        "r",
        "black",
        "darkgreen",
    ]



def _get_color(label: str, idx: int = 0) -> Tuple[str, int]:
    for key, val in _COLORS.items():
        if key in label:
            return val
    return _COLORS_LIST[idx % len(_COLORS_LIST)], idx
    return "gray", 0


def _flux_to_mag(arr: np.ndarray, ref_mag: float = 18.0) -> np.ndarray:
    """Convert (t, flux, err_flux) to (t, mag, err_mag)."""
    out = arr.copy()
    f = np.clip(out[:, 1], 1e-300, None)
    ef = out[:, 2]
    out[:, 1] = ref_mag - 2.5 * np.log10(f)
    out[:, 2] = np.where(f > 0, ef / f / (0.4 * np.log(10.0)), np.nan)
    return out


def _rescale_to_ref(data_flux: np.ndarray, fs: float, fb: float, fs_ref: float, fb_ref: float) -> np.ndarray:
    """Affine rescale a dataset's flux to reference (fs_ref, fb_ref)."""
    out = data_flux.copy()
    # deblend and rescale: ((f - fb)/fs) * fs_ref + fb_ref
    out[:, 1] = ((data_flux[:, 1] - fb) / fs) * fs_ref + fb_ref
    out[:, 2] = data_flux[:, 2] / max(abs(fs), 1e-30) * abs(fs_ref)
    return out


def _read_best(best_csv: Path) -> Dict[str, float]:
    """Read best-row CSV written by sampler; return dict of parameters.

    The file contains commented metadata lines starting with '#', and a header
    row with names: chi2, <params...>, <blobs...>. We parse a single data row.
    """
    with open(best_csv) as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue
            names = line.strip().split(",")
            skiprows = i + 1
            break
    arr = np.loadtxt(best_csv, delimiter=",", comments="#", skiprows=skiprows)
    return dict(zip(names, arr))


def plot_lightcurve(config_path: str | Path) -> Path:
    cfg = toml.load(config_path)
    phot = load_photometry(cfg)

    # Build model from event coordinates
    event_cfg = cfg.get("event", {})
    coords = event_cfg.get("coords", "").split()
    if len(coords) != 2:
        raise ValueError("event.coords must be 'RA DEC', e.g. '17:31:42.61 -30:46:17.04'")

    model = globals()[cfg.get("mcmc").get("model")](*coords)
    model.precalculate_parallax([ds.data[:, 0] for ds in phot])

    # Output files from sampler
    out_dir = Path(cfg.get("paths").get("output"))
    out_cfg = cfg.get("mcmc", {}).get("outputs", {})
    best_file = out_dir / out_cfg.get("best_file")
    if not best_file.exists():
        raise FileNotFoundError(f"Best-fit CSV not found: {best_file}")
    best = _read_best(best_file)
    best = model.normalize(best)

    # Reference magnification scaling from first dataset (if available)
    if len(phot) == 0 or len(phot[0].data) == 0:
        raise ValueError("No photometry loaded; cannot plot light curve")
    mag_ref = model.magnification(phot[0].data[:, 0], best, dataset_id=0)
    _, ref_params = solve_blending_chi2(phot[0].flux, mag_ref, blending=phot[0].blending)
    fs_ref, fb_ref = float(ref_params[0] if np.isfinite(ref_params[0]) else 1.0), float(ref_params[1] if np.isfinite(ref_params[1]) else 0.0)
    if abs(fs_ref) < 1e-12:
        fs_ref = 1.0

    # Figure with mosaic panels
    fig, axd = plt.subplot_mosaic("EEE\nEEE\nAAA\nAAA\nBBB", figsize=(8, 8))

    # Plot datasets
    tmin, tmax = np.inf, -np.inf
    mm = 0.01  # residual scale seed

    for i, ds in enumerate(phot):
        if len(ds.data) == 0:
            continue
        t = ds.data[:, 0]
        tmin = min(tmin, float(t.min()))
        tmax = max(tmax, float(t.max()))

        mag_model = model.magnification(t, best, dataset_id=i)
        _, lin = solve_blending_chi2(ds.flux, mag_model, blending=ds.blending)
        fs_i = float(lin[0]) if np.isfinite(lin[0]) else 1.0
        fb_i = float(lin[1]) if np.isfinite(lin[1]) else 0.0

        # rescale to reference and convert to magnitudes
        flux_rescaled = _rescale_to_ref(ds.flux, fs_i, fb_i, fs_ref, fb_ref)
        mag_data = _flux_to_mag(flux_rescaled)
        mag_model_rescaled = _flux_to_mag(np.column_stack([t, mag_model * fs_ref + fb_ref, np.zeros_like(t)]))

        color, zorder = _get_color(ds.label, i)

        errorbar_kwargs = dict(fmt="o", ms=4, capsize=0, fillstyle="none", mew=1.5, c=color, zorder=zorder, alpha=0.7)
        for key in ("A", "E"):
            axd[key].errorbar(
                mag_data[:, 0],
                mag_data[:, 1],
                yerr=np.abs(mag_data[:, 2]),
                label=ds.label if key == "E" else None,
                **errorbar_kwargs,
            )

        axd["B"].errorbar(
            mag_data[:, 0],
            mag_data[:, 1] - mag_model_rescaled[:, 1],
            yerr=np.abs(mag_data[:, 2]),
            **errorbar_kwargs,
        )

        if i == 0:
            mm = float(np.median(np.abs(mag_data[:, 1] - mag_model_rescaled[:, 1])))

    # Model curve over range
    t0 = best["t0"]
    tE = best["tE"]

    t_ref = cfg.get("mcmc").get("t_ref", np.nan)

    t_model = np.arange(tmin - 0.1 * tE, max(tmax, t_ref) + 0.1 * tE, 0.5)
    mag_curve = _flux_to_mag(np.column_stack([t_model, model.magnification(t_model, best) * fs_ref + fb_ref, np.zeros_like(t_model)]))

    for key in ("A", "E"):
        axd[key].plot(t_model, mag_curve[:, 1], lw=1, c="tab:cyan", zorder=100)
    axd["B"].axhline(0.0, c="tab:cyan", lw=1, zorder=100)

    # Axes limits
    ymin, ymax = float(np.max(mag_curve[:, 1])) + 0.2, float(np.min(mag_curve[:, 1])) - 0.2
    axd["A"].set_ylim(ymin, ymax)
    axd["E"].set_ylim(ymin + 0.1, ymax - 0.1)

    # Zoom window around peak and around t_ref
    t_min = max(t0 - 3 * tE, t_ref - 360)
    t_max = min(t0 + tE, t_ref + 120)
    # expand to include first dataset coverage loosely
    if np.isfinite(t_ref):
        t_min = min(t_min, t_ref - 30)
        t_max = max(t_max, t_ref + 15)

    axd["A"].set_xlim(t_min, t_max)
    axd["B"].set_xlim(t_min, t_max)
    axd["B"].set_ylim((mm * 5, -mm * 5) if mm * 5 > 0.05 else (0.06, -0.06))

    # Labels, title, legend
    axd["E"].set_ylabel(f"${phot[0].filter}$ mag")
    axd["A"].set_ylabel(f"${phot[0].filter}$ mag")
    axd["B"].set_ylabel("Residual")
    axd["B"].set_xlabel("HJD $-$ 2450000")
    axd["A"].sharex(axd["B"])
    axd["A"].tick_params(axis="x", which="both", bottom=True, labelbottom=False)

    ev_name = event_cfg.get("name") or event_cfg.get("id") or "Light Curve"
    blend_flags = "".join("1" if ds.blending else "0" for ds in phot)
    axd["E"].set_title(f"{ev_name}; blending={blend_flags}")
    if len(phot) > 10:
        axd["E"].set_title(f"{ev_name}")

    if np.isfinite(t_ref):
        for key in ("A", "E"):
            axd[key].axvline(t_ref, c="r")
    axd["E"].legend(ncol=2, loc="upper left")

    fig.align_labels()
    fig.tight_layout()

    pos = axd["B"].get_position()
    axd["B"].set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])

    outfile = out_dir / "lc.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return outfile
