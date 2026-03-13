"""Lightweight I/O helpers: config + photometry loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class PhotDataset:
    """Single dataset descriptor with time-series arrays."""
    label: str
    filename: str
    filter: str | None
    blending: bool
    data: np.ndarray   # columns: time, mag/flux, err
    flux: np.ndarray   # same shape as data, converted to flux


def load_photometry_file(path: str, subtract_jd: bool = True, jd_offset: float = 2450000.0) -> np.ndarray:
    """Load text photometry with 3 columns; auto-subtract jd_offset if needed."""
    usecols = (1, -2, -1) if str(path).endswith(".pysis5") else (0, 1, 2)
    arr = np.loadtxt(path, usecols=usecols)
    if subtract_jd and arr.ndim > 0 and arr.shape[0] > 0 and arr[0, 0] > jd_offset:
        arr[:, 0] -= jd_offset
    return arr


def mag_to_flux(arr: np.ndarray, ref_mag: float = 18.0) -> np.ndarray:
    """Convert (time, mag, err_mag) to (time, flux, err_flux)."""
    out = arr.copy()
    mag, err = out[:, 1], out[:, 2]
    flux = 10.0 ** (0.4 * (ref_mag - mag))
    out[:, 1] = flux
    out[:, 2] = flux * err * 0.4 * np.log(10.0)
    return out


def load_photometry(cfg: dict, mask: Callable[[np.ndarray], np.ndarray] | None = None) -> list[PhotDataset]:
    """Load photometry datasets; filenames are resolved relative to config file."""
    data_opts = cfg.get("data", {})
    data_dir = Path(cfg.get("paths").get("input"))
    jd_offset = float(data_opts.get("jd_offset", 2450000.0))
    subtract_jd = bool(data_opts.get("subtract_jd_offset", True))

    def _apply_mask(data: np.ndarray, indices) -> np.ndarray:
        if mask is None and not indices:
            return data
        m = np.zeros(len(data), dtype=bool)
        if mask is not None:
            m |= np.asarray(mask(data), dtype=bool)
        if indices:
            m[np.asarray(indices, dtype=int)] = True
        return data[~m]

    phot: list[PhotDataset] = []
    for ent in cfg.get("phot", []):
        name = ent.get("label", ent.get("filename"))
        fname = ent["filename"]
        filt = ent.get("filter")
        blend = bool(ent.get("blending", False))
        err_floor = float(ent.get("error_floor", 0.0))
        err_scale = float(ent.get("error_scale", 1.0))
        mask_rows = list(ent.get("mask_rows", []))

        fpath = data_dir / fname
        raw = load_photometry_file(str(fpath), subtract_jd=subtract_jd, jd_offset=jd_offset)
        raw[:, 2] = np.sqrt(raw[:, 2] ** 2 + err_floor**2) * err_scale
        data = _apply_mask(raw, mask_rows)
        phot.append(PhotDataset(label=name, filename=fname, filter=filt, blending=blend, data=data, flux=mag_to_flux(data)))
    return phot
