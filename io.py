"""Lightweight I/O helpers: config + photometry loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PhotDataset:
    """Single dataset descriptor with time-series arrays."""
    label: str
    filename: str
    filter: str | None
    blending: bool

    data: NDArray   # columns: time, mag/flux, err
    flux: NDArray   # same shape as data, converted to flux

    data_masked: Optional[NDArray] = None
    flux_masked: Optional[NDArray] = None

    color: Optional[str|None] = None
    zorder: Optional[int] = None

def load_photometry_file(path: str, subtract_jd: bool = True, jd_offset: float = 2450000.0) -> NDArray:
    """Load text photometry with 3 columns; auto-subtract jd_offset if needed."""
    usecols = (1, -2, -1) if str(path).endswith(".pysis5") else (0, 1, 2)
    arr = np.loadtxt(path, usecols=usecols)
    if subtract_jd and arr.ndim > 0 and arr.shape[0] > 0 and arr[0, 0] > jd_offset:
        arr[:, 0] -= jd_offset
    return arr


def mag_to_flux(arr: NDArray, ref_mag: float = 18.0) -> np.ndarray:
    """Convert (time, mag, err_mag) to (time, flux, err_flux)."""
    out = arr.copy()
    mag, err = out[:, 1], out[:, 2]
    flux = 10.0 ** (0.4 * (ref_mag - mag))
    out[:, 1] = flux
    out[:, 2] = flux * err * 0.4 * np.log(10.0)
    return out


def load_photometry(cfg: dict, mask: Callable[[NDArray], np.ndarray] | None = None) -> list[PhotDataset]:
    """Load photometry datasets; filenames are resolved relative to config file."""
    data_dir = Path(cfg.get("input"))

    def _apply_mask(data: NDArray, indices) -> Tuple[np.ndarray, np.ndarray]:
        if mask is None and not indices:
            return data, np.array([])
        m = np.zeros(len(data), dtype=bool)
        if mask is not None:
            m |= np.asarray(mask(data), dtype=bool)
        if indices:
            m[np.asarray(indices, dtype=int)] = True
        return data[~m], data[m]

    phot: list[PhotDataset] = []
    for ent in cfg["phot"]:

        err_floor = ent.pop("error_floor", 0.0)
        err_scale = ent.pop("error_scale", 1.0)
        mask_rows = ent.pop("mask_rows", [])

        fpath = data_dir / ent["filename"]
        raw = load_photometry_file(str(fpath), subtract_jd=True, jd_offset=2450000)
        raw[:, 2] = np.sqrt(raw[:, 2] ** 2 + err_floor**2) * err_scale
        data, data_masked = _apply_mask(raw, mask_rows)
        flux_masked = np.array([])

        if len(data_masked):
            flux_masked = mag_to_flux(data_masked)
        phot.append(PhotDataset(**ent, data=data, flux=mag_to_flux(data), data_masked=data_masked, flux_masked=flux_masked))
    return phot


def write_csv_with_metadata(path: Path, header: List[str], data: NDArray, metadata: List[str]):
    """Write CSV with metadata and header using fast NumPy I/O for rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = {
        "chi2": "%.2f",
        "u0": "%.5f",
        "teff": "%.5f",
        "A_ref": "%.5f",
        "t0": "%.5f",
        "pi1": "%.5f",
        "pi2": "%.5f",
    }
    fmt = [formatter.get(col, "%.6e") for col in header]

    with open(path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(f"# {line}\n")
        f.write(",".join(header) + "\n")
        np.savetxt(f, data, delimiter=",", fmt=fmt)

