"""Outlier-rejecting preprocessing workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import toml
import tomlkit
from scipy.optimize import least_squares
from tomlkit import item, table
from typing import Any

from . import models
from .config import FitConfig, build_fit_config
from .io import load_photometry, load_photometry_file
from .likelihood import solve_blending_chi2


def _default_output_path(config_path: Path) -> Path:
    """Return the default cleaned-config path next to the source config."""
    return config_path.with_name(f"{config_path.stem}.clean.toml")


def _current_mask_rows(entry: dict) -> list[int]:
    """Return sorted row indices already masked in one ``[[phot]]`` entry."""
    return sorted({int(idx) for idx in entry.get("mask_rows", [])})


def _unmasked_indices(size: int, mask_rows: list[int]) -> np.ndarray:
    """Map local unmasked rows back to indices in the raw photometry file."""
    mask = np.zeros(size, dtype=bool)
    if mask_rows:
        mask[np.asarray(mask_rows, dtype=int)] = True
    return np.flatnonzero(~mask)


def _build_model(cfg: dict[str, Any], fit_config: FitConfig, phot):
    """Instantiate the configured model and precache parallax trajectories."""
    coords = cfg.get("coords", "").split()
    if len(coords) != 2:
        raise ValueError("event.coords must be 'RA DEC' string, e.g. '17:31:42.61 -30:46:17.04'")
    model = getattr(models, fit_config.model)(*coords)
    model.precalculate_parallax([ds.data[:, 0] for ds in phot])
    return model


def _params_from_theta(theta: np.ndarray, fit_config: FitConfig) -> dict[str, float]:
    """Merge free-vector values with fixed parameters into one parameter dict."""
    params = {name: float(value) for name, value in fit_config.param_fix.items()}
    params.update({name: float(value) for name, value in zip(fit_config.free, theta, strict=True)})
    return params


def _bounds_from_config(fit_config: FitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-parameter bounds into ``least_squares`` lower/upper arrays."""
    lower: list[float] = []
    upper: list[float] = []
    for name in fit_config.free:
        bounds = fit_config.bounds.get(name)
        if bounds is None:
            lower.append(-np.inf)
            upper.append(np.inf)
        else:
            lower.append(bounds[0])
            upper.append(bounds[1])
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _dataset_residuals(ds, mag: np.ndarray) -> np.ndarray:
    """Return normalized flux residuals after solving linear blending terms."""
    _, lin_params = solve_blending_chi2(ds.flux, mag, blending=ds.blending)
    if np.isfinite(lin_params).all():
        fs, fb = lin_params
        model_flux = fs * mag + fb
        resid = (ds.flux[:, 1] - model_flux) / ds.flux[:, 2]
        return resid
    else:
        return np.zeros(len(ds.data), dtype=float)



def _all_residuals(theta: np.ndarray, fit_config: FitConfig, phot, model, masks: list[np.ndarray]) -> np.ndarray:
    """Stack residuals from all currently unmasked datasets for least-squares."""
    params = _params_from_theta(theta, fit_config)
    residuals: list[np.ndarray] = []
    for dataset_id, ds in enumerate(phot):
        mask = masks[dataset_id]
        if len(ds.data) == 0 or np.all(mask):
            continue
        mag = model.magnification(ds.data[:, 0], params, dataset_id=dataset_id, a1=ds.a1)
        residuals.append(_dataset_residuals(ds, mag)[~mask])
    if not residuals:
        return np.zeros(0, dtype=float)
    return np.concatenate(residuals)


def _fit_theta(fit_config: FitConfig, phot, model, masks: list[np.ndarray], theta0: np.ndarray) -> np.ndarray:
    """Run one nonlinear least-squares step starting from ``theta0``."""
    lower, upper = _bounds_from_config(fit_config)
    result = least_squares(
        _all_residuals,
        theta0,
        jac='2-point',
        bounds=(lower, upper),
        args=(fit_config, phot, model, masks),
        max_nfev=2000,
    )
    return result.x


def _outlier_rows(cfg: dict, masks: list[np.ndarray]) -> list[list[int]]:
    """Convert local boolean masks back to raw ``mask_rows`` indices per dataset."""
    input_dir = Path(cfg.get("input"))
    rows: list[list[int]] = []
    for entry, local_mask in zip(cfg["phot"], masks, strict=True):
        raw = load_photometry_file(str(input_dir / entry["filename"]), subtract_jd=True, jd_offset=2450000)
        existing = _current_mask_rows(entry)
        keep = _unmasked_indices(len(raw), existing)
        rows.append(keep[local_mask].astype(int).tolist())
    return rows


def _updated_error_scales(cfg: dict, phot, model, masks: list[np.ndarray], best_params: dict[str, float]) -> list[float]:
    """Rescale each dataset error bar by the RMS of retained normalized residuals."""
    scales: list[float] = []
    for dataset_id, (entry, ds, mask) in enumerate(zip(cfg["phot"], phot, masks, strict=True)):
        current_scale = float(entry.get("error_scale", 1.0))
        if len(ds.data) == 0 or np.all(mask):
            scales.append(current_scale)
            continue
        mag = model.magnification(ds.data[:, 0], best_params, dataset_id=dataset_id, a1=ds.a1)
        resid = _dataset_residuals(ds, mag)
        kept = resid[~mask]
        if kept.size == 0:
            scales.append(current_scale)
            continue
        rms = float(np.sqrt(np.mean(kept**2)))
        if not np.isfinite(rms) or rms <= 0:
            scales.append(current_scale)
            continue
        scales.append(current_scale * rms)
    return scales


def _format_fixed_float(value: float, digits: int) -> float:
    """Round a float to a fixed number of decimals for cleaner TOML output."""
    return float(f"{float(value):.{digits}f}")


def _inline_array(values: list[int]):
    """Create a TOML array item and keep it on one line when possible."""
    arr = item([int(value) for value in values])
    try:
        arr.multiline(False)
    except Exception:
        pass
    return arr


def _update_clean_doc(
    source_path: Path,
    outliers: list[list[int]],
    error_scales: list[float],
    sigma: float,
    maxiters: int,
    best_params: dict[str, float],
) -> str:
    """Apply mask/scale/start updates while preserving TOML formatting."""
    doc = tomlkit.parse(source_path.read_text(encoding="utf-8"))

    for phot_entry, new_outliers, error_scale in zip(doc["phot"], outliers, error_scales, strict=True):
        phot_entry["mask_rows"] = _inline_array(new_outliers)
        phot_entry["error_scale"] = item(_format_fixed_float(error_scale, 2))

    param_table = doc["mcmc"]["parameters"]
    for name, value in best_params.items():
        if name in param_table:
            param_table[name]["start"] = item(_format_fixed_float(value, 3))

    rejection = table()
    rejection["source_config"] = str(source_path)
    rejection["sigma"] = float(sigma)
    rejection["maxiters"] = int(maxiters)
    doc["outlier_rejection"] = rejection

    return tomlkit.dumps(doc)


def _reject_outliers(fit_config: FitConfig, phot, model, sigma: float, maxiters: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Alternate between least-squares fitting and sigma clipping."""
    masks = [np.zeros(len(ds.data), dtype=bool) for ds in phot]
    theta = np.asarray(fit_config.initial_theta(), dtype=float)

    for _ in range(maxiters):
        # Warm-start each iteration from the previous best-fit parameters.
        theta = _fit_theta(fit_config, phot, model, masks, theta)
        params = _params_from_theta(theta, fit_config)
        changed = False

        for dataset_id, ds in enumerate(phot):
            if len(ds.data) == 0 or np.all(masks[dataset_id]):
                continue
            mag = model.magnification(ds.data[:, 0], params, dataset_id=dataset_id, a1=ds.a1)
            resid = _dataset_residuals(ds, mag)
            # Clip on normalized flux residuals after the current best fit.
            chi2_per_point = (resid[~masks[dataset_id]] ** 2).mean()
            new_mask = masks[dataset_id] | (resid ** 2 / chi2_per_point > sigma ** 2)
            if not np.array_equal(new_mask, masks[dataset_id]):
                masks[dataset_id] = new_mask
                changed = True

        if not changed:
            break
    for dataset_id, ds in enumerate(phot):
        if "diapl" in ds.filename: continue
        masks[dataset_id] |= (ds.data[:, 2]) > 0.2

    return theta, masks


def fit(
    config_path: str | Path,
    *,
    sigma: float = 5.0,
    maxiters: int = 3,
    output_path: str | Path | None = None,
) -> Path:
    """Run a deterministic pre-fit, reject outliers, and write a cleaned config.

    The cleaned TOML keeps the original structure/comments via ``tomlkit`` and
    updates three things:
    - ``mask_rows`` for rejected points
    - ``error_scale`` per dataset after rejection
    - parameter ``start`` values from the deterministic best fit
    """
    config_path = Path(config_path)
    cfg = toml.load(config_path)
    fit_config = build_fit_config(cfg)
    phot = load_photometry(cfg)
    model = _build_model(cfg, fit_config, phot)

    theta, masks = _reject_outliers(fit_config, phot, model, sigma, maxiters)
    best_params = _params_from_theta(theta, fit_config)
    outliers = _outlier_rows(cfg, masks)
    error_scales = _updated_error_scales(cfg, phot, model, masks, best_params)

    destination = Path(output_path) if output_path is not None else _default_output_path(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _update_clean_doc(
            config_path,
            outliers,
            error_scales,
            sigma,
            maxiters,
            best_params,
        ),
        encoding="utf-8",
    )
    return destination
