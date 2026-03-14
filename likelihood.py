"""Likelihood and blob packing.

Caches the blob tuple layout so lnprob stays fast.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Any

import numpy as np

from .models import Parallax
from .io import PhotDataset

# Cache for blob layout to avoid recomputation for each call.
# Key: (blob_names tuple, dataset names tuple) -> (fs_pos list, fb_pos list, idx_Aref)
_BLOB_LAYOUT_CACHE: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[List[int], List[int], int ]] = {}


def _get_blob_layout(blob_names: Sequence[str], datasets: List[PhotDataset]) -> Tuple[List[int], List[int], int]:
    key = (tuple(blob_names), tuple(ds.label for ds in datasets))
    if key in _BLOB_LAYOUT_CACHE:
        return _BLOB_LAYOUT_CACHE[key]
    name_to_idx = {n: i for i, n in enumerate(blob_names)}
    fs_pos = [-1] * len(datasets)
    fb_pos = [-1] * len(datasets)
    for i, ds in enumerate(datasets):
        fs_name = f"fs_{ds.label}"
        fb_name = f"fb_{ds.label}"
        if fs_name in name_to_idx:
            fs_pos[i] = name_to_idx[fs_name]
        if fb_name in name_to_idx:
            fb_pos[i] = name_to_idx[fb_name]
    idx_Aref = name_to_idx.get("A_ref", -1)
    _BLOB_LAYOUT_CACHE[key] = (fs_pos, fb_pos, idx_Aref)
    return _BLOB_LAYOUT_CACHE[key]


def solve_blending_chi2(data_flux: np.ndarray, model: np.ndarray, blending: bool) -> Tuple[float, np.ndarray]:
    inv_err = 1.0 / data_flux[:, 2]
    y_norm = data_flux[:, 1] * inv_err
    if blending:
        A0 = model * inv_err
        A = np.stack([A0, inv_err], axis=1)
        AtA = A.T @ A
        det = AtA[0, 0] * AtA[1, 1] - AtA[0, 1] * AtA[1, 0]
        if abs(det) < 1e-20:
            return 1e16, np.array([0.0, 0.0])
        inv_det = 1.0 / det
        inv_AtA = np.empty((2, 2), dtype=A.dtype)
        inv_AtA[0, 0] = AtA[1, 1] * inv_det
        inv_AtA[0, 1] = -AtA[0, 1] * inv_det
        inv_AtA[1, 0] = -AtA[1, 0] * inv_det
        inv_AtA[1, 1] = AtA[0, 0] * inv_det
        Aty = A.T @ y_norm
        params = inv_AtA @ Aty
        resid = y_norm - A @ params
        return float(resid @ resid), params
    else:
        model_norm = model * inv_err
        dot_mm = float(model_norm @ model_norm)
        fs = float(model_norm @ y_norm) / dot_mm
        resid = y_norm - fs * model_norm
        chi2 = float(resid @ resid)
        return chi2, np.array([fs, 0.0])


def log_likelihood(
    params: Dict[str, float],
    datasets: List[PhotDataset],
    model: Parallax,
    blob_names: Sequence[str],
) -> Tuple[float, Any]:
    # Precompute parallax trajectories for these datasets (simple LC fitting)
    if not getattr(model, "is_precal", False):
        model.precalculate_parallax([ds.data[:, 0] for ds in datasets])
    chi2_total = 0.0
    blob = [None] * len(blob_names)
    fs_pos, fb_pos, idx_Aref = _get_blob_layout(blob_names, datasets)

    for dataset_id, ds in enumerate(datasets):
        t = ds.data[:, 0]
        mag = model.magnification(t, params, dataset_id=dataset_id)
        chi2, lin_params = solve_blending_chi2(ds.flux, mag, blending=ds.blending)
        chi2_total += chi2
        if fs_pos[dataset_id] >= 0:
            blob[fs_pos[dataset_id]] = float(lin_params[0])
        if fb_pos[dataset_id] >= 0:
            blob[fb_pos[dataset_id]] = float(lin_params[1])

    if idx_Aref >= 0:
        blob[idx_Aref] = model.magnification(params["t_ref"], params, dataset_id=-1)

    return -0.5 * chi2_total, blob
