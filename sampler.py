"""Sampling entrypoints and result container.

Public API:
- fit(path) -> Results
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import emcee
import numpy as np

from .config import FitConfig, build_fit_config
from .io import read_config, load_photometry
from .likelihood import log_likelihood
from .models import PSPL



@dataclass
class Results:
    """Container for MCMC outputs.

    - samples: flattened array (n_samples, n_dim)
    - log_prob: flattened log probability for each sample
    - blobs: dict of flattened blob arrays (if requested)
    - param_names: names aligned with samples columns
    - best: dict with best parameters and chi2
    """
    samples: np.ndarray  # (n_samples, n_dim)
    log_prob: np.ndarray  # (n_samples,)
    blobs: Dict[str, np.ndarray]
    param_names: List[str]
    best: Dict[str, float]

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        q = np.percentile(self.samples, [16, 50, 84], axis=0)
        for i, name in enumerate(self.param_names):
            out[name] = {"p16": float(q[0, i]), "median": float(q[1, i]), "p84": float(q[2, i])}
        out["chi2_best"] = {"value": float(self.best.get("chi2", np.nan))}
        return out


def _theta_to_params(theta: np.ndarray, fit: FitConfig) -> List[Dict[str, float]]:
    """Map theta rows to parameter dicts, merging fixed values and tE derivation."""
    names = fit.free
    theta = np.atleast_2d(theta)
    params_list: List[Dict[str, float]] = []
    for i in range(theta.shape[0]):
        p: Dict[str, float] = dict(fit.param_fix)
        for j, name in enumerate(names):
            p[name] = float(theta[i, j])
        if "teff" in p and "u0" in p:
            p["tE"] = abs(p["teff"] / p["u0"])
        params_list.append(p)
    return params_list


def _should_stop(tau: np.ndarray | None, last_tau: np.ndarray | None, steps_done: int, cfg: FitConfig) -> bool:
    if tau is None:
        return False
    if not np.isfinite(tau).all():
        return False
    if steps_done <= cfg.min_tau_mult * tau.max():
        return False
    if last_tau is None:
        return False
    rel_change = (np.abs(tau - last_tau) / tau).max()
    return rel_change < cfg.convergence_tol


def _initialize_walkers(fit: FitConfig, rng: np.random.Generator) -> np.ndarray:
    """Fast, vectorized walker init with simple bounds.

    - Finite bounds: draw uniformly in [low, high].
    - Otherwise: draw N(start, step).
    - Grow batch size until enough valid walkers are found.
    """
    names = fit.free
    d = len(names)
    centers = np.array([fit.start[n] for n in names], dtype=float)
    widths = np.array([fit.step[n] for n in names], dtype=float)
    lows = np.full(d, -np.inf)
    highs = np.full(d, np.inf)
    for j, n in enumerate(names):
        b = fit.bounds.get(n)
        if b is not None:
            lows[j], highs[j] = b

    def sample_batch(m: int) -> np.ndarray:
        X = np.empty((m, d), dtype=float)
        for j in range(d):
            low, high = lows[j], highs[j]
            if np.isfinite(low) and np.isfinite(high):
                X[:, j] = rng.uniform(low, high, size=m)
            else:
                X[:, j] = rng.normal(centers[j], widths[j], size=m)
        return X

    need = fit.n_walkers
    batch = max(need * 4, 64)
    acc_list: list[np.ndarray] = []
    while need > 0:
        X = sample_batch(batch)
        mask = np.ones(batch, dtype=bool)
        for j in range(d):
            if np.isfinite(lows[j]):
                mask &= X[:, j] >= lows[j]
            if np.isfinite(highs[j]):
                mask &= X[:, j] <= highs[j]
        if mask.any():
            Y = X[mask]
            take = min(len(Y), need)
            if take > 0:
                acc_list.append(Y[:take])
                need -= take
        if need > 0:
            batch = min(batch * 2, max(1024, fit.n_walkers * 64))
        if batch > 1_000_000:  # safety valve
            break

    if not acc_list:
        # Fallback: repeat clipped start
        x0 = np.clip(centers, lows, highs)
        return np.tile(x0, (fit.n_walkers, 1))
    return np.vstack(acc_list)[: fit.n_walkers]


def _write_csv_with_metadata(path: Path, header: List[str], data: np.ndarray, metadata: List[str]):
    """Write CSV with metadata and header using fast NumPy I/O for rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(f"# {line}\n")
        f.write(",".join(header) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.8g")


def fit(config_path: str | Path) -> Results:
    """Run emcee on a configuration file and return Results.

    The function is dependency-light and prints no plots. See README for
    the TOML schema and examples.
    """
    # Load configuration and data
    cfg, base_dir = read_config(config_path)
    phot = load_photometry(cfg)

    # Build model and precompute parallax
    event_cfg = cfg.get("event", {})
    coords = event_cfg.get("coords", "").split()
    if len(coords) != 2:
        raise ValueError("event.coords must be 'RA DEC' string, e.g. '17:31:42.61 -30:46:17.04'")
    model = PSPL(*coords)
    model.precalculate_parallax([ds.data[:, 0] for ds in phot])

    # Fit configuration
    fit_config = build_fit_config(cfg)

    # scalar log-prob with emcee-supported blob tuples
    def lnprob(theta):
        p = _theta_to_params(np.asarray(theta)[None, :], fit_config)[0]
        p["t_ref"] = np.asarray(fit_config.t_ref)
        # flat prior with optional bounds
        null = -np.inf, *tuple(np.nan for _ in fit_config.blobs)
        for name, b in fit_config.bounds.items():
            if b is None or name not in p:
                continue
            if not (b[0] <= p[name] <= b[1]):
                if fit_config.blobs:
                    return null
                return -np.inf
        # also pass bounds into likelihood for internal checking
        ll, blob = log_likelihood(p, phot, model, fit_config.blobs)
        lp = ll / fit_config.temperature
        return (lp, *blob) if fit_config.blobs else lp

    # Initialize walkers around starts with Gaussian scatter
    rng = np.random.default_rng(fit_config.seed)
    p0 = _initialize_walkers(fit_config, rng)

    # Run sampler (scalar lnprob)
    n_dim = fit_config.n_param
    blob_dtype = [(name, float) for name in fit_config.blobs] if fit_config.blobs else None
    sampler = emcee.EnsembleSampler(
        fit_config.n_walkers,
        n_dim,
        lnprob,
        blobs_dtype=blob_dtype,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
    )

    max_steps = fit_config.n_steps
    state = sampler.run_mcmc(p0, fit_config.check_interval, progress=False)
    steps_done = fit_config.check_interval
    last_tau = None
    while steps_done < max_steps:
        run = min(fit_config.check_interval, max_steps - steps_done)
        if steps_done > 2 * fit_config.burn_in:
            tau = sampler.get_autocorr_time(tol=0, discard=fit_config.burn_in)
            print("\rN_eff =", int(steps_done / tau.max()), end="", flush=True)
            if _should_stop(tau, last_tau, steps_done, fit_config):
                break
            if tau is not None and np.isfinite(tau).all():
                last_tau = tau
        state = sampler.run_mcmc(state, run, progress=False)
        steps_done += run

    # flatten with burn-in and thinning
    burn = fit_config.burn_in
    thin = max(1, fit_config.thin)
    chain = sampler.get_chain(discard=burn, thin=thin)
    logp = sampler.get_log_prob(discard=burn, thin=thin)
    blobs = sampler.get_blobs(discard=burn, thin=thin)

    # parameter space samples directly
    W, S, D = chain.shape
    samples = chain.reshape(W * S, D)
    names = fit_config.free
    logp_flat = logp.reshape(W * S)

    blob_dict: Dict[str, np.ndarray] = {}
    if fit_config.blobs and blobs is not None:
        for bn in fit_config.blobs:
            b = blobs[bn].reshape(W * S)
            blob_dict[bn] = np.asarray(b)

    # best row
    idx_best = int(np.argmax(logp_flat))
    best_params = {n: float(samples[idx_best, i]) for i, n in enumerate(names)}
    # compute chi2_best from logp and temperature
    chi2_best = -2.0 * float(logp_flat[idx_best]) * fit_config.temperature
    best = {**best_params, "chi2": chi2_best}

    # write outputs
    out_dir = Path(cfg.get("paths").get("output"))
    out_cfg = cfg.get("mcmc", {}).get("output", {})
    chain_file = out_dir / out_cfg.get("chain_file")
    best_file = out_dir / out_cfg.get("best_file")

    header = ["chi2", *names, *fit_config.blobs]
    chi2_all = -2.0 * logp_flat * fit_config.temperature
    data_cols = [chi2_all, *[samples[:, i] for i in range(samples.shape[1])]]
    for bn in fit_config.blobs:
        data_cols.append(blob_dict.get(bn, np.full_like(logp_flat, np.nan)))
    data_all = np.column_stack(data_cols)

    metadata = [
        f"generated=emcee; steps={steps_done}; walkers={fit_config.n_walkers}",
        f"burn_in={fit_config.burn_in}; thin={fit_config.thin}; temperature={fit_config.temperature}",
        f"seed={fit_config.seed}; samples={data_all.shape[0]}",
    ]

    _write_csv_with_metadata(chain_file, header, data_all, metadata)
    # best: single row
    best_row = np.array([[best["chi2"], *[best[n] for n in names], *[np.nan for _ in fit_config.blobs]]])
    _write_csv_with_metadata(best_file, header, best_row, metadata)

    return Results(samples=samples, log_prob=logp_flat, blobs=blob_dict, param_names=names, best=best)
