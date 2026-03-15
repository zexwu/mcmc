"""Sampling entrypoints and result container.

Public API:
- fit(path) -> Results
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from numpy.typing import NDArray
import toml
import emcee
import warnings
import numpy as np

from .config import FitConfig, build_fit_config
from .io import load_photometry, write_csv_with_metadata
from .likelihood import log_likelihood
from . import models
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=RuntimeWarning)

def colored_text(text: str, color: str="white", bold: bool=False) -> str:
    # Dictionary mapping color names to ANSI codes
    colors = {
        "red": "\033[91m", "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }

    code = colors.get(color.lower(), colors["white"])
    if bold:
        return f"\033[1m{code}{text}{colors['reset']}\033[0m"
    return f"{code}{text}{colors['reset']}"


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


def _initialize_walkers(fit: FitConfig) -> np.ndarray:
    """Fast, vectorized walker init with simple bounds.

    - Finite bounds: draw uniformly in [low, high].
    - Otherwise: draw N(start, step).
    - Grow batch size until enough valid walkers are found.
    """
    names = fit.free
    d = len(names)
    centers = np.array([fit.start[n] for n in names], dtype=float)
    widths = np.array([fit.sigma[n] for n in names], dtype=float)
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
                X[:, j] = np.random.uniform(low, high, size=m)
            else:
                X[:, j] = np.random.normal(centers[j], widths[j], size=m)
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


# scalar log-prob with emcee-supported blob tuples
def lnprob(theta: NDArray, datasets: Sequence[Any], model: Any, fit_config: FitConfig):
    p = dict(zip(fit_config.free, theta))
    p.update(fit_config.param_fix)

    if fit_config.t_ref:
        p["t_ref"] = np.asarray(fit_config.t_ref)

    # flat prior with optional bounds
    null = -np.inf, *tuple(np.nan for _ in fit_config.blob_names)
    for name, b in fit_config.bounds.items():
        if b is None:
            continue
        if not (b[0] <= p[name] <= b[1]):
            if fit_config.blob_names: return null
            return -np.inf
    # also pass bounds into likelihood for internal checking
    ll, blob = log_likelihood(p, datasets, model, fit_config.blob_names)
    lp = ll / fit_config.temperature
    return (lp, *blob) if fit_config.blob_names else lp


def fit(config_path: str | Path) -> Results:
    """Run emcee on a configuration file and return Results.

    The function is dependency-light and prints no plots. See README for
    the TOML schema and examples.
    """
    # Load configuration and data
    cfg = toml.load(config_path)
    phot = load_photometry(cfg)

    # Build model and precompute parallax
    coords = cfg.get("coords", "").split()
    if len(coords) != 2:
        raise ValueError("event.coords must be 'RA DEC' string, e.g. '17:31:42.61 -30:46:17.04'")

    # Fit configuration
    fit_config = build_fit_config(cfg)

    model = getattr(models, fit_config.model)(*coords)

    # Initialize walkers around starts with Gaussian scatter
    np.random.seed(fit_config.seed)  # for any non-NumPy code that uses global seed
    p0 = _initialize_walkers(fit_config)

    # Run sampler (scalar lnprob)
    blob_dtype = [(name, float) for name in fit_config.blob_names] if fit_config.blob_names else None
    pool = None
    if fit_config.n_processes > 1:
        pool = Pool(fit_config.n_processes)
    sampler = emcee.EnsembleSampler(
        fit_config.n_walkers,
        fit_config.n_param,
        lnprob,
        args=(phot, model, fit_config),
        blobs_dtype=blob_dtype,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        pool=pool
    )

    max_steps = fit_config.n_steps
    state = sampler.run_mcmc(p0, fit_config.check_interval, progress=False)
    steps_done = fit_config.check_interval
    last_tau = np.inf
    while steps_done < max_steps:
        run = min(fit_config.check_interval, max_steps - steps_done)
        discard = max(0, steps_done - 3 * fit_config.check_interval)

        tau = sampler.get_autocorr_time(tol=0, discard=discard)
        tau = np.clip(tau, 1e-8, None)
        rel_err = (np.abs(tau - last_tau) / tau).max()
        rel_err = np.clip(rel_err, 0, 0.999)

        window = fit_config.check_interval
        chain = sampler.get_chain()[-(window+1):]
        accepted = np.any(chain[1:] != chain[:-1], axis=2)
        rolling_af = np.mean(accepted)
        tau_str = ",".join(f"{x:3.0f}" for x in tau)

        print(colored_text("\r"+cfg.get("event")+">", bold=True),
              colored_text(f"N={steps_done:5d};", "green"),
              colored_text(f"tau={tau_str};"),
              colored_text(f"N/tau={steps_done / tau.max():4.0f};", "blue"),
              colored_text(f"relerr={rel_err:5.1%};"),
              colored_text(f"acc={rolling_af:5.1%} ", "green"),
              end="", flush=True)
        print()
        if _should_stop(tau, last_tau, steps_done, fit_config):
            break
        if tau is not None and np.isfinite(tau).all():
            last_tau = tau

        state = sampler.run_mcmc(state, run, progress=False)
        steps_done += run

    print("\n")
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
    for bn in fit_config.blob_names:
        blob_dict[bn] = blobs[bn].reshape(W * S)

    # best row
    idx_best = int(np.argmax(logp_flat))
    best_params = {n: float(samples[idx_best, i]) for i, n in enumerate(names)}
    # compute chi2_best from logp and temperature
    chi2_best = -2.0 * float(logp_flat[idx_best]) * fit_config.temperature
    best = {**best_params, "chi2": chi2_best}

    # write outputs
    out_dir = Path(cfg.get("output"))
    out_cfg = cfg.get("mcmc", {}).get("outputs", {})
    chain_file = out_dir / out_cfg.get("chain_file")
    best_file = out_dir / out_cfg.get("best_file")

    header = ["chi2", *names, *fit_config.blob_names]
    chi2_all = -2.0 * logp_flat * fit_config.temperature
    data_cols = [chi2_all, *[samples[:, i] for i in range(samples.shape[1])]]
    for bn in fit_config.blob_names:
        data_cols.append(blob_dict.get(bn, np.full_like(logp_flat, np.nan)))
    data_all = np.column_stack(data_cols)

    metadata = [
        f"generated=emcee; steps={steps_done}; walkers={fit_config.n_walkers}",
        f"burn_in={fit_config.burn_in}; thin={fit_config.thin}; temperature={fit_config.temperature}",
        f"seed={fit_config.seed}; samples={data_all.shape[0]}",
    ]

    write_csv_with_metadata(chain_file, header, data_all, metadata)
    best_row = data_all[idx_best]

    # add fixed parameters to results
    for par in fit_config.fixed:
        header.append(par)
        best_row = np.insert(best_row, header.index(par), fit_config.param_fix[par])
    best_row = np.atleast_2d(best_row)

    write_csv_with_metadata(best_file, header, best_row, metadata)

    return Results(samples=samples, log_prob=logp_flat, blobs=blob_dict, param_names=names, best=best)
