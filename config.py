"""Configuration helpers for sampler setup.

The public entrypoint is :func:`build_fit_config`, which converts a loaded
TOML configuration dictionary into a normalized :class:`FitConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FitConfig:
    """Normalized sampler configuration.

    Attributes
    ----------
    model
        Model class name from :mod:`mcmc.models`.
    sigma
        Proposal scales keyed by parameter name.
    fixed
        Names of fixed parameters.
    start
        Starting values keyed by parameter name.
    blob_names
        Optional sampler blob names to persist in outputs.
    """

    # model name (e.g. "PSPL", "PSBL")
    model: str

    # parameter tables
    sigma: Mapping[str, float]
    fixed: Sequence[str]
    start: Mapping[str, float]

    # blob names (e.g. "t_ref", "u0_sign")
    blob_names: Sequence[str]
    t_ref: float = -1

    # sampling controls
    temperature: float = 1.0
    n_processes: int = 1
    n_walkers: int = 10
    n_steps: int = 50000
    seed: int = 0
    burn_in: int = 0
    thin: int = 1
    check_interval: int = 500
    min_tau_mult: int = 200
    convergence_tol: float = 0.01

    # optional simple bounds per parameter (flat prior inside, -inf outside)
    bounds: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize mutable inputs and derive free-parameter order."""
        self.sigma = dict(self.sigma)
        self.start = dict(self.start)
        self.fixed = list(self.fixed)
        self.free = [name for name in self.sigma if name not in self.fixed]
        self.blob_names = list(self.blob_names or [])

    @property
    def param_fix(self) -> Dict[str, float]:
        return {name: self.start[name] for name in self.fixed}

    @property
    def n_param(self) -> int:
        return len(self.free)

    @property
    def n_blobs(self) -> int:
        return len(self.blob_names)

    def initial_theta(self) -> List[float]:
        return [self.start[name] for name in self.free]


def build_fit_config(config: dict) -> FitConfig:
    """Build :class:`FitConfig` from a loaded TOML dictionary."""
    mcmc = config.get("mcmc", {})
    params = mcmc.get("parameters", {})
    blobs = mcmc.get("blobs", {})

    # parameter tables
    start = {k: float(v["start"]) for k, v in params.items()}
    sigma = {k: v.get("sigma", 0) for k, v in params.items()}
    fixed = [k for k, v in params.items() if v.get("fixed", False)]

    bounds: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, spec in params.items():
        bounds[name] = spec.get("bounds", [-np.inf, np.inf])

    fit = FitConfig(
        model=mcmc.get("model", "PSPL"),
        **mcmc.get("config", {}),
        sigma=sigma,
        fixed=fixed,
        start=start,
        bounds=bounds,
        t_ref=blobs.get("t_ref", -1),
        blob_names=blobs.get("names", []),
    )

    return fit
