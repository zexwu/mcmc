"""Tiny config mapping for microlens_mcmc.

Reads a TOML dict (already loaded) and returns a minimal FitConfig.
Supports either flat `[mcmc]` keys or legacy `[mcmc.config]`.
Flat priors only, with optional simple bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class FitConfig:
    """Minimal container for sampler knobs and parameter tables.

    Use `.free` for the ordered free-parameter names and `.fixed_values`
    for a dict of fixed names to values.
    """
    step: Mapping[str, float]
    fixed: Sequence[str]
    start: Mapping[str, float]
    blob_names: Sequence[str]
    model: str = "PSPL"
    temperature: float = 1.0
    n_walkers: int = 10
    n_steps: int = 50000
    n_processes: int = 1
    t_ref: float = -1

    # sampling controls
    seed: int = 0
    burn_in: int = 0
    thin: int = 1
    check_interval: int = 500
    min_tau_mult: int = 200
    convergence_tol: float = 0.01

    # internal ordering of free parameters (derived from step_sizes keys)
    _fit_order: List[str] = field(init=False, repr=False)
    # optional simple bounds per parameter (flat prior inside, -inf outside)
    bounds: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize inputs and capture a stable free-parameter order.
        self.step = dict(self.step)
        self.start = dict(self.start)
        self.fixed = list(self.fixed)
        self.blob_names = list(self.blob_names or [])
        # dict preserves TOML order; keep it for consistent sampling/outputs
        self._fit_order = list(self.step.keys())

    @property
    def free(self) -> List[str]:
        return self._fit_order

    @property
    def param_fix(self) -> Dict[str, float]:
        return {name: self.start[name] for name in self.fixed}

    @property
    def n_param(self) -> int:
        return len(self._fit_order)

    @property
    def n_blobs(self) -> int:
        return len(self.blob_names)

    def initial_theta(self) -> List[float]:
        return [self.start[name] for name in self._fit_order]


def build_fit_config(config: dict) -> FitConfig:
    """Build FitConfig from a loaded TOML dict.

    Supports both flat `[mcmc]` keys and legacy `[mcmc.config]`.
    """
    mcmc = config.get("mcmc", {})
    params = mcmc.get("parameters")
    if not params:
        raise ValueError("Missing [mcmc.parameters]")

    # parameter tables
    start = {k: float(v["start"]) for k, v in params.items()}
    steps = {k: float(v["step"]) for k, v in params.items() if not v.get("fixed", False)}
    fixed = [k for k, v in params.items() if v.get("fixed", False)]

    # sampler knobs (flat preferred, fallback to legacy [mcmc.config])
    cfg = {**mcmc.get("config", {}), **{k: v for k, v in mcmc.items() if k not in ["parameters", "outputs"]}}

    fit = FitConfig(
        step=steps,
        fixed=fixed,
        start=start,
        **cfg
    )

    # simple bounds: u0 sign rule, then explicit per-parameter bounds
    bounds: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, spec in params.items():
        b = spec.get("bounds")
        if b and len(b) == 2:
            bounds[name] = (float(b[0]), float(b[1]))
        elif name not in bounds:
            bounds[name] = None
    fit.bounds = bounds
    return fit
