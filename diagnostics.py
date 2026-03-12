from __future__ import annotations

import numpy as np


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """
    Compute split-Rhat for an array shaped (n_walkers, n_samples, n_dim).
    Simple, dependency-free Gelman-Rubin diagnostic.
    """
    W, N, D = chains.shape
    if N < 4 or W < 2:
        return np.full(D, np.nan)
    # split chains
    half = N // 2
    parts = np.concatenate([chains[:, :half, :], chains[:, half:2 * half, :]], axis=0)
    M = parts.shape[0]
    # per-chain means and variances
    chain_means = parts.mean(axis=1)
    chain_vars = parts.var(axis=1, ddof=1)
    Wn = chain_vars.mean(axis=0)
    Bn = (N / (M - 1)) * ((chain_means - chain_means.mean(axis=0)) ** 2).sum(axis=0)
    var_hat = ((N - 1) / N) * Wn + (1 / N) * Bn
    Rhat = np.sqrt(var_hat / Wn)
    return Rhat


def ess_per_dim(chains: np.ndarray) -> np.ndarray:
    """
    Rough ESS estimate per dimension using batch means.
    """
    W, N, D = chains.shape
    if N < 10:
        return np.full(D, np.nan)
    b = max(5, N // 20)
    nb = N // b
    # reshape: (W, nb, b, D)
    x = chains[:, : nb * b, :].reshape(W, nb, b, D)
    batch_means = x.mean(axis=2)
    chain_means = x.reshape(W, nb * b, D).mean(axis=1)
    s2 = ((batch_means - chain_means[:, None, :]) ** 2).sum(axis=(0, 1)) / (W * (nb - 1))
    var = x.reshape(W, nb * b, D).var(axis=(0, 1), ddof=1)
    ess = (W * nb) * var / s2
    return ess

