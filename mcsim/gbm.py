"""Geometric Brownian motion path simulation (fully vectorized)."""
from __future__ import annotations

import numpy as np


def normals(rng: np.random.Generator, shape: tuple, antithetic: bool = False) -> np.ndarray:
    """Standard normal draws along axis 0. With antithetic=True the second half mirrors the first."""
    if not antithetic:
        return rng.standard_normal(shape)
    n = shape[0]
    if n % 2:
        raise ValueError("antithetic sampling needs an even number of paths")
    z = rng.standard_normal((n // 2,) + tuple(shape[1:]))
    return np.concatenate([z, -z], axis=0)


def simulate_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate GBM price paths, shape (n_paths, n_steps + 1), column 0 = S0.

    Uses the exact log-space solution
        S_{t+dt} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z),
    so there is no discretization error regardless of step size.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    z = normals(rng, (n_paths, n_steps), antithetic)
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def simulate_terminal(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Terminal prices S_T only. European payoffs are path-independent, so one exact step suffices."""
    rng = rng or np.random.default_rng()
    z = normals(rng, (n_paths,), antithetic)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
