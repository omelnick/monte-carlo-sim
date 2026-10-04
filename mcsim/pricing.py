"""Black-Scholes closed form and risk-neutral Monte Carlo pricing of European options."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .gbm import simulate_terminal


def black_scholes(S0: float, K: float, r: float, sigma: float, T: float, kind: str = "call") -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if kind == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    raise ValueError(f"unknown option kind: {kind}")


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def mc_european(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    kind: str = "call",
    antithetic: bool = False,
    rng: np.random.Generator | None = None,
) -> MCResult:
    """Price a European option under the risk-neutral measure (drift = r).

    With antithetic sampling, path i and path i + n/2 are negatively correlated, so the
    standard error is computed over the n/2 independent pair-averages, not the n raw payoffs.
    """
    ST = simulate_terminal(S0, r, sigma, T, n_paths, rng=rng, antithetic=antithetic)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T) * payoff
    if antithetic:
        half = n_paths // 2
        samples = 0.5 * (disc[:half] + disc[half:])
    else:
        samples = disc
    return MCResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(len(samples))),
        n_paths=n_paths,
    )
