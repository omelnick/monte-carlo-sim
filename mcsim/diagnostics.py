"""Statistics that expose where GBM's i.i.d.-normal-returns assumption breaks."""
from __future__ import annotations

import numpy as np
from scipy import stats


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation for lags 1..max_lag."""
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def tail_counts(returns: np.ndarray, k: float) -> tuple[int, float]:
    """Observed number of |z| > k days vs the count a normal distribution predicts."""
    z = (returns - returns.mean()) / returns.std(ddof=1)
    observed = int(np.sum(np.abs(z) > k))
    expected = float(len(z) * 2 * stats.norm.sf(k))
    return observed, expected


def summary(returns: np.ndarray, max_lag: int = 20) -> dict:
    jb_stat, jb_p = stats.jarque_bera(returns)
    lb_crit = stats.chi2.ppf(0.95, max_lag)
    n = len(returns)
    rho = acf(returns**2, max_lag)
    ljung_box = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, max_lag + 1)))
    return {
        "skew": float(stats.skew(returns)),
        "excess_kurtosis": float(stats.kurtosis(returns)),
        "jarque_bera_p": float(jb_p),
        "ljung_box_sq": float(ljung_box),
        "ljung_box_crit95": float(lb_crit),
    }
