"""Load real price history and calibrate GBM parameters."""
from __future__ import annotations

from pathlib import Path

import numpy as np

TRADING_DAYS = 252
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str = "SPY", start: str = "2010-01-01", end: str | None = None) -> np.ndarray:
    """Adjusted daily closes. Downloads via yfinance once, then reads the CSV cache in data/."""
    cache = DATA_DIR / f"{ticker}_{start}.csv"
    if cache.exists():
        return np.loadtxt(cache, delimiter=",", skiprows=1, usecols=1)

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"no data returned for {ticker}; check network or ticker")
    close = df["Close"].squeeze().dropna()
    DATA_DIR.mkdir(exist_ok=True)
    with open(cache, "w") as f:
        f.write("date,close\n")
        for d, p in close.items():
            f.write(f"{d.date()},{p:.6f}\n")
    return close.to_numpy(dtype=float)


def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


def calibrate_gbm(prices: np.ndarray) -> tuple[float, float]:
    """MLE of annualized (mu, sigma) from daily closes.

    Daily log returns ~ N((mu - sigma^2/2) dt, sigma^2 dt), so
    sigma = std * sqrt(252) and mu = mean * 252 + sigma^2 / 2.
    """
    r = log_returns(prices)
    sigma = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    mu = r.mean() * TRADING_DAYS + 0.5 * sigma**2
    return float(mu), float(sigma)
