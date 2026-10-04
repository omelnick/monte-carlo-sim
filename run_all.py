"""Run every experiment, print a report, and write figures to figures/."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mcsim.data import TRADING_DAYS, calibrate_gbm, load_prices, log_returns
from mcsim.diagnostics import acf, summary, tail_counts
from mcsim.gbm import simulate_paths
from mcsim.pricing import black_scholes, mc_european

FIG = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "legend.frameon": False, "lines.linewidth": 2,
})


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def fig_paths(S0, mu, sigma, T, rng):
    section("1. GBM price paths (1,000 paths, 1 year, daily steps)")
    paths = simulate_paths(S0, mu, sigma, T, TRADING_DAYS, 1000, rng=rng)
    ST = paths[:, -1]
    t = np.linspace(0, T, TRADING_DAYS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.plot(t, paths[:200].T, color=BLUE, alpha=0.06, linewidth=0.7)
    for q, ls in [(5, "--"), (50, "-"), (95, "--")]:
        ax1.plot(t, np.percentile(paths, q, axis=0), color=INK, linestyle=ls, linewidth=1.5)
        ax1.annotate(f"p{q}", (t[-1], np.percentile(ST, q)), xytext=(4, 0),
                     textcoords="offset points", va="center", color=INK2)
    ax1.set(title="Simulated paths (200 of 1,000 shown) with 5/50/95th percentiles",
            xlabel="Years", ylabel="Price")

    ax2.hist(ST, bins=50, density=True, color=BLUE, alpha=0.75, edgecolor="#fcfcfb", linewidth=1)
    x = np.linspace(ST.min(), ST.max(), 400)
    pdf = stats.lognorm.pdf(x, s=sigma * np.sqrt(T), scale=S0 * np.exp((mu - 0.5 * sigma**2) * T))
    ax2.plot(x, pdf, color=ORANGE, label="Theoretical lognormal")
    ax2.axvline(S0, color=INK2, linestyle=":", linewidth=1)
    ax2.set(title="Distribution of terminal price $S_T$", xlabel="Price", ylabel="Density")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG / "1_paths.png", dpi=150)
    plt.close(fig)

    var95 = S0 - np.percentile(ST, 5)
    print(f"  S0={S0:.2f}  mu={mu:.2%}  sigma={sigma:.2%}")
    print(f"  E[S_T] sim={ST.mean():.2f}  theory={S0 * np.exp(mu * T):.2f}")
    print(f"  P(loss)={np.mean(ST < S0):.1%}   1y 95% VaR = {var95:.2f} ({var95 / S0:.1%} of S0)")


def fig_pricing(S0, r, sigma, T, rng):
    section("2. European options: Monte Carlo vs Black-Scholes (100,000 paths, antithetic)")
    print(f"  {'K':>8} {'type':>5} {'BS':>9} {'MC':>9} {'SE':>7} {'|err|/SE':>9}")
    for K in S0 * np.array([0.8, 0.9, 1.0, 1.1, 1.2]):
        for kind in ("call", "put"):
            bs = black_scholes(S0, K, r, sigma, T, kind)
            mc = mc_european(S0, K, r, sigma, T, 100_000, kind, antithetic=True, rng=rng)
            print(f"  {K:8.2f} {kind:>5} {bs:9.4f} {mc.price:9.4f} {mc.std_error:7.4f} "
                  f"{abs(mc.price - bs) / mc.std_error:9.2f}")

    section("3. Variance reduction: antithetic variates (ATM call)")
    K = S0
    bs = black_scholes(S0, K, r, sigma, T)
    ns = np.unique(np.logspace(2, 5.5, 25).astype(int) // 2 * 2)
    plain = [mc_european(S0, K, r, sigma, T, n, rng=rng) for n in ns]
    anti = [mc_european(S0, K, r, sigma, T, n, antithetic=True, rng=rng) for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for res, color, label in [(plain, ORANGE, "Plain MC"), (anti, BLUE, "Antithetic")]:
        p = np.array([x.price for x in res])
        se = np.array([x.std_error for x in res])
        ax1.fill_between(ns, p - 1.96 * se, p + 1.96 * se, color=color, alpha=0.15, linewidth=0)
        ax1.plot(ns, p, color=color, marker="o", markersize=3, label=label)
        ax2.loglog(ns, se, color=color, marker="o", markersize=3, label=label)
    ax1.axhline(bs, color=INK, linestyle="--", linewidth=1.2, label=f"Black-Scholes = {bs:.3f}")
    ax1.set_xscale("log")
    ax1.set(title="ATM call estimate with 95% CI", xlabel="Number of paths", ylabel="Price")
    ax1.legend()
    ref = plain[0].std_error * np.sqrt(ns[0] / ns)
    ax2.loglog(ns, ref, color=INK2, linestyle=":", linewidth=1.2, label=r"$\propto 1/\sqrt{N}$")
    ax2.set(title="Standard error vs paths", xlabel="Number of paths", ylabel="Standard error")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG / "2_pricing_convergence.png", dpi=150)
    plt.close(fig)

    # Fair comparison at a fixed budget of payoff evaluations, repeated to estimate estimator variance
    n, reps = 10_000, 500
    est_p = [mc_european(S0, K, r, sigma, T, n, rng=rng).price for _ in range(reps)]
    est_a = [mc_european(S0, K, r, sigma, T, n, antithetic=True, rng=rng).price for _ in range(reps)]
    ratio = np.var(est_p) / np.var(est_a)
    print(f"  BS price {bs:.4f}")
    print(f"  {reps} repetitions x {n:,} paths each:")
    print(f"    plain      mean={np.mean(est_p):.4f}  std={np.std(est_p):.4f}")
    print(f"    antithetic mean={np.mean(est_a):.4f}  std={np.std(est_a):.4f}")
    print(f"  variance reduction factor = {ratio:.2f}x  (same accuracy with ~{ratio:.1f}x fewer paths)")


def fig_failures(prices, ticker, mu, sigma, rng):
    section(f"4. Where GBM fails: {ticker} daily returns vs GBM with the same mu, sigma")
    real = log_returns(prices)
    n = len(real)
    sim_prices = simulate_paths(prices[0], mu, sigma, n / TRADING_DAYS, n, 1, rng=rng)[0]
    sim = log_returns(sim_prices)

    s_real, s_sim = summary(real), summary(sim)
    print(f"  {'':24}{ticker:>12}{'GBM':>12}")
    for k in s_real:
        print(f"  {k:24}{s_real[k]:12.4g}{s_sim[k]:12.4g}")
    for k in (3, 4, 5):
        o, e = tail_counts(real, k)
        print(f"  days with |z| > {k}: observed {o:4d}, normal predicts {e:8.2f}  ({o / e:,.0f}x)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    z = (real - real.mean()) / real.std()
    bins = np.linspace(-8, 8, 81)
    ax.hist(z, bins=bins, density=True, color=BLUE, alpha=0.75, edgecolor="#fcfcfb", label=ticker)
    xs = np.linspace(-8, 8, 400)
    ax.plot(xs, stats.norm.pdf(xs), color=ORANGE, label="Normal (GBM)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set(title="Fat tails: standardized returns (log scale)", xlabel="z-score", ylabel="Density")
    ax.legend()

    ax = axes[0, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm")
    ax.scatter(osm, osr, s=8, color=BLUE, alpha=0.6, label=ticker)
    ax.plot(osm, osm, color=ORANGE, label="Normal")
    ax.set(title="Q-Q plot vs normal", xlabel="Theoretical quantile", ylabel="Sample quantile")
    ax.legend()

    ax = axes[1, 0]
    lags = np.arange(1, 61)
    band = 1.96 / np.sqrt(n)
    ax.bar(lags - 0.2, acf(real**2, 60), width=0.4, color=BLUE, label=ticker)
    ax.bar(lags + 0.2, acf(sim**2, 60), width=0.4, color=ORANGE, label="GBM")
    ax.axhspan(-band, band, color=INK2, alpha=0.12, linewidth=0, label="95% noise band")
    ax.set(title="Volatility clustering: ACF of squared returns", xlabel="Lag (days)",
           ylabel="Autocorrelation")
    ax.legend()

    ax = axes[1, 1]
    w = 21
    roll = lambda x: np.sqrt(np.convolve(x**2, np.ones(w) / w, mode="valid") * TRADING_DAYS)
    ax.plot(roll(real), color=BLUE, linewidth=1.2, label=ticker)
    ax.plot(roll(sim), color=ORANGE, linewidth=1.2, label="GBM")
    ax.axhline(sigma, color=INK, linestyle="--", linewidth=1, label=f"Calibrated $\\sigma$ = {sigma:.1%}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(title="Rolling 21-day annualized volatility", xlabel="Trading day", ylabel="Volatility")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG / "3_gbm_failures.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--rate", type=float, default=0.04, help="annual risk-free rate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prices = load_prices(args.ticker, args.start)
    mu, sigma = calibrate_gbm(prices)
    section(f"0. Calibration from {len(prices):,} daily closes of {args.ticker} since {args.start}")
    print(f"  mu = {mu:.2%}   sigma = {sigma:.2%}   last close = {prices[-1]:.2f}")

    S0, T = 100.0, 1.0  # normalize to 100 so the numbers read as percentages
    fig_paths(S0, mu, sigma, T, rng)
    fig_pricing(S0, args.rate, sigma, T, rng)
    fig_failures(prices, args.ticker, mu, sigma, rng)
    print(f"\nFigures written to {FIG}/")


if __name__ == "__main__":
    main()
