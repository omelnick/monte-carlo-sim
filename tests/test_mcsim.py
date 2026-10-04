import numpy as np
import pytest

from mcsim.data import calibrate_gbm
from mcsim.gbm import simulate_paths, simulate_terminal
from mcsim.pricing import black_scholes, mc_european

S0, K, r, sigma, T = 100.0, 105.0, 0.04, 0.2, 1.0


def test_paths_shape_and_start():
    p = simulate_paths(S0, 0.05, sigma, T, 50, 100, rng=np.random.default_rng(0))
    assert p.shape == (100, 51)
    assert np.all(p[:, 0] == S0)
    assert np.all(p > 0)


def test_terminal_moments_match_lognormal():
    ST = simulate_terminal(S0, 0.05, sigma, T, 400_000, rng=np.random.default_rng(1))
    assert ST.mean() == pytest.approx(S0 * np.exp(0.05 * T), rel=2e-3)
    assert np.log(ST).std() == pytest.approx(sigma * np.sqrt(T), rel=5e-3)


def test_put_call_parity():
    c = black_scholes(S0, K, r, sigma, T, "call")
    p = black_scholes(S0, K, r, sigma, T, "put")
    assert c - p == pytest.approx(S0 - K * np.exp(-r * T))


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_matches_black_scholes(kind, antithetic):
    res = mc_european(S0, K, r, sigma, T, 200_000, kind, antithetic, rng=np.random.default_rng(2))
    assert abs(res.price - black_scholes(S0, K, r, sigma, T, kind)) < 4 * res.std_error


def test_antithetic_reduces_variance():
    rng = np.random.default_rng(3)
    plain = [mc_european(S0, K, r, sigma, T, 2_000, rng=rng).price for _ in range(300)]
    anti = [mc_european(S0, K, r, sigma, T, 2_000, antithetic=True, rng=rng).price for _ in range(300)]
    assert np.var(anti) < np.var(plain)


def test_calibration_recovers_parameters():
    prices = simulate_paths(S0, 0.08, 0.25, 40.0, 40 * 252, 1, rng=np.random.default_rng(4))[0]
    _, sig_hat = calibrate_gbm(prices)
    assert sig_hat == pytest.approx(0.25, rel=0.02)
