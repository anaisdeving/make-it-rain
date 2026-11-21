import numpy as np
import pandas as pd
import pytest

from src.config import RegimeConfig
from src.domain.models import BarSeries, RegimeState
from src.strategy.pead_screener import evaluate_event
from src.strategy.regime_filter import classify_regime, compute_regimes
from tests.conftest import make_event, make_regime, make_snapshot

CFG = RegimeConfig()


@pytest.mark.parametrize(
    "close,sma,sigma,expected",
    [
        (110, 100, 0.15, RegimeState.NORMAL),
        (110, 100, 0.22, RegimeState.CAUTION),   # sigma >= 0.22 boundary
        (110, 100, 0.40, RegimeState.CAUTION),   # high vol but above SMA: not PANIC
        (100, 100, 0.10, RegimeState.CAUTION),   # SPY <= SMA boundary
        (95, 100, 0.30, RegimeState.CAUTION),
        (95, 100, 0.35, RegimeState.CAUTION),    # PANIC needs sigma > 0.35 strictly
        (95, 100, 0.36, RegimeState.PANIC),
        (float("nan"), 100, 0.1, RegimeState.PANIC),  # no history -> no entries
    ],
)
def test_classification_boundaries(close, sma, sigma, expected):
    assert classify_regime(close, sma, sigma, CFG) is expected


def test_regime_transitions_on_price_path():
    rng = np.random.default_rng(0)
    up = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.004, 260)))
    crash = up[-1] * np.exp(np.cumsum(rng.normal(-0.006, 0.035, 60)))
    calm_low = crash[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.003, 40)))
    recovery = calm_low[-1] * np.exp(np.cumsum(rng.normal(0.006, 0.004, 200)))
    close = np.concatenate([up, crash, calm_low, recovery])
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    df = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
                       "volume": 1e8, "raw_close": close}, index=idx)
    reg = compute_regimes(BarSeries(symbol="SPY", frame=df), CFG)
    states = [reg[t].state for t in idx]
    assert all(s is RegimeState.PANIC for s in states[:199])        # warm-up
    assert states[255] is RegimeState.NORMAL
    assert RegimeState.PANIC in states[270:320]
    assert RegimeState.CAUTION in states[320:400]                     # below SMA, calm
    assert states[-1] is RegimeState.NORMAL
    first_panic = next(i for i in range(260, len(states)) if states[i] is RegimeState.PANIC)
    assert states[first_panic - 1] in (RegimeState.NORMAL, RegimeState.CAUTION)
    for t in idx[200:]:
        r = reg[t]
        assert r.max_positions == {RegimeState.NORMAL: 3, RegimeState.CAUTION: 2, RegimeState.PANIC: 0}[r.state]


def test_regime_uses_only_past_prices():
    close = np.linspace(100, 200, 300)
    idx = pd.bdate_range("2020-01-01", periods=300)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0,
                       "raw_close": close}, index=idx)
    full = compute_regimes(BarSeries(symbol="SPY", frame=df), CFG)
    cut = compute_regimes(BarSeries(symbol="SPY", frame=df.iloc[:250]), CFG)
    for t in idx[:250]:
        a, b = full[t].model_dump(), cut[t].model_dump()
        for k in a:
            if isinstance(a[k], float):
                assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k
            else:
                assert a[k] == b[k], k


def test_regime_gates_min_gap_and_entries(settings):
    ev = make_event()
    snap = make_snapshot(close=53.0, high=53.2, low=51.0, prev_close=50.0)  # +6% day
    assert evaluate_event(ev, snap, make_regime(RegimeState.NORMAL), settings.screener)[1] == "ok"
    assert evaluate_event(ev, snap, make_regime(RegimeState.CAUTION), settings.screener)[1] == "event_gap"
    assert evaluate_event(ev, snap, make_regime(RegimeState.PANIC), settings.screener)[1] == "regime_panic"
