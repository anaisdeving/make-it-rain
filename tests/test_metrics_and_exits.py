from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest.engine import manage_position
from src.backtest.metrics import cagr, deflated_sharpe, max_drawdown, sharpe, trade_stats
from src.domain.models import ClosedTrade, ExitReason, Position, RegimeState
from tests.conftest import make_signal


def _pos(**kw):
    base = dict(symbol="TST", shares=10, entry_date=date(2024, 5, 2), entry_price=50.0, initial_stop=47.0,
                stop_price=47.0, highest_high=50.0, entry_commission=0.38, risk_fraction=0.05,
                regime_at_entry=RegimeState.NORMAL, signal=make_signal())
    base.update(kw)
    return Position(**base)


def _row(h, c, atr=1.0):
    return pd.Series({"open": c, "high": h, "low": c * 0.99, "close": c, "atr14": atr})


def test_breakeven_at_1_5r(settings):
    p, ex = manage_position(_pos(), _row(54.4, 54.0), settings)
    assert p.stop_price == 47.0 and not p.breakeven_armed
    p, ex = manage_position(p, _row(54.5, 54.0), settings)          # +1.5R = 54.5
    assert p.stop_price == 50.0 and p.breakeven_armed and ex is None


def test_trail_at_2r(settings):
    p, _ = manage_position(_pos(), _row(56.0, 55.5, atr=1.0), settings)   # +2R
    assert p.trail_armed and p.stop_price == pytest.approx(56.0 - 2.5)
    p, _ = manage_position(p, _row(55.0, 54.0, atr=1.0), settings)        # stop never lowered
    assert p.stop_price == pytest.approx(53.5)


def test_time_stop_after_45_sessions_below_1r(settings):
    p, ex = manage_position(_pos(bars_held=44), _row(52.0, 51.0), settings)
    assert p.bars_held == 45 and ex is ExitReason.TIME_STOP
    p, ex = manage_position(_pos(bars_held=44, highest_high=56), _row(54.0, 53.5), settings)  # +1.17R
    assert ex is None
    _, ex = manage_position(_pos(bars_held=10), _row(51.0, 50.5), settings)
    assert ex is None


def test_dsr_single_trial_equals_psr():
    r = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 1000))
    sr = r.mean() / r.std(ddof=1)
    sk, ku = stats.skew(r, bias=False), stats.kurtosis(r, fisher=False, bias=False)
    psr = stats.norm.cdf(sr * np.sqrt(len(r) - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2))
    assert deflated_sharpe(r, 252, 1, 0.5) == pytest.approx(psr)
    assert deflated_sharpe(r, 252, 50, 0.5) < psr


def test_drawdown_cagr_sharpe():
    idx = pd.date_range("2020-01-01", periods=3, freq="365D")
    eq = pd.Series([120.0, 60.0, 121.0], index=idx)
    assert max_drawdown(eq, 100.0) == pytest.approx(-0.5)
    assert cagr(pd.Series([100.0, 200.0], index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")]), 100) \
        == pytest.approx(1.0, abs=0.01)
    assert np.isnan(sharpe(pd.Series([0.0, 0.0, 0.0]), 252))


def test_trade_stats():
    def tr(pnl, r):
        return ClosedTrade(symbol="X", signal_date=date(2024, 1, 1), entry_date=date(2024, 1, 2),
                           exit_date=date(2024, 1, 3), shares=1, entry_price=1, exit_price=1, initial_stop=0.5,
                           exit_reason=ExitReason.STOP, bars_held=1, gross_pnl=pnl, commissions=0, net_pnl=pnl,
                           initial_risk_usd=10, r_multiple=r, regime_at_entry=RegimeState.NORMAL,
                           event_timing="BMO", eps_surprise_pct=1.0, event_day_return=0.1)
    s = trade_stats([tr(30, 3.0), tr(-10, -1.0), tr(-10, -1.0)])
    assert s["win_rate"] == pytest.approx(1 / 3)
    assert s["profit_factor"] == pytest.approx(1.5)
    assert s["expectancy_r"] == pytest.approx(1 / 3)
    assert s["avg_win_r"] == 3.0 and s["avg_loss_r"] == -1.0
