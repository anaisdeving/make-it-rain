"""Zero look-ahead guarantees."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import prepare_data, reaction_session, run_backtest
from src.config import load_settings
from src.data.provider_base import MarketDataProvider
from src.data.sample_provider import SampleProvider
from src.domain.models import AnnouncementTiming, BarSeries
from src.strategy.indicators import add_sue, compute_features
from tests.conftest import make_bars, make_event

CAL = pd.bdate_range("2024-04-29", periods=10)  # Mon 29 Apr 2024 ...


@pytest.mark.parametrize(
    "when,timing,expected",
    [
        (datetime(2024, 4, 30, 7, 0), AnnouncementTiming.BMO, "2024-04-30"),
        (datetime(2024, 4, 30, 12, 0), AnnouncementTiming.DMH, "2024-04-30"),
        (datetime(2024, 4, 30, 16, 5), AnnouncementTiming.AMC, "2024-05-01"),
        (datetime(2024, 4, 30, 0, 0), AnnouncementTiming.UNKNOWN, "2024-05-01"),
        (datetime(2024, 5, 4, 7, 0), AnnouncementTiming.BMO, "2024-05-06"),   # Saturday -> Monday
        (datetime(2024, 5, 3, 16, 5), AnnouncementTiming.AMC, "2024-05-06"),  # Friday AMC -> Monday
    ],
)
def test_reaction_session_never_precedes_information(when, timing, expected):
    ev = make_event(when=when, timing=timing)
    assert reaction_session(ev, CAL) == pd.Timestamp(expected)


def test_features_truncation_invariant(settings):
    rng = np.random.default_rng(1)
    bars = make_bars(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    shares = pd.Series([1e7, 1.2e7], index=[bars.frame.index[10], bars.frame.index[300]])
    full = compute_features(bars, settings.screener, settings.risk, shares)
    for cut in (260, 299, 300, 350):
        trunc = BarSeries(symbol="TST", frame=bars.frame.iloc[: cut + 1])
        part = compute_features(trunc, settings.screener, settings.risk, shares[shares.index <= trunc.frame.index[-1]])
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)


def test_market_cap_not_backfilled(settings):
    bars = make_bars(np.full(300, 20.0))
    shares = pd.Series([5e7], index=[bars.frame.index[100]])
    f = compute_features(bars, settings.screener, settings.risk, shares)
    assert f["market_cap"].iloc[:100].isna().all()
    assert f["market_cap"].iloc[100] == pytest.approx(1e9)


def test_sue_uses_only_prior_quarters():
    evs = [make_event(when=datetime(2020 + q // 4, 1 + 3 * (q % 4), 15, 7), eps_surprise=s)
           for q, s in enumerate([1, 2, 3, 4, 5, 50])]
    out = sorted(add_sue(evs, lookback=8, min_quarters=4), key=lambda e: e.announced_at)
    assert all(e.sue is None for e in out[:4])
    past = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    expected = 0.5 / past.std(ddof=1)
    assert out[5].sue == pytest.approx(expected, rel=1e-6)


class _Perturbed(MarketDataProvider):
    """Wraps a provider and scrambles every bar on/after ``cutoff``."""

    def __init__(self, inner: SampleProvider, cutoff: pd.Timestamp) -> None:
        self.inner, self.cutoff = inner, cutoff
        self.name, self.is_synthetic = inner.name, True

    def _scramble(self, b: BarSeries, seed: int) -> BarSeries:
        df = b.frame.copy()
        m = df.index >= self.cutoff
        rng = np.random.default_rng(seed)
        k = np.exp(rng.normal(0, 0.3, m.sum()))
        for c in ("open", "high", "low", "close", "raw_close"):
            df.loc[m, c] = df.loc[m, c] * k
        df.loc[m, "volume"] = df.loc[m, "volume"] * rng.uniform(0.1, 10, m.sum())
        return BarSeries(symbol=b.symbol, frame=df)

    def get_universe(self, as_of=None):
        return self.inner.get_universe(as_of)

    def get_daily_bars(self, symbols, start, end):
        return {s: self._scramble(b, i) for i, (s, b) in enumerate(self.inner.get_daily_bars(symbols, start, end).items())}

    def get_benchmark_bars(self, start, end, symbol="SPY"):
        return self._scramble(self.inner.get_benchmark_bars(start, end, symbol), 999)

    def get_earnings_calendar(self, symbols, start, end):
        evs = self.inner.get_earnings_calendar(symbols, start, end)
        return [e if pd.Timestamp(e.announced_at.date()) < self.cutoff
                else e.model_copy(update={"eps_surprise_pct": -e.eps_surprise_pct if e.eps_surprise_pct else 1.0})
                for e in evs]

    def get_shares_outstanding(self, symbols, start, end):
        return self.inner.get_shares_outstanding(symbols, start, end)


def test_engine_decisions_unchanged_by_future_data():
    s = load_settings()
    s = s.model_copy(update={"backtest": s.backtest.model_copy(update={
        "start_date": date(2023, 1, 2), "end_date": date(2025, 6, 30)})})
    inner = SampleProvider(date(2021, 11, 1), date(2025, 6, 30), n_symbols=40, seed=11)
    cutoff = pd.Timestamp("2024-06-03")
    base = run_backtest(prepare_data(s, inner), s)
    pert = run_backtest(prepare_data(s, _Perturbed(inner, cutoff)), s)
    assert len(base.trades) > 5

    pre = base.equity_curve.index < cutoff
    pd.testing.assert_frame_equal(base.equity_curve[pre].drop(columns="benchmark_close"),
                                  pert.equity_curve[pre].drop(columns="benchmark_close"))
    closed_before = lambda r: [t for t in r.trades if pd.Timestamp(t.exit_date) < cutoff]  # noqa: E731
    assert closed_before(base) == closed_before(pert)
    entered_before = lambda r: [(t.symbol, t.entry_date, t.shares, t.entry_price) for t in r.trades  # noqa: E731
                                if pd.Timestamp(t.entry_date) < cutoff]
    assert entered_before(base) == entered_before(pert)
    assert base.equity_curve["equity"].iloc[-1] != pert.equity_curve["equity"].iloc[-1]


def test_entries_occur_strictly_after_signal_session():
    s = load_settings()
    s = s.model_copy(update={"backtest": s.backtest.model_copy(update={
        "start_date": date(2023, 1, 2), "end_date": date(2025, 6, 30)})})
    res = run_backtest(prepare_data(s, SampleProvider(date(2021, 11, 1), date(2025, 6, 30), 40, 11)), s)
    assert res.trades
    for t in res.trades:
        assert t.entry_date > t.signal_date
