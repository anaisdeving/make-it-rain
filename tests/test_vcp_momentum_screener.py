"""Unit tests for Strategy v3 (VCP + 52-Week High Momentum Breakout).

Covers:
- Config DI (slippage 0.15%, commission $0.38, +1.5% gap cap, 2.5x ATR14 / 15% cap, exits)
- Stage 2 trend filters (MA stack, rising SMA200, 52w high & low thresholds)
- Strictly cross-sectional RS top-15% ranking across eligible universe only
- VCP contraction (ATR10/ATR50 <= 0.75 at t-1, prior 5-day base [t-5..t-1] <= 8%)
- Breakout trigger (Close > MaxHigh_5d_prior, Vol >= 1.5x AvgVol20, CloseLoc >= 0.60)
- Entry gap cap (+1.5%), stop wiring via untouched risk_manager, and exit management
- Equivalence between vectorised ``screen_all_sessions`` and ``screen_vcp_day``
- Zero look-ahead (feature truncation invariance + post-cutoff scrambling test)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import manage_position
from src.config import load_settings
from src.data.provider_base import MarketDataProvider
from src.data.sample_provider import SampleProvider
from src.domain.models import BarSeries, ExitReason, RegimeState, UniverseMember
from src.execution.backtest_broker import BacktestBroker
from src.risk.risk_manager import build_order_intent
from src.strategy.vcp_momentum_screener import (
    VcpConfig,
    VcpSnapshot,
    apply_v3_settings,
    compute_vcp_features,
    prepare_vcp_data,
    run_backtest_v3,
    screen_all_sessions,
    screen_vcp_day,
    vcp_snapshot,
)
from tests.conftest import make_bars, make_regime


def make_vcp_snap(**overrides) -> VcpSnapshot:
    defaults = dict(
        symbol="AAA",
        date=date(2024, 5, 1),
        open=99.0,
        high=103.0,
        low=98.5,
        close=102.0,
        prev_close=98.0,
        raw_close=102.0,
        volume=600_000.0,
        avg_volume_20=300_000.0,
        avg_dollar_volume_20=30_000_000.0,
        sma50=92.0,
        sma150=84.0,
        sma200=78.0,
        sma200_prev=75.0,
        high_252d=108.0,
        low_252d=60.0,
        rs_composite=0.35,
        atr14=2.2,
        atr_ratio=0.68,
        max_high_5d_prior=100.0,
        min_low_5d_prior=95.0,
        base_range_5d_prior=5.0 / 98.0,  # ~5.1% <= 8%
        volume_ratio=2.0,
        close_location=(102.0 - 98.5) / (103.0 - 98.5),  # 0.778 >= 0.60
    )
    defaults.update(overrides)
    return VcpSnapshot(**defaults)


def test_apply_v3_settings_config_di(settings):
    cfg = VcpConfig()
    eff = apply_v3_settings(settings, cfg)
    assert eff.execution.slippage_per_side_pct == pytest.approx(0.0015)
    assert eff.execution.commission_per_side_usd == pytest.approx(0.38)
    assert eff.execution.entry_limit_cap_pct == pytest.approx(0.015)
    assert eff.execution.breakeven_trigger_r == pytest.approx(1.5)
    assert eff.execution.trail_trigger_r == pytest.approx(2.0)
    assert eff.execution.trail_atr_multiple == pytest.approx(2.5)
    assert eff.execution.time_stop_days == 45
    assert eff.execution.time_stop_min_r == pytest.approx(1.0)
    assert eff.risk.atr_stop_multiple == pytest.approx(2.5)
    assert eff.risk.max_stop_pct == pytest.approx(0.15)
    # Original settings object remains untouched
    assert settings.execution.slippage_per_side_pct == pytest.approx(0.0025)
    assert settings.execution.entry_limit_cap_pct == pytest.approx(0.003)
    assert settings.risk.atr_stop_multiple == pytest.approx(1.5)


def test_stage2_trend_filters_reject_each_violation():
    cfg = VcpConfig()
    reg = make_regime(RegimeState.NORMAL)

    # Baseline valid snapshot passes
    sigs, why = screen_vcp_day([make_vcp_snap()], reg, cfg)
    assert len(sigs) == 1 and why == {"ok": 1}

    cases = [
        ("price", {"raw_close": 9.99}),
        ("price", {"raw_close": 150.01}),
        ("liquidity", {"avg_dollar_volume_20": 9_999_999.0}),
        ("stage2_ma_stack", {"sma50": 80.0, "sma150": 84.0}),
        ("stage2_ma_stack", {"close": 91.0, "sma50": 92.0}),
        ("stage2_sma200_slope", {"sma200": 75.0, "sma200_prev": 75.0}),
        ("stage2_52w_high", {"high_252d": 125.0}),  # 102 < 0.85 * 125 = 106.25
        ("stage2_52w_low", {"low_252d": 80.0}),  # 102 < 1.30 * 80 = 104.0
        ("vcp_atr_ratio", {"atr_ratio": 0.76}),
        ("vcp_base_range", {"base_range_5d_prior": 0.081}),
        ("breakout_price", {"close": 100.0, "max_high_5d_prior": 100.0}),
        ("breakout_volume", {"volume_ratio": 1.49}),
        ("breakout_close_loc", {"close_location": 0.59}),
    ]
    for expected_reason, overrides in cases:
        s, w = screen_vcp_day([make_vcp_snap(**overrides)], reg, cfg)
        assert s == [], f"expected rejection {expected_reason} for {overrides}"
        assert w == {expected_reason: 1}, f"got {w}, expected {expected_reason}"

    # PANIC regime rejects all candidates immediately
    s, w = screen_vcp_day([make_vcp_snap()], make_regime(RegimeState.PANIC), cfg)
    assert s == [] and w == {"regime_panic": 1}


def test_rs_top_15_percent_cross_sectional_eligible_only():
    cfg = VcpConfig()
    reg = make_regime(RegimeState.NORMAL)

    # Build 20 eligible stocks with RS = 0.01 .. 0.20, plus 10 INELIGIBLE stocks with RS = 9.99
    snaps = [
        make_vcp_snap(symbol=f"E{i:02d}", rs_composite=0.01 * i)
        for i in range(1, 21)
    ]
    ineligible = [
        make_vcp_snap(symbol=f"X{i:02d}", raw_close=5.0, rs_composite=9.99)
        for i in range(5)
    ] + [
        make_vcp_snap(symbol=f"Y{i:02d}", avg_dollar_volume_20=1e6, rs_composite=9.99)
        for i in range(5)
    ]

    sigs, why = screen_vcp_day(snaps + ineligible, reg, cfg)
    # Ranks 17, 18, 19, 20 out of 20 have pct = 0.85, 0.90, 0.95, 1.00 >= 0.85
    assert [s.symbol for s in sigs] == ["E20", "E19", "E18", "E17"]
    assert why["price"] == 5
    assert why["liquidity"] == 5
    assert why["rs_below_top15"] == 16
    assert why["ok"] == 4


def test_vcp_feature_bar_indices_prior_base_and_atr_lag(settings):
    """Verify 5-day base [t-5..t-1] and ATR10/ATR50 at t-1 are unaffected by day-t breakout bar."""
    cfg = VcpConfig()
    n = 300
    idx = pd.bdate_range("2022-01-03", periods=n)
    # Build steady Stage 2 uptrend with wider earlier volatility then tight last 10 days before t
    close = np.linspace(50.0, 100.0, n)
    # Keep bars [n-11 .. n-2] (the 10 bars ending at t-1) very tight around 98.0-99.5
    close[n - 11 : n - 1] = 99.0
    high = close * 1.04  # earlier bars ~8% daily range
    low = close * 0.96
    # Last 10 bars before breakout (indices n-11 .. n-2): tight ~1% daily range
    high[n - 11 : n - 1] = 99.5
    low[n - 11 : n - 1] = 98.5
    open_ = close.copy()
    vol = np.full(n, 300_000.0)

    # Day t (last bar, index n-1): strong expanding breakout bar above 99.5 on 2x volume!
    open_[n - 1] = 99.4
    low[n - 1] = 99.2
    high[n - 1] = 105.0
    close[n - 1] = 104.5
    vol[n - 1] = 750_000.0

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "raw_close": close},
        index=idx,
    )
    bars = BarSeries(symbol="VCP", frame=df)
    feat = compute_vcp_features(bars, cfg, settings.risk)
    last = feat.iloc[-1]

    # 5-day prior base [n-6 .. n-2] has high=99.5, low=98.5, prev_close=99.0 -> range = 1/99 (~1.01%)
    assert last["max_high_5d_prior"] == pytest.approx(99.5)
    assert last["min_low_5d_prior"] == pytest.approx(98.5)
    assert last["base_range_5d_prior"] == pytest.approx(1.0 / 99.0)
    # ATR10/ATR50 at t-1 reflects the tight 10-day window (~1.0 / ~7.5 << 0.75), NOT day-t's 5.8 range
    assert last["atr_ratio"] < 0.30
    assert last["volume_ratio"] == pytest.approx(2.5)
    assert last["close_location"] == pytest.approx((104.5 - 99.2) / (105.0 - 99.2))
    assert last["close"] > last["max_high_5d_prior"]

    # Screening day t passes cleanly
    snap = vcp_snapshot("VCP", idx[-1], last)
    sigs, why = screen_vcp_day([snap], make_regime(RegimeState.NORMAL), cfg)
    assert len(sigs) == 1 and why == {"ok": 1}


def test_stop_wiring_gap_cap_and_exits(settings):
    cfg = VcpConfig()
    eff = apply_v3_settings(settings, cfg)
    reg = make_regime(RegimeState.NORMAL)

    # Even if breakout day low is far below close (e.g. 88 vs 100), pure_atr_stop=True
    # wires event_day_low = limit so untouched risk_manager yields exactly min(2.5*ATR14, 0.15*limit)
    snap = make_vcp_snap(close=100.0, low=88.0, high=101.0, max_high_5d_prior=99.0, atr14=2.0)
    sigs, _ = screen_vcp_day([snap], reg, cfg)
    assert len(sigs) == 1
    sig = sigs[0]

    intent, reason = build_order_intent(sig, reg, 10_000.0, 10_000.0, eff.risk, eff.execution)
    assert reason == "ok" and intent is not None
    assert intent.limit_price == pytest.approx(101.5)  # +1.5% cap
    assert intent.stop_distance == pytest.approx(5.0)  # 2.5 * 2.0 = 5.0 (< 15% of 101.5)
    assert intent.stop_price == pytest.approx(96.5)

    # Cap at 15% when 2.5 * ATR14 > 0.15 * limit
    snap_hi_atr = make_vcp_snap(close=100.0, low=96.0, high=101.0, max_high_5d_prior=99.0, atr14=8.0)
    sig_hi = screen_vcp_day([snap_hi_atr], reg, cfg)[0][0]
    intent_hi, _ = build_order_intent(sig_hi, reg, 10_000.0, 10_000.0, eff.risk, eff.execution)
    assert intent_hi is not None
    assert intent_hi.stop_distance == pytest.approx(0.15 * 101.5)

    # Verify +1.5% gap cap + 0.15% slippage + $0.38 commission in untouched BacktestBroker
    cal = pd.bdate_range("2024-05-01", periods=50)
    broker_skip = BacktestBroker(cal, 10_000.0, eff.execution)
    broker_skip.submit_bracket_order(intent)
    # Gap > +1.5% (101.51 > 101.50) -> skipped as gap_above_limit
    broker_skip.process_session(
        cal[1],
        {"AAA": pd.Series({"open": 101.51, "high": 103.0, "low": 101.0, "close": 102.0})},
    )
    assert len(broker_skip.positions) == 0
    assert len(broker_skip.skipped) == 1 and broker_skip.skipped[0].reason == "gap_above_limit"

    # Gap <= +1.5% (open = 100.5) -> fills at open * 1.0015 with $0.38 commission
    broker_fill = BacktestBroker(cal, 10_000.0, eff.execution)
    broker_fill.submit_bracket_order(intent)
    broker_fill.process_session(
        cal[1],
        {"AAA": pd.Series({"open": 100.5, "high": 101.0, "low": 99.0, "close": 100.8})},
    )
    assert "AAA" in broker_fill.positions
    pos = broker_fill.positions["AAA"]
    assert pos.entry_price == pytest.approx(100.5 * 1.0015)
    assert pos.entry_commission == pytest.approx(0.38)

    # Verify breakeven (+1.5R), trailing (+2.0R -> HH - 2.5*ATR14), and 45d time stop (< +1.0R)
    r = pos.r_distance
    p_be, ex = manage_position(
        pos, pd.Series({"high": pos.entry_price + 1.5 * r, "close": pos.entry_price + 1.2 * r, "atr14": 2.0}), eff
    )
    assert p_be.breakeven_armed and p_be.stop_price == pytest.approx(pos.entry_price) and ex is None

    hh_2r = pos.entry_price + 2.5 * r
    p_tr, ex = manage_position(
        p_be, pd.Series({"high": hh_2r, "close": hh_2r - 0.5, "atr14": 2.0}), eff
    )
    assert p_tr.trail_armed and p_tr.stop_price == pytest.approx(max(pos.entry_price, hh_2r - 2.5 * 2.0)) and ex is None

    # Time stop after 45 bars with close < +1.0R
    p_old = pos.evolve(bars_held=44)
    _, ex_time = manage_position(
        p_old, pd.Series({"high": pos.entry_price + 0.5 * r, "close": pos.entry_price + 0.5 * r, "atr14": 2.0}), eff
    )
    assert ex_time is ExitReason.TIME_STOP


def test_vcp_features_truncation_invariant(settings):
    rng = np.random.default_rng(7)
    bars = make_bars(50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 400))))
    cfg = VcpConfig()
    full = compute_vcp_features(bars, cfg, settings.risk)
    for cut in (252, 275, 320, 399):
        trunc = BarSeries(symbol="TST", frame=bars.frame.iloc[: cut + 1])
        part = compute_vcp_features(trunc, cfg, settings.risk)
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)


class _VcpSyntheticProvider(MarketDataProvider):
    """Synthetic multi-symbol provider guaranteed to generate deterministic VCP breakouts."""

    name = "synthetic_vcp"
    is_synthetic = True

    def __init__(self, n_symbols: int = 25, seed: int = 13) -> None:
        self.idx = pd.bdate_range("2022-01-03", periods=460)
        rng = np.random.default_rng(seed)
        self._bars: dict[str, BarSeries] = {}

        # Bullish SPY so regime is NORMAL across the test window
        spy_close = 400.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.006, len(self.idx))))
        self._spy = self._mk("SPY", spy_close, 0.005, 5e7, rng)

        for i in range(n_symbols):
            sym = f"V{i:02d}"
            # Top 5 symbols have strong Stage 2 uptrends + periodic tight bases + breakouts
            drift = 0.0018 if i < 6 else 0.0002
            close = 30.0 * np.exp(np.cumsum(rng.normal(drift, 0.014, len(self.idx))))
            high = close * rng.uniform(1.015, 1.035, len(self.idx))
            low = close * rng.uniform(0.965, 0.985, len(self.idx))
            open_ = close * rng.uniform(0.99, 1.01, len(self.idx))
            high = np.maximum(high, np.maximum(open_, close))
            low = np.minimum(low, np.minimum(open_, close))
            vol = np.full(len(self.idx), 400_000.0)

            # Inject explicit clean VCP tight bases + volume breakouts on leaders (both before & after cutoff)
            if i < 6:
                for bday in (270 + i * 7, 320 + i * 6, 390 + i * 5):
                    if bday + 5 < len(self.idx):
                        anchor = float(close[bday - 11])
                        # 10-day tight base [bday-10 .. bday-1]
                        close[bday - 10 : bday] = anchor
                        open_[bday - 10 : bday] = anchor
                        high[bday - 10 : bday] = anchor * 1.005
                        low[bday - 10 : bday] = anchor * 0.995
                        vol[bday - 10 : bday] = 350_000.0
                        # Breakout on bday
                        open_[bday] = anchor * 1.002
                        low[bday] = anchor * 1.000
                        high[bday] = anchor * 1.035
                        close[bday] = anchor * 1.032
                        vol[bday] = 900_000.0
                        # Post-breakout continuation from new price level
                        scale = close[bday] / close[bday + 1] if bday + 1 < len(self.idx) else 1.0
                        close[bday + 1 :] *= scale
                        open_[bday + 1 :] *= scale
                        high[bday + 1 :] *= scale
                        low[bday + 1 :] *= scale
                        # Ensure t+1 Open stays within +1.5% cap
                        open_[bday + 1] = close[bday] * 1.003
                        high[bday + 1] = max(high[bday + 1], open_[bday + 1], close[bday + 1])
                        low[bday + 1] = min(low[bday + 1], open_[bday + 1], close[bday + 1])

            df = pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "raw_close": close},
                index=self.idx,
            )
            self._bars[sym] = BarSeries(symbol=sym, frame=df)

    def _mk(self, sym: str, close: np.ndarray, spread: float, vol_base: float, rng) -> BarSeries:
        open_ = close * rng.uniform(1 - spread / 2, 1 + spread / 2, len(close))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        vol = np.full(len(close), vol_base)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "raw_close": close},
            index=self.idx,
        )
        return BarSeries(symbol=sym, frame=df)

    def get_universe(self, as_of=None):
        return [UniverseMember(symbol=s, source="synthetic_vcp") for s in sorted(self._bars)]

    def get_daily_bars(self, symbols, start, end):
        return {s: self._bars[s] for s in symbols if s in self._bars}

    def get_benchmark_bars(self, start, end, symbol="SPY"):
        return self._spy

    def get_earnings_calendar(self, symbols, start, end):
        return []

    def get_shares_outstanding(self, symbols, start, end):
        return {}


class _PerturbedVcpProvider(MarketDataProvider):
    """Scrambles all bars (including SPY) strictly on/after ``cutoff``."""

    def __init__(self, inner: _VcpSyntheticProvider, cutoff: pd.Timestamp) -> None:
        self.inner = inner
        self.cutoff = cutoff
        self.name = inner.name
        self.is_synthetic = True

    def _scramble(self, b: BarSeries, seed: int) -> BarSeries:
        df = b.frame.copy()
        m = df.index >= self.cutoff
        rng = np.random.default_rng(seed)
        k = np.exp(rng.normal(0, 0.35, m.sum()))
        for c in ("open", "high", "low", "close", "raw_close"):
            df.loc[m, c] = df.loc[m, c] * k
        df.loc[m, "volume"] = df.loc[m, "volume"] * rng.uniform(0.1, 10.0, m.sum())
        return BarSeries(symbol=b.symbol, frame=df)

    def get_universe(self, as_of=None):
        return self.inner.get_universe(as_of)

    def get_daily_bars(self, symbols, start, end):
        return {
            s: self._scramble(b, i)
            for i, (s, b) in enumerate(self.inner.get_daily_bars(symbols, start, end).items())
        }

    def get_benchmark_bars(self, start, end, symbol="SPY"):
        return self._scramble(self.inner.get_benchmark_bars(start, end, symbol), 999)

    def get_earnings_calendar(self, symbols, start, end):
        return []

    def get_shares_outstanding(self, symbols, start, end):
        return {}


def test_screen_all_sessions_matches_screen_vcp_day_exactly():
    s = load_settings()
    prov = _VcpSyntheticProvider()
    s = s.model_copy(
        update={
            "backtest": s.backtest.model_copy(
                update={"start_date": prov.idx[260].date(), "end_date": prov.idx[-1].date()}
            )
        }
    )
    cfg = VcpConfig()
    data = prepare_vcp_data(s, prov, cfg)

    vec_sigs, vec_rej, _ = screen_all_sessions(data, cfg)

    # Run day-by-day screen_vcp_day and compare exact signals and rejection counts
    day_sigs = {}
    day_rej: dict[str, int] = {}
    for ts in data.calendar:
        snaps = [
            vcp_snapshot(sym, ts, feat.loc[ts])
            for sym, feat in sorted(data.features.items())
            if ts in feat.index
        ]
        sigs, why = screen_vcp_day(snaps, data.regimes[ts], cfg)
        if sigs:
            day_sigs[ts] = sigs
        for k, v in why.items():
            day_rej[k] = day_rej.get(k, 0) + v

    assert len(vec_sigs) > 0
    assert vec_sigs == day_sigs
    assert vec_rej == day_rej


def test_v3_zero_lookahead_post_cutoff_scrambling():
    s = load_settings()
    inner = _VcpSyntheticProvider()
    cutoff = inner.idx[360]
    s = s.model_copy(
        update={
            "backtest": s.backtest.model_copy(
                update={"start_date": inner.idx[260].date(), "end_date": inner.idx[-1].date()}
            )
        }
    )
    cfg = VcpConfig()
    base_data = prepare_vcp_data(s, inner, cfg)
    pert_data = prepare_vcp_data(s, _PerturbedVcpProvider(inner, cutoff), cfg)

    base = run_backtest_v3(base_data, s, cfg, initial_capital_usd=5300.0, vps_enabled=False)
    pert = run_backtest_v3(pert_data, s, cfg, initial_capital_usd=5300.0, vps_enabled=False)

    assert len(base.trades) >= 3
    for t in base.trades:
        assert t.entry_date > t.signal_date

    pre = base.equity_curve.index < cutoff
    pd.testing.assert_frame_equal(
        base.equity_curve[pre].drop(columns="benchmark_close"),
        pert.equity_curve[pre].drop(columns="benchmark_close"),
    )
    closed_before = lambda r: [t for t in r.trades if pd.Timestamp(t.exit_date) < cutoff]  # noqa: E731
    assert closed_before(base) == closed_before(pert)
    entered_before = lambda r: sorted(  # noqa: E731
        (t.entry_date, t.symbol, t.shares, t.entry_price)
        for t in r.trades
        if pd.Timestamp(t.entry_date) < cutoff
    )
    assert entered_before(base) == entered_before(pert)
    # Sanity check: post-cutoff scrambling actually altered post-cutoff outcomes
    assert base.equity_curve["equity"].iloc[-1] != pert.equity_curve["equity"].iloc[-1]
