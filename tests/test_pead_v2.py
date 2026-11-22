"""Unit tests for Variant v2 screener, config DI, v2b breakout, and signal diagnostics."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_signal import (
    classify_gap_bucket,
    collect_signal_diagnostics,
    summarize_diagnostics,
)
from src.backtest.engine import PreparedData, prepare_data, run_backtest
from src.config import load_settings
from src.data.provider_base import MarketDataProvider
from src.data.sample_provider import SampleProvider
from src.domain.models import BarSeries, MarketRegime, RegimeState
from src.risk.risk_manager import build_order_intent, stop_distance
from src.strategy.pead_v2_screener import (
    ConsolidationTracker,
    TrendSnapshot,
    V2Config,
    apply_v2_settings,
    compute_benchmark_trend,
    evaluate_event_v2,
    run_backtest_v2,
    screen_v2,
    trend_gate_passes,
)
from tests.conftest import make_bars, make_event, make_regime, make_snapshot


def test_apply_v2_settings_uses_config_di(settings):
    cfg = V2Config(entry_limit_cap_pct=0.02, atr_stop_multiple=2.5)
    eff = apply_v2_settings(settings, cfg)
    assert eff.risk.atr_stop_multiple == pytest.approx(2.5)
    assert eff.risk.base_risk_per_trade == pytest.approx(0.05)
    assert eff.risk.max_stop_pct == pytest.approx(0.15)
    assert eff.execution.entry_limit_cap_pct == pytest.approx(0.02)
    # Original settings untouched
    assert settings.risk.atr_stop_multiple == pytest.approx(1.5)
    assert settings.execution.entry_limit_cap_pct == pytest.approx(0.003)


def test_wider_stop_scales_position_size_via_risk_manager_di(settings):
    sig, why = evaluate_event_v2(
        make_event(),
        make_snapshot(close=55.0, low=53.5, high=55.2, atr14=2.0),
        make_regime(),
        settings.screener,
        V2Config(require_spy_above_sma200=False, require_iwm_above_sma200=False),
    )
    assert why == "ok" and sig is not None

    eff_v1 = apply_v2_settings(settings, V2Config(atr_stop_multiple=1.5, entry_limit_cap_pct=0.02))
    eff_v2 = apply_v2_settings(settings, V2Config(atr_stop_multiple=2.5, entry_limit_cap_pct=0.02))

    # limit = 55 * 1.02 = 56.10; entry - low = 2.60; 1.5*ATR = 3.0; 2.5*ATR = 5.0; 15% cap = 8.415
    intent_v1, r1 = build_order_intent(sig, make_regime(), 10_000.0, 10_000.0, eff_v1.risk, eff_v1.execution)
    intent_v2, r2 = build_order_intent(sig, make_regime(), 10_000.0, 10_000.0, eff_v2.risk, eff_v2.execution)
    assert r1 == "ok" and r2 == "ok" and intent_v1 is not None and intent_v2 is not None
    assert intent_v1.limit_price == pytest.approx(55.0 * 1.02)
    assert intent_v1.stop_distance == pytest.approx(3.0)
    assert intent_v2.stop_distance == pytest.approx(5.0)
    # Risk fraction stays 5%, so wider stop yields fewer shares (floor(500/5)=100 -> capped by 50% notional = 89)
    assert intent_v1.risk_fraction == pytest.approx(0.05)
    assert intent_v2.risk_fraction == pytest.approx(0.05)

    # At $5,000 equity where risk budget ($250) binds before 50% notional ($2,500):
    # v1: floor(250 / 3.0) = 83 shares -> capped at 50% notional floor(2500 / 56.1) = 44
    # Test with equity where neither hits notional cap: atr=2.5 -> d_v1=3.75, d_v2=6.25 -> 15% cap=8.415
    # With equity=5000, risk=250: 250/3.75=66 (notional=44); try lower price or higher stop%
    sig2, _ = evaluate_event_v2(
        make_event(),
        make_snapshot(close=55.0, low=54.0, high=55.2, atr14=3.0),
        make_regime(),
        settings.screener,
        V2Config(require_spy_above_sma200=False, require_iwm_above_sma200=False),
    )
    assert sig2 is not None
    # limit=56.10, 1.5*3=4.5 (<8.415), 2.5*3=7.5 (<8.415), notional cap = 50% -> 2500/56.1 = 44
    # Risk shares: 250/4.5 = 55 (capped at 44); 250/7.5 = 33 (<44 -> risk binds!)
    i1, _ = build_order_intent(sig2, make_regime(), 5000.0, 5000.0, eff_v1.risk, eff_v1.execution)
    i2, _ = build_order_intent(sig2, make_regime(), 5000.0, 5000.0, eff_v2.risk, eff_v2.execution)
    assert i1 is not None and i2 is not None
    assert i2.shares == 33
    assert i2.shares < i1.shares
    assert i2.shares * i2.stop_distance <= 5000.0 * 0.05 + 1e-6


def test_benchmark_trend_is_truncation_invariant():
    rng = np.random.default_rng(42)
    bars = make_bars(200 * np.exp(np.cumsum(rng.normal(0, 0.015, 320))), symbol="IWM")
    full = compute_benchmark_trend(bars, sma_window=200)
    # First 199 rows have NaN SMA -> not bullish
    for ts in bars.frame.index[:199]:
        assert np.isnan(full[ts].sma)
        assert not full[ts].is_bullish
    for cut in (200, 250, 319):
        trunc = BarSeries(symbol="IWM", frame=bars.frame.iloc[: cut + 1])
        part = compute_benchmark_trend(trunc, sma_window=200)
        for ts in trunc.frame.index:
            assert part[ts].close == pytest.approx(full[ts].close)
            if np.isnan(full[ts].sma):
                assert np.isnan(part[ts].sma)
            else:
                assert part[ts].sma == pytest.approx(full[ts].sma)


def test_dual_trend_gate_blocks_spy_or_iwm_below_sma200(settings):
    v2_cfg = V2Config()
    iwm_bull = TrendSnapshot(date=date(2024, 5, 1), close=210.0, sma=200.0)
    iwm_bear = TrendSnapshot(date=date(2024, 5, 1), close=190.0, sma=200.0)
    iwm_nan = TrendSnapshot(date=date(2024, 5, 1), close=210.0, sma=float("nan"))

    # Both bullish -> passes
    sig, why = evaluate_event_v2(make_event(), make_snapshot(), make_regime(), settings.screener, v2_cfg, iwm_bull)
    assert why == "ok" and sig is not None

    # SPY > SMA200 but IWM <= SMA200 -> rejected
    sig, why = evaluate_event_v2(make_event(), make_snapshot(), make_regime(), settings.screener, v2_cfg, iwm_bear)
    assert why == "iwm_below_sma200" and sig is None

    # IWM warmup NaN or missing -> rejected (fail-closed)
    assert trend_gate_passes(make_regime(), iwm_nan, v2_cfg) == (False, "iwm_below_sma200")
    assert trend_gate_passes(make_regime(), None, v2_cfg) == (False, "iwm_below_sma200")

    # CAUTION regime where SPY <= SMA200 (even if IWM > SMA200) -> rejected
    spy_bear_caution = MarketRegime(
        date=date(2024, 5, 1),
        state=RegimeState.CAUTION,
        spy_close=470.0,
        sma200=480.0,
        sigma_20=0.25,
        risk_multiplier=0.5,
        min_event_gap=0.08,
        max_positions=2,
    )
    sig, why = evaluate_event_v2(make_event(), make_snapshot(), spy_bear_caution, settings.screener, v2_cfg, iwm_bull)
    assert why == "spy_below_sma200" and sig is None

    # PANIC still reports regime_panic first
    sig, why = evaluate_event_v2(
        make_event(), make_snapshot(), make_regime(RegimeState.PANIC), settings.screener, v2_cfg, iwm_bull
    )
    assert why == "regime_panic" and sig is None


def test_consolidation_tracker_v2b_lifecycle(settings):
    v2_cfg = V2Config(consolidation_breakout=True, consolidation_min_days=2, consolidation_max_days=5)
    tracker = ConsolidationTracker(v2_cfg)
    iwm_bull = TrendSnapshot(date=date(2024, 5, 1), close=210.0, sma=200.0)
    reg = make_regime()

    # Reaction day t0: high=55.5, low=51.5, close=55.0
    r_why = tracker.register_reaction_day([(make_event("AAA"), make_snapshot(symbol="AAA"))], reg, settings.screener)
    assert r_why == {"ok": 1}
    assert "AAA" in tracker.watches

    # Day t+1: consolidates (close=54.5, between 51.5 and 55.5) -> no signal yet
    sigs, why = tracker.step_existing(
        pd.Timestamp("2024-05-02"),
        {"AAA": make_snapshot(symbol="AAA", date=date(2024, 5, 2), close=54.5, high=55.0, low=53.0)},
        reg,
        iwm_bull,
    )
    assert sigs == [] and why == {}
    assert "AAA" in tracker.watches

    # Day t+2: breaks out above event_day_high (55.5) with close=56.2, atr14=1.8
    sigs, why = tracker.step_existing(
        pd.Timestamp("2024-05-03"),
        {"AAA": make_snapshot(symbol="AAA", date=date(2024, 5, 3), close=56.2, high=56.5, low=54.2, atr14=1.8)},
        reg,
        iwm_bull,
    )
    assert len(sigs) == 1 and why == {"breakout_ok": 1}
    assert sigs[0].signal_date == date(2024, 5, 3)
    assert sigs[0].signal_close == pytest.approx(56.2)
    assert sigs[0].atr14 == pytest.approx(1.8)
    assert sigs[0].event_day_low == pytest.approx(51.5)
    assert "AAA" not in tracker.watches


def test_consolidation_tracker_invalidations_and_expiry(settings):
    v2_cfg = V2Config(consolidation_breakout=True, consolidation_min_days=2, consolidation_max_days=3)
    tracker = ConsolidationTracker(v2_cfg)
    iwm_bull = TrendSnapshot(date=date(2024, 5, 1), close=210.0, sma=200.0)
    reg = make_regime()

    tracker.register_reaction_day(
        [
            (make_event("EARLY"), make_snapshot(symbol="EARLY")),
            (make_event("BROKE"), make_snapshot(symbol="BROKE")),
            (make_event("STALE"), make_snapshot(symbol="STALE")),
        ],
        reg,
        settings.screener,
    )

    # Day 1: EARLY breaks out too early; BROKE closes below event_day_low (51.5); STALE consolidates
    sigs, why = tracker.step_existing(
        pd.Timestamp("2024-05-02"),
        {
            "EARLY": make_snapshot(symbol="EARLY", date=date(2024, 5, 2), close=56.0, high=56.5, low=54.0),
            "BROKE": make_snapshot(symbol="BROKE", date=date(2024, 5, 2), close=50.5, high=54.0, low=50.0),
            "STALE": make_snapshot(symbol="STALE", date=date(2024, 5, 2), close=54.0, high=55.0, low=53.0),
        },
        reg,
        iwm_bull,
    )
    assert sigs == []
    assert why == {"consolidation_early_breakout": 1, "consolidation_broke_low": 1}
    assert list(tracker.watches) == ["STALE"]

    # Day 2 and Day 3 without breakout -> STALE expires on Day 3
    for d in ("2024-05-03", "2024-05-06"):
        sigs, why = tracker.step_existing(
            pd.Timestamp(d),
            {"STALE": make_snapshot(symbol="STALE", date=pd.Timestamp(d).date(), close=54.5, high=55.2, low=53.5)},
            reg,
            iwm_bull,
        )
        assert sigs == []
    assert why == {"consolidation_expired": 1}
    assert tracker.watches == {}


def test_run_backtest_v2_matches_v1_exactly_when_v2_levers_disabled():
    s = load_settings()
    s = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": date(2023, 1, 2), "end_date": date(2025, 6, 30)})}
    )
    prov = SampleProvider(date(2021, 11, 1), date(2025, 6, 30), n_symbols=30, seed=11)
    data = prepare_data(s, prov)
    v1_res = run_backtest(data, s)
    v1_equiv_cfg = V2Config(
        require_spy_above_sma200=False,
        require_iwm_above_sma200=False,
        entry_limit_cap_pct=s.execution.entry_limit_cap_pct,
        atr_stop_multiple=s.risk.atr_stop_multiple,
        consolidation_breakout=False,
    )
    v2_res = run_backtest_v2(data, s, v1_equiv_cfg)
    pd.testing.assert_frame_equal(v1_res.equity_curve, v2_res.equity_curve)
    assert v1_res.trades == v2_res.trades
    assert v1_res.skipped == v2_res.skipped
    assert v1_res.rejections == v2_res.rejections
    assert v1_res.sizing_rejections == v2_res.sizing_rejections


class _PerturbedWithIWM(MarketDataProvider):
    """Scrambles all bars (including SPY and IWM) and earnings on/after ``cutoff``."""

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
        return {
            s: self._scramble(b, i)
            for i, (s, b) in enumerate(self.inner.get_daily_bars(symbols, start, end).items())
        }

    def get_benchmark_bars(self, start, end, symbol="SPY"):
        seed = 999 if symbol == "SPY" else 1001
        return self._scramble(self.inner.get_benchmark_bars(start, end, symbol), seed)

    def get_earnings_calendar(self, symbols, start, end):
        evs = self.inner.get_earnings_calendar(symbols, start, end)
        return [
            e
            if pd.Timestamp(e.announced_at.date()) < self.cutoff
            else e.model_copy(update={"eps_surprise_pct": -e.eps_surprise_pct if e.eps_surprise_pct else 1.0})
            for e in evs
        ]

    def get_shares_outstanding(self, symbols, start, end):
        return self.inner.get_shares_outstanding(symbols, start, end)


@pytest.mark.parametrize("breakout", [False, True])
def test_v2_and_v2b_zero_lookahead(breakout):
    s = load_settings()
    s = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": date(2023, 1, 2), "end_date": date(2025, 6, 30)})}
    )
    inner = SampleProvider(date(2021, 11, 1), date(2025, 6, 30), n_symbols=40, seed=11)
    pert_prov = _PerturbedWithIWM(inner, pd.Timestamp("2024-06-03"))
    cutoff = pd.Timestamp("2024-06-03")
    v2_cfg = V2Config(consolidation_breakout=breakout)

    base_data = prepare_data(s, inner)
    pert_data = prepare_data(s, pert_prov)
    base_iwm = inner.get_benchmark_bars(date(2021, 11, 1), date(2025, 6, 30), "IWM")
    pert_iwm = pert_prov.get_benchmark_bars(date(2021, 11, 1), date(2025, 6, 30), "IWM")

    base = run_backtest_v2(base_data, s, v2_cfg, iwm=base_iwm)
    pert = run_backtest_v2(pert_data, s, v2_cfg, iwm=pert_iwm)

    pre = base.equity_curve.index < cutoff
    pd.testing.assert_frame_equal(
        base.equity_curve[pre].drop(columns="benchmark_close"),
        pert.equity_curve[pre].drop(columns="benchmark_close"),
    )
    closed_before = lambda r: [t for t in r.trades if pd.Timestamp(t.exit_date) < cutoff]  # noqa: E731
    assert closed_before(base) == closed_before(pert)
    entered_before = lambda r: [  # noqa: E731
        (t.symbol, t.entry_date, t.shares, t.entry_price)
        for t in r.trades
        if pd.Timestamp(t.entry_date) < cutoff
    ]
    assert entered_before(base) == entered_before(pert)
    for t in base.trades:
        assert t.entry_date > t.signal_date


def test_diagnose_signal_mae_mfe_and_buckets(settings):
    assert classify_gap_bucket(-0.01) == "<= +0.3%"
    assert classify_gap_bucket(0.003) == "<= +0.3%"
    assert classify_gap_bucket(0.0031) == "(+0.3%, +2.5%]"
    assert classify_gap_bucket(0.025) == "(+0.3%, +2.5%]"
    assert classify_gap_bucket(0.026) == "> +2.5%"

    s = settings.model_copy(
        update={"backtest": settings.backtest.model_copy(update={"start_date": date(2023, 1, 2), "end_date": date(2025, 6, 30)})}
    )
    prov = SampleProvider(date(2021, 11, 1), date(2025, 6, 30), n_symbols=30, seed=11)
    data = prepare_data(s, prov)
    iwm = prov.get_benchmark_bars(date(2021, 11, 1), date(2025, 6, 30), "IWM")
    df = collect_signal_diagnostics(data, s, iwm=iwm)
    assert len(df) > 0
    assert {"ret_5d", "ret_10d", "ret_20d", "ret_45d", "ret_60d", "mae_atr", "mfe_atr"}.issubset(df.columns)
    w = df[df["has_45d_window"] == True]  # noqa: E712
    assert len(w) > 0
    assert (w["mae_atr"] >= -1e-9).all()
    assert (w["mfe_atr"] >= -1e-9).all()

    summary = summarize_diagnostics(df)
    assert summary["total_signals"] == len(df)
    assert summary["by_regime"]["PANIC"]["n"] == 0
    assert set(summary["by_gap_bucket"].keys()) == {"<= +0.3%", "(+0.3%, +2.5%]", "> +2.5%"}
    assert "winners_pos_45d" in summary["mae_mfe_45d"]
    assert "v2_2_5x_atr" in summary["stop_binding"]
