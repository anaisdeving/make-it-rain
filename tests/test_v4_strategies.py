"""Unit and zero-lookahead tests for Strategy v4a (Panic Mean-Reversion) and v4b (Momentum Rotation).

Covers:
- v4a: Wilder vs simple RSI(2), every filter, lowest-RSI(2) ranking, vectorised vs day-level
  equivalence, 0.08% slippage + $0.38 commission, 3.0*ATR14 catastrophe stop, gap-through stop,
  Close > SMA5 exit at next Open, 10-trading-day max hold exit at next Open (zero look-ahead),
  whole shares, T+1 settled cash (no GFV), f=5% vs f=10%, truncation invariance, and
  post-cutoff scrambling zero-lookahead test.
- v4b: Point-in-time Top-100 liquidity ranking ($50M+, $15+), SPY regime gate (SPY <= SMA200 or
  sigma_20 > 0.28 -> cash at t+1 Open), Close > SMA50 & Return_63d/RealizedVol_63d top-3
  selection, keeping overlapping positions without churn, honest T+1 rotation (sell t+1 Open,
  buy t+2 Open once settled — zero GFV), -15% trailing catastrophe stop with post-close ratchet
  and overnight gap-through fill, 0.05% slippage + $0.38 commission, whole shares, truncation
  invariance, and post-cutoff scrambling zero-lookahead test.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.data.provider_base import MarketDataProvider
from src.domain.models import BarSeries, ExitReason, RegimeState, UniverseMember
from src.strategy.momentum_rotation_screener import (
    MomentumRotationConfig,
    MomentumSnapshot,
    compute_momentum_features,
    compute_spy_risk_on,
    prepare_momentum_data,
    run_backtest_v4b,
    select_momentum_leaders_day,
)
from src.strategy.panic_reversion_screener import (
    PanicReversionConfig,
    PanicSnapshot,
    compute_panic_features,
    compute_rsi,
    panic_snapshot,
    prepare_panic_data,
    run_backtest_v4a,
    screen_panic_day,
    screen_panic_sessions,
)
from tests.conftest import make_bars, make_regime


# ---------------------------------------------------------------------------
# Strategy v4a unit tests
# ---------------------------------------------------------------------------


def make_panic_snap(**overrides) -> PanicSnapshot:
    defaults = dict(
        symbol="AAA",
        date=date(2024, 5, 1),
        open=102.0,
        high=103.0,
        low=99.0,
        close=100.0,
        raw_close=100.0,
        volume=500_000.0,
        avg_dollar_volume_20=50_000_000.0,
        sma5=106.0,
        sma200=90.0,
        atr14=2.0,
        ret_3d=-0.06,  # -6% <= -2.5 * (2 / 100) = -5%
        rsi2=2.5,
    )
    defaults.update(overrides)
    return PanicSnapshot(**defaults)


def test_rsi2_wilder_vs_simple_formulas():
    # Two down days after an up day: simple RSI(2) hits exact 0.0, Wilder stays > 0.0
    s = pd.Series([100.0, 104.0, 102.0, 100.0, 100.0, 100.0])
    r_simple = compute_rsi(s, window=2, method="simple")
    r_wilder = compute_rsi(s, window=2, method="wilder")

    assert math.isnan(r_simple.iloc[0]) and math.isnan(r_simple.iloc[1])
    assert r_simple.iloc[3] == pytest.approx(0.0)  # two consecutive down days -> 0.0
    assert r_wilder.iloc[3] == pytest.approx(40.0)  # Wilder retains decaying memory: ag=1.0, al=1.5 -> 40.0
    assert r_simple.iloc[5] == pytest.approx(50.0)  # two flat days -> 50.0

    # All-up series -> 100.0
    up = pd.Series([10.0, 11.0, 12.0, 13.0])
    assert compute_rsi(up, window=2, method="wilder").iloc[-1] == pytest.approx(100.0)
    assert compute_rsi(up, window=2, method="simple").iloc[-1] == pytest.approx(100.0)


def test_v4a_screen_filters_and_lowest_rsi2_ranking():
    cfg = PanicReversionConfig()
    reg = make_regime(RegimeState.NORMAL)

    sigs, why = screen_panic_day([make_panic_snap()], reg, cfg)
    assert len(sigs) == 1 and why == {"ok": 1}

    cases = [
        ("price", {"raw_close": 14.99}),
        ("liquidity", {"avg_dollar_volume_20": 24_999_999.0}),
        ("below_sma200", {"close": 90.0, "sma200": 90.0}),
        ("rsi_too_high", {"rsi2": 5.0}),
        ("drop_too_small", {"ret_3d": -0.049, "atr14": 2.0, "close": 100.0}),
    ]
    for expected, ov in cases:
        s, w = screen_panic_day([make_panic_snap(**ov)], reg, cfg)
        assert s == [] and w == {expected: 1}, f"expected {expected}, got {w}"

    # Lowest RSI(2) ranked first (tiebreak ascending symbol)
    snaps = [
        make_panic_snap(symbol="CCC", rsi2=4.2),
        make_panic_snap(symbol="BBB", rsi2=1.1),
        make_panic_snap(symbol="AAA", rsi2=1.1),
        make_panic_snap(symbol="DDD", rsi2=0.4),
    ]
    ranked, _ = screen_panic_day(snaps, reg, cfg)
    assert [x.symbol for x in ranked] == ["DDD", "AAA", "BBB", "CCC"]


def test_v4a_features_truncation_invariant():
    rng = np.random.default_rng(42)
    bars = make_bars(60.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, 350))))
    cfg = PanicReversionConfig()
    full = compute_panic_features(bars, cfg)
    for cut in (205, 240, 300, 349):
        trunc = BarSeries(symbol="TST", frame=bars.frame.iloc[: cut + 1])
        part = compute_panic_features(trunc, cfg)
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)


def _single_symbol_provider(
    df_sym: pd.DataFrame,
    df_spy: pd.DataFrame,
    symbol: str = "AAA",
) -> MarketDataProvider:
    class _OneProv(MarketDataProvider):
        name = "test_one"
        is_synthetic = True

        def get_universe(self, as_of=None):
            return [UniverseMember(symbol=symbol, source="test")]

        def get_daily_bars(self, symbols, start, end):
            return {symbol: BarSeries(symbol=symbol, frame=df_sym)} if symbol in symbols else {}

        def get_benchmark_bars(self, start, end, symbol_b="SPY"):
            return BarSeries(symbol=symbol_b, frame=df_spy)

        def get_earnings_calendar(self, symbols, start, end):
            return []

        def get_shares_outstanding(self, symbols, start, end):
            return {}

    return _OneProv()


def _build_uptrend_with_panic(n_pre: int = 220, n_post: int = 20, drop_pct: float = 0.115):
    """Build deterministic bars with a guaranteed v4a panic signal strictly at index n_pre."""
    n = n_pre + n_post
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = np.linspace(50.0, 110.0, n)
    anchor = float(close[n_pre - 3])
    # Gentle 2-day pullback on n_pre-2, n_pre-1 (above threshold), then sharp panic drop on n_pre
    close[n_pre - 2] = anchor * 0.995
    close[n_pre - 1] = anchor * 0.985
    close[n_pre] = anchor * (1.0 - drop_pct)
    # Hold flat post-signal by default so tests can inject exits/gaps
    close[n_pre + 1 :] = close[n_pre]
    open_ = close.copy()
    high = close * 1.006
    low = close * 0.994
    vol = np.full(n, 1_000_000.0)  # ~$100M dollar volume >= $25M
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "raw_close": close},
        index=idx,
    )
    spy_c = np.linspace(400.0, 480.0, n)
    spy = pd.DataFrame(
        {
            "open": spy_c,
            "high": spy_c * 1.004,
            "low": spy_c * 0.996,
            "close": spy_c,
            "volume": np.full(n, 5e7),
            "raw_close": spy_c,
        },
        index=idx,
    )
    return idx, df, spy


def test_v4a_sma5_exit_gap_stop_time_stop_and_t1_settlement():
    s = load_settings()
    cfg = PanicReversionConfig(risk_per_trade=0.05)

    # 1) Close > SMA5 exit -> executes at next Open (zero look-ahead) with 0.08% slip + $0.38 comm
    idx, df, spy = _build_uptrend_with_panic(220, 20)
    entry_open_expected = float(df.iloc[221]["open"])
    # Day 221 (t+1): enter at Open; close stays <= SMA5
    # Day 222 (t+2): bounce close = 112.0 > SMA5 -> queues SIGNAL_EXIT for day 223 Open
    # Day 223 (t+3): open = 113.0 -> fills at 113.0 * (1 - 0.0008)
    df.iloc[222, df.columns.get_loc("close")] = 112.0
    df.iloc[222, df.columns.get_loc("raw_close")] = 112.0
    df.iloc[222, df.columns.get_loc("high")] = 112.5
    df.iloc[223, df.columns.get_loc("open")] = 113.0
    df.iloc[223, df.columns.get_loc("high")] = 113.5
    df.iloc[223, df.columns.get_loc("close")] = 113.0
    df.iloc[223, df.columns.get_loc("raw_close")] = 113.0

    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": idx[215].date(), "end_date": idx[-1].date()})}
    )
    data = prepare_panic_data(s_win, _single_symbol_provider(df, spy), cfg)
    res = run_backtest_v4a(data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.signal_date == idx[220].date()
    assert t.entry_date == idx[221].date()
    assert t.exit_date == idx[223].date()
    assert t.exit_reason is ExitReason.SIGNAL_EXIT
    assert t.entry_price == pytest.approx(entry_open_expected * 1.0008)
    assert t.exit_price == pytest.approx(113.0 * (1.0 - 0.0008))
    assert t.commissions == pytest.approx(0.76)
    assert isinstance(t.shares, int) and t.shares >= 1
    # Verify T+1 settlement: sale on day 223 is unsettled on day 223 Close, settled on day 224
    assert res.equity_curve.loc[idx[223], "unsettled_cash"] > 0
    assert res.equity_curve.loc[idx[224], "unsettled_cash"] == pytest.approx(0.0)

    # 2) Overnight gap-down worse than 3*ATR14 stop -> GAP_STOP at Open * (1 - 0.0008)
    idx2, df2, spy2 = _build_uptrend_with_panic(220, 20)
    # Day 222 opens at 70.0 (far below stop)
    df2.iloc[222, df2.columns.get_loc("open")] = 70.0
    df2.iloc[222, df2.columns.get_loc("high")] = 71.0
    df2.iloc[222, df2.columns.get_loc("low")] = 69.0
    df2.iloc[222, df2.columns.get_loc("close")] = 70.5
    df2.iloc[222, df2.columns.get_loc("raw_close")] = 70.5
    data2 = prepare_panic_data(s_win, _single_symbol_provider(df2, spy2), cfg)
    res2 = run_backtest_v4a(data2, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)
    assert len(res2.trades) == 1
    t2 = res2.trades[0]
    assert t2.exit_reason is ExitReason.GAP_STOP
    assert t2.exit_date == idx2[222].date()
    assert t2.exit_price == pytest.approx(70.0 * (1.0 - 0.0008))

    # 3) 10-trading-day max hold -> exits at day 11 Open (idx 231) when Close stays <= SMA5
    idx3, df3, spy3 = _build_uptrend_with_panic(220, 20)
    # Gentle drift down from entry open so Close stays below SMA5 without hitting 3*ATR14 stop
    start_p = float(df3.iloc[220]["close"])
    drift = np.linspace(start_p, start_p - 2.0, 15)
    for k, val in enumerate(drift):
        j = 221 + k
        df3.iloc[j, df3.columns.get_loc("open")] = val
        df3.iloc[j, df3.columns.get_loc("high")] = val + 0.1
        df3.iloc[j, df3.columns.get_loc("low")] = val - 0.1
        df3.iloc[j, df3.columns.get_loc("close")] = val
        df3.iloc[j, df3.columns.get_loc("raw_close")] = val
    data3 = prepare_panic_data(s_win, _single_symbol_provider(df3, spy3), cfg)
    res3 = run_backtest_v4a(data3, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)
    t3 = res3.trades[0]
    assert t3.entry_date == idx3[221].date()
    assert t3.bars_held == 10
    assert t3.exit_date == idx3[231].date()  # next Open after 10th bar Close (idx 230)
    assert t3.exit_reason is ExitReason.TIME_STOP


def test_v4a_f5_vs_f10_sizing_and_t1_no_gfv():
    """Verify f=10% allocates more shares than f=5% when risk-bound, and never spends unsettled cash."""
    s = load_settings()
    idx, df, spy = _build_uptrend_with_panic(220, 20, drop_pct=0.18)
    # Make ATR larger (~4.8, so 3*ATR ~ 14.4 > 15% of price) so f=5% is strictly risk-bound
    df["high"] = df["close"] * 1.024
    df["low"] = df["close"] * 0.976
    df.iloc[221:, df.columns.get_loc("high")] = df.iloc[221:]["close"] * 1.005
    df.iloc[221:, df.columns.get_loc("low")] = df.iloc[221:]["close"] * 0.995
    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": idx[215].date(), "end_date": idx[-1].date()})}
    )

    cfg5 = PanicReversionConfig(risk_per_trade=0.05)
    cfg10 = PanicReversionConfig(risk_per_trade=0.10)
    data = prepare_panic_data(s_win, _single_symbol_provider(df, spy), cfg5)
    r5 = run_backtest_v4a(data, s_win, cfg5, initial_capital_usd=5300.0, vps_enabled=False)
    r10 = run_backtest_v4a(data, s_win, cfg10, initial_capital_usd=5300.0, vps_enabled=False)

    assert len(r5.trades) >= 1 and len(r10.trades) >= 1
    assert r10.trades[0].shares > r5.trades[0].shares
    assert (r5.equity_curve["settled_cash"] >= -1e-6).all()
    assert (r10.equity_curve["settled_cash"] >= -1e-6).all()


def test_v4a_does_not_spend_same_day_unsettled_exit_proceeds():
    """Selling a position at t+1 Open leaves proceeds unsettled until t+2; new t+1 Open buy uses settled cash only."""
    s = load_settings()
    idx, df_a, spy = _build_uptrend_with_panic(220, 20)
    # AAA signals on day 220, enters day 221 (~$500 with max_pos=1.0 on $600 account), bounces day 222 -> exits day 223 Open
    df_a.iloc[222, df_a.columns.get_loc("close")] = 112.0
    df_a.iloc[222, df_a.columns.get_loc("raw_close")] = 112.0
    df_a.iloc[222, df_a.columns.get_loc("high")] = 112.5
    df_a.iloc[223, df_a.columns.get_loc("open")] = 112.0
    df_a.iloc[223, df_a.columns.get_loc("close")] = 112.0
    df_a.iloc[223, df_a.columns.get_loc("raw_close")] = 112.0

    # BBB signals on day 222 (wants to enter on day 223 Open — SAME open AAA sells!)
    _, df_b, _ = _build_uptrend_with_panic(222, 18)
    df_b.index = idx

    class _TwoProv(MarketDataProvider):
        name = "test_two"
        is_synthetic = True

        def get_universe(self, as_of=None):
            return [UniverseMember(symbol="AAA", source="test"), UniverseMember(symbol="BBB", source="test")]

        def get_daily_bars(self, symbols, start, end):
            m = {"AAA": BarSeries(symbol="AAA", frame=df_a), "BBB": BarSeries(symbol="BBB", frame=df_b)}
            return {k: m[k] for k in symbols if k in m}

        def get_benchmark_bars(self, start, end, symbol="SPY"):
            return BarSeries(symbol=symbol, frame=spy)

        def get_earnings_calendar(self, symbols, start, end):
            return []

        def get_shares_outstanding(self, symbols, start, end):
            return {}

    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": idx[215].date(), "end_date": idx[-1].date()})}
    )
    # Capital $160, AAA price ~$97 -> buys 1 share, leaving ~$62 settled cash (< BBB ~$97)
    cfg = PanicReversionConfig(risk_per_trade=0.10, max_position_pct_equity=1.0)
    data = prepare_panic_data(s_win, _TwoProv(), cfg)
    res = run_backtest_v4a(data, s_win, cfg, initial_capital_usd=160.0, vps_enabled=False)

    # AAA fills on day 221 and exits on day 223 Open; BBB MUST be skipped on day 223 Open
    # (unsettled proceeds from AAA cannot be used on day 223!) and may only enter on/after day 224 (T+1 settled).
    assert any(
        sk.symbol == "BBB" and sk.attempt_date == idx[223].date() and sk.reason == "insufficient_settled_cash"
        for sk in res.skipped
    )
    bbb_trades = [t for t in res.trades if t.symbol == "BBB"]
    assert all(t.entry_date >= idx[224].date() for t in bbb_trades)
    assert (res.equity_curve["settled_cash"] >= -1e-6).all()


# ---------------------------------------------------------------------------
# Strategy v4b unit tests
# ---------------------------------------------------------------------------


def make_mom_snap(**overrides) -> MomentumSnapshot:
    defaults = dict(
        symbol="AAA",
        date=date(2024, 5, 1),
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        raw_close=101.0,
        volume=1_000_000.0,
        avg_dollar_volume_20=100_000_000.0,
        sma50=95.0,
        ret_63d=0.25,
        realized_vol_63d=0.25,
        risk_adj_mom=1.0,
    )
    defaults.update(overrides)
    return MomentumSnapshot(**defaults)


def test_v4b_point_in_time_top100_liquidity_and_top3_momentum():
    cfg = MomentumRotationConfig(top_liquid_n=5, top_k=3)

    # Risk-off immediately rejects everyone
    cands_off, why_off = select_momentum_leaders_day([make_mom_snap()], risk_on=False, cfg=cfg)
    assert cands_off == [] and why_off == {"regime_risk_off": 1}

    # Build 8 stocks meeting >= $50M & >= $15, plus 2 ineligible
    # Note: S06 has HUGE momentum (9.9) but ranks #6 in liquidity -> MUST be excluded by Top-5 liquidity cap!
    snaps = [
        make_mom_snap(symbol="S01", avg_dollar_volume_20=500e6, risk_adj_mom=1.5),
        make_mom_snap(symbol="S02", avg_dollar_volume_20=400e6, risk_adj_mom=2.8),
        make_mom_snap(symbol="S03", avg_dollar_volume_20=300e6, risk_adj_mom=0.9),
        make_mom_snap(symbol="S04", avg_dollar_volume_20=200e6, risk_adj_mom=3.1),
        make_mom_snap(symbol="S05", avg_dollar_volume_20=150e6, risk_adj_mom=2.2, close=90.0, sma50=95.0),  # below SMA50
        make_mom_snap(symbol="S06", avg_dollar_volume_20=100e6, risk_adj_mom=9.9),  # outside Top-5 liquid!
        make_mom_snap(symbol="S07", avg_dollar_volume_20=49e6, risk_adj_mom=9.9),   # below $50M
        make_mom_snap(symbol="S08", avg_dollar_volume_20=600e6, raw_close=14.5, risk_adj_mom=9.9),  # below $15
    ]
    cands, why = select_momentum_leaders_day(snaps, risk_on=True, cfg=cfg)
    assert [c.symbol for c in cands] == ["S04", "S02", "S01"]
    assert why["price"] == 1
    assert why["liquidity"] == 1
    assert why["outside_top_liquid"] == 1
    assert why["below_sma50"] == 1
    assert why["outside_top_k"] == 1
    assert why["selected"] == 3


def test_v4b_features_and_spy_regime_truncation_invariant():
    rng = np.random.default_rng(19)
    bars = make_bars(80.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.018, 350))))
    spy = make_bars(420.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 350))), symbol="SPY")
    cfg = MomentumRotationConfig()

    full = compute_momentum_features(bars, cfg)
    full_on, full_sma, full_sig = compute_spy_risk_on(spy, cfg)
    for cut in (205, 250, 310, 349):
        trunc = BarSeries(symbol="TST", frame=bars.frame.iloc[: cut + 1])
        part = compute_momentum_features(trunc, cfg)
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)

        spy_t = BarSeries(symbol="SPY", frame=spy.frame.iloc[: cut + 1])
        part_on, part_sma, part_sig = compute_spy_risk_on(spy_t, cfg)
        pd.testing.assert_series_equal(full_sma.iloc[: cut + 1], part_sma)
        pd.testing.assert_series_equal(full_sig.iloc[: cut + 1], part_sig)
        for ts in spy_t.frame.index:
            assert full_on[ts] == part_on[ts]


def test_v4b_t1_rotation_keep_overlap_regime_exit_and_trailing_stop():
    """Verify honest T+1 rotation (sell t+1 Open, buy t+2 Open), keep-overlap, regime cash-out, and -15% trail."""
    s = load_settings()
    n = 260
    idx = pd.bdate_range("2022-01-03", periods=n)

    def mk_df(closes: np.ndarray, highs=None, lows=None, opens=None) -> pd.DataFrame:
        c = np.asarray(closes, dtype=float)
        o = c.copy() if opens is None else np.asarray(opens, dtype=float)
        h = np.maximum(o, c) * 1.005 if highs is None else np.asarray(highs, dtype=float)
        lo = np.minimum(o, c) * 0.995 if lows is None else np.asarray(lows, dtype=float)
        return pd.DataFrame(
            {"open": o, "high": h, "low": lo, "close": c, "volume": np.full(n, 2_000_000.0), "raw_close": c},
            index=idx,
        )

    # SPY steady uptrend (risk-on), then sharp crash on bar 240 -> below SMA200 (risk-off)
    spy_c = np.linspace(400.0, 500.0, n)
    spy_c[240:] = 360.0
    spy_df = mk_df(spy_c)

    # 4 liquid stocks with identical noise pattern so higher return strictly wins risk-adjusted rank:
    # At bar 220 (rebalance 0): A, B, C are top 3; D is lagging.
    # At bar 230 (rebalance 1): A & B stay strong, C stalls, D surges -> keep A & B, sell C at 231 Open, buy D at 232 Open!
    # At bar 235: B gaps down -22% -> GAP_STOP!
    # At bar 240 (rebalance 2): SPY < SMA200 -> REGIME_EXIT remaining positions at 241 Open!
    rng = np.random.default_rng(7)
    noise = np.exp(np.cumsum(rng.normal(0.0, 0.008, n)))
    a_c = 40.0 * np.exp(np.linspace(0.0, 0.90, n)) * noise
    b_c = 40.0 * np.exp(np.linspace(0.0, 0.80, n)) * noise
    c_c = 40.0 * np.exp(np.linspace(0.0, 0.70, n)) * noise
    c_c[221:] = c_c[220] * np.linspace(1.0, 0.96, n - 221)  # stalls after bar 220 -> drops out of top 3 at bar 230
    d_c = 40.0 * np.exp(np.linspace(0.0, 0.25, n)) * noise
    d_c[221:] = d_c[220] * np.exp(np.linspace(0.04, 0.75, n - 221))  # surges after bar 220 -> enters top 3 at bar 230

    # Inject overnight gap-down > 15% on B at bar 235
    b_o = b_c.copy()
    b_h = b_c * 1.005
    b_l = b_c * 0.995
    b_o[235:] = b_c[234] * 0.78
    b_c[235:] = b_c[234] * 0.78
    b_h[235:] = b_c[234] * 0.79
    b_l[235:] = b_c[234] * 0.77

    bars_map = {
        "AAA": BarSeries(symbol="AAA", frame=mk_df(a_c)),
        "BBB": BarSeries(symbol="BBB", frame=mk_df(b_c, highs=b_h, lows=b_l, opens=b_o)),
        "CCC": BarSeries(symbol="CCC", frame=mk_df(c_c)),
        "DDD": BarSeries(symbol="DDD", frame=mk_df(d_c)),
    }

    class _FourProv(MarketDataProvider):
        name = "test_four"
        is_synthetic = True

        def get_universe(self, as_of=None):
            return [UniverseMember(symbol=k, source="test") for k in sorted(bars_map)]

        def get_daily_bars(self, symbols, start, end):
            return {k: bars_map[k] for k in symbols if k in bars_map}

        def get_benchmark_bars(self, start, end, symbol="SPY"):
            return BarSeries(symbol=symbol, frame=spy_df)

        def get_earnings_calendar(self, symbols, start, end):
            return []

        def get_shares_outstanding(self, symbols, start, end):
            return {}

    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": idx[220].date(), "end_date": idx[250].date()})}
    )
    cfg = MomentumRotationConfig(top_liquid_n=100, top_k=3, rebalance_every_days=10)
    data = prepare_momentum_data(s_win, _FourProv(), cfg)
    res = run_backtest_v4b(data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)

    by_sym = {t.symbol: t for t in res.trades}
    # AAA, BBB, CCC entered from cash on bar 221 (t+1 Open after bar 220 rebalance)
    assert by_sym["AAA"].entry_date == idx[221].date()
    assert by_sym["BBB"].entry_date == idx[221].date()
    assert by_sym["CCC"].entry_date == idx[221].date()

    # CCC rotated out on bar 231 Open (t+1 after bar 230 rebalance)
    assert by_sym["CCC"].exit_date == idx[231].date()
    assert by_sym["CCC"].exit_reason is ExitReason.REBALANCE_EXIT
    # AAA and BBB were KEPT across bar 230 rebalance (zero churn!)
    assert res.kept_positions_total >= 2
    # DDD bought on bar 232 Open (t+2 Open once CCC's T+1 cash settled!)
    assert by_sym["DDD"].entry_date == idx[232].date()
    assert by_sym["DDD"].entry_price == pytest.approx(d_c[232] * 1.0005)

    # BBB gapped down -22% on bar 235 -> GAP_STOP at Open * (1 - 0.0005)
    assert by_sym["BBB"].exit_date == idx[235].date()
    assert by_sym["BBB"].exit_reason is ExitReason.GAP_STOP
    assert by_sym["BBB"].exit_price == pytest.approx(b_o[235] * (1.0 - 0.0005))

    # Rebalance at bar 240 is risk-off -> AAA and DDD liquidated at bar 241 Open via REGIME_EXIT
    assert by_sym["AAA"].exit_date == idx[241].date()
    assert by_sym["AAA"].exit_reason is ExitReason.REGIME_EXIT
    assert by_sym["DDD"].exit_date == idx[241].date()
    assert by_sym["DDD"].exit_reason is ExitReason.REGIME_EXIT

    # Whole shares, 0.05% slippage, $0.38/side commission, ZERO Good Faith Violations (settled_cash >= 0)
    assert all(isinstance(t.shares, int) and t.shares >= 1 for t in res.trades)
    assert all(t.commissions == pytest.approx(0.76) for t in res.trades)
    assert (res.equity_curve["settled_cash"] >= -1e-6).all()


def test_v4b_trailing_stop_updates_post_close_no_intraday_lookahead():
    """Verify intraday high on bar tau does NOT ratchet the stop used during bar tau itself."""
    s = load_settings()
    n = 240
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(11)
    c = 60.0 * np.exp(np.linspace(0.0, 0.5, n) + np.cumsum(rng.normal(0.0, 0.005, n)))
    o = c.copy()
    h = c * 1.005
    lo = c * 0.995
    # Entry on bar 221 at ~100. On bar 223, intraday high spikes +30% while low drops only -5% below entry.
    # If intraday high wrongly ratcheted stop intraday (1.30 * 0.85 = 1.105 > 0.95), it would falsely stop out on bar 223!
    # Causal post-close update keeps stop at ~0.85 * prior HH on bar 223, then triggers on bar 224 if low <= 1.105.
    base_entry = float(o[221])
    h[223] = base_entry * 1.30
    lo[223] = base_entry * 0.95
    c[223] = base_entry * 1.15
    o[224] = base_entry * 1.14
    lo[224] = base_entry * 1.05  # <= 1.30 * 0.85 = 1.105 -> TRAIL_STOP on bar 224!
    c[224] = base_entry * 1.06

    df = pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": np.full(n, 2_000_000.0), "raw_close": c},
        index=idx,
    )
    spy_c = np.linspace(400.0, 480.0, n)
    spy_df = pd.DataFrame(
        {"open": spy_c, "high": spy_c * 1.004, "low": spy_c * 0.996, "close": spy_c, "volume": np.full(n, 5e7), "raw_close": spy_c},
        index=idx,
    )

    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": idx[220].date(), "end_date": idx[235].date()})}
    )
    cfg = MomentumRotationConfig(top_liquid_n=100, top_k=1, rebalance_every_days=10)
    data = prepare_momentum_data(s_win, _single_symbol_provider(df, spy_df), cfg)
    res = run_backtest_v4b(data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)

    assert len(res.trades) >= 1
    t0 = res.trades[0]
    assert t0.entry_date == idx[221].date()
    assert t0.exit_date == idx[224].date()  # NOT 223!
    assert t0.exit_reason is ExitReason.TRAIL_STOP
    assert t0.exit_price == pytest.approx(h[223] * 0.85 * (1.0 - 0.0005))


# ---------------------------------------------------------------------------
# Post-cutoff scrambling zero-lookahead tests for BOTH v4a and v4b
# ---------------------------------------------------------------------------


class _MultiV4SyntheticProvider(MarketDataProvider):
    """Multi-symbol synthetic provider guaranteed to trigger both v4a panics and v4b rotations."""

    name = "synthetic_v4"
    is_synthetic = True

    def __init__(self, n_symbols: int = 18, seed: int = 23) -> None:
        self.idx = pd.bdate_range("2022-01-03", periods=420)
        rng = np.random.default_rng(seed)
        self._bars: dict[str, BarSeries] = {}

        spy_c = 400.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.006, len(self.idx))))
        self._spy = self._mk("SPY", spy_c, 0.005, 5e7, rng)

        for i in range(n_symbols):
            sym = f"M{i:02d}"
            drift = 0.0012 + (i % 5) * 0.0003
            c = 40.0 * np.exp(np.cumsum(rng.normal(drift, 0.011, len(self.idx))))
            # Inject periodic 3-day panics inside uptrends for v4a before and after cutoff (330)
            if i < 8:
                for pday in (235 + i * 5, 290 + i * 4, 355 + i * 4):
                    if pday + 6 < len(self.idx):
                        base_p = c[pday - 3]
                        c[pday - 2] = base_p * 0.965
                        c[pday - 1] = base_p * 0.915
                        c[pday] = base_p * 0.855
                        c[pday + 1] = base_p * 0.870
                        c[pday + 2] = base_p * 0.940
                        c[pday + 3 :] *= (base_p * 0.960) / c[pday + 3]
            self._bars[sym] = self._mk(sym, c, 0.008, 2_000_000.0, rng)

    def _mk(self, sym: str, close: np.ndarray, spread: float, vol_base: float, rng) -> BarSeries:
        open_ = np.concatenate([[close[0]], close[:-1]]) * rng.uniform(0.996, 1.004, len(close))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        vol = np.full(len(close), vol_base)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "raw_close": close},
            index=self.idx,
        )
        return BarSeries(symbol=sym, frame=df)

    def get_universe(self, as_of=None):
        return [UniverseMember(symbol=s, source="synthetic_v4") for s in sorted(self._bars)]

    def get_daily_bars(self, symbols, start, end):
        return {s: self._bars[s] for s in symbols if s in self._bars}

    def get_benchmark_bars(self, start, end, symbol="SPY"):
        return self._spy

    def get_earnings_calendar(self, symbols, start, end):
        return []

    def get_shares_outstanding(self, symbols, start, end):
        return {}


class _PerturbedProvider(MarketDataProvider):
    """Scrambles all bars (including SPY) strictly on/after ``cutoff``."""

    def __init__(self, inner: _MultiV4SyntheticProvider, cutoff: pd.Timestamp) -> None:
        self.inner = inner
        self.cutoff = cutoff
        self.name = inner.name
        self.is_synthetic = True

    def _scramble(self, b: BarSeries, seed: int) -> BarSeries:
        df = b.frame.copy()
        m = df.index >= self.cutoff
        rng = np.random.default_rng(seed)
        k = np.exp(rng.normal(0, 0.30, m.sum()))
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


def test_v4a_vectorised_matches_day_screener_and_zero_lookahead():
    s = load_settings()
    inner = _MultiV4SyntheticProvider()
    cutoff = inner.idx[330]
    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": inner.idx[220].date(), "end_date": inner.idx[-1].date()})}
    )
    cfg = PanicReversionConfig()
    base_data = prepare_panic_data(s_win, inner, cfg)
    pert_data = prepare_panic_data(s_win, _PerturbedProvider(inner, cutoff), cfg)

    # 1. Exact equivalence between screen_panic_sessions and day-by-day screen_panic_day
    vec_sigs, vec_rej, _ = screen_panic_sessions(base_data, cfg)
    day_sigs = {}
    day_rej: dict[str, int] = {}
    for ts in base_data.calendar:
        snaps = [
            panic_snapshot(sym, ts, feat.loc[ts])
            for sym, feat in sorted(base_data.features.items())
            if ts in feat.index
        ]
        sigs, why = screen_panic_day(snaps, base_data.regimes[ts], cfg)
        if sigs:
            day_sigs[ts] = sigs
        for k, v in why.items():
            day_rej[k] = day_rej.get(k, 0) + v
    assert len(vec_sigs) >= 5
    assert vec_sigs == day_sigs
    assert vec_rej == day_rej

    # 2. Zero look-ahead post-cutoff scrambling
    base = run_backtest_v4a(base_data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)
    pert = run_backtest_v4a(pert_data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)

    assert len(base.trades) >= 4
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
    assert base.equity_curve["equity"].iloc[-1] != pert.equity_curve["equity"].iloc[-1]


def test_v4b_zero_lookahead_post_cutoff_scrambling():
    s = load_settings()
    inner = _MultiV4SyntheticProvider()
    cutoff = inner.idx[330]
    s_win = s.model_copy(
        update={"backtest": s.backtest.model_copy(update={"start_date": inner.idx[220].date(), "end_date": inner.idx[-1].date()})}
    )
    cfg = MomentumRotationConfig(top_liquid_n=100, top_k=3, rebalance_every_days=10)
    base_data = prepare_momentum_data(s_win, inner, cfg)
    pert_data = prepare_momentum_data(s_win, _PerturbedProvider(inner, cutoff), cfg)

    base = run_backtest_v4b(base_data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)
    pert = run_backtest_v4b(pert_data, s_win, cfg, initial_capital_usd=5300.0, vps_enabled=False)

    assert len(base.trades) >= 4
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
    assert base.equity_curve["equity"].iloc[-1] != pert.equity_curve["equity"].iloc[-1]
