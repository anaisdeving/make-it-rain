"""Unit and zero-look-ahead tests for Strategy v7 (VIX Term-Structure Tail-Capture & Rotation).

Covers:
- 3-state causal hysteresis state machine (TAIL_HEDGE, RISK_ON, NEUTRAL) & boundary values
- Strict us_date < eu_date cross-timezone alignment, US/EU holidays, & stray CBOE prints
- Truncation & post-cutoff scrambling zero-look-ahead invariance
- v7c 50% LVO / 50% EUR Cash tail-hedge allocation & exact sleeve P&L conservation
- Strict T+2 settlement (zero Good Faith Violations), T+2 wait tracking on direct
  RISK_ON -> TAIL_HEDGE switches vs immediate deployment from settled NEUTRAL cash,
  and missed short VIX spikes during T+2 settlement
- Whole as-traded shares, split handling (CL2.PA 300:1 as-traded vs post-split),
  exact EUR 1.25 commission + 0.10%/side slippage, and home-exchange holiday guards
- Diagnostic v7b_cash_tail ablation (tail_hedge_weight=0.0)
- VixTermLoader EUR currency enforcement, stale-flatline / reverse-split jump rejection,
  fallback ticker resolution, and Parquet+SQLite caching.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.bar_utils import raw_close_from_splits
from src.data.cross_asset_eu_loader import TickerMetadata
from src.data.vix_term_loader import VixTermLoader
from src.domain.models import BarSeries
from src.strategy.ucits_etf_regime import SizingMode
from src.strategy.vix_term_regime import (
    VixRegimeState,
    VixTermRegimeConfig,
    align_vix_signals_to_eu_calendar,
    compute_sleeve_attribution,
    compute_vix_regime_metrics,
    compute_vix_term_signals,
    run_vix_term_backtest,
)


def _mk_series(
    symbol: str,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    open_: np.ndarray | None = None,
    splits: pd.Series | None = None,
    volume: float = 250_000.0,
) -> BarSeries:
    c = np.asarray(close, dtype=float)
    o = c.copy() if open_ is None else np.asarray(open_, dtype=float)
    h = np.maximum(o, c) * 1.005
    lo = np.minimum(o, c) * 0.995
    s_close = pd.Series(c, index=dates)
    rc = raw_close_from_splits(s_close, splits)
    df = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": np.full(len(dates), volume),
            "raw_close": rc.to_numpy(dtype=float),
        },
        index=dates,
    )
    return BarSeries(symbol=symbol, frame=df)


# ---------------------------------------------------------------------------
# 1. Three-State Hysteresis Regime State Machine
# ---------------------------------------------------------------------------


def test_three_state_hysteresis_transitions():
    """Verify every transition and boundary in the 3-state VIX/QQQ hysteresis machine."""
    idx = pd.bdate_range("2022-01-03", periods=20)
    # sma_window=5 so bars 0..3 are NaN warmup -> NEUTRAL.
    # Bar 4: QQQ=100, SMA=100, ratio=0.90 -> QQQ not > 1.01*SMA -> NEUTRAL
    # Bar 5: QQQ=105 (>1.01*101=102.01), ratio=0.95 (NOT < 0.95) -> stays NEUTRAL
    # Bar 6: QQQ=106, ratio=0.94 (< 0.95) -> enters RISK_ON!
    # Bar 7: QQQ=102 (inside [0.99, 1.01]*SMA), ratio=0.98 (<= 1.00) -> stays RISK_ON (hysteresis!)
    # Bar 8: QQQ=105, ratio=1.00 (not > 1.00) -> stays RISK_ON
    # Bar 9: QQQ=105, ratio=1.01 (> 1.00, <= 1.03) -> exits RISK_ON to NEUTRAL
    # Bar 10: QQQ=108, ratio=0.94 -> re-enters RISK_ON
    # Bar 11: QQQ=108, ratio=1.04 (> 1.03) -> exits RISK_ON directly to TAIL_HEDGE!
    # Bar 12: QQQ=108, ratio=0.97 (>= 0.97) -> stays in TAIL_HEDGE!
    # Bar 13: QQQ=108, ratio=0.96 (< 0.97, >= 0.95) -> exits TAIL_HEDGE to NEUTRAL (not RISK_ON)
    # Bar 14: QQQ=90, ratio=1.05 (> 1.03) -> enters TAIL_HEDGE from NEUTRAL
    # Bar 15: QQQ=120 (>1.01*SMA), ratio=0.94 (< 0.95) -> exits TAIL_HEDGE directly to RISK_ON!
    # Bar 16: QQQ=95 (< 0.99*SMA), ratio=0.90 -> exits RISK_ON to NEUTRAL due to QQQ < 0.99*SMA
    # Bar 17: QQQ=100, ratio=0.90 -> still below 1.01*SMA -> stays NEUTRAL
    # Bar 18: QQQ=100, ratio=1.03 (not > 1.03) -> stays NEUTRAL
    # Bar 19: QQQ=100, ratio=1.031 (> 1.03) -> enters TAIL_HEDGE
    qqq_c = np.array(
        [
            100.0, 100.0, 100.0, 100.0, 100.0,
            105.0, 106.0, 102.0, 105.0, 105.0,
            108.0, 108.0, 108.0, 108.0, 90.0,
            120.0, 95.0, 100.0, 100.0, 100.0,
        ]
    )
    ratios = np.array(
        [
            0.90, 0.90, 0.90, 0.90, 0.90,
            0.95, 0.94, 0.98, 1.00, 1.01,
            0.94, 1.04, 0.97, 0.96, 1.05,
            0.94, 0.90, 0.90, 1.03, 1.031,
        ]
    )
    vix3m_c = np.full(len(idx), 20.0)
    vix_c = vix3m_c * ratios

    cfg = VixTermRegimeConfig(sma_window=5)
    sig = compute_vix_term_signals(
        _mk_series("QQQ", idx, qqq_c),
        _mk_series("^VIX", idx, vix_c),
        _mk_series("^VIX3M", idx, vix3m_c),
        cfg,
    )

    expected = [
        "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL",
        "NEUTRAL", "RISK_ON", "RISK_ON", "RISK_ON", "NEUTRAL",
        "RISK_ON", "TAIL_HEDGE", "TAIL_HEDGE", "NEUTRAL", "TAIL_HEDGE",
        "RISK_ON", "NEUTRAL", "NEUTRAL", "NEUTRAL", "TAIL_HEDGE",
    ]
    assert list(sig["state"]) == expected
    assert bool(sig["switched"].iloc[6]) is True
    assert bool(sig["switched"].iloc[7]) is False
    assert bool(sig["switched"].iloc[11]) is True
    assert bool(sig["switched"].iloc[15]) is True


# ---------------------------------------------------------------------------
# 2. Strict Cross-Timezone Alignment (us_date < eu_date) & Holidays
# ---------------------------------------------------------------------------


def test_strict_us_to_eu_alignment_and_holidays():
    """US close on date d is NEVER actionable on EU date d; stray VIX holiday bars are ignored."""
    us_dates = pd.DatetimeIndex(
        ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-07"]  # 2022-01-06 is US-only holiday
    )
    # Add a stray CBOE ^VIX bar on US holiday 2022-01-06 that is missing from QQQ & ^VIX3M
    vix_dates = us_dates.union(pd.DatetimeIndex(["2022-01-06"]))
    eu_dates = pd.DatetimeIndex(
        ["2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-10"]
    )

    qqq = _mk_series("QQQ", us_dates, np.array([100.0, 100.0, 110.0, 112.0]))
    vix3m = _mk_series("^VIX3M", us_dates, np.array([20.0, 20.0, 20.0, 20.0]))
    vix = _mk_series("^VIX", vix_dates, np.array([18.0, 18.0, 18.0, 35.0, 22.0]))  # stray 35.0 on holiday

    cfg = VixTermRegimeConfig(sma_window=2)
    us_sig = compute_vix_term_signals(qqq, vix, vix3m, cfg)

    # Stray holiday bar 2022-01-06 must NOT appear in US signals
    assert pd.Timestamp("2022-01-06") not in us_sig.index
    assert list(us_sig.index) == list(us_dates)

    aligned = align_vix_signals_to_eu_calendar(us_sig, eu_dates)
    for eu_ts, row in aligned.iterrows():
        assert row["signal_us_date"] < eu_ts

    # EU 2022-01-06 (US holiday) and 2022-01-07 BOTH map strictly to US 2022-01-05 (< eu_date)
    assert aligned.loc[pd.Timestamp("2022-01-06"), "signal_us_date"] == pd.Timestamp("2022-01-05")
    assert aligned.loc[pd.Timestamp("2022-01-07"), "signal_us_date"] == pd.Timestamp("2022-01-05")
    # US 2022-01-07 (ratio=22/20=1.10 -> TAIL_HEDGE) only becomes actionable on Monday 2022-01-10!
    assert aligned.loc[pd.Timestamp("2022-01-07"), "target_state"] == VixRegimeState.RISK_ON.value
    assert aligned.loc[pd.Timestamp("2022-01-10"), "signal_us_date"] == pd.Timestamp("2022-01-07")
    assert aligned.loc[pd.Timestamp("2022-01-10"), "target_state"] == VixRegimeState.TAIL_HEDGE.value


# ---------------------------------------------------------------------------
# 3. Zero Look-Ahead: Truncation & Future-Scrambling Invariance
# ---------------------------------------------------------------------------


def test_zero_lookahead_truncation_and_scrambling_invariance():
    """Mutating US and EU bars after cutoff must not change a single fill or equity value <= cutoff."""
    dates = pd.bdate_range("2021-10-01", periods=140)
    rng = np.random.default_rng(42)
    qqq_c = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.012, len(dates)))
    vix3m_c = np.full(len(dates), 20.0)
    vix_c = 18.0 + 4.0 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    ro_c = 50.0 * np.cumprod(1.0 + rng.normal(0.002, 0.025, len(dates)))
    lvo_c = 2.0 * np.cumprod(1.0 + rng.normal(-0.001, 0.03, len(dates)))

    cutoff = dates[95].date()
    start = dates[25].date()
    end = dates[-1].date()

    cfg = VixTermRegimeConfig(
        name="v7b_test",
        risk_on_symbol="QQQ3.MI",
        tail_hedge_symbol="LVO.MI",
        tail_hedge_weight=1.0,
        sma_window=15,
    )

    res_trunc = run_vix_term_backtest(
        _mk_series("QQQ", dates, qqq_c),
        _mk_series("^VIX", dates, vix_c),
        _mk_series("^VIX3M", dates, vix3m_c),
        _mk_series("QQQ3.MI", dates, ro_c),
        cfg,
        start_date=start,
        end_date=cutoff,
        initial_capital_eur=5000.0,
        tail_hedge_bars=_mk_series("LVO.MI", dates, lvo_c),
    )

    # Wildly scramble all bars strictly AFTER cutoff
    qqq_scr = qqq_c.copy()
    vix_scr = vix_c.copy()
    ro_scr = ro_c.copy()
    lvo_scr = lvo_c.copy()
    mask_after = dates > pd.Timestamp(cutoff)
    qqq_scr[mask_after] *= 0.2
    vix_scr[mask_after] = 90.0
    ro_scr[mask_after] *= 5.0
    lvo_scr[mask_after] *= 10.0

    res_full = run_vix_term_backtest(
        _mk_series("QQQ", dates, qqq_scr),
        _mk_series("^VIX", dates, vix_scr),
        _mk_series("^VIX3M", dates, vix3m_c),
        _mk_series("QQQ3.MI", dates, ro_scr),
        cfg,
        start_date=start,
        end_date=end,
        initial_capital_eur=5000.0,
        tail_hedge_bars=_mk_series("LVO.MI", dates, lvo_scr),
    )

    c1 = res_trunc.equity_curve[["equity", "settled_cash", "unsettled_cash", "market_value"]]
    c2 = res_full.equity_curve.loc[: pd.Timestamp(cutoff), ["equity", "settled_cash", "unsettled_cash", "market_value"]]
    pd.testing.assert_frame_equal(c1, c2, check_exact=False, rtol=1e-12, atol=1e-12)

    fills_full_cut = [f for f in res_full.fills if f.date <= cutoff]
    assert res_trunc.fills == fills_full_cut


# ---------------------------------------------------------------------------
# 4. v7c 50% / 50% Tail-Hedge Allocation & Exact P&L Conservation
# ---------------------------------------------------------------------------


def test_v7c_half_hedge_allocation_and_attribution_conservation():
    """v7c allocates 50% settled EUR Cash to LVO.MI and keeps 50% in EUR Cash; sleeve P&Ls sum exactly."""
    dates = pd.bdate_range("2022-01-03", periods=16)
    # Bars 0..4 warmup (sma_window=4): NEUTRAL
    # Bar 5: TAIL_HEDGE (from settled NEUTRAL cash -> immediate entry on Bar 6 Open!)
    # Bars 6..8: stay TAIL_HEDGE while LVO rises 2.00 -> 2.50
    # Bar 9: drop to NEUTRAL (exit LVO on Bar 10 Open)
    # Bars 10..11: stay NEUTRAL so cash settles at T+2 (Bar 12)
    # Bar 11: RISK_ON (buy QQQ3 on Bar 12 Open with full settled cash!)
    # Bars 12..15: hold QQQ3 as it rises 50 -> 55
    qqq_c = np.array(
        [100, 100, 100, 100, 100, 98, 98, 98, 98, 99, 100, 110, 112, 114, 116, 118], dtype=float
    )
    ratios = np.array(
        [0.98, 0.98, 0.98, 0.98, 0.98, 1.05, 1.04, 1.02, 0.98, 0.96, 0.96, 0.90, 0.90, 0.90, 0.90, 0.90]
    )
    vix3m_c = np.full(len(dates), 20.0)
    vix_c = vix3m_c * ratios
    ro_c = np.array(
        [50, 50, 50, 50, 50, 48, 48, 48, 48, 49, 50, 50, 50, 52, 54, 55], dtype=float
    )
    lvo_c = np.array(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.2, 2.4, 2.5, 2.5, 2.3, 2.1, 2.0, 2.0, 2.0], dtype=float
    )

    cfg = VixTermRegimeConfig(
        name="v7c_3x_half_hedge",
        risk_on_symbol="QQQ3.MI",
        tail_hedge_symbol="LVO.MI",
        tail_hedge_weight=0.50,
        sma_window=4,
        commission_eur=1.25,
        slippage_pct=0.001,
        settlement_days=2,
    )
    res = run_vix_term_backtest(
        _mk_series("QQQ", dates, qqq_c),
        _mk_series("^VIX", dates, vix_c),
        _mk_series("^VIX3M", dates, vix3m_c),
        _mk_series("QQQ3.MI", dates, ro_c),
        cfg,
        start_date=dates[4].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=1000.0,
        tail_hedge_bars=_mk_series("LVO.MI", dates, lvo_c),
    )

    # First fill must be BUY LVO.MI on Bar 6 Open with ~50% of EUR 1,000 (~EUR 500)
    assert len(res.fills) >= 3
    f0 = res.fills[0]
    assert f0.role == "TAIL_HEDGE"
    assert f0.side == "BUY"
    assert f0.date == dates[6].date()
    assert f0.reason == "REGIME_TAIL_HEDGE_50PCT"
    # Budget = 500.0, comm = 1.25 -> available = 498.75, fill_raw = 2.0 * 1.001 = 2.002 -> 249 shares
    assert f0.raw_shares == math.floor((500.0 - 1.25) / (2.0 * 1.001))
    row_entry = res.equity_curve.loc[dates[6]]
    assert 0.48 <= float(row_entry["w_TAIL_HEDGE"]) <= 0.51
    assert 0.49 <= float(row_entry["w_CASH"]) <= 0.52

    # Immediate entry from settled NEUTRAL cash => 0 T+2 delay sessions
    assert len(res.tail_hedge_episodes) == 1
    assert res.tail_hedge_episodes[0].t2_delay_sessions == 0
    assert res.tail_hedge_episodes[0].net_pnl_eur > 100.0  # ~25% gain on ~EUR 500

    # Exact additive P&L conservation across sleeves
    attr = compute_sleeve_attribution(res)
    total_attr_pnl = sum(a.net_pnl_eur for a in attr.values())
    actual_pnl = float(res.equity_curve["equity"].iloc[-1]) - 1000.0
    assert abs(total_attr_pnl - actual_pnl) < 1e-6


# ---------------------------------------------------------------------------
# 5. Strict T+2 Settlement, Zero GFV, & Spike Delay / Missed Episode Tracking
# ---------------------------------------------------------------------------


def test_strict_t2_settlement_and_spike_delay_tracking():
    """Direct RISK_ON -> TAIL_HEDGE must wait 2 sessions for T+2; short spikes can be missed."""
    dates = pd.bdate_range("2022-02-01", periods=20)
    # Bar 3: RISK_ON -> buys QQQ3 on Bar 4 Open
    # Bar 6: sudden backwardation (ratio=1.08) -> sells QQQ3 on Bar 7 Open (proceeds settle Bar 9)
    #        Bar 7 & Bar 8 are T+2 wait days while LVO spikes 2.00 -> 2.40!
    #        Bar 9 Open: T+2 settled -> buys LVO at 2.40!
    # Bar 10: exits TAIL_HEDGE to RISK_ON -> sells LVO on Bar 11 Open (proceeds settle Bar 13)
    # Bar 13: buys QQQ3 on Bar 13 Open
    # Bar 14: 1-day VIX spike (ratio=1.06) -> sells QQQ3 on Bar 15 Open (settles Bar 17)
    # Bar 15: VIX collapses immediately (ratio=0.90) -> TAIL_HEDGE episode ends on Bar 16 BEFORE T+2!
    #        => Episode #2 is logged as missed_due_to_t2=True!
    qqq_c = np.array(
        [
            100, 100, 100, 110, 112,
            114, 108, 106, 107, 108,
            118, 119, 120, 121, 116,
            125, 126, 127, 128, 129,
        ],
        dtype=float,
    )
    ratios = np.array(
        [
            0.98, 0.98, 0.98, 0.90, 0.90,
            0.90, 1.08, 1.07, 1.05, 1.04,
            0.92, 0.90, 0.90, 0.90, 1.06,
            0.90, 0.90, 0.90, 0.90, 0.90,
        ]
    )
    vix3m_c = np.full(len(dates), 20.0)
    vix_c = vix3m_c * ratios
    ro_c = np.full(len(dates), 100.0)
    lvo_c = np.array(
        [
            2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.2, 2.4,
            2.4, 2.3, 2.1, 2.0, 2.0,
            2.3, 2.0, 2.0, 2.0, 2.0,
        ],
        dtype=float,
    )

    cfg = VixTermRegimeConfig(
        name="v7b_t2_test",
        risk_on_symbol="QQQ3.MI",
        tail_hedge_symbol="LVO.MI",
        tail_hedge_weight=1.0,
        sma_window=3,
        settlement_days=2,
    )
    res = run_vix_term_backtest(
        _mk_series("QQQ", dates, qqq_c),
        _mk_series("^VIX", dates, vix_c),
        _mk_series("^VIX3M", dates, vix3m_c),
        _mk_series("QQQ3.MI", dates, ro_c),
        cfg,
        start_date=dates[4].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=1000.0,
        tail_hedge_bars=_mk_series("LVO.MI", dates, lvo_c),
    )

    # Zero Good Faith Violations across every single session
    assert res.min_settled_cash_eur >= -1e-6
    assert (res.equity_curve["settled_cash"] >= -1e-6).all()

    # Episode 1: Bar 7 sell QQQ3, wait Bar 7 & Bar 8 (2 T+2 wait days), buy LVO on Bar 9 Open!
    assert len(res.tail_hedge_episodes) == 2
    ep1 = res.tail_hedge_episodes[0]
    assert ep1.first_actionable_eu_date == dates[7].date()
    assert ep1.entry_date == dates[9].date()
    assert ep1.t2_delay_sessions == 2
    assert ep1.missed_due_to_t2 is False
    # LVO moved from 2.00 on signal day (Bar 7) to 2.40 on fill day (Bar 9) => +20.0% missed during T+2!
    assert ep1.lvo_move_during_t2_pct is not None
    assert abs(ep1.lvo_move_during_t2_pct - 0.20) < 1e-6

    # Episode 2: Bar 15 sell QQQ3, wait Bar 15 (1 T+2 wait day), signal reverses before T+2 settles -> missed!
    ep2 = res.tail_hedge_episodes[1]
    assert ep2.first_actionable_eu_date == dates[15].date()
    assert ep2.entry_date is None
    assert ep2.t2_delay_sessions == 1
    assert ep2.missed_due_to_t2 is True
    assert ep2.exit_reason == "MISSED_T2_SETTLEMENT"

    m = compute_vix_regime_metrics(res)
    assert m["tail_hedge_t2_wait_days"] == 3
    assert m["tail_hedge_episodes"] == 2
    assert m["tail_hedge_episodes_t2_delayed"] == 2
    assert m["tail_hedge_episodes_missed_t2"] == 1


# ---------------------------------------------------------------------------
# 6. Whole Shares, Split Handling (300:1 CL2.PA), & Exact IBKR Friction
# ---------------------------------------------------------------------------


def test_whole_shares_split_as_traded_vs_post_split_and_friction():
    """Pre-split CL2.PA (> EUR 1,000) is uninvestable at EUR 1,000 as-traded until split date."""
    dates = pd.bdate_range("2023-10-02", periods=10)
    # 300:1 split on dates[6]; adjusted close=12.0 => pre-split raw_close=3600.0, post-split=12.0
    splits = pd.Series(0.0, index=dates)
    splits.iloc[6] = 300.0
    cl2 = _mk_series("CL2.PA", dates, np.full(len(dates), 12.0), splits=splits)
    lvo = _mk_series("LVO.MI", dates, np.full(len(dates), 2.0))
    qqq = _mk_series("QQQ", dates, np.array([100, 100, 110, 112, 114, 115, 116, 117, 118, 119], dtype=float))
    vix = _mk_series("^VIX", dates, np.full(len(dates), 16.0))
    vix3m = _mk_series("^VIX3M", dates, np.full(len(dates), 20.0))

    cfg_as = VixTermRegimeConfig(
        name="v7a_as",
        risk_on_symbol="CL2.PA",
        tail_hedge_symbol="LVO.MI",
        sma_window=2,
        sizing_mode=SizingMode.AS_TRADED,
    )
    res_as = run_vix_term_backtest(
        qqq, vix, vix3m, cl2, cfg_as,
        start_date=dates[3].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=1000.0,
        tail_hedge_bars=lvo,
    )
    # Bars 3, 4, 5 are pre-split (raw price ~3600 > 1000) => 3 skipped_zero_shares!
    # Bar 6 is post-split (raw price = 12.0) => buys immediately on Bar 6 Open!
    assert res_as.skipped_zero_shares == 3
    assert len(res_as.fills) == 1
    assert res_as.fills[0].date == dates[6].date()
    expected_shares = math.floor((1000.0 - 1.25) / (12.0 * 1.001))
    assert res_as.fills[0].raw_shares == expected_shares
    assert abs(res_as.total_commissions_eur - 1.25) < 1e-9
    assert abs(res_as.total_slippage_eur - expected_shares * 12.0 * 0.001) < 1e-9

    cfg_ps = cfg_as.model_copy(update={"name": "v7a_ps", "sizing_mode": SizingMode.POST_SPLIT})
    res_ps = run_vix_term_backtest(
        qqq, vix, vix3m, cl2, cfg_ps,
        start_date=dates[3].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=1000.0,
        tail_hedge_bars=lvo,
    )
    assert res_ps.skipped_zero_shares == 0
    assert res_ps.fills[0].date == dates[3].date()


# ---------------------------------------------------------------------------
# 7. Home-Exchange Holiday Guard & Cash-Tail Ablation
# ---------------------------------------------------------------------------


def test_home_exchange_holiday_delays_fill_causally():
    """Fills never execute on a session where the traded EU ticker has no bar on its exchange."""
    dates = pd.bdate_range("2022-04-11", periods=10)
    # CL2.PA (PAR) has a holiday on dates[4]; LVO.MI (MIL) trades on dates[4]
    cl2_dates = dates.drop(dates[4])
    qqq = _mk_series("QQQ", dates, np.array([100, 100, 100, 110, 112, 113, 114, 115, 116, 117], dtype=float))
    vix = _mk_series("^VIX", dates, np.full(len(dates), 16.0))
    vix3m = _mk_series("^VIX3M", dates, np.full(len(dates), 20.0))

    cfg = VixTermRegimeConfig(
        name="v7a_hol",
        risk_on_symbol="CL2.PA",
        tail_hedge_symbol="LVO.MI",
        sma_window=3,
    )
    res = run_vix_term_backtest(
        qqq, vix, vix3m,
        _mk_series("CL2.PA", cl2_dates, np.full(len(cl2_dates), 25.0)),
        cfg,
        start_date=dates[3].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=1000.0,
        tail_hedge_bars=_mk_series("LVO.MI", dates, np.full(len(dates), 2.0)),
    )
    # Signal turns RISK_ON at US close dates[3]; EU dates[4] has no CL2.PA bar -> fills on dates[5]!
    assert len(res.fills) == 1
    assert res.fills[0].date == dates[5].date()


def test_cash_tail_ablation_holds_cash_during_backwardation():
    """v7b_cash_tail (tail_hedge_weight=0.0) sits in EUR Cash during TAIL_HEDGE with zero LVO fills."""
    dates = pd.bdate_range("2022-03-01", periods=10)
    qqq = _mk_series("QQQ", dates, np.array([100, 100, 110, 112, 105, 104, 106, 115, 116, 117], dtype=float))
    ratios = np.array([0.90, 0.90, 0.90, 1.08, 1.06, 1.02, 0.92, 0.90, 0.90, 0.90])
    vix3m = _mk_series("^VIX3M", dates, np.full(len(dates), 20.0))
    vix = _mk_series("^VIX", dates, 20.0 * ratios)
    ro = _mk_series("QQQ3.MI", dates, np.full(len(dates), 50.0))

    cfg = VixTermRegimeConfig(
        name="v7b_cash_tail",
        risk_on_symbol="QQQ3.MI",
        tail_hedge_symbol="LVO.MI",
        tail_hedge_weight=0.0,
        sma_window=2,
    )
    res = run_vix_term_backtest(
        qqq, vix, vix3m, ro, cfg,
        start_date=dates[3].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=1000.0,
    )
    assert all(f.role == "RISK_ON" for f in res.fills)
    assert len(res.tail_hedge_episodes) == 1
    assert res.tail_hedge_episodes[0].exit_reason == "CASH_TAIL_ABLATION"
    assert res.tail_hedge_episodes[0].net_pnl_eur == 0.0


# ---------------------------------------------------------------------------
# 8. VixTermLoader: EUR Guard, Stale-Flatline / Jump Guard, Fallback & Caching
# ---------------------------------------------------------------------------


def test_vix_term_loader_eur_integrity_and_fallback(tmp_path):
    """Loader rejects empty, non-EUR, stale flatlines, and unsplit jumps, then caches clean EUR hit."""
    dates = pd.bdate_range("2020-10-19", "2026-11-20")
    n = len(dates)
    clean_prices = 2.0 + 0.5 * np.sin(np.linspace(0, 20.0, n))

    def _raw_df(prices: np.ndarray, vol: float = 100_000.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": np.full(len(prices), vol),
                "Stock Splits": np.zeros(len(prices)),
            },
            index=dates,
        )

    flat_prices = clean_prices.copy()
    flat_prices[400:460] = 1.50  # 60 consecutive identical closes -> flatline guard must reject!

    jump_prices = clean_prices.copy()
    jump_prices[600:] *= 50.0  # +4900% unsplit single-day jump (like VIXL.MI) -> jump guard must reject!

    meta_map = {
        "LVO.PA": TickerMetadata("LVO.PA", None, None, None),
        "LVX.BR": TickerMetadata("LVX.BR", "USD", "BRU", "USD VIX ETF"),
        "VIXL.DE": TickerMetadata("VIXL.DE", "EUR", "GER", "Flatlined VIX ETP"),
        "VIXL.MI": TickerMetadata("VIXL.MI", "EUR", "MIL", "Broken Reverse-Split VIX ETP"),
        "LVO.MI": TickerMetadata("LVO.MI", "EUR", "MIL", "Amundi S&P 500 VIX Futures Enhanced Roll"),
        "CL2.PA": TickerMetadata("CL2.PA", "EUR", "PAR", "Amundi 2x USA"),
        "QQQ3.MI": TickerMetadata("QQQ3.MI", "EUR", "MIL", "WisdomTree 3x QQQ"),
        "CSPX.AS": TickerMetadata("CSPX.AS", "EUR", "AMS", "iShares S&P 500"),
        "QQQ": TickerMetadata("QQQ", "USD", "NMS", "Invesco QQQ"),
        "^VIX": TickerMetadata("^VIX", "USD", "CXI", "CBOE VIX"),
        "^VIX3M": TickerMetadata("^VIX3M", "USD", "WCB", "CBOE VIX3M"),
    }
    hist_calls: list[str] = []

    def _fake_meta(sym: str) -> TickerMetadata:
        return meta_map[sym]

    def _fake_hist(sym: str, start: date, end: date):
        hist_calls.append(sym)
        if sym == "LVO.PA":
            return None, None
        if sym == "VIXL.DE":
            df = _raw_df(flat_prices, vol=0.0)
            return df, df["Stock Splits"]
        if sym == "VIXL.MI":
            df = _raw_df(jump_prices)
            return df, df["Stock Splits"]
        if sym in ("^VIX", "^VIX3M"):
            df = _raw_df(18.0 + 3.0 * np.sin(np.linspace(0, 15.0, n)), vol=0.0)
            return df, df["Stock Splits"]
        df = _raw_df(clean_prices)
        return df, df["Stock Splits"]

    loader = VixTermLoader(tmp_path, meta_fetcher=_fake_meta, history_fetcher=_fake_hist)
    ds = loader.load_dataset(
        tail_hedge_candidates=("LVO.PA", "LVX.BR", "VIXL.DE", "VIXL.MI", "LVO.MI"),
    )

    assert ds.tail_hedge_symbol == "LVO.MI"
    audit = ds.audits["tail_hedge"]
    assert audit.currency == "EUR"
    assert audit.synthetic is False
    assert set(audit.rejected_candidates.keys()) == {"LVO.PA", "LVX.BR", "VIXL.DE", "VIXL.MI"}
    assert "non-EUR" in audit.rejected_candidates["LVX.BR"]
    assert "flatline" in audit.rejected_candidates["VIXL.DE"]
    assert "jump" in audit.rejected_candidates["VIXL.MI"]

    # Second load must hit Parquet+SQLite cache with ZERO additional history_fetcher calls!
    n_calls_first = len(hist_calls)
    ds2 = loader.load_dataset(
        tail_hedge_candidates=("LVO.PA", "LVX.BR", "VIXL.DE", "VIXL.MI", "LVO.MI"),
    )
    assert ds2.tail_hedge_symbol == "LVO.MI"
    assert len(hist_calls) == n_calls_first
