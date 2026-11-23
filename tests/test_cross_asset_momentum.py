"""Unit tests for Strategy v6: Cross-Asset Regime-Filtered Momentum Rotation & EU Loader."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.bar_utils import normalise_ohlcv
from src.data.cross_asset_eu_loader import (
    BucketSpec,
    CrossAssetEULoader,
    TickerMetadata,
)
from src.domain.models import BarSeries
from src.strategy.cross_asset_momentum import (
    CrossAssetConfig,
    challenger_beats_held,
    compute_asset_attribution,
    compute_asset_indicators,
    compute_cross_asset_metrics,
    run_cross_asset_backtest,
    select_target_assets,
)
from src.strategy.ucits_etf_regime import SizingMode


def _make_series(
    symbol: str,
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
    *,
    opens: np.ndarray | None = None,
    split_date: pd.Timestamp | None = None,
    split_ratio: float = 1.0,
) -> BarSeries:
    closes = np.asarray(closes, dtype=float)
    opens_arr = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    raw = closes.copy()
    if split_date is not None and split_ratio != 1.0:
        raw = np.where(dates < split_date, closes * split_ratio, closes)
    df = pd.DataFrame(
        {
            "open": opens_arr,
            "high": np.maximum(opens_arr, closes) * 1.005,
            "low": np.minimum(opens_arr, closes) * 0.995,
            "close": closes,
            "volume": np.full(len(dates), 100_000.0),
            "raw_close": raw,
        },
        index=dates,
    )
    return BarSeries(symbol=symbol, frame=df)


def _raw_yf_frame(dates: pd.DatetimeIndex, closes: np.ndarray, splits: dict[pd.Timestamp, float] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    closes = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(len(dates), 50_000.0),
        },
        index=dates,
    )
    sp = pd.Series(0.0, index=dates, name="Stock Splits")
    if splits:
        for ts, ratio in splits.items():
            if ts in sp.index:
                sp.loc[ts] = ratio
    return df, sp


# ---------------------------------------------------------------------------
# 1. Ticker Resolution, EUR Enforcement & Stale-Flatline Rejection
# ---------------------------------------------------------------------------


def test_loader_rejects_non_eur_and_corrupted_flatline_and_resolves_fallback(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-10-19", "2026-11-20")
    n = len(dates)
    rng = np.random.default_rng(42)
    clean_prices = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))

    # Corrupted series: flatlined for 300 days then jumps +150%
    flat_prices = clean_prices.copy()
    flat_prices[400:700] = 18.85
    flat_prices[700:] = flat_prices[700:] * 2.5

    meta_map = {
        "USD_BAD": TickerMetadata("USD_BAD", "USD", "PCX", "USD ETF"),
        "GBP_BAD": TickerMetadata("GBP_BAD", "GBp", "LSE", "GBp ETF"),
        "FLAT_EUR": TickerMetadata("FLAT_EUR", "EUR", "AMS", "Corrupted Flat EUR"),
        "SHORT_EUR": TickerMetadata("SHORT_EUR", "EUR", "GER", "Too Short EUR"),
        "GOOD_EUR": TickerMetadata("GOOD_EUR", "EUR", "MIL", "Clean EUR ETC"),
    }

    def fake_meta(sym: str) -> TickerMetadata:
        return meta_map[sym]

    def fake_hist(sym: str, start: date, end: date):
        if sym == "FLAT_EUR":
            return _raw_yf_frame(dates, flat_prices)
        if sym == "SHORT_EUR":
            short_dates = dates[-200:]
            return _raw_yf_frame(short_dates, clean_prices[-200:])
        if sym == "GOOD_EUR":
            return _raw_yf_frame(dates, clean_prices, splits={pd.Timestamp("2023-10-10"): 10.0})
        return _raw_yf_frame(dates, clean_prices)

    loader = CrossAssetEULoader(tmp_path, meta_fetcher=fake_meta, history_fetcher=fake_hist)
    spec = BucketSpec(
        bucket="silver",
        label="Silver ETC",
        asset_class="metal",
        vol_cap=0.45,
        candidates=("USD_BAD", "GBP_BAD", "FLAT_EUR", "SHORT_EUR", "GOOD_EUR"),
    )

    uni = loader.resolve_universe((spec,), warmup_start=date(2020, 10, 18), eval_start=date(2021, 11, 22), eval_end=date(2026, 11, 20))
    audit = uni.audits["silver"]
    assert audit.symbol == "GOOD_EUR"
    assert audit.currency == "EUR"
    assert audit.synthetic is False
    assert "USD_BAD" in audit.rejected_candidates and "non-EUR" in audit.rejected_candidates["USD_BAD"]
    assert "GBP_BAD" in audit.rejected_candidates and "non-EUR" in audit.rejected_candidates["GBP_BAD"]
    assert "FLAT_EUR" in audit.rejected_candidates and "flatline" in audit.rejected_candidates["FLAT_EUR"]
    assert "SHORT_EUR" in audit.rejected_candidates and "insufficient 5Y" in audit.rejected_candidates["SHORT_EUR"]
    assert audit.splits_in_5y == {"2023-10-10": 10.0}

    # Re-run with failing fetchers to prove disk cache serves everything without network
    def boom(*_args, **_kwargs):
        raise AssertionError("Should have hit disk cache!")

    cached_loader = CrossAssetEULoader(tmp_path, meta_fetcher=boom, history_fetcher=boom)
    uni2 = cached_loader.resolve_universe((spec,), warmup_start=date(2020, 10, 18), eval_start=date(2021, 11, 22), eval_end=date(2026, 11, 20))
    assert uni2.audits["silver"].symbol == "GOOD_EUR"
    pd.testing.assert_frame_equal(uni.bars["GOOD_EUR"].frame, uni2.bars["GOOD_EUR"].frame, check_freq=False)


# ---------------------------------------------------------------------------
# 2. Trend & Volatility Eligibility Caps + Score Guardrails
# ---------------------------------------------------------------------------


def test_eligibility_trend_and_vol_caps_differ_for_equity_vs_crypto() -> None:
    dates = pd.bdate_range("2023-01-02", periods=150)
    # Create an uptrending series with ~60% annualised volatility:
    # Passes crypto vol_cap (0.85), FAILS equity/metal vol_cap (0.45)!
    steps = np.where(np.arange(len(dates)) % 2 == 0, 0.039, -0.035)
    prices = 100.0 * np.exp(np.cumsum(steps))
    assert prices[-1] > np.mean(prices[-100:])
    assert prices[-1] > np.mean(prices[-50:])

    bs_eq = _make_series("QQQ3.MI", dates, prices)
    bs_btc = _make_series("VBTC.DE", dates, prices)
    cfg = CrossAssetConfig(vol_caps={"QQQ3.MI": 0.45, "VBTC.DE": 0.85})

    ind_eq = compute_asset_indicators(bs_eq, cfg)
    ind_btc = compute_asset_indicators(bs_btc, cfg)

    sig_last = float(ind_eq["sigma_20"].iloc[-1])
    assert 0.45 < sig_last < 0.85
    assert bool(ind_eq["eligible"].iloc[-1]) is False
    assert bool(ind_btc["eligible"].iloc[-1]) is True

    # Flat constant price -> sigma_20 == 0 -> guarded against div-by-zero / NaN -> NOT eligible
    bs_flat = _make_series("FLAT", dates, np.full(len(dates), 100.0))
    ind_flat = compute_asset_indicators(bs_flat, cfg)
    assert math.isnan(float(ind_flat["score"].iloc[-1]))
    assert bool(ind_flat["eligible"].iloc[-1]) is False

    # Close below SMA(50) fails eligibility even with low vol
    down_prices = np.concatenate([np.linspace(100.0, 140.0, 120), np.linspace(140.0, 110.0, 30)])
    bs_down = _make_series("DOWN", dates, down_prices)
    ind_down = compute_asset_indicators(bs_down, cfg)
    assert bool(ind_down["eligible"].iloc[-1]) is False


# ---------------------------------------------------------------------------
# 3. Hysteresis Buffer (Including Non-Positive Score Edge Cases!)
# ---------------------------------------------------------------------------


def test_hysteresis_positive_zero_and_negative_scores() -> None:
    # Positive held score: must strictly exceed 1.15 * held_score
    assert challenger_beats_held(1.149, 1.0, 0.15) is False
    assert challenger_beats_held(1.150, 1.0, 0.15) is False
    assert challenger_beats_held(1.151, 1.0, 0.15) is True

    # Zero held score: any positive challenger beats 0; 0 or negative does not
    assert challenger_beats_held(0.0, 0.0, 0.15) is False
    assert challenger_beats_held(-0.01, 0.0, 0.15) is False
    assert challenger_beats_held(0.001, 0.0, 0.15) is True

    # Negative held score: MUST NOT invert! Worse score (-1.10) must NOT beat -1.00.
    # Threshold is -1.00 + 0.15 * 1.00 = -0.85.
    assert challenger_beats_held(-1.10, -1.0, 0.15) is False
    assert challenger_beats_held(-0.90, -1.0, 0.15) is False
    assert challenger_beats_held(-0.85, -1.0, 0.15) is False
    assert challenger_beats_held(-0.84, -1.0, 0.15) is True
    assert challenger_beats_held(0.05, -1.0, 0.15) is True

    # Target selection with top_n=1 and top_n=2
    # top_n=1: held A (score=1.0), challenger B (1.10) -> keep A
    assert select_target_assets({"A": 1.0, "B": 1.10}, {"A"}, top_n=1, hysteresis_pct=0.15) == ["A"]
    # challenger B (1.20) > 1.15 -> rotate to B
    assert select_target_assets({"A": 1.0, "B": 1.20}, {"A"}, top_n=1, hysteresis_pct=0.15) == ["B"]
    # held A disqualified (not in eligible_scores) -> rotate to B
    assert select_target_assets({"B": 0.2}, {"A"}, top_n=1, hysteresis_pct=0.15) == ["B"]

    # top_n=2: held {A: 1.0, B: 0.8}, challenger C (0.90 <= 1.15*0.8=0.92) -> keep {A, B}
    assert select_target_assets({"A": 1.0, "B": 0.8, "C": 0.90}, {"A", "B"}, top_n=2, hysteresis_pct=0.15) == ["A", "B"]
    # challenger C (0.95 > 0.92) replaces weaker incumbent B -> {A, C}
    assert select_target_assets({"A": 1.0, "B": 0.8, "C": 0.95}, {"A", "B"}, top_n=2, hysteresis_pct=0.15) == ["A", "C"]


# ---------------------------------------------------------------------------
# 4. Intra-Week Catastrophe Stop (Entry DD >= 15% & Peak DD >= 20%)
# ---------------------------------------------------------------------------


def test_catastrophe_stop_entry_drawdown_and_cash_until_weekly_rebalance() -> None:
    # Build 120 warmup days of smooth uptrend + 10 eval days (2 weeks: Mon..Fri, Mon..Fri)
    dates = pd.bdate_range("2024-06-03", periods=135)
    eval_start_idx = 120  # Ensure eval_start is a Monday
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1

    closes = np.linspace(50.0, 100.0, len(dates))
    opens = closes.copy()

    # On eval_start_idx (Mon), enters at open=100.0.
    # On Tue (eval_start_idx + 1), closes at 84.0 (-16% from entry >= 15% trigger!).
    # On Wed (eval_start_idx + 2), opens at 83.0 -> MUST execute stop sell at Wed Open,
    # then stay in EUR Cash on Thu & Fri even if price rebounds to 105!
    closes[eval_start_idx] = 100.0
    opens[eval_start_idx] = 100.0
    closes[eval_start_idx + 1] = 84.0
    opens[eval_start_idx + 1] = 98.0
    opens[eval_start_idx + 2] = 83.0
    closes[eval_start_idx + 2] = 105.0
    opens[eval_start_idx + 3] = 105.0
    closes[eval_start_idx + 3] = 106.0
    opens[eval_start_idx + 4] = 106.0
    closes[eval_start_idx + 4] = 107.0

    bs = _make_series("ASSET_A", dates, closes, opens=opens)
    cfg = CrossAssetConfig(
        name="test_stop_entry",
        top_n=1,
        vol_caps={"ASSET_A": 2.0},  # High vol cap so only catastrophe stop drives exit
        settlement_days=2,
    )

    res = run_cross_asset_backtest(
        {"ASSET_A": bs},
        cfg,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[eval_start_idx + 4].date(),
        initial_capital_eur=1000.0,
    )

    assert res.catastrophe_stops_triggered == 1
    assert len(res.fills) == 2
    assert res.fills[0].side == "BUY" and res.fills[0].date == dates[eval_start_idx].date()
    assert res.fills[1].side == "SELL"
    assert res.fills[1].date == dates[eval_start_idx + 2].date()  # Wed Open (t+1 after Tue Close)
    assert res.fills[1].signal_date == dates[eval_start_idx + 1].date()
    assert res.fills[1].reason.startswith("CATASTROPHE_STOP_ENTRY")
    # Remained flat (EUR Cash) on Wed, Thu, Fri until next weekly rebalance
    assert res.equity_curve.loc[dates[eval_start_idx + 2], "open_positions"] == 0
    assert res.equity_curve.loc[dates[eval_start_idx + 3], "open_positions"] == 0
    assert res.equity_curve.loc[dates[eval_start_idx + 4], "open_positions"] == 0


def test_catastrophe_stop_peak_drawdown_20pct_after_rally() -> None:
    dates = pd.bdate_range("2024-06-03", periods=135)
    eval_start_idx = 120
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1

    closes = np.linspace(50.0, 100.0, len(dates))
    opens = closes.copy()

    # Mon: buy at 100. Tue: rally to 150 (new peak!). Wed: drop to 118 (-21.3% from peak 150,
    # even though still +18% ABOVE entry 100!). Thu Open: MUST exit via CATASTROPHE_STOP_PEAK!
    opens[eval_start_idx] = 100.0
    closes[eval_start_idx] = 102.0
    opens[eval_start_idx + 1] = 105.0
    closes[eval_start_idx + 1] = 150.0
    opens[eval_start_idx + 2] = 145.0
    closes[eval_start_idx + 2] = 118.0  # 118 / 150 - 1 = -21.33% <= -20%
    opens[eval_start_idx + 3] = 117.0
    closes[eval_start_idx + 3] = 116.0
    opens[eval_start_idx + 4] = 116.0
    closes[eval_start_idx + 4] = 115.0

    bs = _make_series("ASSET_A", dates, closes, opens=opens)
    cfg = CrossAssetConfig(
        name="test_stop_peak",
        top_n=1,
        vol_caps={"ASSET_A": 3.0},
        settlement_days=2,
    )

    res = run_cross_asset_backtest(
        {"ASSET_A": bs},
        cfg,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[eval_start_idx + 4].date(),
        initial_capital_eur=1000.0,
    )

    assert res.catastrophe_stops_triggered == 1
    assert len(res.round_trips) == 1
    rt = res.round_trips[0]
    assert rt.exit_reason == "CATASTROPHE_STOP_PEAK"
    assert rt.exit_date == dates[eval_start_idx + 3].date()
    assert rt.peak_close_adj == pytest.approx(150.0)
    assert rt.net_pnl_eur > 0.0  # Profitable trade exited on peak trailing stop


# ---------------------------------------------------------------------------
# 5. Strict T+2 Settlement Rotation (Zero GFV) & Whole Shares
# ---------------------------------------------------------------------------


def test_t2_settlement_rotation_waits_two_sessions_and_zero_gfv() -> None:
    dates = pd.bdate_range("2024-06-03", periods=145)
    eval_start_idx = 120
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1

    # Asset A is in a strong uptrend before Week 1; Asset B is in a mild downtrend (below SMA100)
    # so A is the sole eligible leader at Week 1 Open. During Week 1, A drops below SMA100 while
    # B rallies sharply above SMA100, triggering an A -> B rotation at Week 2 Monday Open!
    n = len(dates)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.002, n)
    close_a = 50.0 * np.exp(np.cumsum(0.005 + noise))
    close_b = np.linspace(100.0, 80.0, n) + rng.normal(0.0, 0.2, n)

    for k in range(5):
        close_a[eval_start_idx + k] = close_a[eval_start_idx - 1] * (0.92 - 0.08 * k)
        close_b[eval_start_idx + k] = 90.0 + 5.0 * k  # Fri close = 110 (> SMA100 ~88)
    close_a[eval_start_idx + 5 :] = close_a[eval_start_idx + 4]
    close_b[eval_start_idx + 5 :] = 112.0

    bs_a = _make_series("ASSET_A", dates, close_a)
    bs_b = _make_series("ASSET_B", dates, close_b)

    cfg = CrossAssetConfig(
        name="test_t2",
        top_n=1,
        vol_caps={"ASSET_A": 1.5, "ASSET_B": 1.5},
        catastrophe_entry_dd=0.90,  # Disable cat stop to test pure rebalance rotation
        catastrophe_peak_dd=0.90,
        settlement_days=2,
        commission_eur=1.25,
        slippage_pct=0.001,
    )

    res = run_cross_asset_backtest(
        {"ASSET_A": bs_a, "ASSET_B": bs_b},
        cfg,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[eval_start_idx + 12].date(),
        initial_capital_eur=1000.0,
    )

    # Expect: BUY A on Week 1 Mon (idx), SELL A on Week 2 Mon (idx+5),
    # wait Mon & Tue (2 settlement_wait_days), BUY B on Week 2 Wed (idx+7)!
    assert len(res.fills) == 3
    assert res.fills[0].symbol == "ASSET_A" and res.fills[0].side == "BUY"
    assert res.fills[0].date == dates[eval_start_idx].date()
    assert res.fills[1].symbol == "ASSET_A" and res.fills[1].side == "SELL"
    assert res.fills[1].date == dates[eval_start_idx + 5].date()
    assert res.fills[2].symbol == "ASSET_B" and res.fills[2].side == "BUY"
    assert res.fills[2].date == dates[eval_start_idx + 7].date()  # Exact T+2!
    assert res.settlement_wait_days == 2
    assert res.min_settled_cash_eur >= -1e-9
    # Whole shares only
    for f in res.fills:
        assert isinstance(f.raw_shares, int) and f.raw_shares >= 1
    # Unsettled cash > 0 on sell day & T+1, zero on T+2
    assert res.equity_curve.loc[dates[eval_start_idx + 5], "unsettled_cash"] > 0
    assert res.equity_curve.loc[dates[eval_start_idx + 6], "unsettled_cash"] > 0
    assert res.equity_curve.loc[dates[eval_start_idx + 7], "unsettled_cash"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Zero Look-Ahead & Truncation / Post-Cutoff Scrambling Invariance
# ---------------------------------------------------------------------------


def test_zero_lookahead_truncation_and_scrambling_invariance() -> None:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2021-01-04", periods=320)
    n = len(dates)

    bars_full: dict[str, BarSeries] = {}
    for sym, drift, vol in [("EQ3", 0.0008, 0.015), ("BTC", 0.0012, 0.028), ("GLD", 0.0004, 0.008)]:
        px = 50.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        bars_full[sym] = _make_series(sym, dates, px)

    start_d = dates[150].date()
    cutoff_d = dates[240].date()
    end_d = dates[-1].date()

    cfg = CrossAssetConfig(
        name="v6b_invariance",
        top_n=2,
        vol_caps={"EQ3": 0.45, "BTC": 0.85, "GLD": 0.45},
    )

    res_full = run_cross_asset_backtest(bars_full, cfg, start_date=start_d, end_date=end_d, initial_capital_eur=5000.0)

    # Case A: Strict truncation at cutoff_d
    bars_trunc = {
        sym: BarSeries(symbol=sym, frame=bs.frame.loc[: pd.Timestamp(cutoff_d)].copy())
        for sym, bs in bars_full.items()
    }
    res_trunc = run_cross_asset_backtest(bars_trunc, cfg, start_date=start_d, end_date=cutoff_d, initial_capital_eur=5000.0)

    # Case B: Wildly scramble every price strictly AFTER cutoff_d
    bars_scram: dict[str, BarSeries] = {}
    for sym, bs in bars_full.items():
        df = bs.frame.copy()
        after = df.index > pd.Timestamp(cutoff_d)
        df.loc[after, ["open", "high", "low", "close", "raw_close"]] *= 9.99
        bars_scram[sym] = BarSeries(symbol=sym, frame=df)
    res_scram = run_cross_asset_backtest(bars_scram, cfg, start_date=start_d, end_date=end_d, initial_capital_eur=5000.0)

    cut_full = res_full.equity_curve.loc[: pd.Timestamp(cutoff_d)]
    cut_scram = res_scram.equity_curve.loc[: pd.Timestamp(cutoff_d)]
    pd.testing.assert_frame_equal(cut_full, res_trunc.equity_curve)
    pd.testing.assert_frame_equal(cut_full, cut_scram)

    fills_full_cut = [f for f in res_full.fills if f.date <= cutoff_d]
    fills_scram_cut = [f for f in res_scram.fills if f.date <= cutoff_d]
    assert fills_full_cut == res_trunc.fills
    assert fills_full_cut == fills_scram_cut


# ---------------------------------------------------------------------------
# 7. v6a_top1 vs v6b_top2 Allocation Rules (1 Eligible = 50% Cash in v6b) & Attribution Identity
# ---------------------------------------------------------------------------


def test_v6b_top2_single_eligible_holds_50pct_cash_and_no_churn_incumbent() -> None:
    dates = pd.bdate_range("2024-06-03", periods=145)
    eval_start_idx = 120
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1
    n = len(dates)

    # Asset A is smoothly uptrending (always eligible)
    # Asset B is downtrending in Week 1 (ineligible), then flat/down
    close_a = np.linspace(20.0, 40.0, n)
    close_b = np.linspace(100.0, 60.0, n)

    bs_a = _make_series("ASSET_A", dates, close_a)
    bs_b = _make_series("ASSET_B", dates, close_b)

    cfg_top1 = CrossAssetConfig(name="v6a_top1", top_n=1, vol_caps={"ASSET_A": 0.45, "ASSET_B": 0.45})
    cfg_top2 = CrossAssetConfig(name="v6b_top2", top_n=2, vol_caps={"ASSET_A": 0.45, "ASSET_B": 0.45})

    res1 = run_cross_asset_backtest(
        {"ASSET_A": bs_a, "ASSET_B": bs_b},
        cfg_top1,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=2000.0,
    )
    res2 = run_cross_asset_backtest(
        {"ASSET_A": bs_a, "ASSET_B": bs_b},
        cfg_top2,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[-1].date(),
        initial_capital_eur=2000.0,
    )

    # Only ASSET_A qualifies across all weeks:
    # - v6a_top1 deploys ~100% into ASSET_A with exactly 1 BUY order (no churn across subsequent weeks)
    # - v6b_top2 deploys ~50% into ASSET_A and holds ~50% in EUR Cash, with exactly 1 BUY order
    assert len(res1.fills) == 1
    assert len(res2.fills) == 1
    w_a_1 = float(res1.equity_curve["w_ASSET_A"].iloc[0])
    w_a_2 = float(res2.equity_curve["w_ASSET_A"].iloc[0])
    w_cash_2 = float(res2.equity_curve["w_CASH"].iloc[0])
    assert w_a_1 > 0.95
    assert 0.45 <= w_a_2 <= 0.52
    assert 0.48 <= w_cash_2 <= 0.55

    # Exact additive attribution identity: sum(net_pnl_eur) == final_equity - initial_capital
    attr2 = compute_asset_attribution(res2)
    total_attr_pnl = sum(a.net_pnl_eur for a in attr2.values())
    expected_pnl = float(res2.equity_curve["equity"].iloc[-1]) - 2000.0
    assert total_attr_pnl == pytest.approx(expected_pnl, abs=1e-6)
    assert attr2["ASSET_B"].days_held == 0
    assert attr2["ASSET_B"].net_pnl_eur == 0.0

    # Metrics include standalone and cumulative DSR
    m2 = compute_cross_asset_metrics(res2)
    assert "deflated_sharpe" in m2 and "deflated_sharpe_cum30" in m2 and "deflated_sharpe_cum50" in m2
    assert m2["total_orders"] == 1


# ---------------------------------------------------------------------------
# 8. Split-Consistent As-Traded Sizing Continuity (No Fake Equity Jump on Split)
# ---------------------------------------------------------------------------


def test_split_continuity_as_traded_sizing_has_zero_equity_jump() -> None:
    dates = pd.bdate_range("2024-06-03", periods=140)
    eval_start_idx = 120
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1

    split_ts = dates[eval_start_idx + 3]
    closes = np.linspace(10.0, 20.0, len(dates))
    # 10:1 stock split on split_ts -> pre-split raw price ~€180, post-split ~€18
    bs = _make_series("SPLIT_ETF", dates, closes, split_date=split_ts, split_ratio=10.0)

    cfg = CrossAssetConfig(
        name="split_test",
        top_n=1,
        vol_caps={"SPLIT_ETF": 0.45},
        sizing_mode=SizingMode.AS_TRADED,
    )
    res = run_cross_asset_backtest(
        {"SPLIT_ETF": bs},
        cfg,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[eval_start_idx + 8].date(),
        initial_capital_eur=1000.0,
    )
    eq = res.equity_curve["equity"]
    # Daily return across the 10:1 split date must be tiny (~0.4%), NOT -90% or +900%!
    ret_on_split = float(eq.loc[split_ts] / eq.loc[dates[eval_start_idx + 2]] - 1.0)
    assert abs(ret_on_split) < 0.02
    assert res.fills[0].raw_shares == int(math.floor((1000.0 - 1.25) / (closes[eval_start_idx] * 10.0 * 1.001)))


# ---------------------------------------------------------------------------
# 9. v6b_top2 Two-Slot Allocation, Single-Slot Rotation & Cross-Exchange Holidays
# ---------------------------------------------------------------------------


def test_v6b_top2_two_slot_buy_and_one_slot_rotation_keeps_incumbent_untouched() -> None:
    dates = pd.bdate_range("2024-06-03", periods=145)
    eval_start_idx = 120
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1
    n = len(dates)
    rng = np.random.default_rng(19)

    # Week 1: A and B are both strong uptrends (eligible), C is below SMA100 (ineligible).
    # During Week 1: A stays strong, B collapses below SMA100, C surges above SMA100!
    close_a = 20.0 * np.exp(np.cumsum(0.004 + rng.normal(0.0, 0.002, n)))
    close_b = 25.0 * np.exp(np.cumsum(0.003 + rng.normal(0.0, 0.002, n)))
    close_c = np.linspace(40.0, 30.0, n) + rng.normal(0.0, 0.1, n)

    for k in range(5):
        close_b[eval_start_idx + k] = close_b[eval_start_idx - 1] * (0.90 - 0.08 * k)
        close_c[eval_start_idx + k] = 36.0 + 3.0 * k  # Fri close = 48 (> SMA100)
    close_b[eval_start_idx + 5 :] = close_b[eval_start_idx + 4]
    close_c[eval_start_idx + 5 :] = 49.0

    bars = {
        "A": _make_series("A", dates, close_a),
        "B": _make_series("B", dates, close_b),
        "C": _make_series("C", dates, close_c),
    }
    cfg = CrossAssetConfig(
        name="v6b_rot",
        top_n=2,
        vol_caps={"A": 1.5, "B": 1.5, "C": 1.5},
        catastrophe_entry_dd=0.90,
        catastrophe_peak_dd=0.90,
        settlement_days=2,
    )
    res = run_cross_asset_backtest(
        bars,
        cfg,
        start_date=dates[eval_start_idx].date(),
        end_date=dates[eval_start_idx + 12].date(),
        initial_capital_eur=2000.0,
    )

    # Week 1 Mon: BUY A (~50%) and BUY B (~50%).
    # Week 2 Mon: SELL B (incumbent A stays completely untouched -> NO fills for A in Week 2!).
    # Week 2 Wed (T+2): BUY C with settled proceeds from B!
    assert len(res.fills) == 4
    buys_w1 = [f for f in res.fills if f.date == dates[eval_start_idx].date()]
    assert {f.symbol for f in buys_w1} == {"A", "B"}
    sell_b = [f for f in res.fills if f.side == "SELL"]
    assert len(sell_b) == 1 and sell_b[0].symbol == "B" and sell_b[0].date == dates[eval_start_idx + 5].date()
    buy_c = [f for f in res.fills if f.symbol == "C"]
    assert len(buy_c) == 1 and buy_c[0].date == dates[eval_start_idx + 7].date()  # Exact T+2!
    # Incumbent A had ZERO churn after initial entry
    assert sum(1 for f in res.fills if f.symbol == "A") == 1
    assert res.min_settled_cash_eur >= -1e-9


def test_cross_exchange_holiday_delays_fill_to_next_open_session() -> None:
    dates = pd.bdate_range("2024-06-03", periods=140)
    eval_start_idx = 120
    while dates[eval_start_idx].weekday() != 0:
        eval_start_idx += 1

    # Asset A (Milan) is closed on Monday eval_start_idx (exchange holiday);
    # Asset B (Xetra) is open on Monday (downtrending / ineligible).
    # Rebalance runs on Monday Open from Friday close, selects A, and A executes cleanly on Tuesday Open!
    n = len(dates)
    close_a = np.linspace(20.0, 50.0, n)
    close_b = np.linspace(100.0, 60.0, n)

    mon_ts = dates[eval_start_idx]
    tue_ts = dates[eval_start_idx + 1]
    bs_a_full = _make_series("A_MIL", dates, close_a)
    bs_a_hol = BarSeries(symbol="A_MIL", frame=bs_a_full.frame.drop(index=[mon_ts]))
    bs_b = _make_series("B_GER", dates, close_b)

    cfg = CrossAssetConfig(name="hol_test", top_n=1, vol_caps={"A_MIL": 1.0, "B_GER": 1.0})
    res = run_cross_asset_backtest(
        {"A_MIL": bs_a_hol, "B_GER": bs_b},
        cfg,
        start_date=mon_ts.date(),
        end_date=dates[eval_start_idx + 4].date(),
        initial_capital_eur=1000.0,
    )

    assert len(res.fills) == 1
    assert res.fills[0].symbol == "A_MIL"
    assert res.fills[0].date == tue_ts.date()  # Executed Tuesday Open when Milan re-opened!
    assert res.equity_curve.loc[mon_ts, "open_positions"] == 0
    assert res.equity_curve.loc[tue_ts, "open_positions"] == 1

