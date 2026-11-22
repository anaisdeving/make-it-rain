"""Unit and anti-look-ahead tests for Strategy v8b-clean (EUR Spot Crypto Trend Breakout).

Covers:
- Prior 20d high & prior 10d low strict exclusion of day ``t``
- Zero look-ahead truncation & future-scrambling invariance
- Gap-through-stop fills (``Open <= stop`` fills at ``Open``, NEVER inside the gap)
- Intra-day catastrophe stop fills (``Low <= stop`` fills at ``stop``, including on entry bar)
- Trailing stop (``HighestCloseSinceEntry - 2.5 * ATR14``) & 10d Donchian low signal exit
- BTC-EUR SMA(100) macro regime entry gate and forced macro exit
- Volatility-adjusted tie-breaker ranking ``Return_20d / (ATR14 / Close_t)``
- Fractional EUR sizing, EUR 10 min-order enforcement, T+0 instant settlement, and exact
  per-coin P&L conservation
- Bitvavo Tier 1 (0.25% taker + 0.10% slip) vs Kraken Pro Tier 1 (0.80% + 0.10%) fee impact
- Causal ``EURUSD=X`` conversion math (``EUR = USD / EURUSD``), weekend forward-fill without
  back-fill, loader fallback auditing, and caching.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.crypto_eur_loader import (
    CryptoEurLoader,
    convert_usd_bars_to_eur,
)
from src.domain.models import BarSeries
from src.strategy.crypto_eur_breakout import (
    CRYPTO_PERIODS_PER_YEAR,
    CryptoBreakoutConfig,
    compute_coin_attribution,
    compute_coin_indicators,
    compute_crypto_metrics,
    run_crypto_breakout_backtest,
)


def _mk_crypto_series(
    symbol: str,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    *,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: float = 1_000_000.0,
) -> BarSeries:
    c = np.asarray(close, dtype=float)
    o = c.copy() if open_ is None else np.asarray(open_, dtype=float)
    h = np.maximum(o, c) * 1.01 if high is None else np.asarray(high, dtype=float)
    lo = np.minimum(o, c) * 0.99 if low is None else np.asarray(low, dtype=float)
    h = np.maximum(h, np.maximum(o, c))
    lo = np.minimum(lo, np.minimum(o, c))
    df = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": np.full(len(dates), volume, dtype=float),
            "raw_close": c.copy(),
        },
        index=dates,
    )
    return BarSeries(symbol=symbol, frame=df)


# ---------------------------------------------------------------------------
# 1. Prior 20d High & Prior 10d Low Strictly Exclude Day t
# ---------------------------------------------------------------------------


def test_prior_high_and_low_strictly_exclude_day_t():
    """Prior_20d_High = max(High_{t-20..t-1}) and Prior_10d_Low = min(Low_{t-10..t-1})."""
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    close = np.full(40, 100.0)
    high = np.full(40, 102.0)
    low = np.full(40, 98.0)

    # Set a known high peak at index 10 and low trough at index 22
    high[10] = 110.0
    # On index 25, spike High[25] to 200.0 and Close[25] to 150.0
    high[25] = 200.0
    close[25] = 150.0
    # On index 30, crash Low[30] to 50.0 and Close[30] to 60.0
    low[22] = 90.0
    low[30] = 50.0
    close[30] = 60.0

    cfg = CryptoBreakoutConfig(sma_window=10, entry_high_window=20, exit_low_window=10, atr_window=5)
    ind = compute_coin_indicators(
        _mk_crypto_series("BTC-EUR", dates, close, high=high, low=low), cfg
    )

    # First 20 rows (0..19) must have NaN prior_high because 20 prior bars don't exist yet
    assert ind["prior_high"].iloc[:20].isna().all()
    assert math.isfinite(float(ind["prior_high"].iloc[20]))
    # On day 25 itself, prior_high MUST be max(High[5..24]) == 110.0 (NOT 200.0!)
    assert float(ind["prior_high"].iloc[25]) == pytest.approx(110.0)
    assert float(ind["close"].iloc[25]) > float(ind["prior_high"].iloc[25])
    # On day 26, day 25's high (200.0) enters the prior window
    assert float(ind["prior_high"].iloc[26]) == pytest.approx(200.0)

    # First 10 rows (0..9) must have NaN prior_low
    assert ind["prior_low"].iloc[:10].isna().all()
    # On day 30 itself, prior_low MUST be min(Low[20..29]) == 90.0 (NOT 50.0!)
    assert float(ind["prior_low"].iloc[30]) == pytest.approx(90.0)
    assert float(ind["close"].iloc[30]) < float(ind["prior_low"].iloc[30])
    # On day 31, day 30's low (50.0) enters the prior window
    assert float(ind["prior_low"].iloc[31]) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 2. Zero Look-Ahead: Truncation & Post-Cutoff Future-Scrambling Invariance
# ---------------------------------------------------------------------------


def test_zero_lookahead_truncation_and_scrambling_invariance():
    """Scrambling bars strictly after cutoff must not change any fill or equity value <= cutoff."""
    dates = pd.date_range("2021-07-01", periods=220, freq="D")
    rng = np.random.default_rng(123)

    def _rand_walk(start_px: float, drift: float, vol: float) -> np.ndarray:
        return start_px * np.cumprod(1.0 + rng.normal(drift, vol, len(dates)))

    btc_c = _rand_walk(30_000.0, 0.002, 0.03)
    eth_c = _rand_walk(2_000.0, 0.0025, 0.04)
    sol_c = _rand_walk(40.0, 0.003, 0.05)

    start = dates[115].date()
    cutoff = dates[175].date()
    end = dates[-1].date()

    cfg = CryptoBreakoutConfig(
        name="v8b_lookahead_test",
        universe=("BTC-EUR", "ETH-EUR", "SOL-EUR"),
        max_slots=2,
        sma_window=50,
    )

    bars_clean = {
        "BTC-EUR": _mk_crypto_series("BTC-EUR", dates, btc_c),
        "ETH-EUR": _mk_crypto_series("ETH-EUR", dates, eth_c),
        "SOL-EUR": _mk_crypto_series("SOL-EUR", dates, sol_c),
    }

    res_trunc = run_crypto_breakout_backtest(
        bars_clean, cfg, start_date=start, end_date=cutoff, initial_capital_eur=5000.0
    )

    mask_after = dates > pd.Timestamp(cutoff)
    btc_scr = btc_c.copy()
    eth_scr = eth_c.copy()
    sol_scr = sol_c.copy()
    btc_scr[mask_after] *= 0.05
    eth_scr[mask_after] *= 8.0
    sol_scr[mask_after] *= 0.01

    bars_scr = {
        "BTC-EUR": _mk_crypto_series("BTC-EUR", dates, btc_scr),
        "ETH-EUR": _mk_crypto_series("ETH-EUR", dates, eth_scr),
        "SOL-EUR": _mk_crypto_series("SOL-EUR", dates, sol_scr),
    }
    res_full = run_crypto_breakout_backtest(
        bars_scr, cfg, start_date=start, end_date=end, initial_capital_eur=5000.0
    )

    cols = ["equity", "cash", "market_value", "open_positions"]
    pd.testing.assert_frame_equal(
        res_trunc.equity_curve[cols],
        res_full.equity_curve.loc[: pd.Timestamp(cutoff), cols],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    fills_cut = [f for f in res_full.fills if f.date <= cutoff]
    assert res_trunc.fills == fills_cut


# ---------------------------------------------------------------------------
# 3. Hard Catastrophe Stop: Gap-Through-Stop (at Open) & Intra-Day (at Stop)
# ---------------------------------------------------------------------------


def test_gap_through_hard_stop_fills_at_open_not_inside_gap():
    """If Open <= hard_stop on a held bar, fill MUST occur at Open (with sell fee + slip)."""
    dates = pd.date_range("2022-01-01", periods=35, freq="D")
    # Flat 100 with range [99, 101] -> ATR14 ~ 2.0 -> 3*ATR ~ 6.0
    open_ = np.full(35, 100.0)
    high = np.full(35, 101.0)
    low = np.full(35, 99.0)
    close = np.full(35, 100.0)

    # Day 24 close breaks out above Prior_20d_High (101) & SMA(10) -> buy at Day 25 Open (105.0)
    close[24] = 105.0
    high[24] = 105.0
    open_[25] = 105.0
    high[25] = 106.0
    low[25] = 104.0
    close[25] = 105.5

    # Day 26: massive overnight gap-down crash to Open=40.0 (way below hard stop ~98.6!)
    open_[26] = 40.0
    high[26] = 45.0
    low[26] = 35.0
    close[26] = 38.0

    # Keep BTC healthy above SMA so BTC macro exit does not interfere
    btc = _mk_crypto_series("BTC-EUR", dates, np.linspace(30_000.0, 35_000.0, 35))
    alt = _mk_crypto_series("ALT-EUR", dates, close, open_=open_, high=high, low=low)

    cfg = CryptoBreakoutConfig(
        name="gap_stop_test",
        universe=("ALT-EUR",),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=14,
        taker_fee_pct=0.0025,
        slippage_pct=0.0010,
    )
    res = run_crypto_breakout_backtest(
        {"BTC-EUR": btc, "ALT-EUR": alt},
        cfg,
        start_date=dates[22].date(),
        end_date=dates[28].date(),
        initial_capital_eur=1000.0,
    )

    assert len(res.round_trips) == 1
    trip = res.round_trips[0]
    assert trip.entry_date == dates[25].date()
    assert trip.exit_date == dates[26].date()
    assert trip.exit_reason == "HARD_STOP_GAP"
    assert trip.exit_category == "HARD_STOP"
    # MUST fill at gap Open (40.0), NOT at hard_stop (~98)!
    assert trip.hard_stop_price > 90.0
    assert trip.exit_raw_price == pytest.approx(40.0)
    assert trip.exit_fill_price == pytest.approx(40.0 * (1.0 - 0.0010))
    # Sell fee + slippage applied to stop fill
    sell_fill = res.fills[1]
    assert sell_fill.side == "SELL"
    assert sell_fill.fee_eur == pytest.approx(sell_fill.notional_eur * 0.0025)
    assert sell_fill.slippage_eur == pytest.approx(sell_fill.qty * 40.0 * 0.0010)


def test_intraday_hard_stop_fills_at_stop_including_on_entry_bar():
    """If Open > stop and Low <= stop (even on entry bar after open), fill at stop with fee + slip."""
    dates = pd.date_range("2022-01-01", periods=35, freq="D")
    open_ = np.full(35, 100.0)
    high = np.full(35, 101.0)
    low = np.full(35, 99.0)
    close = np.full(35, 100.0)

    # Day 24 breakout -> buy on Day 25 Open (105.0)
    close[24] = 105.0
    high[24] = 105.0
    open_[25] = 105.0
    high[25] = 106.0
    # Intraday flash-crash on the VERY SAME entry bar (Day 25) down to Low=80.0!
    low[25] = 80.0
    close[25] = 95.0

    btc = _mk_crypto_series("BTC-EUR", dates, np.linspace(30_000.0, 35_000.0, 35))
    alt = _mk_crypto_series("ALT-EUR", dates, close, open_=open_, high=high, low=low)

    cfg = CryptoBreakoutConfig(
        name="intraday_entry_bar_stop",
        universe=("ALT-EUR",),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=14,
        taker_fee_pct=0.0025,
        slippage_pct=0.0010,
    )
    res = run_crypto_breakout_backtest(
        {"BTC-EUR": btc, "ALT-EUR": alt},
        cfg,
        start_date=dates[22].date(),
        end_date=dates[28].date(),
        initial_capital_eur=1000.0,
    )

    assert len(res.round_trips) == 1
    trip = res.round_trips[0]
    assert trip.entry_date == dates[25].date()
    assert trip.exit_date == dates[25].date()
    assert trip.exit_reason == "HARD_STOP_INTRADAY"
    assert trip.exit_category == "HARD_STOP"
    expected_entry_fill = 105.0 * (1.0 + 0.0010)
    expected_stop = expected_entry_fill - 3.0 * trip.entry_atr14
    assert trip.hard_stop_price == pytest.approx(expected_stop)
    assert trip.exit_raw_price == pytest.approx(expected_stop)
    assert trip.exit_fill_price == pytest.approx(expected_stop * (1.0 - 0.0010))
    # Position is already flat by Day 25 close
    assert int(res.equity_curve.loc[dates[25], "open_positions"]) == 0


# ---------------------------------------------------------------------------
# 4. BTC Macro Regime Gate & Forced Exit + ATR Trailing Stop
# ---------------------------------------------------------------------------


def test_btc_macro_gate_blocks_entry_and_forces_exit():
    """BTC <= SMA(100) blocks new entries; BTC < SMA(100) exits held altcoin at t+1 open."""
    dates = pd.date_range("2022-01-01", periods=45, freq="D")
    # BTC starts bear (< SMA10), turns bull on day 25, then crashes < SMA10 on day 33
    btc_c = np.concatenate(
        [
            np.linspace(40_000.0, 30_000.0, 24),  # 0..23 bear
            np.array([29_000.0]),  # 24 still bear
            np.linspace(38_000.0, 42_000.0, 8),  # 25..32 bull (> SMA10)
            np.full(12, 25_000.0),  # 33..44 bear (< SMA10)
        ]
    )
    # Altcoin breaks out twice: once on day 22 (while BTC is bear -> MUST be blocked!)
    # and once on day 27 (while BTC is bull -> MUST enter on day 28 Open, then exit on day 34 Open!)
    alt_c = np.full(45, 100.0)
    alt_c[22] = 115.0  # breakout while BTC < SMA10 -> blocked
    alt_c[23:27] = 105.0
    alt_c[27] = 125.0  # breakout while BTC > SMA10 -> enter day 28
    alt_c[28:] = 126.0  # stays flat/strong so only BTC_MACRO_EXIT can close it

    cfg = CryptoBreakoutConfig(
        name="btc_gate_test",
        universe=("ALT-EUR",),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=5,
    )
    res = run_crypto_breakout_backtest(
        {
            "BTC-EUR": _mk_crypto_series("BTC-EUR", dates, btc_c),
            "ALT-EUR": _mk_crypto_series("ALT-EUR", dates, alt_c),
        },
        cfg,
        start_date=dates[21].date(),
        end_date=dates[40].date(),
        initial_capital_eur=1000.0,
    )

    assert len(res.round_trips) == 1
    trip = res.round_trips[0]
    assert trip.entry_date == dates[28].date()  # day 22 breakout was blocked by BTC regime!
    assert trip.exit_date == dates[34].date()  # day 33 BTC bear close -> day 34 open exit!
    assert trip.exit_reason == "BTC_MACRO_EXIT"
    assert trip.exit_category == "SIGNAL_EXIT"


def test_atr_trailing_stop_exit_uses_highest_close_since_entry():
    """Close_t < HighestCloseSinceEntry - 2.5 * ATR14 triggers exit at day t+1 open."""
    dates = pd.date_range("2022-01-01", periods=45, freq="D")
    btc = _mk_crypto_series("BTC-EUR", dates, np.linspace(30_000.0, 40_000.0, 45))

    close = np.full(45, 100.0)
    high = np.full(45, 101.0)
    low = np.full(45, 99.0)

    # Day 22 breakout -> enter Day 23 Open (110.0)
    close[22] = 110.0
    high[22] = 110.5
    # Rally steadily to 140 on day 30 so Prior_10d_Low stays around ~109 while HighestClose reaches 140
    close[23:31] = np.linspace(112.0, 140.0, 8)
    high[23:31] = close[23:31] + 0.5
    low[23:31] = close[23:31] - 0.5
    # On day 31, drop to 118 (still ABOVE Prior_10d_Low ~ 99..111, still ABOVE hard_stop ~ 103,
    # and strictly BELOW HighestClose(140) - 2.5 * ATR5!)
    close[31] = 118.0
    high[31] = 135.0
    low[31] = 117.5
    close[32:] = 118.0
    high[32:] = 119.0
    low[32:] = 117.0

    cfg = CryptoBreakoutConfig(
        name="trail_test",
        universe=("ALT-EUR",),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=5,
        trail_atr_mult=2.5,
        hard_stop_atr_mult=10.0,  # isolate ATR trailing stop from hard catastrophe stop
    )
    res = run_crypto_breakout_backtest(
        {
            "BTC-EUR": btc,
            "ALT-EUR": _mk_crypto_series("ALT-EUR", dates, close, open_=close, high=high, low=low),
        },
        cfg,
        start_date=dates[21].date(),
        end_date=dates[38].date(),
        initial_capital_eur=1000.0,
    )

    assert len(res.round_trips) == 1
    trip = res.round_trips[0]
    assert trip.entry_date == dates[23].date()
    assert trip.exit_date == dates[32].date()
    assert trip.exit_reason == "ATR_TRAILING_STOP"
    assert trip.highest_close == pytest.approx(140.0)


# ---------------------------------------------------------------------------
# 5. Tie-Breaker Ranking: Return_20d / (ATR14 / Close_t)
# ---------------------------------------------------------------------------


def test_tiebreaker_ranks_by_return_over_normalized_atr():
    """When 2 candidates trigger for 1 slot, coin with higher Return_20d / (ATR14 / Close_t) wins."""
    dates = pd.date_range("2022-01-01", periods=35, freq="D")
    btc = _mk_crypto_series("BTC-EUR", dates, np.linspace(30_000.0, 35_000.0, 35))

    # Keep both flat through day 24 so neither breaks out prior to day 25.
    # Coin A: +25% 20d return with LOW daily volatility (high Return_20d / (ATR14 / Close))
    a_close = np.full(35, 100.0)
    a_high = np.full(35, 101.0)
    a_low = np.full(35, 99.0)
    a_close[25] = 125.0
    a_high[25] = 125.5
    a_low[25] = 124.5
    a_close[26:] = 126.0
    a_high[26:] = 126.5
    a_low[26:] = 125.5

    # Coin B: +30% 20d return with MUCH HIGHER daily volatility (lower Return_20d / (ATR14 / Close))
    b_close = np.full(35, 100.0)
    b_high = np.full(35, 115.0)
    b_low = np.full(35, 85.0)
    b_close[25] = 130.0
    b_high[25] = 135.0
    b_low[25] = 120.0
    b_close[26:] = 131.0
    b_high[26:] = 136.0
    b_low[26:] = 125.0

    cfg = CryptoBreakoutConfig(
        name="tiebreak_test",
        universe=("COIN_A-EUR", "COIN_B-EUR"),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=14,
    )
    res = run_crypto_breakout_backtest(
        {
            "BTC-EUR": btc,
            "COIN_A-EUR": _mk_crypto_series("COIN_A-EUR", dates, a_close, high=a_high, low=a_low),
            "COIN_B-EUR": _mk_crypto_series("COIN_B-EUR", dates, b_close, high=b_high, low=b_low),
        },
        cfg,
        start_date=dates[25].date(),
        end_date=dates[28].date(),
        initial_capital_eur=1000.0,
    )

    buy_fills = [f for f in res.fills if f.side == "BUY"]
    assert len(buy_fills) == 1
    assert buy_fills[0].symbol == "COIN_A-EUR"
    assert buy_fills[0].date == dates[26].date()


# ---------------------------------------------------------------------------
# 6. Fractional EUR Sizing, EUR 10 Min Order, T+0 Instant Settlement, Conservation
# ---------------------------------------------------------------------------


def test_fractional_sizing_min_order_t0_settlement_and_pnl_conservation():
    """Verify EUR 10 min order skip, T+0 same-open redeployment, and exact P&L conservation."""
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    btc = _mk_crypto_series("BTC-EUR", dates, np.linspace(30_000.0, 40_000.0, 40))

    # Coin A breaks out on Day 22 (buy Day 23 Open), then breaks 10d low on Day 28 (sell Day 29 Open).
    # Coin B breaks out on Day 28 (buy Day 29 Open using T+0 instant proceeds from Coin A!).
    a_c = np.full(40, 100.0)
    a_c[22] = 110.0
    a_c[23:28] = 112.0
    a_c[28] = 90.0  # < prior 10d low -> sell Day 29 Open
    a_c[29:] = 90.0

    b_c = np.full(40, 200.0)
    b_c[28] = 220.0  # breakout on Day 28 close -> buy Day 29 Open!
    b_c[29:] = 230.0

    cfg = CryptoBreakoutConfig(
        name="t0_test",
        universe=("A-EUR", "B-EUR"),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=5,
        min_order_eur=10.0,
        hard_stop_atr_mult=10.0,
    )
    res = run_crypto_breakout_backtest(
        {
            "BTC-EUR": btc,
            "A-EUR": _mk_crypto_series("A-EUR", dates, a_c),
            "B-EUR": _mk_crypto_series("B-EUR", dates, b_c),
        },
        cfg,
        start_date=dates[22].date(),
        end_date=dates[35].date(),
        initial_capital_eur=1000.0,
    )

    # T+0 instant settlement: SELL A-EUR and BUY B-EUR both execute on dates[29] Open!
    sell_a = [f for f in res.fills if f.symbol == "A-EUR" and f.side == "SELL"]
    buy_b = [f for f in res.fills if f.symbol == "B-EUR" and f.side == "BUY"]
    assert len(sell_a) == 1 and len(buy_b) == 1
    assert sell_a[0].date == dates[29].date()
    assert buy_b[0].date == dates[29].date()
    # Fractional quantity (not integer-truncated)
    assert abs(buy_b[0].qty - round(buy_b[0].qty)) > 1e-4

    # Exact additive P&L conservation across per-coin attributions
    attr = compute_coin_attribution(res)
    total_attr_pnl = sum(a.net_pnl_eur for a in attr.values())
    actual_pnl = float(res.equity_curve["equity"].iloc[-1]) - 1000.0
    assert total_attr_pnl == pytest.approx(actual_pnl, abs=1e-6)

    # EUR 10 minimum order enforcement: EUR 15 capital split across 2 slots = EUR 7.50 < EUR 10 -> skipped!
    cfg_min = cfg.model_copy(update={"max_slots": 2, "min_order_eur": 10.0})
    res_min = run_crypto_breakout_backtest(
        {
            "BTC-EUR": btc,
            "A-EUR": _mk_crypto_series("A-EUR", dates, a_c),
            "B-EUR": _mk_crypto_series("B-EUR", dates, b_c),
        },
        cfg_min,
        start_date=dates[22].date(),
        end_date=dates[30].date(),
        initial_capital_eur=15.0,
    )
    assert len(res_min.fills) == 0
    assert res_min.skipped_min_order >= 1


# ---------------------------------------------------------------------------
# 7. Bitvavo (0.25% + 0.10%) vs Kraken (0.80% + 0.10%) Fee Sensitivity
# ---------------------------------------------------------------------------


def test_bitvavo_vs_kraken_fee_model_impact():
    """Kraken 0.80% taker fee produces exact higher fee drag and lower final equity than Bitvavo 0.25%."""
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    btc = _mk_crypto_series("BTC-EUR", dates, np.linspace(30_000.0, 40_000.0, 40))
    alt_c = np.full(40, 100.0)
    alt_c[22] = 110.0
    alt_c[23:29] = 115.0
    alt_c[29] = 95.0
    alt_c[30:] = 95.0
    bars = {"BTC-EUR": btc, "ALT-EUR": _mk_crypto_series("ALT-EUR", dates, alt_c)}

    cfg_bitvavo = CryptoBreakoutConfig(
        name="bitvavo",
        universe=("ALT-EUR",),
        max_slots=1,
        sma_window=10,
        entry_high_window=20,
        exit_low_window=10,
        atr_window=5,
        taker_fee_pct=0.0025,
        slippage_pct=0.0010,
        hard_stop_atr_mult=10.0,
    )
    cfg_kraken = cfg_bitvavo.model_copy(update={"name": "kraken", "taker_fee_pct": 0.0080})

    res_bv = run_crypto_breakout_backtest(
        bars, cfg_bitvavo, start_date=dates[22].date(), end_date=dates[32].date(), initial_capital_eur=1000.0
    )
    res_kr = run_crypto_breakout_backtest(
        bars, cfg_kraken, start_date=dates[22].date(), end_date=dates[32].date(), initial_capital_eur=1000.0
    )

    assert len(res_bv.fills) == 2 and len(res_kr.fills) == 2
    assert res_kr.total_fees_eur > res_bv.total_fees_eur * 3.0
    assert float(res_kr.equity_curve["equity"].iloc[-1]) < float(res_bv.equity_curve["equity"].iloc[-1])

    m = compute_crypto_metrics(res_bv)
    assert m["periods_per_year"] == CRYPTO_PERIODS_PER_YEAR
    assert m["daily_sample_length"] == len(res_bv.equity_curve)
    assert "deflated_sharpe_n10" in m and "deflated_sharpe_cum40" in m and "deflated_sharpe_cum60" in m


# ---------------------------------------------------------------------------
# 8. Causal EURUSD=X Conversion Math & CryptoEurLoader Fallback / Caching
# ---------------------------------------------------------------------------


def test_usd_to_eur_conversion_math_and_causal_weekend_ffill():
    """Verify EUR_price = USD_price / EURUSD and strict causal forward-fill over Sat/Sun."""
    # Thu 2022-01-06, Fri 2022-01-07, Sat 2022-01-08, Sun 2022-01-09, Mon 2022-01-10
    crypto_dates = pd.date_range("2022-01-06", periods=5, freq="D")
    fx_dates = pd.DatetimeIndex(["2022-01-06", "2022-01-07", "2022-01-10"])  # Mon-Fri only

    usd_series = _mk_crypto_series(
        "SOL-USD",
        crypto_dates,
        np.array([110.0, 120.0, 120.0, 120.0, 100.0]),
        open_=np.array([110.0, 120.0, 120.0, 120.0, 100.0]),
        high=np.array([110.0, 120.0, 120.0, 120.0, 100.0]),
        low=np.array([110.0, 120.0, 120.0, 120.0, 100.0]),
    )
    # Thu EURUSD=1.10, Fri EURUSD=1.20, Mon EURUSD=1.00
    fx_series = _mk_crypto_series("EURUSD=X", fx_dates, np.array([1.10, 1.20, 1.00]))

    eur_series, ffill_days = convert_usd_bars_to_eur(usd_series, fx_series, "SOL-EUR")
    assert eur_series.symbol == "SOL-EUR"
    assert ffill_days == 2  # Saturday and Sunday forward-filled from Friday's 1.20!

    # Thu: 110 / 1.10 = 100.0
    assert float(eur_series.frame.loc[pd.Timestamp("2022-01-06"), "close"]) == pytest.approx(100.0)
    # Fri: 120 / 1.20 = 100.0
    assert float(eur_series.frame.loc[pd.Timestamp("2022-01-07"), "close"]) == pytest.approx(100.0)
    # Sat & Sun MUST use Friday's 1.20 (NOT Monday's 1.00!) -> 120 / 1.20 = 100.0
    assert float(eur_series.frame.loc[pd.Timestamp("2022-01-08"), "close"]) == pytest.approx(100.0)
    assert float(eur_series.frame.loc[pd.Timestamp("2022-01-09"), "close"]) == pytest.approx(100.0)
    # Mon: 100 / 1.00 = 100.0
    assert float(eur_series.frame.loc[pd.Timestamp("2022-01-10"), "close"]) == pytest.approx(100.0)


def test_crypto_eur_loader_fallback_conversion_and_caching(tmp_path):
    """Loader uses native -EUR when clean, falls back to -USD/EURUSD=X when missing, and caches."""
    all_dates = pd.date_range("2021-06-01", "2026-11-20", freq="D")
    fx_dates = pd.bdate_range("2021-05-18", "2026-11-20")

    def _raw_df(idx: pd.DatetimeIndex, px: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": px,
                "High": px * 1.02,
                "Low": px * 0.98,
                "Close": px,
                "Volume": np.full(len(idx), 500_000.0),
                "Stock Splits": np.zeros(len(idx)),
            },
            index=idx,
        )

    calls: list[str] = []

    def _fake_hist(sym: str, start: date, end: date):
        calls.append(sym)
        if sym == "EURUSD=X":
            df = _raw_df(fx_dates, np.full(len(fx_dates), 1.10))
            return df, df["Stock Splits"]
        if sym == "BTC-EUR":
            df = _raw_df(all_dates, 30_000.0 + 5_000.0 * np.sin(np.linspace(0, 10, len(all_dates))))
            return df, df["Stock Splits"]
        if sym == "AVAX-EUR":
            # Missing native EUR history -> must trigger causal AVAX-USD / EURUSD=X fallback
            return None, None
        if sym == "AVAX-USD":
            df = _raw_df(all_dates, np.full(len(all_dates), 55.0) + np.linspace(0, 5.0, len(all_dates)))
            return df, df["Stock Splits"]
        return None, None

    loader = CryptoEurLoader(tmp_path, history_fetcher=_fake_hist)
    ds = loader.load_dataset(
        symbols=("BTC-EUR", "AVAX-EUR"),
        inspect_lunc=False,
    )

    assert "BTC-EUR" in ds.bars and "AVAX-EUR" in ds.bars
    assert ds.audits["BTC-EUR"].conversion_used is False
    assert ds.audits["AVAX-EUR"].conversion_used is True
    assert ds.audits["AVAX-EUR"].source_symbol == "AVAX-USD / EURUSD=X"
    # First close: 55.0 / 1.10 = 50.0 EUR
    assert float(ds.bars["AVAX-EUR"].frame["close"].iloc[0]) == pytest.approx(50.0)

    # Second load hits disk cache with zero additional fetches
    n_first = len(calls)
    ds2 = loader.load_dataset(symbols=("BTC-EUR", "AVAX-EUR"), inspect_lunc=False)
    assert len(calls) == n_first
    assert set(ds2.bars.keys()) == {"BTC-EUR", "AVAX-EUR"}
