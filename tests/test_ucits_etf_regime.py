"""Unit and zero-look-ahead tests for Strategy v5 (UCITS Leveraged ETF/ETP Regime Rotation).

Covers:
- Hysteresis band state machine (no whipsaw inside ±1%, strict entry > 1.01, exit < 0.99)
- v5b volatility circuit breaker (matches regime_filter.py sigma_20, > 0.35 exit,
  strict re-entry requiring sigma_20 <= 0.35 AND Close > 1.01 * SMA(200))
- Strict US-close to next-EU-open calendar alignment (never same-day EU open;
  handles weekends, US-only holidays, EU-only holidays)
- Truncation invariance and post-cutoff scrambling zero-look-ahead test
- Strict T+2 European settlement enforcement (zero Good Faith Violations, including
  XEON.DE rotation 2-session cash wait vs instant EUR Cash deployment)
- Whole-share sizing, leftover EUR cash tracking, exact EUR 1.25 commission + 0.10%
  slippage, split-consistent as-traded share price reconstruction, and cross-exchange
  EU holiday handling.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import RegimeConfig
from src.data.bar_utils import raw_close_from_splits
from src.domain.models import BarSeries
from src.strategy.regime_filter import compute_regimes
from src.strategy.ucits_etf_regime import (
    RiskOffMode,
    SizingMode,
    UcitsRegimeConfig,
    UcitsRegimeState,
    align_us_signals_to_eu_calendar,
    compute_ucits_metrics,
    compute_us_regime_signals,
    run_buy_and_hold_etf,
    run_ucits_regime_backtest,
)


def _mk_series(
    symbol: str,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    open_: np.ndarray | None = None,
    splits: pd.Series | None = None,
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
            "volume": np.full(len(dates), 500_000.0),
            "raw_close": rc.to_numpy(dtype=float),
        },
        index=dates,
    )
    return BarSeries(symbol=symbol, frame=df)


# ---------------------------------------------------------------------------
# 1. Hysteresis band state machine (v5a)
# ---------------------------------------------------------------------------


def test_hysteresis_band_prevents_whipsaw():
    """Entry requires Close > 1.01*SMA, exit requires Close < 0.99*SMA; hold inside ±1%."""
    idx = pd.bdate_range("2022-01-03", periods=15)
    # Use sma_window=5 to verify exact numerical thresholds cleanly
    # Bars 0..4: flat 100 -> bar 4 SMA(5) = 100, close = 100 -> RISK_OFF (not > 101)
    # Bar 5: close = 101.0 -> SMA = 100.2, 1.01*SMA = 101.202 -> still RISK_OFF
    # Bar 6: close = 104.0 -> SMA = 101.0, 1.01*SMA = 102.01 -> enters RISK_ON!
    # Bars 7..9: hover inside [0.99*SMA, 1.01*SMA] (e.g. slightly below SMA) -> MUST stay RISK_ON!
    # Bar 10: close = 98.0 -> strictly < 0.99*SMA -> exits to RISK_OFF!
    # Bar 11..12: bounce slightly above SMA (~100.5, inside band) -> MUST stay RISK_OFF!
    # Bar 13: close = 106.0 -> strictly > 1.01*SMA -> re-enters RISK_ON!
    closes = np.array(
        [100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 104.0, 100.8, 100.7, 100.9, 98.0, 100.5, 100.8, 106.0, 106.0]
    )
    bars = _mk_series("QQQ", idx, closes)
    cfg = UcitsRegimeConfig(sma_window=5, hysteresis_pct=0.01, vol_circuit_breaker=None)
    sig = compute_us_regime_signals(bars, cfg)

    # Warmup bars 0..3 have NaN SMA -> RISK_OFF
    assert sig["sma200"].iloc[:4].isna().all()
    assert (sig["state"].iloc[:4] == UcitsRegimeState.RISK_OFF.value).all()

    # Bar 4 (100 vs SMA 100) and Bar 5 (101 vs 1.01*100.2=101.202) -> still RISK_OFF
    assert sig["state"].iloc[4] == UcitsRegimeState.RISK_OFF.value
    assert sig["state"].iloc[5] == UcitsRegimeState.RISK_OFF.value

    # Bar 6 (104 > 102.01) -> switches to RISK_ON
    assert sig["state"].iloc[6] == UcitsRegimeState.RISK_ON.value
    assert bool(sig["switched"].iloc[6]) is True

    # Bars 7, 8, 9 oscillate inside [0.99*SMA, 1.01*SMA] -> NO whipsaw, stays RISK_ON!
    for j in (7, 8, 9):
        c_j, m_j = sig["close"].iloc[j], sig["sma200"].iloc[j]
        assert 0.99 * m_j <= c_j <= 1.01 * m_j
        assert sig["state"].iloc[j] == UcitsRegimeState.RISK_ON.value
        assert bool(sig["switched"].iloc[j]) is False

    # Bar 10 drops strictly below 0.99 * SMA -> switches to RISK_OFF
    assert sig["close"].iloc[10] < 0.99 * sig["sma200"].iloc[10]
    assert sig["state"].iloc[10] == UcitsRegimeState.RISK_OFF.value
    assert bool(sig["switched"].iloc[10]) is True

    # Bars 11, 12 bounce inside hysteresis band -> stays RISK_OFF (no premature re-entry!)
    for j in (11, 12):
        c_j, m_j = sig["close"].iloc[j], sig["sma200"].iloc[j]
        assert 0.99 * m_j <= c_j <= 1.01 * m_j
        assert sig["state"].iloc[j] == UcitsRegimeState.RISK_OFF.value

    # Bar 13 breaks > 1.01 * SMA -> re-enters RISK_ON
    assert sig["close"].iloc[13] > 1.01 * sig["sma200"].iloc[13]
    assert sig["state"].iloc[13] == UcitsRegimeState.RISK_ON.value
    assert bool(sig["switched"].iloc[13]) is True


# ---------------------------------------------------------------------------
# 2. v5b volatility circuit breaker + locked re-entry rule
# ---------------------------------------------------------------------------


def test_v5b_vol_circuit_breaker_matches_regime_filter_and_reentry_rule():
    """sigma_20 matches regime_filter.py; vol > 0.35 forces RISK_OFF; re-entry needs <= 0.35 AND > 1.01*SMA."""
    idx = pd.bdate_range("2022-01-03", periods=100)
    # Build calm uptrend (bars 0..49) so Close > 1.01*SMA(20) and sigma_20 is tiny (~0.05)
    closes = np.linspace(100.0, 125.0, 100)
    # Inject violent +/-4.5% whipsaw on bars 50..56 while STAYING WELL ABOVE SMA(20) -> sigma_20 >> 0.35!
    base = closes[49]
    for k, mult in enumerate([1.045, 0.995, 1.050, 0.995, 1.055, 1.000, 1.055]):
        closes[50 + k] = base * mult
    # Bars 57..80: calm flat plateau right AT the SMA (~1.002 * SMA) while 20-day vol decays back <= 0.35
    closes[57:82] = base * 1.028
    # Bars 82..99: clear bullish breakout strictly > 1.01 * SMA -> allows re-entry!
    closes[82:] = np.linspace(base * 1.06, base * 1.15, 18)

    bars = _mk_series("QQQ", idx, closes)

    # Verify exact sigma_20 match with src/strategy/regime_filter.py
    reg_map = compute_regimes(bars, RegimeConfig(sma_window=20, vol_window=20))
    cfg = UcitsRegimeConfig(
        name="v5b",
        sma_window=20,
        hysteresis_pct=0.01,
        vol_window=20,
        vol_circuit_breaker=0.35,
    )
    sig = compute_us_regime_signals(bars, cfg)
    for ts in idx[20:]:
        assert sig.loc[ts, "sigma_20"] == pytest.approx(reg_map[ts].sigma_20, rel=1e-12)

    # Bar 49: calm uptrend -> RISK_ON
    assert sig["state"].iloc[49] == UcitsRegimeState.RISK_ON.value
    assert bool(sig["vol_breaker_active"].iloc[49]) is False

    # During vol spike (bars 54..57): Close is still > SMA, yet sigma_20 > 0.35 forces RISK_OFF!
    spike_bars = sig.iloc[52:60]
    triggered = spike_bars[spike_bars["sigma_20"] > 0.35]
    assert len(triggered) >= 1
    assert (triggered["state"] == UcitsRegimeState.RISK_OFF.value).all()
    assert triggered["vol_breaker_active"].all()

    # Once sigma_20 decays <= 0.35 on flat plateau (inside ±1% band), MUST remain RISK_OFF!
    calm_after = sig.iloc[76:82]
    assert (calm_after["sigma_20"] <= 0.35).all()
    assert (calm_after["close"] <= 1.01 * calm_after["sma200"]).all()
    assert (calm_after["state"] == UcitsRegimeState.RISK_OFF.value).all()

    # On breakout (> 1.01 * SMA and sigma_20 <= 0.35), re-enters RISK_ON!
    assert sig["sigma_20"].iloc[85] <= 0.35
    assert sig["close"].iloc[85] > 1.01 * sig["sma200"].iloc[85]
    assert sig["state"].iloc[85] == UcitsRegimeState.RISK_ON.value


# ---------------------------------------------------------------------------
# 3. Cross-timezone US-close -> next-EU-open alignment (weekends & holidays)
# ---------------------------------------------------------------------------


def test_cross_timezone_alignment_never_same_day_and_handles_holidays():
    """US close on d is strictly actionable on next EU date > d (handles US & EU holidays)."""
    # Week 1: Normal Mon-Fri + Mon MLK Day (US holiday: 2024-01-15)
    # Week 2: Easter Monday (EU holiday: 2024-04-01; US is OPEN, EU is CLOSED)
    us_dates = pd.DatetimeIndex(
        [
            "2024-01-12",  # Fri -> RISK_OFF
            # 2024-01-15 Mon is US holiday (MLK Day) - US CLOSED
            "2024-01-16",  # Tue -> switches RISK_ON
            "2024-01-17",  # Wed -> RISK_ON
            "2024-03-28",  # Thu -> RISK_ON
            # 2024-03-29 Fri Good Friday (both closed)
            "2024-04-01",  # Mon -> US OPEN, switches RISK_OFF (EU holiday!)
            "2024-04-02",  # Tue -> RISK_OFF
        ]
    )
    us_sig = pd.DataFrame(
        {
            "close": [100.0, 110.0, 111.0, 112.0, 95.0, 94.0],
            "sma200": [100.0] * 6,
            "sigma_20": [0.15] * 6,
            "state": ["RISK_OFF", "RISK_ON", "RISK_ON", "RISK_ON", "RISK_OFF", "RISK_OFF"],
            "vol_breaker_active": [False] * 6,
            "switched": [False, True, False, False, True, False],
        },
        index=us_dates,
    )

    eu_dates = pd.DatetimeIndex(
        [
            "2024-01-15",  # Mon (US holiday, EU open) -> must see Fri 2024-01-12 (RISK_OFF)
            "2024-01-16",  # Tue (US switches ON at 22:00 CET!) -> EU 09:00 open MUST still see Fri 2024-01-12 (RISK_OFF)!
            "2024-01-17",  # Wed -> first EU open seeing Tue 2024-01-16 (RISK_ON)!
            "2024-01-18",  # Thu -> sees Wed 2024-01-17 (RISK_ON)
            # 2024-04-01 Mon is EU Easter Monday (EU CLOSED while US switches RISK_OFF!)
            "2024-04-02",  # Tue -> first EU open after Easter Monday, MUST see Mon 2024-04-01 (RISK_OFF)!
        ]
    )

    aligned = align_us_signals_to_eu_calendar(us_sig, eu_dates)

    # Strict inequality on EVERY row: signal_us_date < eu_date (NEVER same-day!)
    assert (aligned["signal_us_date"] < aligned.index).all()

    assert aligned.loc["2024-01-15", "signal_us_date"] == pd.Timestamp("2024-01-12")
    assert aligned.loc["2024-01-15", "target_state"] == "RISK_OFF"

    # Tue EU Open MUST NOT peek at Tue US Close!
    assert aligned.loc["2024-01-16", "signal_us_date"] == pd.Timestamp("2024-01-12")
    assert aligned.loc["2024-01-16", "target_state"] == "RISK_OFF"

    # Wed EU Open acts on Tue US Close!
    assert aligned.loc["2024-01-17", "signal_us_date"] == pd.Timestamp("2024-01-16")
    assert aligned.loc["2024-01-17", "target_state"] == "RISK_ON"

    # Tue after EU Easter Monday acts on Mon US Close!
    assert aligned.loc["2024-04-02", "signal_us_date"] == pd.Timestamp("2024-04-01")
    assert aligned.loc["2024-04-02", "target_state"] == "RISK_OFF"


# ---------------------------------------------------------------------------
# 4. T+2 European settlement enforcement (Cash vs XEON.DE) & whole shares
# ---------------------------------------------------------------------------


def test_t2_settlement_zero_gfv_cash_vs_xeon_and_whole_shares():
    """Verify exact T+2 cash settlement, XEON rotation 2-day delay vs instant Cash, and exact costs."""
    # 30 business days: sma_window=5
    idx = pd.bdate_range("2024-01-02", periods=30)
    # US signal:
    # Bars 0..5: 100 -> RISK_OFF
    # Bar 6 (idx[6]): jumps to 110 -> RISK_ON (actionable EU idx[7] Open!)
    # Bars 7..13: 110 -> RISK_ON
    # Bar 14 (idx[14]): drops to 90 -> RISK_OFF (actionable EU idx[15] Open!)
    # Bars 15..21: 90 -> RISK_OFF
    # Bar 22 (idx[22]): jumps to 120 -> RISK_ON (actionable EU idx[23] Open!)
    us_c = np.full(30, 100.0)
    us_c[6:14] = 110.0
    us_c[14:22] = 90.0
    us_c[22:] = 120.0
    us_bars = _mk_series("QQQ", idx, us_c)

    # Risk-On ETF priced around EUR 260/share (like QQQ3.MI) -> EUR 1,000 buys 3 whole shares!
    ro_c = np.full(30, 260.0)
    ro_c[15:] = 280.0
    ro_bars = _mk_series("QQQ3.MI", idx, ro_c)

    # Risk-Off XEON.DE priced at EUR 140/share -> EUR 1,000 buys 7 whole shares!
    xeon_c = np.full(30, 140.0)
    xeon_bars = _mk_series("XEON.DE", idx, xeon_c)

    # A) EUR Cash mode: instant deployment on idx[7] and idx[23]
    cfg_cash = UcitsRegimeConfig(
        name="v5a_cash",
        risk_on_symbol="QQQ3.MI",
        risk_off_mode=RiskOffMode.CASH,
        sma_window=5,
        hysteresis_pct=0.01,
        commission_eur=1.25,
        slippage_pct=0.001,
        settlement_days=2,
    )
    res_cash = run_ucits_regime_backtest(
        us_bars,
        ro_bars,
        cfg_cash,
        start_date=idx[6].date(),
        end_date=idx[28].date(),
        initial_capital_eur=1000.0,
    )

    # Zero Good Faith Violations
    assert res_cash.min_settled_cash_eur >= -1e-9
    assert (res_cash.equity_curve["settled_cash"] >= -1e-9).all()

    # Exact fills & whole-share sizing:
    # Fill 0: BUY 3 shares at idx[7] Open (260 * 1.001 = 260.26), cost = 3*260.26 + 1.25 = 782.03 -> leftover cash = 217.97
    assert len(res_cash.fills) == 3  # BUY idx[7], SELL idx[15], BUY idx[23]
    f0, f1, f2 = res_cash.fills
    assert f0.side == "BUY" and f0.date == idx[7].date() and f0.signal_us_date == idx[6].date()
    assert f0.raw_shares == 3
    assert f0.raw_fill_price == pytest.approx(260.0 * 1.001)
    assert f0.commission_eur == pytest.approx(1.25)
    assert res_cash.equity_curve.loc[idx[7], "settled_cash"] == pytest.approx(1000.0 - (3 * 260.26 + 1.25))

    # Fill 1: SELL 3 shares at idx[15] Open (280 * 0.999 = 279.72), net proceeds = 3*279.72 - 1.25 = 837.91
    # Unsettled on idx[15] and idx[16], settles at idx[17]!
    assert f1.side == "SELL" and f1.date == idx[15].date() and f1.signal_us_date == idx[14].date()
    expected_proceeds = 3 * 280.0 * 0.999 - 1.25
    assert res_cash.equity_curve.loc[idx[15], "unsettled_cash"] == pytest.approx(expected_proceeds)
    assert res_cash.equity_curve.loc[idx[16], "unsettled_cash"] == pytest.approx(expected_proceeds)
    assert res_cash.equity_curve.loc[idx[17], "unsettled_cash"] == pytest.approx(0.0)

    # Fill 2: Cash is already settled -> buys Risk-On IMMEDIATELY at idx[23] Open!
    assert f2.side == "BUY" and f2.date == idx[23].date() and f2.signal_us_date == idx[22].date()

    # B) XEON.DE mode: strict T+2 delay on BOTH rotations (XEON -> Risk-On AND Risk-On -> XEON)!
    cfg_xeon = cfg_cash.model_copy(update={"name": "v5a_xeon", "risk_off_mode": RiskOffMode.XEON})
    res_xeon = run_ucits_regime_backtest(
        us_bars,
        ro_bars,
        cfg_xeon,
        risk_off_bars=xeon_bars,
        start_date=idx[6].date(),
        end_date=idx[28].date(),
        initial_capital_eur=1000.0,
    )

    assert res_xeon.min_settled_cash_eur >= -1e-9
    assert (res_xeon.equity_curve["settled_cash"] >= -1e-9).all()
    assert res_xeon.settlement_wait_days >= 2

    # Timeline in XEON mode:
    # - idx[6]: target RISK_OFF (from idx[5]) -> BUYS 7 shares of XEON.DE (7*140.14+1.25=982.23, leftover=17.77)
    # - idx[7]: target RISK_ON (from idx[6]) -> SELLS XEON.DE at idx[7] Open; leftover settled cash = 17.77 (< 260) -> CANNOT buy QQQ3.MI!
    # - idx[8]: still unsettled -> CANNOT buy QQQ3.MI!
    # - idx[9]: T+2 settled! -> BUYS QQQ3.MI at idx[9] Open! (2-session T+2 delay!)
    # - idx[15]: target RISK_OFF -> SELLS QQQ3.MI at idx[15] Open; proceeds unsettled on idx[15], idx[16]
    # - idx[17]: T+2 settled! -> BUYS XEON.DE at idx[17] Open!
    # - idx[23]: target RISK_ON -> SELLS XEON.DE at idx[23] Open; proceeds unsettled on idx[23], idx[24]
    # - idx[25]: T+2 settled! -> BUYS QQQ3.MI at idx[25] Open!
    fill_summary = [(f.date, f.symbol, f.side) for f in res_xeon.fills]
    assert fill_summary == [
        (idx[6].date(), "XEON.DE", "BUY"),
        (idx[7].date(), "XEON.DE", "SELL"),
        (idx[9].date(), "QQQ3.MI", "BUY"),
        (idx[15].date(), "QQQ3.MI", "SELL"),
        (idx[17].date(), "XEON.DE", "BUY"),
        (idx[23].date(), "XEON.DE", "SELL"),
        (idx[25].date(), "QQQ3.MI", "BUY"),
    ]


# ---------------------------------------------------------------------------
# 5. Split-consistent as-traded whole-share sizing (like CL2.PA 300:1 split)
# ---------------------------------------------------------------------------


def test_split_consistent_as_traded_whole_shares():
    """Pre-split EUR 4,000 share cannot be bought with EUR 1,000; EUR 5,000 holds 1 share seamlessly across 300:1 split."""
    idx = pd.bdate_range("2023-10-02", periods=15)
    # Split 300:1 on bar 6 (2023-10-10)
    splits = pd.Series(0.0, index=idx)
    splits.iloc[6] = 300.0
    # Split-adjusted price is ~EUR 13.33 -> pre-split as-traded price is EUR 4,000!
    adj_c = np.full(15, 13.3333333333)
    adj_c[6:] = 13.50  # slight +1.25% move on split day
    cl2 = _mk_series("CL2.PA", idx, adj_c, splits=splits)

    # Check EUR 1,000 Buy & Hold (as-traded): CANNOT buy before split (bar 0..5), buys post-split on bar 6!
    res_1k = run_buy_and_hold_etf(
        cl2,
        start_date=idx[0].date(),
        end_date=idx[-1].date(),
        initial_capital_eur=1000.0,
        sizing_mode=SizingMode.AS_TRADED,
    )
    assert res_1k.skipped_zero_shares == 6
    assert len(res_1k.fills) == 1
    assert res_1k.fills[0].date == idx[6].date()
    assert res_1k.fills[0].raw_shares == int(math.floor((1000.0 - 1.25) / (13.50 * 1.001)))

    # Check EUR 5,000 Buy & Hold (as-traded): buys 1 share at ~EUR 4,004 on bar 0, NO jump across 300:1 split!
    res_5k = run_buy_and_hold_etf(
        cl2,
        start_date=idx[0].date(),
        end_date=idx[-1].date(),
        initial_capital_eur=5000.0,
        sizing_mode=SizingMode.AS_TRADED,
    )
    assert len(res_5k.fills) == 1
    assert res_5k.fills[0].date == idx[0].date()
    assert res_5k.fills[0].raw_shares == 1
    assert res_5k.fills[0].adj_shares == pytest.approx(300.0)
    eq_pre = res_5k.equity_curve.loc[idx[5], "equity"]
    eq_post = res_5k.equity_curve.loc[idx[6], "equity"]
    # 300 adj shares * (13.50 - 13.333333) = +EUR 50.00 gain (smooth, zero 300x artifact!)
    assert eq_post - eq_pre == pytest.approx(300.0 * (13.50 - 13.3333333333), abs=0.05)


# ---------------------------------------------------------------------------
# 6. Post-cutoff scrambling & truncation zero-look-ahead test
# ---------------------------------------------------------------------------


def test_truncation_and_post_cutoff_scrambling_zero_lookahead():
    """Both v5a and v5b are strictly truncation-invariant and immune to post-cutoff scrambling."""
    rng = np.random.default_rng(123)
    n = 360
    us_idx = pd.bdate_range("2021-01-04", periods=n)
    # Create an EU calendar with different holidays (drop every 31st day, add a few shifted days)
    eu_mask = np.ones(n, dtype=bool)
    eu_mask[::31] = False
    eu_idx = us_idx[eu_mask]

    # Deterministic regime swings: bull -> bear -> bull -> high-vol spike -> bull
    us_c = 300.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    us_c[210:250] *= np.linspace(1.0, 0.72, 40)
    us_c[250:290] *= np.linspace(0.72, 1.12, 40)
    us_c[290:] *= 1.12
    # Inject vol spike around bar 275 (before cutoff 300)
    for k, mult in enumerate([1.05, 0.94, 1.06, 0.93, 1.05, 0.94]):
        us_c[270 + k] *= mult

    us_bars = _mk_series("QQQ", us_idx, us_c)
    ro_c = 150.0 * np.exp(np.cumsum(rng.normal(0.0007, 0.025, len(eu_idx))))
    xeon_c = 140.0 * np.exp(np.linspace(0.0, 0.04, len(eu_idx)))
    ro_bars = _mk_series("QQQ3.MI", eu_idx, ro_c)
    xeon_bars = _mk_series("XEON.DE", eu_idx, xeon_c)

    # 1. Truncation invariance of signal generator (v5a & v5b)
    for vol_cap in (None, 0.35):
        cfg = UcitsRegimeConfig(sma_window=50, vol_window=20, vol_circuit_breaker=vol_cap)
        full_sig = compute_us_regime_signals(us_bars, cfg)
        for cut in (60, 220, 280, 359):
            trunc_us = BarSeries(symbol="QQQ", frame=us_bars.frame.iloc[: cut + 1])
            part_sig = compute_us_regime_signals(trunc_us, cfg)
            pd.testing.assert_frame_equal(full_sig.iloc[: cut + 1], part_sig)

    # 2. Post-cutoff scrambling on BOTH US signal bars and EU ETF bars
    cutoff = us_idx[300]

    def _scramble(b: BarSeries, seed: int) -> BarSeries:
        df = b.frame.copy()
        m = df.index >= cutoff
        r = np.random.default_rng(seed)
        k = np.exp(r.normal(0.0, 0.35, int(m.sum())))
        for col in ("open", "high", "low", "close", "raw_close"):
            df.loc[m, col] = df.loc[m, col] * k
        return BarSeries(symbol=b.symbol, frame=df)

    us_pert = _scramble(us_bars, 11)
    ro_pert = _scramble(ro_bars, 22)
    xeon_pert = _scramble(xeon_bars, 33)

    for mode in (RiskOffMode.CASH, RiskOffMode.XEON):
        cfg = UcitsRegimeConfig(
            name=f"v5b_{mode.value}",
            risk_on_symbol="QQQ3.MI",
            risk_off_mode=mode,
            sma_window=50,
            vol_window=20,
            vol_circuit_breaker=0.35,
        )
        base = run_ucits_regime_backtest(
            us_bars,
            ro_bars,
            cfg,
            risk_off_bars=xeon_bars if mode is RiskOffMode.XEON else None,
            start_date=us_idx[180].date(),
            end_date=us_idx[-1].date(),
            initial_capital_eur=5000.0,
        )
        pert = run_ucits_regime_backtest(
            us_pert,
            ro_pert,
            cfg,
            risk_off_bars=xeon_pert if mode is RiskOffMode.XEON else None,
            start_date=us_idx[180].date(),
            end_date=us_idx[-1].date(),
            initial_capital_eur=5000.0,
        )

        assert len(base.fills) >= 3
        # Strict cross-timezone check on EVERY fill
        for f in base.fills:
            assert f.signal_us_date < f.date

        pre = base.equity_curve.index < cutoff
        pd.testing.assert_frame_equal(
            base.equity_curve[pre].drop(columns=["benchmark_close"]),
            pert.equity_curve[pre].drop(columns=["benchmark_close"]),
        )
        fills_pre_base = [f for f in base.fills if pd.Timestamp(f.date) < cutoff]
        fills_pre_pert = [f for f in pert.fills if pd.Timestamp(f.date) < cutoff]
        assert fills_pre_base == fills_pre_pert
        assert base.equity_curve["equity"].iloc[-1] != pert.equity_curve["equity"].iloc[-1]

        # Smoke-check metrics helper returns finite key fields
        m = compute_ucits_metrics(base)
        assert not math.isnan(m["cagr"])
        assert m["min_settled_cash_eur"] >= -1e-9


def test_cross_exchange_eu_holiday_waits_for_open_exchange():
    """If XEON.DE exchange (Xetra) is closed on an EU day while Risk-On (Paris) is open, wait cleanly."""
    us_idx = pd.bdate_range("2024-05-01", periods=15)
    us_c = np.full(15, 100.0)
    us_c[7:] = 80.0  # flips RISK_OFF at bar 7 -> actionable bar 8
    us_bars = _mk_series("QQQ", us_idx, us_c)

    ro_bars = _mk_series("CL2.PA", us_idx, np.full(15, 25.0))
    # Remove bar 10 (T+2 after bar 8 sale!) from XEON.DE to simulate a Xetra-only holiday
    xeon_idx = us_idx.delete(10)
    xeon_bars = _mk_series("XEON.DE", xeon_idx, np.full(len(xeon_idx), 140.0))

    cfg = UcitsRegimeConfig(
        name="v5a_xeon_hol",
        risk_on_symbol="CL2.PA",
        risk_off_mode=RiskOffMode.XEON,
        sma_window=5,
        hysteresis_pct=0.01,
        settlement_days=2,
    )
    # Start on bar 5 where US signal is RISK_OFF (wait: make bars 0..6 RISK_ON so we hold CL2.PA then sell!)
    us_c2 = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0])
    us_bars2 = _mk_series("QQQ", us_idx, us_c2)

    res = run_ucits_regime_backtest(
        us_bars2,
        ro_bars,
        cfg,
        risk_off_bars=xeon_bars,
        start_date=us_idx[6].date(),
        end_date=us_idx[14].date(),
        initial_capital_eur=1000.0,
    )
    # Bar 6: BUY CL2.PA (from bar 5 RISK_ON)
    # Bar 8: SELL CL2.PA (from bar 7 RISK_OFF) -> settles at bar 10
    # Bar 10: Xetra is CLOSED (missing from xeon_bars) -> cannot fill XEON.DE on bar 10!
    # Bar 11: Xetra re-opens -> BUYS XEON.DE on bar 11!
    xeon_buys = [f for f in res.fills if f.symbol == "XEON.DE" and f.side == "BUY"]
    assert len(xeon_buys) == 1
    assert xeon_buys[0].date == us_idx[11].date()
    assert res.min_settled_cash_eur >= -1e-9
