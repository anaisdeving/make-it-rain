"""Strategy v3: Volatility Contraction (VCP) & 52-Week High Momentum Breakout.

Pure price-and-volume daily screener and event-driven backtest runner operating
strictly on cached OHLCV bars (zero external fundamental APIs, zero look-ahead).

Leaves ``src/risk/risk_manager.py`` and ``src/execution/`` completely untouched:
- Injects liquid-tier slippage (0.15% per side), IBKR commissions ($0.38/side),
  entry gap cap (+1.5%), initial stop multiple (2.5 * ATR14, capped at 15%),
  and exit parameters strictly via Pydantic ``Settings`` dependency injection.
- Passes ``event_day_low = signal_close * (1 + entry_limit_cap_pct)`` when
  ``pure_atr_stop=True`` so untouched ``risk_manager.stop_distance`` resolves
  cleanly to ``min(2.5 * ATR14, 0.15 * limit)``.

Exact bar-index conventions for a breakout signal evaluated at the close of day ``t``:
- Universe liquidity: ``avg_dollar_volume_20`` and ``avg_volume_20`` use prior 20
  sessions ``[t-20 .. t-1]`` so day ``t``'s breakout volume does not inflate them.
- Stage 2 trend & RS: ``SMA50``, ``SMA150``, ``SMA200``, ``High_252d``, ``Low_252d``,
  ``Return_63d``, ``Return_126d`` use bars ``<= t``; ``SMA200`` rising checks
  ``SMA200_t > SMA200_{t-20}``. Cross-sectional RS percentile is computed strictly
  across the eligible universe on day ``t``.
- Tight base & volatility contraction: 5-day consolidation window is the 5 sessions
  immediately *prior* to breakout ``[t-5 .. t-1]`` with
  ``(MaxHigh_5d_prior - MinLow_5d_prior) / Close_{t-1} <= 0.08``, and
  ``ATR10 / ATR50 <= 0.75`` is evaluated at ``t-1`` (end of the base, before day
  ``t``'s breakout bar expands True Range) using simple rolling-mean TR over
  ``[t-10 .. t-1]`` vs ``[t-50 .. t-1]``.
- Breakout trigger on day ``t``: ``Close_t > MaxHigh_5d_prior``,
  ``Volume_t >= 1.5 * AvgVol20``, ``(Close_t - Low_t) / (High_t - Low_t) >= 0.60``.
- Entry on day ``t+1`` Open via ``BacktestBroker`` with ``+1.5%`` limit cap.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.backtest.engine import BacktestResult, manage_position
from src.config import RiskConfig, Settings
from src.data.provider_base import MarketDataProvider
from src.domain.models import (
    AnnouncementTiming,
    BarSeries,
    EarningsEvent,
    ExitReason,
    MarketRegime,
    RegimeState,
    TradeSignal,
    UniverseMember,
)
from src.execution.backtest_broker import BacktestBroker
from src.risk.risk_manager import build_order_intent
from src.strategy.indicators import wilder_atr
from src.strategy.regime_filter import compute_regimes

log = logging.getLogger(__name__)


class VcpConfig(BaseModel):
    """Configuration for Strategy v3 (VCP + 52-Week High Momentum Breakout)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Universe (point-in-time)
    min_price: float = Field(10.0, gt=0)
    max_price: float = Field(150.0, gt=0)
    min_avg_dollar_volume: float = Field(10_000_000.0, ge=0)
    avg_window: int = Field(20, ge=1)

    # Stage 2 Trend & Relative Strength
    sma_fast: int = Field(50, ge=1)
    sma_mid: int = Field(150, ge=1)
    sma_slow: int = Field(200, ge=1)
    sma_slow_slope_lookback: int = Field(20, ge=1)
    high_low_window: int = Field(252, ge=1)
    min_pct_of_52w_high: float = Field(0.85, gt=0, le=1.0)
    min_multiple_of_52w_low: float = Field(1.30, ge=1.0)
    rs_short_window: int = Field(63, ge=1)
    rs_long_window: int = Field(126, ge=1)
    rs_short_weight: float = Field(0.60, ge=0, le=1.0)
    rs_long_weight: float = Field(0.40, ge=0, le=1.0)
    rs_top_quantile: float = Field(0.85, ge=0, le=1.0)

    # Volatility Contraction (Tight Base)
    atr_short_window: int = Field(10, ge=1)
    atr_long_window: int = Field(50, ge=1)
    max_atr_ratio: float = Field(0.75, gt=0)
    atr_contraction_lag: int = Field(1, ge=0)
    atr_contraction_mode: Literal["sma_atr", "sma_natr", "wilder_atr"] = "sma_atr"
    base_window: int = Field(5, ge=1)
    max_base_range_pct: float = Field(0.08, gt=0)

    # Breakout Trigger
    min_volume_ratio: float = Field(1.5, gt=0)
    min_close_location: float = Field(0.60, ge=0, le=1.0)

    # Execution & Risk DI overrides (applied via Settings, never editing risk_manager/execution)
    entry_limit_cap_pct: float = Field(0.015, ge=0.0)
    slippage_per_side_pct: float = Field(0.0015, ge=0.0)
    commission_per_side_usd: float = Field(0.38, ge=0.0)
    atr_stop_multiple: float = Field(2.5, gt=0.0)
    max_stop_pct: float = Field(0.15, gt=0.0, lt=1.0)
    breakeven_trigger_r: float = Field(1.5, gt=0.0)
    trail_trigger_r: float = Field(2.0, gt=0.0)
    trail_atr_multiple: float = Field(2.5, gt=0.0)
    time_stop_days: int = Field(45, ge=1)
    time_stop_min_r: float = Field(1.0)
    pure_atr_stop: bool = True

    @model_validator(mode="after")
    def _validate_bounds(self) -> "VcpConfig":
        if self.max_price < self.min_price:
            raise ValueError("max_price must be >= min_price")
        if not (self.sma_fast < self.sma_mid < self.sma_slow):
            raise ValueError("SMA windows must satisfy sma_fast < sma_mid < sma_slow")
        if self.atr_short_window >= self.atr_long_window:
            raise ValueError("atr_short_window must be < atr_long_window")
        return self


class VcpSnapshot(BaseModel):
    """Everything the v3 screener may know about one symbol at the close of ``date``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    raw_close: float
    volume: float
    avg_volume_20: float
    avg_dollar_volume_20: float
    sma50: float
    sma150: float
    sma200: float
    sma200_prev: float
    high_252d: float
    low_252d: float
    rs_composite: float
    atr14: float
    atr_ratio: float
    max_high_5d_prior: float
    min_low_5d_prior: float
    base_range_5d_prior: float
    volume_ratio: float
    close_location: float


class PreparedVcpData(BaseModel):
    """Prepared pure-OHLCV dataset for Strategy v3 (zero fundamental data)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calendar: pd.DatetimeIndex
    benchmark: BarSeries
    regimes: dict[pd.Timestamp, MarketRegime]
    features: dict[str, pd.DataFrame]
    universe: list[UniverseMember]
    data_stats: dict[str, int | float | str]
    provenance: dict[str, str]
    is_synthetic: bool


def apply_v3_settings(base: Settings, cfg: VcpConfig) -> Settings:
    """Return a new Settings object with v3 risk and execution overrides via DI."""
    return base.model_copy(
        update={
            "risk": base.risk.model_copy(
                update={
                    "atr_stop_multiple": cfg.atr_stop_multiple,
                    "max_stop_pct": cfg.max_stop_pct,
                }
            ),
            "execution": base.execution.model_copy(
                update={
                    "entry_limit_cap_pct": cfg.entry_limit_cap_pct,
                    "slippage_per_side_pct": cfg.slippage_per_side_pct,
                    "commission_per_side_usd": cfg.commission_per_side_usd,
                    "breakeven_trigger_r": cfg.breakeven_trigger_r,
                    "trail_trigger_r": cfg.trail_trigger_r,
                    "trail_atr_multiple": cfg.trail_atr_multiple,
                    "time_stop_days": cfg.time_stop_days,
                    "time_stop_min_r": cfg.time_stop_min_r,
                }
            ),
        }
    )


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def compute_vcp_features(bars: BarSeries, cfg: VcpConfig, rcfg: RiskConfig) -> pd.DataFrame:
    """Compute strictly causal v3 features for one symbol. Row ``t`` uses rows ``<= t`` only."""
    df = bars.frame
    out = pd.DataFrame(index=df.index)
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["volume"] = df["volume"]
    out["raw_close"] = df["raw_close"]
    out["prev_close"] = df["close"].shift(1)

    w = cfg.avg_window
    out["avg_volume_20"] = df["volume"].shift(1).rolling(w, min_periods=w).mean()
    out["avg_dollar_volume_20"] = (df["close"] * df["volume"]).shift(1).rolling(w, min_periods=w).mean()

    out["sma50"] = df["close"].rolling(cfg.sma_fast, min_periods=cfg.sma_fast).mean()
    out["sma150"] = df["close"].rolling(cfg.sma_mid, min_periods=cfg.sma_mid).mean()
    out["sma200"] = df["close"].rolling(cfg.sma_slow, min_periods=cfg.sma_slow).mean()
    out["sma200_prev"] = out["sma200"].shift(cfg.sma_slow_slope_lookback)

    out["high_252d"] = df["high"].rolling(cfg.high_low_window, min_periods=cfg.high_low_window).max()
    out["low_252d"] = df["low"].rolling(cfg.high_low_window, min_periods=cfg.high_low_window).min()

    ret_short = df["close"] / df["close"].shift(cfg.rs_short_window) - 1.0
    ret_long = df["close"] / df["close"].shift(cfg.rs_long_window) - 1.0
    out["ret_63d"] = ret_short
    out["ret_126d"] = ret_long
    out["rs_composite"] = cfg.rs_short_weight * ret_short + cfg.rs_long_weight * ret_long

    # Wilder ATR14 matches risk_manager and manage_position trailing stops.
    out["atr14"] = wilder_atr(df, rcfg.atr_window)

    # Volatility contraction ratio ATR10 / ATR50, evaluated at t - atr_contraction_lag.
    tr = _true_range(df)
    if cfg.atr_contraction_mode == "sma_atr":
        atr_s = tr.rolling(cfg.atr_short_window, min_periods=cfg.atr_short_window).mean()
        atr_l = tr.rolling(cfg.atr_long_window, min_periods=cfg.atr_long_window).mean()
    elif cfg.atr_contraction_mode == "sma_natr":
        tr_pct = tr / out["prev_close"]
        tr_pct.iloc[0] = (df["high"].iloc[0] - df["low"].iloc[0]) / df["close"].iloc[0]
        atr_s = tr_pct.rolling(cfg.atr_short_window, min_periods=cfg.atr_short_window).mean()
        atr_l = tr_pct.rolling(cfg.atr_long_window, min_periods=cfg.atr_long_window).mean()
    else:
        atr_s = wilder_atr(df, cfg.atr_short_window)
        atr_l = wilder_atr(df, cfg.atr_long_window)

    raw_ratio = atr_s / atr_l.replace(0.0, np.nan)
    out["atr10"] = atr_s.shift(cfg.atr_contraction_lag) if cfg.atr_contraction_lag else atr_s
    out["atr50"] = atr_l.shift(cfg.atr_contraction_lag) if cfg.atr_contraction_lag else atr_l
    out["atr_ratio"] = raw_ratio.shift(cfg.atr_contraction_lag) if cfg.atr_contraction_lag else raw_ratio

    # Prior 5-day consolidation base [t - base_window .. t - 1]
    bw = cfg.base_window
    out["max_high_5d_prior"] = df["high"].shift(1).rolling(bw, min_periods=bw).max()
    out["min_low_5d_prior"] = df["low"].shift(1).rolling(bw, min_periods=bw).min()
    out["base_range_5d_prior"] = (out["max_high_5d_prior"] - out["min_low_5d_prior"]) / out["prev_close"]

    out["volume_ratio"] = np.where(out["avg_volume_20"] > 0, df["volume"] / out["avg_volume_20"], np.nan)
    rng = df["high"] - df["low"]
    out["close_location"] = np.where(rng > 0, (df["close"] - df["low"]) / rng, 0.0)
    return out


def vcp_snapshot(symbol: str, ts: pd.Timestamp, row: pd.Series) -> VcpSnapshot:
    """Build an immutable point-in-time snapshot for one symbol at close ``ts``."""
    return VcpSnapshot(
        symbol=symbol,
        date=ts.date(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        prev_close=float(row["prev_close"]),
        raw_close=float(row["raw_close"]),
        volume=float(row["volume"]),
        avg_volume_20=float(row["avg_volume_20"]),
        avg_dollar_volume_20=float(row["avg_dollar_volume_20"]),
        sma50=float(row["sma50"]),
        sma150=float(row["sma150"]),
        sma200=float(row["sma200"]),
        sma200_prev=float(row["sma200_prev"]),
        high_252d=float(row["high_252d"]),
        low_252d=float(row["low_252d"]),
        rs_composite=float(row["rs_composite"]),
        atr14=float(row["atr14"]),
        atr_ratio=float(row["atr_ratio"]),
        max_high_5d_prior=float(row["max_high_5d_prior"]),
        min_low_5d_prior=float(row["min_low_5d_prior"]),
        base_range_5d_prior=float(row["base_range_5d_prior"]),
        volume_ratio=float(row["volume_ratio"]),
        close_location=float(row["close_location"]),
    )


def _make_signal(s: VcpSnapshot, regime: MarketRegime, cfg: VcpConfig) -> TradeSignal:
    """Build a ``TradeSignal`` compatible with untouched ``risk_manager`` & ``BacktestBroker``."""
    # Setting event_day_low = limit makes (limit - event_day_low) == 0 so
    # risk_manager.stop_distance returns min(2.5 * ATR14, 0.15 * limit) cleanly.
    eff_event_low = s.close * (1.0 + cfg.entry_limit_cap_pct) if cfg.pure_atr_stop else s.low
    placeholder_event = EarningsEvent(
        symbol=s.symbol,
        announced_at=datetime(s.date.year, s.date.month, s.date.day, 16, 0),
        timing=AnnouncementTiming.AMC,
        source="vcp_breakout",
    )
    return TradeSignal(
        symbol=s.symbol,
        signal_date=s.date,
        event=placeholder_event,
        regime=regime.state,
        event_day_return=s.close / s.prev_close - 1.0,
        volume_ratio=s.volume_ratio,
        close_location=s.close_location,
        event_day_low=eff_event_low,
        signal_close=s.close,
        atr14=s.atr14,
        market_cap=None,
        rank_score=s.rs_composite,
    )


def screen_vcp_day(
    snapshots: list[VcpSnapshot],
    regime: MarketRegime,
    cfg: VcpConfig,
) -> tuple[list[TradeSignal], dict[str, int]]:
    """Screen all symbols at the close of one session ``t`` (pure, zero look-ahead).

    Cross-sectional RS percentile is computed strictly across the eligible universe
    on day ``t`` (price in ``[min_price, max_price]``, 20d avg dollar volume
    ``>= min_avg_dollar_volume``, and non-NaN ``rs_composite``).
    """
    reasons: dict[str, int] = {}

    def _inc(key: str, n: int = 1) -> None:
        reasons[key] = reasons.get(key, 0) + n

    if regime.state is RegimeState.PANIC:
        if snapshots:
            _inc("regime_panic", len(snapshots))
        return [], reasons

    eligible: list[VcpSnapshot] = []
    for s in snapshots:
        if math.isnan(s.raw_close) or not (cfg.min_price <= s.raw_close <= cfg.max_price):
            _inc("price")
            continue
        if math.isnan(s.avg_dollar_volume_20) or s.avg_dollar_volume_20 < cfg.min_avg_dollar_volume:
            _inc("liquidity")
            continue
        if math.isnan(s.rs_composite):
            _inc("insufficient_rs_history")
            continue
        eligible.append(s)

    if not eligible:
        return [], reasons

    # Strictly cross-sectional percentile rank on day t across eligible stocks
    rs_series = pd.Series([s.rs_composite for s in eligible])
    rs_pct = rs_series.rank(method="average", pct=True).to_numpy()

    signals: list[TradeSignal] = []
    for idx, s in enumerate(eligible):
        if rs_pct[idx] < cfg.rs_top_quantile:
            _inc("rs_below_top15")
            continue

        needed = (
            s.prev_close,
            s.sma50,
            s.sma150,
            s.sma200,
            s.sma200_prev,
            s.high_252d,
            s.low_252d,
            s.atr14,
            s.atr_ratio,
            s.max_high_5d_prior,
            s.min_low_5d_prior,
            s.base_range_5d_prior,
            s.volume_ratio,
        )
        if any(math.isnan(x) for x in needed) or s.atr14 <= 0:
            _inc("insufficient_history")
            continue

        if not (s.close > s.sma50 > s.sma150 > s.sma200):
            _inc("stage2_ma_stack")
            continue
        if not (s.sma200 > s.sma200_prev):
            _inc("stage2_sma200_slope")
            continue
        if s.close < cfg.min_pct_of_52w_high * s.high_252d:
            _inc("stage2_52w_high")
            continue
        if s.close < cfg.min_multiple_of_52w_low * s.low_252d:
            _inc("stage2_52w_low")
            continue

        if s.atr_ratio > cfg.max_atr_ratio:
            _inc("vcp_atr_ratio")
            continue
        if s.base_range_5d_prior > cfg.max_base_range_pct:
            _inc("vcp_base_range")
            continue

        if not (s.close > s.max_high_5d_prior):
            _inc("breakout_price")
            continue
        if s.volume_ratio < cfg.min_volume_ratio:
            _inc("breakout_volume")
            continue
        if s.close_location < cfg.min_close_location:
            _inc("breakout_close_loc")
            continue

        signals.append(_make_signal(s, regime, cfg))
        _inc("ok")

    signals.sort(key=lambda sig: (-sig.rank_score, sig.symbol))
    return signals, reasons


def prepare_vcp_data(
    settings: Settings,
    provider: MarketDataProvider,
    cfg: VcpConfig,
    max_symbols: int | None = None,
) -> PreparedVcpData:
    """Load cached OHLCV bars + SPY benchmark and compute strictly causal v3 features."""
    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    universe = [m for m in provider.get_universe(bt.end_date) if not m.is_etf]
    if max_symbols:
        universe = universe[:max_symbols]
    symbols = [m.symbol for m in universe]
    log.info("v3 universe candidates: %d", len(symbols))

    benchmark = provider.get_benchmark_bars(fetch_start, bt.end_date, bt.benchmark_symbol)
    bars = provider.get_daily_bars(symbols, fetch_start, bt.end_date)

    eff_settings = apply_v3_settings(settings, cfg)
    features = {s: compute_vcp_features(b, cfg, eff_settings.risk) for s, b in bars.items()}

    full_cal = benchmark.frame.index
    calendar = full_cal[(full_cal >= pd.Timestamp(bt.start_date)) & (full_cal <= pd.Timestamp(bt.end_date))]
    regimes = compute_regimes(benchmark, eff_settings.regime)

    pre_start_bars = [int((b.frame.index < pd.Timestamp(bt.start_date)).sum()) for b in bars.values()]
    removed = [m for m in universe if m.removed_on is not None]
    stats: dict[str, int | float | str] = {
        "universe_candidates": len(universe),
        "removed_index_members_in_candidates": len(removed),
        "symbols_with_bars": len(bars),
        "removed_members_with_bars": sum(1 for m in removed if m.symbol in bars),
        "symbols_missing_bars": len(universe) - len(bars),
        "symbols_with_252d_warmup": sum(1 for n in pre_start_bars if n >= cfg.high_low_window),
        "benchmark_warmup_bars": int((full_cal < pd.Timestamp(bt.start_date)).sum()),
        "sessions": len(calendar),
        "first_session": str(calendar[0].date()) if len(calendar) else "",
        "last_session": str(calendar[-1].date()) if len(calendar) else "",
    }
    return PreparedVcpData(
        calendar=calendar,
        benchmark=benchmark,
        regimes=regimes,
        features=features,
        universe=universe,
        data_stats=stats,
        provenance=provider.describe(),
        is_synthetic=provider.is_synthetic,
    )


def slice_vcp_data(data: PreparedVcpData, start_date: date, end_date: date) -> PreparedVcpData:
    """Return a view of ``PreparedVcpData`` restricted to calendar ``[start_date, end_date]``.

    Because every column in ``data.features`` and ``data.regimes`` is strictly causal
    (truncation-invariant), slicing the simulation calendar introduces zero look-ahead.
    """
    full_cal = data.benchmark.frame.index
    cal = full_cal[(full_cal >= pd.Timestamp(start_date)) & (full_cal <= pd.Timestamp(end_date))]
    stats = dict(data.data_stats)
    stats["sessions"] = len(cal)
    stats["first_session"] = str(cal[0].date()) if len(cal) else ""
    stats["last_session"] = str(cal[-1].date()) if len(cal) else ""
    return data.model_copy(update={"calendar": cal, "data_stats": stats})


def screen_all_sessions(
    data: PreparedVcpData,
    cfg: VcpConfig,
) -> tuple[dict[pd.Timestamp, list[TradeSignal]], dict[str, int], dict[pd.Timestamp, int]]:
    """Vectorised cross-sectional screening across ``data.calendar`` (100% equivalent to ``screen_vcp_day``).

    Returns ``(signals_by_date, rejection_counts, eligible_count_by_date)``.
    Every day ``t`` uses ONLY row ``t`` of strictly causal features (zero look-ahead).
    """
    cal = data.calendar
    syms = sorted(data.features.keys())
    if not len(cal) or not syms:
        return {}, {}, {}

    def _panel(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: data.features[s][col] for s in syms}).reindex(cal)

    close = _panel("close")
    present = close.notna()
    raw_close = _panel("raw_close")
    avg_dv20 = _panel("avg_dollar_volume_20")
    rs_comp = _panel("rs_composite")

    rejections: dict[str, int] = defaultdict(int)
    panic_mask = pd.Series(
        [data.regimes[ts].state is RegimeState.PANIC for ts in cal], index=cal
    )
    not_panic = ~panic_mask

    n_panic = int(present.loc[panic_mask].sum().sum())
    if n_panic:
        rejections["regime_panic"] += n_panic

    active = present.mul(not_panic, axis=0)
    price_ok = active & raw_close.notna() & (raw_close >= cfg.min_price) & (raw_close <= cfg.max_price)
    n_price = int((active & ~price_ok).sum().sum())
    if n_price:
        rejections["price"] += n_price

    liq_ok = price_ok & avg_dv20.notna() & (avg_dv20 >= cfg.min_avg_dollar_volume)
    n_liq = int((price_ok & ~liq_ok).sum().sum())
    if n_liq:
        rejections["liquidity"] += n_liq

    elig = liq_ok & rs_comp.notna()
    n_no_rs = int((liq_ok & ~elig).sum().sum())
    if n_no_rs:
        rejections["insufficient_rs_history"] += n_no_rs

    elig_counts = {ts: int(c) for ts, c in elig.sum(axis=1).items()}

    # Cross-sectional RS percentile rank strictly across eligible stocks on each day t
    rs_pct = rs_comp.where(elig).rank(axis=1, method="average", pct=True)
    top_rs = elig & (rs_pct >= cfg.rs_top_quantile)
    n_low_rs = int((elig & ~top_rs).sum().sum())
    if n_low_rs:
        rejections["rs_below_top15"] += n_low_rs

    # Evaluate remaining filters strictly in the same order as screen_vcp_day
    prev_close = _panel("prev_close")
    sma50 = _panel("sma50")
    sma150 = _panel("sma150")
    sma200 = _panel("sma200")
    sma200_prev = _panel("sma200_prev")
    high_252d = _panel("high_252d")
    low_252d = _panel("low_252d")
    atr14 = _panel("atr14")
    atr_ratio = _panel("atr_ratio")
    max_h5 = _panel("max_high_5d_prior")
    min_l5 = _panel("min_low_5d_prior")
    base_rng = _panel("base_range_5d_prior")
    vol_ratio = _panel("volume_ratio")
    close_loc = _panel("close_location")

    has_hist = (
        top_rs
        & prev_close.notna()
        & sma50.notna()
        & sma150.notna()
        & sma200.notna()
        & sma200_prev.notna()
        & high_252d.notna()
        & low_252d.notna()
        & atr14.notna()
        & (atr14 > 0)
        & atr_ratio.notna()
        & max_h5.notna()
        & min_l5.notna()
        & base_rng.notna()
        & vol_ratio.notna()
    )

    steps: list[tuple[str, pd.DataFrame]] = [
        ("insufficient_history", has_hist),
        ("stage2_ma_stack", has_hist & (close > sma50) & (sma50 > sma150) & (sma150 > sma200)),
        ("stage2_sma200_slope", (sma200 > sma200_prev)),
        ("stage2_52w_high", (close >= cfg.min_pct_of_52w_high * high_252d)),
        ("stage2_52w_low", (close >= cfg.min_multiple_of_52w_low * low_252d)),
        ("vcp_atr_ratio", (atr_ratio <= cfg.max_atr_ratio)),
        ("vcp_base_range", (base_rng <= cfg.max_base_range_pct)),
        ("breakout_price", (close > max_h5)),
        ("breakout_volume", (vol_ratio >= cfg.min_volume_ratio)),
        ("breakout_close_loc", (close_loc >= cfg.min_close_location)),
    ]

    cur = top_rs
    for reason, cond in steps:
        nxt = cur & cond
        dropped = int((cur & ~nxt).sum().sum())
        if dropped:
            rejections[reason] += dropped
        cur = nxt

    signals_by_date: dict[pd.Timestamp, list[TradeSignal]] = defaultdict(list)
    passed = cur.stack()
    passed = passed[passed]
    for ts, sym in passed.index:
        snap = vcp_snapshot(sym, ts, data.features[sym].loc[ts])
        signals_by_date[ts].append(_make_signal(snap, data.regimes[ts], cfg))
        rejections["ok"] += 1

    for ts in signals_by_date:
        signals_by_date[ts].sort(key=lambda sig: (-sig.rank_score, sig.symbol))

    return dict(signals_by_date), dict(rejections), elig_counts


def run_backtest_v3(
    data: PreparedVcpData,
    settings: Settings,
    cfg: VcpConfig,
    initial_capital_usd: float | None = None,
    vps_enabled: bool | None = None,
    precomputed_signals: tuple[dict[pd.Timestamp, list[TradeSignal]], dict[str, int]] | None = None,
) -> BacktestResult:
    """Event-driven daily backtest loop for Strategy v3 (zero look-ahead).

    Reuses untouched ``BacktestBroker``, ``manage_position``, and
    ``risk_manager.build_order_intent`` with ``Settings`` configured via DI.
    """
    eff_settings = apply_v3_settings(settings, cfg)
    cash0 = initial_capital_usd if initial_capital_usd is not None else eff_settings.backtest.initial_capital_usd
    vps = eff_settings.execution.vps_enabled if vps_enabled is None else vps_enabled
    cal = data.calendar
    broker = BacktestBroker(data.benchmark.frame.index, cash0, eff_settings.execution)

    if precomputed_signals is not None:
        signals_by_date, rejections = precomputed_signals
        rejections = dict(rejections)
    else:
        signals_by_date, rejections, _ = screen_all_sessions(data, cfg)

    sizing_rej: dict[str, int] = defaultdict(int)
    rows = []
    last_month: tuple[int, int] | None = None
    bench_close = data.benchmark.frame["close"]

    def bar(sym: str, ts: pd.Timestamp) -> pd.Series | None:
        f = data.features.get(sym)
        if f is None or ts not in f.index:
            return None
        return f.loc[ts]

    for i, ts in enumerate(cal):
        month = (ts.year, ts.month)
        if vps and month != last_month:
            broker.charge_fee(eff_settings.execution.vps_monthly_eur * eff_settings.backtest.eur_usd_rate)
        last_month = month

        active = set(broker.positions) | set(broker.pending_orders)
        today = {s: b for s in active if (b := bar(s, ts)) is not None}
        broker.process_session(ts, today)

        for sym, p in list(broker.positions.items()):
            row = today.get(sym)
            if row is None:
                f = data.features[sym]
                if ts > f.index[-1]:
                    broker.force_exit(ts, sym, float(f["close"].iloc[-1]), ExitReason.DATA_END)
                continue
            newp, exit_reason = manage_position(p, row, eff_settings)
            broker.positions[sym] = newp
            if exit_reason is not None:
                broker.close_position(sym, exit_reason)

        regime = data.regimes[ts]
        next_ts = cal[i + 1] if i + 1 < len(cal) else None
        acct = broker.get_account_state()
        signals = signals_by_date.get(ts, [])

        if signals and next_ts is not None:
            slots = regime.max_positions - len(broker.positions) - len(broker.pending_orders)
            avail = broker.ledger.settled_by(next_ts)
            for sig in signals:
                if slots <= 0:
                    sizing_rej["no_free_slot"] += 1
                    continue
                if sig.symbol in broker.positions or sig.symbol in broker.pending_orders:
                    sizing_rej["already_held"] += 1
                    continue
                intent, reason = build_order_intent(
                    sig, regime, acct.equity, avail, eff_settings.risk, eff_settings.execution
                )
                if intent is None:
                    sizing_rej[reason] += 1
                    continue
                broker.submit_bracket_order(intent)
                avail -= intent.shares * intent.limit_price + eff_settings.execution.commission_per_side_usd
                slots -= 1

        acct = broker.get_account_state()
        rows.append(
            {
                "date": ts,
                "equity": acct.equity,
                "settled_cash": acct.settled_cash,
                "unsettled_cash": acct.unsettled_cash,
                "market_value": acct.market_value,
                "open_positions": acct.open_positions,
                "regime": regime.state.value,
                "benchmark_close": float(bench_close.loc[ts]),
            }
        )

    last = cal[-1]
    if eff_settings.backtest.force_close_at_end:
        for sym in list(broker.positions):
            f = data.features[sym]
            px = float(f["close"].loc[:last].iloc[-1])
            broker.force_exit(last, sym, px, ExitReason.END_OF_TEST)
        acct = broker.get_account_state()
        rows[-1].update(
            equity=acct.equity,
            settled_cash=acct.settled_cash,
            unsettled_cash=acct.unsettled_cash,
            market_value=acct.market_value,
            open_positions=acct.open_positions,
        )

    curve = pd.DataFrame(rows).set_index("date")
    return BacktestResult(
        initial_capital_usd=cash0,
        equity_curve=curve,
        trades=broker.closed_trades,
        skipped=broker.skipped,
        rejections=rejections,
        sizing_rejections=dict(sizing_rej),
        fees_paid=broker.fees_paid,
        vps_enabled=vps,
        is_synthetic=data.is_synthetic,
    )
