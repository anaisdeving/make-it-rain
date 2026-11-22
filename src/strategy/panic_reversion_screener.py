"""Strategy v4a: Uptrend Panic Mean-Reversion (Pure OHLCV, Zero Look-Ahead).

Structurally opposite to breakout/momentum strategies: buys extreme 3-day panic
selloffs inside established long-term uptrends (``Close > SMA200``) and exits on
short-term mean reversion (``Close > SMA5``) or a 10-trading-day time stop.

Guardrails:
- Leaves ``src/risk/risk_manager.py`` and ``src/execution/`` completely untouched.
- Directly reuses ``SettlementLedger`` (T+1 settled cash, zero Good Faith
  Violations) and ``position_size`` (whole shares, risk-fraction & notional caps)
  from ``src/risk/risk_manager.py``.
- Strictly causal features (row ``t`` uses bars ``<= t`` only):
  - Trailing 20d avg dollar volume: ``(close * volume).rolling(20).mean()`` on ``[t-19..t]``
  - ``SMA200`` and ``SMA5``: trailing simple moving averages on ``close``
  - ``ATR14``: Wilder ATR(14) via ``src.strategy.indicators.wilder_atr``
  - ``RSI(2)``: Wilder EWM RSI(2) (``alpha = 1/2, adjust=False, min_periods=2``;
    standard Connors RSI-2 convention, avoiding Cutler/simple RSI's degenerate
    ties at 0.0 whenever two consecutive down days occur; ``simple`` is also
    supported for diagnostics)
  - 3-day return: ``Close_t / Close_{t-3} - 1 <= -2.5 * (ATR14_t / Close_t)``
- Realistic execution:
  - Entry at ``t+1`` Open with ``0.08%`` slippage + ``$0.38`` IBKR commission.
  - Catastrophe stop ``3.0 * ATR14`` below entry fill, honoring overnight
    gap-downs worse than stop (``open <= stop -> fill at open * (1 - slip)``).
  - Exit at ``tau+1`` Open the morning after ``Close_tau > SMA5_tau``, or at
    ``tau+1`` Open after ``bars_held >= 10`` (zero look-ahead).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.backtest.engine import BacktestResult
from src.config import RiskConfig, Settings
from src.data.provider_base import MarketDataProvider
from src.domain.models import (
    AnnouncementTiming,
    BarSeries,
    ClosedTrade,
    EarningsEvent,
    ExitReason,
    MarketRegime,
    Position,
    RegimeState,
    SkippedEntry,
    TradeSignal,
    UniverseMember,
)
from src.risk.risk_manager import SettlementLedger, position_size
from src.strategy.indicators import wilder_atr
from src.strategy.regime_filter import compute_regimes

log = logging.getLogger(__name__)


class PanicReversionConfig(BaseModel):
    """Locked configuration for Strategy v4a (Uptrend Panic Mean-Reversion)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Point-in-time universe filters
    min_price: float = Field(15.0, gt=0)
    min_avg_dollar_volume: float = Field(25_000_000.0, ge=0)
    avg_window: int = Field(20, ge=1)

    # Signal thresholds (strictly <= t)
    trend_sma_window: int = Field(200, ge=1)
    exit_sma_window: int = Field(5, ge=1)
    rsi_window: int = Field(2, ge=1)
    rsi_method: Literal["wilder", "simple"] = "wilder"
    max_rsi: float = Field(5.0, gt=0, lt=100)
    drop_window: int = Field(3, ge=1)
    drop_atr_multiple: float = Field(2.5, gt=0)
    atr_window: int = Field(14, ge=1)

    # Portfolio, sizing, stops & frictions
    max_positions: int = Field(3, ge=1)
    risk_per_trade: float = Field(0.05, gt=0, le=0.25)
    max_position_pct_equity: float = Field(0.50, gt=0, le=1.0)
    atr_stop_multiple: float = Field(3.0, gt=0)
    max_hold_days: int = Field(10, ge=1)
    slippage_per_side_pct: float = Field(0.0008, ge=0.0)
    commission_per_side_usd: float = Field(0.38, ge=0.0)
    settlement_days: int = Field(1, ge=0)


class PanicSnapshot(BaseModel):
    """Point-in-time feature snapshot for one symbol at session ``date`` Close."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    raw_close: float
    volume: float
    avg_dollar_volume_20: float
    sma5: float
    sma200: float
    atr14: float
    ret_3d: float
    rsi2: float


class PreparedPanicData(BaseModel):
    """Prepared pure-OHLCV dataset for Strategy v4a."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calendar: pd.DatetimeIndex
    benchmark: BarSeries
    regimes: dict[pd.Timestamp, MarketRegime]
    features: dict[str, pd.DataFrame]
    universe: list[UniverseMember]
    data_stats: dict[str, int | float | str]
    provenance: dict[str, str]
    is_synthetic: bool


class PanicBacktestResult(BacktestResult):
    """BacktestResult augmented with exact turnover and friction accounting."""

    total_buy_notional: float = 0.0
    total_sell_notional: float = 0.0
    total_commissions_usd: float = 0.0
    total_slippage_usd: float = 0.0


def compute_rsi(
    close: pd.Series,
    window: int = 2,
    method: Literal["wilder", "simple"] = "wilder",
) -> pd.Series:
    """Compute strictly causal RSI(window) using Wilder EWM or Cutler simple SMA.

    - ``wilder``: ``ewm(alpha=1/window, adjust=False, min_periods=window).mean()``
      on gains and losses. Standard Connors RSI-2 convention; avoids ties at 0.0.
    - ``simple``: ``rolling(window, min_periods=window).mean()`` (Cutler's RSI).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    if method == "wilder":
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    else:
        avg_gain = gain.rolling(window, min_periods=window).mean()
        avg_loss = loss.rolling(window, min_periods=window).mean()

    valid = avg_gain.notna() & avg_loss.notna()
    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    both_zero = valid & (avg_gain == 0.0) & (avg_loss == 0.0)
    loss_zero = valid & (avg_loss == 0.0) & (avg_gain > 0.0)
    normal = valid & (avg_loss > 0.0)
    rsi.loc[both_zero] = 50.0
    rsi.loc[loss_zero] = 100.0
    rs = avg_gain.loc[normal] / avg_loss.loc[normal]
    rsi.loc[normal] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_panic_features(bars: BarSeries, cfg: PanicReversionConfig) -> pd.DataFrame:
    """Compute strictly causal v4a features for one symbol. Row ``t`` uses rows ``<= t`` only."""
    df = bars.frame
    out = pd.DataFrame(index=df.index)
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["volume"] = df["volume"]
    out["raw_close"] = df["raw_close"]

    dv = df["close"] * df["volume"]
    out["avg_dollar_volume_20"] = dv.rolling(cfg.avg_window, min_periods=cfg.avg_window).mean()
    out["sma5"] = df["close"].rolling(cfg.exit_sma_window, min_periods=cfg.exit_sma_window).mean()
    out["sma200"] = df["close"].rolling(cfg.trend_sma_window, min_periods=cfg.trend_sma_window).mean()
    out["atr14"] = wilder_atr(df, cfg.atr_window)
    out["ret_3d"] = df["close"] / df["close"].shift(cfg.drop_window) - 1.0
    out["rsi2"] = compute_rsi(df["close"], window=cfg.rsi_window, method=cfg.rsi_method)
    return out


def panic_snapshot(symbol: str, ts: pd.Timestamp, row: pd.Series) -> PanicSnapshot:
    """Build an immutable point-in-time snapshot at session ``ts`` Close."""
    return PanicSnapshot(
        symbol=symbol,
        date=ts.date(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        raw_close=float(row["raw_close"]),
        volume=float(row["volume"]),
        avg_dollar_volume_20=float(row["avg_dollar_volume_20"]),
        sma5=float(row["sma5"]),
        sma200=float(row["sma200"]),
        atr14=float(row["atr14"]),
        ret_3d=float(row["ret_3d"]),
        rsi2=float(row["rsi2"]),
    )


def _make_panic_signal(s: PanicSnapshot, regime: MarketRegime) -> TradeSignal:
    """Build a ``TradeSignal`` ranked by lowest RSI(2) (``rank_score = -rsi2``)."""
    placeholder_event = EarningsEvent(
        symbol=s.symbol,
        announced_at=datetime(s.date.year, s.date.month, s.date.day, 16, 0),
        timing=AnnouncementTiming.AMC,
        source="v4a_panic_reversion",
    )
    rng = s.high - s.low
    cloc = (s.close - s.low) / rng if rng > 0 else 0.0
    return TradeSignal(
        symbol=s.symbol,
        signal_date=s.date,
        event=placeholder_event,
        regime=regime.state,
        event_day_return=s.ret_3d,
        volume_ratio=1.0,
        close_location=cloc,
        event_day_low=s.low,
        signal_close=s.close,
        atr14=s.atr14,
        market_cap=None,
        rank_score=-s.rsi2,
    )


def screen_panic_day(
    snapshots: list[PanicSnapshot],
    regime: MarketRegime,
    cfg: PanicReversionConfig,
) -> tuple[list[TradeSignal], dict[str, int]]:
    """Screen all symbols at the close of one session ``t`` (pure, zero look-ahead).

    Candidates are ranked by lowest ``RSI(2)`` (tiebreak ascending symbol).
    """
    reasons: dict[str, int] = {}

    def _inc(key: str, n: int = 1) -> None:
        reasons[key] = reasons.get(key, 0) + n

    signals: list[TradeSignal] = []
    for s in snapshots:
        if math.isnan(s.raw_close) or s.raw_close < cfg.min_price:
            _inc("price")
            continue
        if math.isnan(s.avg_dollar_volume_20) or s.avg_dollar_volume_20 < cfg.min_avg_dollar_volume:
            _inc("liquidity")
            continue
        if (
            math.isnan(s.sma200)
            or math.isnan(s.rsi2)
            or math.isnan(s.ret_3d)
            or math.isnan(s.atr14)
            or s.atr14 <= 0
            or s.close <= 0
        ):
            _inc("insufficient_history")
            continue
        if not (s.close > s.sma200):
            _inc("below_sma200")
            continue
        if not (s.rsi2 < cfg.max_rsi):
            _inc("rsi_too_high")
            continue
        if not (s.ret_3d <= -cfg.drop_atr_multiple * (s.atr14 / s.close)):
            _inc("drop_too_small")
            continue

        signals.append(_make_panic_signal(s, regime))
        _inc("ok")

    signals.sort(key=lambda sig: (-sig.rank_score, sig.symbol))
    return signals, reasons


def build_panic_data_from_bars(
    bars: dict[str, BarSeries],
    benchmark: BarSeries,
    universe: list[UniverseMember],
    settings: Settings,
    cfg: PanicReversionConfig,
    provenance: dict[str, str],
    is_synthetic: bool,
) -> PreparedPanicData:
    """Compute v4a features from already-loaded OHLCV bars."""
    bt = settings.backtest
    features = {s: compute_panic_features(b, cfg) for s, b in bars.items()}
    full_cal = benchmark.frame.index
    calendar = full_cal[(full_cal >= pd.Timestamp(bt.start_date)) & (full_cal <= pd.Timestamp(bt.end_date))]
    regimes = compute_regimes(benchmark, settings.regime)

    removed = [m for m in universe if m.removed_on is not None]
    stats: dict[str, int | float | str] = {
        "universe_candidates": len(universe),
        "removed_index_members_in_candidates": len(removed),
        "symbols_with_bars": len(bars),
        "removed_members_with_bars": sum(1 for m in removed if m.symbol in bars),
        "symbols_missing_bars": len(universe) - len(bars),
        "benchmark_warmup_bars": int((full_cal < pd.Timestamp(bt.start_date)).sum()),
        "sessions": len(calendar),
        "first_session": str(calendar[0].date()) if len(calendar) else "",
        "last_session": str(calendar[-1].date()) if len(calendar) else "",
    }
    return PreparedPanicData(
        calendar=calendar,
        benchmark=benchmark,
        regimes=regimes,
        features=features,
        universe=universe,
        data_stats=stats,
        provenance=provenance,
        is_synthetic=is_synthetic,
    )


def prepare_panic_data(
    settings: Settings,
    provider: MarketDataProvider,
    cfg: PanicReversionConfig,
    max_symbols: int | None = None,
) -> PreparedPanicData:
    """Load cached OHLCV bars + SPY benchmark and compute strictly causal v4a features."""
    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    universe = [m for m in provider.get_universe(bt.end_date) if not m.is_etf]
    if max_symbols:
        universe = universe[:max_symbols]
    symbols = [m.symbol for m in universe]
    log.info("v4a universe candidates: %d", len(symbols))

    benchmark = provider.get_benchmark_bars(fetch_start, bt.end_date, bt.benchmark_symbol)
    bars = provider.get_daily_bars(symbols, fetch_start, bt.end_date)
    return build_panic_data_from_bars(
        bars=bars,
        benchmark=benchmark,
        universe=universe,
        settings=settings,
        cfg=cfg,
        provenance=provider.describe(),
        is_synthetic=provider.is_synthetic,
    )


def slice_panic_data(data: PreparedPanicData, start_date: date, end_date: date) -> PreparedPanicData:
    """Restrict ``PreparedPanicData`` to calendar ``[start_date, end_date]`` (zero look-ahead)."""
    full_cal = data.benchmark.frame.index
    cal = full_cal[(full_cal >= pd.Timestamp(start_date)) & (full_cal <= pd.Timestamp(end_date))]
    stats = dict(data.data_stats)
    stats["sessions"] = len(cal)
    stats["first_session"] = str(cal[0].date()) if len(cal) else ""
    stats["last_session"] = str(cal[-1].date()) if len(cal) else ""
    return data.model_copy(update={"calendar": cal, "data_stats": stats})


def screen_panic_sessions(
    data: PreparedPanicData,
    cfg: PanicReversionConfig,
) -> tuple[dict[pd.Timestamp, list[TradeSignal]], dict[str, int], dict[pd.Timestamp, int]]:
    """Vectorised screening across ``data.calendar`` (100% equivalent to ``screen_panic_day``)."""
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
    sma200 = _panel("sma200")
    rsi2 = _panel("rsi2")
    ret_3d = _panel("ret_3d")
    atr14 = _panel("atr14")

    rejections: dict[str, int] = defaultdict(int)

    price_ok = present & raw_close.notna() & (raw_close >= cfg.min_price)
    n_price = int((present & ~price_ok).sum().sum())
    if n_price:
        rejections["price"] += n_price

    liq_ok = price_ok & avg_dv20.notna() & (avg_dv20 >= cfg.min_avg_dollar_volume)
    n_liq = int((price_ok & ~liq_ok).sum().sum())
    if n_liq:
        rejections["liquidity"] += n_liq

    elig_counts = {ts: int(c) for ts, c in liq_ok.sum(axis=1).items()}

    has_hist = (
        liq_ok
        & sma200.notna()
        & rsi2.notna()
        & ret_3d.notna()
        & atr14.notna()
        & (atr14 > 0.0)
        & (close > 0.0)
    )
    steps: list[tuple[str, pd.DataFrame]] = [
        ("insufficient_history", has_hist),
        ("below_sma200", has_hist & (close > sma200)),
        ("rsi_too_high", rsi2 < cfg.max_rsi),
        ("drop_too_small", ret_3d <= -cfg.drop_atr_multiple * (atr14 / close)),
    ]

    cur = liq_ok
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
        snap = panic_snapshot(sym, ts, data.features[sym].loc[ts])
        signals_by_date[ts].append(_make_panic_signal(snap, data.regimes[ts]))
        rejections["ok"] += 1

    for ts in signals_by_date:
        signals_by_date[ts].sort(key=lambda sig: (-sig.rank_score, sig.symbol))

    return dict(signals_by_date), dict(rejections), elig_counts


def run_backtest_v4a(
    data: PreparedPanicData,
    settings: Settings,
    cfg: PanicReversionConfig,
    initial_capital_usd: float | None = None,
    vps_enabled: bool | None = None,
    precomputed_signals: tuple[dict[pd.Timestamp, list[TradeSignal]], dict[str, int]] | None = None,
) -> PanicBacktestResult:
    """Event-driven daily backtest for Strategy v4a (zero look-ahead, T+1 settled cash).

    Timeline on session ``ts``:
    1. Monthly VPS fee (if enabled) on first session of each month.
    2. ``ledger.roll(ts)`` settles T+1 sale proceeds.
    3. Market Open exits: overnight gap-downs ``open <= stop_price`` exit at Open
       (``GAP_STOP``); queued ``Close > SMA5`` (``SIGNAL_EXIT``) or 10d max hold
       (``TIME_STOP``) exit at Open. Sale proceeds settle T+1.
    4. Market Open entries for signals queued at ``t-1`` Close: fill at
       ``open * (1 + 0.08%)`` + ``$0.38`` commission, whole shares sized via
       untouched ``risk_manager.position_size`` using ``signal_equity`` (from ``t-1``
       Close), ``d = 3.0 * ATR14_{t-1}``, risk ``f``, and ``ledger.settled`` (never
       using unsettled T+1 proceeds).
    5. Intraday catastrophe stop: ``low <= stop_price`` exits at ``stop_price * (1 - 0.08%)``.
    6. Post-close exit check: increment ``bars_held``; if ``Close > SMA5`` queue
       ``SIGNAL_EXIT`` for ``ts+1`` Open; else if ``bars_held >= 10`` queue ``TIME_STOP``.
    7. Screen signals at ``ts`` Close (ranked by lowest ``RSI(2)``) for ``ts+1`` Open.
    """
    cash0 = initial_capital_usd if initial_capital_usd is not None else settings.backtest.initial_capital_usd
    vps = settings.execution.vps_enabled if vps_enabled is None else vps_enabled
    cal = data.calendar
    if not len(cal):
        raise ValueError("empty simulation calendar")

    if precomputed_signals is not None:
        signals_by_date, rejections = precomputed_signals
        rejections = dict(rejections)
    else:
        signals_by_date, rejections, _ = screen_panic_sessions(data, cfg)

    rcfg = RiskConfig(
        base_risk_per_trade=cfg.risk_per_trade,
        max_risk_per_trade=max(cfg.risk_per_trade, settings.risk.max_risk_per_trade),
        atr_window=cfg.atr_window,
        atr_stop_multiple=cfg.atr_stop_multiple,
        max_stop_pct=settings.risk.max_stop_pct,
        max_position_pct_equity=cfg.max_position_pct_equity,
    )

    ledger = SettlementLedger(
        calendar=data.benchmark.frame.index,
        settled=cash0,
        settlement_days=cfg.settlement_days,
    )
    positions: dict[str, Position] = {}
    marks: dict[str, float] = {}
    closed_trades: list[ClosedTrade] = []
    skipped: list[SkippedEntry] = []
    sizing_rej: dict[str, int] = defaultdict(int)

    fees_paid = 0.0
    total_buy_notional = 0.0
    total_sell_notional = 0.0
    total_commissions = 0.0
    total_slippage = 0.0

    pending_signals: list[TradeSignal] = []
    pending_signal_equity: float = cash0

    slip = cfg.slippage_per_side_pct
    comm = cfg.commission_per_side_usd
    bench_close = data.benchmark.frame["close"]

    def get_bar(sym: str, ts: pd.Timestamp) -> pd.Series | None:
        f = data.features.get(sym)
        if f is None or ts not in f.index:
            return None
        return f.loc[ts]

    def close_pos(ts: pd.Timestamp, p: Position, raw_px: float, reason: ExitReason) -> None:
        nonlocal total_sell_notional, total_commissions, total_slippage
        fill_px = raw_px * (1.0 - slip)
        slip_usd = p.shares * (raw_px - fill_px)
        proceeds = p.shares * fill_px - comm
        ledger.credit_sale(ts, proceeds)

        total_sell_notional += p.shares * fill_px
        total_commissions += comm
        total_slippage += slip_usd

        gross = p.shares * (fill_px - p.entry_price)
        commissions = p.entry_commission + comm
        net = gross - commissions
        risk_usd = p.shares * (p.entry_price - p.initial_stop)
        sig = p.signal
        closed_trades.append(
            ClosedTrade(
                symbol=p.symbol,
                signal_date=sig.signal_date,
                entry_date=p.entry_date,
                exit_date=ts.date(),
                shares=p.shares,
                entry_price=p.entry_price,
                exit_price=fill_px,
                initial_stop=p.initial_stop,
                exit_reason=reason,
                bars_held=p.bars_held,
                gross_pnl=gross,
                commissions=commissions,
                net_pnl=net,
                initial_risk_usd=risk_usd,
                r_multiple=net / risk_usd if risk_usd > 0 else 0.0,
                regime_at_entry=p.regime_at_entry,
                event_timing=sig.event.timing,
                eps_surprise_pct=sig.event.eps_surprise_pct,
                event_day_return=sig.event_day_return,
            )
        )
        del positions[p.symbol]
        marks.pop(p.symbol, None)

    def current_equity() -> tuple[float, float]:
        mv = sum(p.shares * marks.get(s, p.entry_price) for s, p in positions.items())
        return ledger.settled + ledger.unsettled + mv, mv

    rows: list[dict[str, object]] = []
    last_month: tuple[int, int] | None = None

    for i, ts in enumerate(cal):
        month = (ts.year, ts.month)
        if vps and month != last_month:
            fee = settings.execution.vps_monthly_eur * settings.backtest.eur_usd_rate
            ledger.debit_fee(fee)
            fees_paid += fee
        last_month = month

        # 1. T+1 Settlement at start of session
        ledger.roll(ts)

        # 2. Open exits & overnight gap-down catastrophe stops on carried positions
        for sym, p in list(positions.items()):
            row = get_bar(sym, ts)
            if row is None:
                f = data.features[sym]
                if ts > f.index[-1]:
                    close_pos(ts, p, float(f["close"].iloc[-1]), ExitReason.DATA_END)
                continue
            open_px = float(row["open"])
            if open_px <= p.stop_price:
                close_pos(ts, p, open_px, ExitReason.GAP_STOP)
            elif p.pending_exit is not None:
                close_pos(ts, p, open_px, p.pending_exit)

        # 3. Open entries for signals queued at t-1 Close
        if pending_signals:
            queued = pending_signals
            sig_eq = pending_signal_equity
            pending_signals = []
            for sig in queued:
                if len(positions) >= cfg.max_positions:
                    sizing_rej["no_free_slot"] += 1
                    continue
                if sig.symbol in positions:
                    sizing_rej["already_held"] += 1
                    continue
                row = get_bar(sig.symbol, ts)
                if row is None:
                    skipped.append(
                        SkippedEntry(
                            symbol=sig.symbol,
                            signal_date=sig.signal_date,
                            attempt_date=ts.date(),
                            reason="no_bar",
                        )
                    )
                    continue
                open_px = float(row["open"])
                fill_px = open_px * (1.0 + slip)
                dist = min(cfg.atr_stop_multiple * sig.atr14, 0.90 * fill_px)
                stop_px = fill_px - dist

                shares = position_size(
                    equity=sig_eq,
                    entry=fill_px,
                    distance=dist,
                    fraction=cfg.risk_per_trade,
                    settled_cash=ledger.settled,
                    commission=comm,
                    cfg=rcfg,
                )
                if shares < 1:
                    reason = "insufficient_settled_cash" if ledger.settled < fill_px + comm else "size_below_one_share"
                    sizing_rej[reason] += 1
                    skipped.append(
                        SkippedEntry(
                            symbol=sig.symbol,
                            signal_date=sig.signal_date,
                            attempt_date=ts.date(),
                            reason=reason,
                            open_price=open_px,
                        )
                    )
                    continue

                cost = shares * fill_px + comm
                ledger.debit_purchase(cost)
                total_buy_notional += shares * fill_px
                total_commissions += comm
                total_slippage += shares * (fill_px - open_px)

                positions[sig.symbol] = Position(
                    symbol=sig.symbol,
                    shares=shares,
                    entry_date=ts.date(),
                    entry_price=fill_px,
                    initial_stop=stop_px,
                    stop_price=stop_px,
                    highest_high=fill_px,
                    bars_held=0,
                    entry_commission=comm,
                    risk_fraction=cfg.risk_per_trade,
                    regime_at_entry=sig.regime,
                    signal=sig,
                )
                marks[sig.symbol] = fill_px

        # 4. Intraday catastrophe stop checks & close marks
        for sym, p in list(positions.items()):
            row = get_bar(sym, ts)
            if row is None:
                continue
            if float(row["low"]) <= p.stop_price:
                close_pos(ts, p, p.stop_price, ExitReason.STOP)
            else:
                marks[sym] = float(row["close"])

        # 5. Post-close exit management (Close > SMA5 or 10d max hold -> next Open)
        for sym, p in list(positions.items()):
            row = get_bar(sym, ts)
            if row is None:
                continue
            hh = max(p.highest_high, float(row["high"]))
            held = p.bars_held + 1
            exit_reason: ExitReason | None = None
            sma5 = float(row["sma5"])
            if not math.isnan(sma5) and float(row["close"]) > sma5:
                exit_reason = ExitReason.SIGNAL_EXIT
            elif held >= cfg.max_hold_days:
                exit_reason = ExitReason.TIME_STOP
            positions[sym] = p.evolve(highest_high=hh, bars_held=held, pending_exit=exit_reason)

        # 6. Queue next-session signals strictly from ts Close
        eq_close, mv_close = current_equity()
        next_ts = cal[i + 1] if i + 1 < len(cal) else None
        day_signals = signals_by_date.get(ts, [])
        if day_signals and next_ts is not None:
            free_tomorrow = cfg.max_positions - sum(
                1 for p in positions.values() if p.pending_exit is None
            )
            if free_tomorrow > 0:
                eligible_sigs = [s for s in day_signals if s.symbol not in positions]
                already = len(day_signals) - len(eligible_sigs)
                if already:
                    sizing_rej["already_held"] += already
                pending_signals = eligible_sigs
                pending_signal_equity = eq_close
            else:
                sizing_rej["no_free_slot"] += len(day_signals)

        regime = data.regimes[ts]
        rows.append(
            {
                "date": ts,
                "equity": eq_close,
                "settled_cash": ledger.settled,
                "unsettled_cash": ledger.unsettled,
                "market_value": mv_close,
                "open_positions": len(positions),
                "regime": regime.state.value,
                "benchmark_close": float(bench_close.loc[ts]),
            }
        )

    last = cal[-1]
    if settings.backtest.force_close_at_end:
        for sym in list(positions):
            f = data.features[sym]
            px = float(f["close"].loc[:last].iloc[-1])
            close_pos(last, positions[sym], px, ExitReason.END_OF_TEST)
        eq_end, mv_end = current_equity()
        rows[-1].update(
            equity=eq_end,
            settled_cash=ledger.settled,
            unsettled_cash=ledger.unsettled,
            market_value=mv_end,
            open_positions=len(positions),
        )

    curve = pd.DataFrame(rows).set_index("date")
    return PanicBacktestResult(
        initial_capital_usd=cash0,
        equity_curve=curve,
        trades=closed_trades,
        skipped=skipped,
        rejections=rejections,
        sizing_rejections=dict(sizing_rej),
        fees_paid=fees_paid,
        vps_enabled=vps,
        is_synthetic=data.is_synthetic,
        total_buy_notional=total_buy_notional,
        total_sell_notional=total_sell_notional,
        total_commissions_usd=total_commissions,
        total_slippage_usd=total_slippage,
    )
