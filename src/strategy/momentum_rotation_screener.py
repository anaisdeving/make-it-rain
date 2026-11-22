"""Strategy v4b: Regime-Gated Bi-Weekly Momentum Leaders (Pure OHLCV, Zero Look-Ahead).

Cross-sectional medium-term momentum rotation over the top 100 most liquid stocks
in the cached universe, gated by SPY trend/volatility regime and protected by a
-15% trailing catastrophe stop between bi-weekly rebalances.

Guardrails:
- Leaves ``src/risk/risk_manager.py`` and ``src/execution/`` completely untouched.
- Directly reuses ``SettlementLedger`` from ``src/risk/risk_manager.py`` to
  enforce T+1 settled cash with ZERO Good Faith Violations.
- Point-in-time universe on every rebalance day ``t``:
  - ``raw_close >= $15`` and trailing 20d avg dollar volume ``>= $50M`` (``[t-19..t]``).
  - Top 100 stocks ranked strictly by trailing 20d avg dollar volume on day ``t``.
- Rebalance rule (every 10 trading days, evaluated at rebalance day ``t`` Close):
  - Risk-off if ``SPY <= SMA200`` or ``sigma_20 > 0.28`` (or warmup NaN): close
    all open positions at ``t+1`` Open and hold 100% cash.
  - Risk-on: filter Top 100 liquid stocks to ``Close > SMA50``, rank by
    ``Return_63d / RealizedVol_63d`` (strictly ``<= t``), select top 3.
  - Keep positions that remain in the top 3 (no needless churn).
  - Honest T+1 settlement rotation:
    - If exiting names are sold at ``t+1`` Open, their proceeds settle on ``t+2``
      Open; new target names are bought at ``t+2`` Open once cash is settled
      (zero Good Faith Violations!).
    - If no positions are being sold at ``t+1`` Open (e.g., entering from cash or
      replacing a previously stopped-out slot whose cash already settled by ``t+1``
      Open), new target names are bought immediately at ``t+1`` Open.
- Between rebalances: ``-15%`` trailing catastrophe stop (``HighestHigh * 0.85``),
  with ``HighestHigh`` updated post-close (zero intraday high-before-low look-ahead)
  and overnight gap-downs worse than stop filled at Open less ``0.05%`` slippage.
- Friction: ``0.05%`` slippage per side + ``$0.38`` IBKR commission per side,
  whole shares only.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.backtest.engine import BacktestResult
from src.config import Settings
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
from src.risk.risk_manager import SettlementLedger
from src.strategy.regime_filter import compute_regimes

log = logging.getLogger(__name__)


class MomentumRotationConfig(BaseModel):
    """Locked configuration for Strategy v4b (Regime-Gated Bi-Weekly Momentum Leaders)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Point-in-time liquid universe
    min_price: float = Field(15.0, gt=0)
    min_avg_dollar_volume: float = Field(50_000_000.0, ge=0)
    avg_window: int = Field(20, ge=1)
    top_liquid_n: int = Field(100, ge=1)

    # SPY regime gate (evaluated at rebalance day t Close, strictly <= t)
    spy_sma_window: int = Field(200, ge=1)
    spy_vol_window: int = Field(20, ge=2)
    max_spy_vol: float = Field(0.28, gt=0)

    # Bi-weekly momentum ranking (strictly <= t)
    rebalance_every_days: int = Field(10, ge=1)
    trend_sma_window: int = Field(50, ge=1)
    momentum_window: int = Field(63, ge=1)
    vol_window: int = Field(63, ge=2)
    top_k: int = Field(3, ge=1)

    # Risk, stops & frictions
    trailing_stop_pct: float = Field(0.15, gt=0, lt=1.0)
    slippage_per_side_pct: float = Field(0.0005, ge=0.0)
    commission_per_side_usd: float = Field(0.38, ge=0.0)
    settlement_days: int = Field(1, ge=0)


class MomentumSnapshot(BaseModel):
    """Point-in-time snapshot for one symbol at rebalance day ``date`` Close."""

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
    sma50: float
    ret_63d: float
    realized_vol_63d: float
    risk_adj_mom: float


class MomentumCandidate(BaseModel):
    """Selected momentum leader at rebalance day ``signal_date`` Close."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    signal_date: date
    signal_close: float
    avg_dollar_volume_20: float
    ret_63d: float
    realized_vol_63d: float
    risk_adj_mom: float
    liquidity_rank: int
    momentum_rank: int


class PreparedMomentumData(BaseModel):
    """Prepared pure-OHLCV dataset for Strategy v4b."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calendar: pd.DatetimeIndex
    benchmark: BarSeries
    regimes: dict[pd.Timestamp, MarketRegime]
    spy_risk_on: dict[pd.Timestamp, bool]
    spy_sma200: pd.Series
    spy_sigma20: pd.Series
    features: dict[str, pd.DataFrame]
    universe: list[UniverseMember]
    data_stats: dict[str, int | float | str]
    provenance: dict[str, str]
    is_synthetic: bool


class MomentumBacktestResult(BacktestResult):
    """BacktestResult augmented with exact turnover, friction, and rebalance stats."""

    total_buy_notional: float = 0.0
    total_sell_notional: float = 0.0
    total_commissions_usd: float = 0.0
    total_slippage_usd: float = 0.0
    rebalance_days_total: int = 0
    rebalance_days_risk_on: int = 0
    rebalance_days_risk_off: int = 0
    kept_positions_total: int = 0
    rotated_out_total: int = 0


def compute_spy_risk_on(
    spy: BarSeries,
    cfg: MomentumRotationConfig,
) -> tuple[dict[pd.Timestamp, bool], pd.Series, pd.Series]:
    """Compute strictly causal SPY risk-on gate: ``SPY > SMA200`` and ``sigma_20 <= 0.28``."""
    close = spy.frame["close"]
    sma200 = close.rolling(cfg.spy_sma_window, min_periods=cfg.spy_sma_window).mean()
    sigma20 = (
        np.log(close).diff().rolling(cfg.spy_vol_window, min_periods=cfg.spy_vol_window).std(ddof=1)
        * math.sqrt(252)
    )
    risk_on: dict[pd.Timestamp, bool] = {}
    for ts in close.index:
        c = float(close.loc[ts])
        s = float(sma200.loc[ts])
        v = float(sigma20.loc[ts])
        risk_on[ts] = (not math.isnan(s)) and (not math.isnan(v)) and (c > s) and (v <= cfg.max_spy_vol)
    return risk_on, sma200, sigma20


def compute_momentum_features(bars: BarSeries, cfg: MomentumRotationConfig) -> pd.DataFrame:
    """Compute strictly causal v4b features for one symbol. Row ``t`` uses rows ``<= t`` only."""
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
    out["sma50"] = df["close"].rolling(cfg.trend_sma_window, min_periods=cfg.trend_sma_window).mean()
    out["ret_63d"] = df["close"] / df["close"].shift(cfg.momentum_window) - 1.0
    out["realized_vol_63d"] = (
        np.log(df["close"]).diff().rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)
        * math.sqrt(252)
    )
    valid_vol = out["realized_vol_63d"].notna() & (out["realized_vol_63d"] > 1e-9) & out["ret_63d"].notna()
    out["risk_adj_mom"] = np.where(valid_vol, out["ret_63d"] / out["realized_vol_63d"], np.nan)
    return out


def momentum_snapshot(symbol: str, ts: pd.Timestamp, row: pd.Series) -> MomentumSnapshot:
    """Build an immutable point-in-time snapshot at session ``ts`` Close."""
    return MomentumSnapshot(
        symbol=symbol,
        date=ts.date(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        raw_close=float(row["raw_close"]),
        volume=float(row["volume"]),
        avg_dollar_volume_20=float(row["avg_dollar_volume_20"]),
        sma50=float(row["sma50"]),
        ret_63d=float(row["ret_63d"]),
        realized_vol_63d=float(row["realized_vol_63d"]),
        risk_adj_mom=float(row["risk_adj_mom"]),
    )


def select_momentum_leaders_day(
    snapshots: list[MomentumSnapshot],
    risk_on: bool,
    cfg: MomentumRotationConfig,
) -> tuple[list[MomentumCandidate], dict[str, int]]:
    """Select top-K risk-adjusted momentum leaders from day ``t``'s Top-100 liquid universe."""
    reasons: dict[str, int] = {}

    def _inc(key: str, n: int = 1) -> None:
        reasons[key] = reasons.get(key, 0) + n

    if not risk_on:
        if snapshots:
            _inc("regime_risk_off", len(snapshots))
        return [], reasons

    liquid: list[MomentumSnapshot] = []
    for s in snapshots:
        if math.isnan(s.raw_close) or s.raw_close < cfg.min_price:
            _inc("price")
            continue
        if math.isnan(s.avg_dollar_volume_20) or s.avg_dollar_volume_20 < cfg.min_avg_dollar_volume:
            _inc("liquidity")
            continue
        liquid.append(s)

    if not liquid:
        return [], reasons

    # Strictly point-in-time Top-N liquidity ranking on day t
    liquid.sort(key=lambda s: (-s.avg_dollar_volume_20, s.symbol))
    top_liquid = liquid[: cfg.top_liquid_n]
    dropped_liq = len(liquid) - len(top_liquid)
    if dropped_liq > 0:
        _inc("outside_top_liquid", dropped_liq)

    qualified: list[tuple[int, MomentumSnapshot]] = []
    for liq_rank, s in enumerate(top_liquid, start=1):
        if math.isnan(s.sma50) or math.isnan(s.risk_adj_mom) or math.isnan(s.realized_vol_63d):
            _inc("insufficient_history")
            continue
        if not (s.close > s.sma50):
            _inc("below_sma50")
            continue
        qualified.append((liq_rank, s))

    if not qualified:
        return [], reasons

    qualified.sort(key=lambda item: (-item[1].risk_adj_mom, item[1].symbol))
    selected = qualified[: cfg.top_k]
    dropped_rank = len(qualified) - len(selected)
    if dropped_rank > 0:
        _inc("outside_top_k", dropped_rank)

    out: list[MomentumCandidate] = []
    for mom_rank, (liq_rank, s) in enumerate(selected, start=1):
        out.append(
            MomentumCandidate(
                symbol=s.symbol,
                signal_date=s.date,
                signal_close=s.close,
                avg_dollar_volume_20=s.avg_dollar_volume_20,
                ret_63d=s.ret_63d,
                realized_vol_63d=s.realized_vol_63d,
                risk_adj_mom=s.risk_adj_mom,
                liquidity_rank=liq_rank,
                momentum_rank=mom_rank,
            )
        )
        _inc("selected")
    return out, reasons


def build_momentum_data_from_bars(
    bars: dict[str, BarSeries],
    benchmark: BarSeries,
    universe: list[UniverseMember],
    settings: Settings,
    cfg: MomentumRotationConfig,
    provenance: dict[str, str],
    is_synthetic: bool,
) -> PreparedMomentumData:
    """Compute v4b features from already-loaded OHLCV bars."""
    bt = settings.backtest
    features = {s: compute_momentum_features(b, cfg) for s, b in bars.items()}
    full_cal = benchmark.frame.index
    calendar = full_cal[(full_cal >= pd.Timestamp(bt.start_date)) & (full_cal <= pd.Timestamp(bt.end_date))]
    regimes = compute_regimes(benchmark, settings.regime)
    spy_risk_on, spy_sma200, spy_sigma20 = compute_spy_risk_on(benchmark, cfg)

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
    return PreparedMomentumData(
        calendar=calendar,
        benchmark=benchmark,
        regimes=regimes,
        spy_risk_on=spy_risk_on,
        spy_sma200=spy_sma200,
        spy_sigma20=spy_sigma20,
        features=features,
        universe=universe,
        data_stats=stats,
        provenance=provenance,
        is_synthetic=is_synthetic,
    )


def prepare_momentum_data(
    settings: Settings,
    provider: MarketDataProvider,
    cfg: MomentumRotationConfig,
    max_symbols: int | None = None,
) -> PreparedMomentumData:
    """Load cached OHLCV bars + SPY benchmark and compute strictly causal v4b features."""
    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    universe = [m for m in provider.get_universe(bt.end_date) if not m.is_etf]
    if max_symbols:
        universe = universe[:max_symbols]
    symbols = [m.symbol for m in universe]
    log.info("v4b universe candidates: %d", len(symbols))

    benchmark = provider.get_benchmark_bars(fetch_start, bt.end_date, bt.benchmark_symbol)
    bars = provider.get_daily_bars(symbols, fetch_start, bt.end_date)
    return build_momentum_data_from_bars(
        bars=bars,
        benchmark=benchmark,
        universe=universe,
        settings=settings,
        cfg=cfg,
        provenance=provider.describe(),
        is_synthetic=provider.is_synthetic,
    )


def slice_momentum_data(data: PreparedMomentumData, start_date: date, end_date: date) -> PreparedMomentumData:
    """Restrict ``PreparedMomentumData`` to calendar ``[start_date, end_date]`` (zero look-ahead)."""
    full_cal = data.benchmark.frame.index
    cal = full_cal[(full_cal >= pd.Timestamp(start_date)) & (full_cal <= pd.Timestamp(end_date))]
    stats = dict(data.data_stats)
    stats["sessions"] = len(cal)
    stats["first_session"] = str(cal[0].date()) if len(cal) else ""
    stats["last_session"] = str(cal[-1].date()) if len(cal) else ""
    return data.model_copy(update={"calendar": cal, "data_stats": stats})


def select_all_rebalance_leaders(
    data: PreparedMomentumData,
    cfg: MomentumRotationConfig,
) -> tuple[dict[pd.Timestamp, list[MomentumCandidate]], dict[str, int], dict[pd.Timestamp, int]]:
    """Select Top-K leaders on every bi-weekly rebalance day ``i % rebalance_every_days == 0``."""
    cal = data.calendar
    syms = sorted(data.features.keys())
    reb_dates = [cal[i] for i in range(0, len(cal), cfg.rebalance_every_days)]
    if not reb_dates or not syms:
        return {}, {}, {}

    leaders_by_date: dict[pd.Timestamp, list[MomentumCandidate]] = {}
    rejections: dict[str, int] = defaultdict(int)
    liq_counts: dict[pd.Timestamp, int] = {}

    for ts in reb_dates:
        snaps: list[MomentumSnapshot] = []
        for sym in syms:
            f = data.features[sym]
            if ts in f.index:
                snaps.append(momentum_snapshot(sym, ts, f.loc[ts]))
        risk_on = data.spy_risk_on.get(ts, False)
        cands, why = select_momentum_leaders_day(snaps, risk_on, cfg)
        leaders_by_date[ts] = cands
        for k, v in why.items():
            rejections[k] += v
        liq_counts[ts] = sum(
            1
            for s in snaps
            if not math.isnan(s.raw_close)
            and s.raw_close >= cfg.min_price
            and not math.isnan(s.avg_dollar_volume_20)
            and s.avg_dollar_volume_20 >= cfg.min_avg_dollar_volume
        )

    return leaders_by_date, dict(rejections), liq_counts


def _candidate_to_signal(c: MomentumCandidate, regime: MarketRegime) -> TradeSignal:
    ev = EarningsEvent(
        symbol=c.symbol,
        announced_at=datetime(c.signal_date.year, c.signal_date.month, c.signal_date.day, 16, 0),
        timing=AnnouncementTiming.AMC,
        source="v4b_momentum_rotation",
    )
    return TradeSignal(
        symbol=c.symbol,
        signal_date=c.signal_date,
        event=ev,
        regime=regime.state,
        event_day_return=c.ret_63d,
        volume_ratio=1.0,
        close_location=0.5,
        event_day_low=c.signal_close * 0.85,
        signal_close=c.signal_close,
        atr14=max(c.signal_close * 0.01, 1e-6),
        market_cap=None,
        rank_score=c.risk_adj_mom,
    )


def run_backtest_v4b(
    data: PreparedMomentumData,
    settings: Settings,
    cfg: MomentumRotationConfig,
    initial_capital_usd: float | None = None,
    vps_enabled: bool | None = None,
    precomputed_leaders: tuple[dict[pd.Timestamp, list[MomentumCandidate]], dict[str, int]] | None = None,
) -> MomentumBacktestResult:
    """Event-driven daily backtest for Strategy v4b (zero look-ahead, T+1 settled cash).

    Timeline on session ``ts`` (index ``i``):
    1. Monthly VPS fee (if enabled) on first session of each month.
    2. ``ledger.roll(ts)`` settles T+1 sale proceeds.
    3. Market Open exits: overnight gap-downs ``open <= stop_price`` exit at Open
       (``GAP_STOP``); queued ``REGIME_EXIT`` or ``REBALANCE_EXIT`` exit at Open.
       Sale proceeds settle at ``ts+1`` Open.
    4. Market Open scheduled buys (executed on ``t+1`` Open when no sells occurred
       at ``t+1`` Open, or on ``t+2`` Open once ``t+1`` Open sale proceeds settle):
       allocates up to ``equity / top_k`` (~33.3%) per new slot from settled cash
       only (whole shares, ``0.05%`` slippage + ``$0.38`` commission).
    5. Intraday ``-15%`` trailing catastrophe stop: ``low <= stop_price`` exits at
       ``stop_price * (1 - 0.05%)``.
    6. Post-close trailing stop update: ``HH = max(HH, high)``,
       ``stop = max(stop, HH * 0.85)`` (strictly post-close).
    7. If ``i % rebalance_every_days == 0``: evaluate SPY risk-on gate and Top-3
       leaders at ``ts`` Close. Keep overlapping positions; queue exits for dropped
       names at ``ts+1`` Open and schedule new buys at ``ts+2`` Open (if rotating)
       or ``ts+1`` Open (if entering from settled cash).
    """
    cash0 = initial_capital_usd if initial_capital_usd is not None else settings.backtest.initial_capital_usd
    vps = settings.execution.vps_enabled if vps_enabled is None else vps_enabled
    cal = data.calendar
    if not len(cal):
        raise ValueError("empty simulation calendar")

    if precomputed_leaders is not None:
        leaders_by_date, rejections = precomputed_leaders
        rejections = dict(rejections)
    else:
        leaders_by_date, rejections, _ = select_all_rebalance_leaders(data, cfg)

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
    reb_total = 0
    reb_risk_on = 0
    reb_risk_off = 0
    kept_total = 0
    rotated_out_total = 0

    scheduled_buy_date: pd.Timestamp | None = None
    scheduled_buy_cands: list[MomentumCandidate] = []
    scheduled_buy_regime: MarketRegime | None = None

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

        # 2. Open exits (gap-through trailing stops + queued regime/rebalance sells)
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

        # 3. Scheduled buys executing at ts Open (strictly settled cash, whole shares)
        if scheduled_buy_date is not None and ts == scheduled_buy_date:
            cands_to_buy = scheduled_buy_cands
            buy_regime = scheduled_buy_regime or data.regimes[ts]
            scheduled_buy_date = None
            scheduled_buy_cands = []
            scheduled_buy_regime = None

            # Mark-to-open equity for ~33.3% target slot sizing
            open_mv = 0.0
            for s_held, p_held in positions.items():
                b_held = get_bar(s_held, ts)
                px_mark = float(b_held["open"]) if b_held is not None else marks.get(s_held, p_held.entry_price)
                open_mv += p_held.shares * px_mark
            open_equity = ledger.settled + ledger.unsettled + open_mv
            slot_target = open_equity / cfg.top_k

            actionable = [c for c in cands_to_buy if c.symbol not in positions]
            rem = min(len(actionable), max(cfg.top_k - len(positions), 0))
            for cand in actionable:
                if len(positions) >= cfg.top_k or rem <= 0:
                    sizing_rej["no_free_slot"] += 1
                    continue
                row = get_bar(cand.symbol, ts)
                if row is None:
                    skipped.append(
                        SkippedEntry(
                            symbol=cand.symbol,
                            signal_date=cand.signal_date,
                            attempt_date=ts.date(),
                            reason="no_bar",
                        )
                    )
                    rem -= 1
                    continue
                open_px = float(row["open"])
                fill_px = open_px * (1.0 + slip)
                budget = min(slot_target, ledger.settled / rem)
                shares = math.floor(max(budget - comm, 0.0) / fill_px + 1e-9)
                while shares > 0 and shares * fill_px + comm > ledger.settled:
                    shares -= 1
                if shares < 1:
                    reason = (
                        "insufficient_settled_cash"
                        if ledger.settled < fill_px + comm
                        else "price_above_slot_budget"
                    )
                    sizing_rej[reason] += 1
                    skipped.append(
                        SkippedEntry(
                            symbol=cand.symbol,
                            signal_date=cand.signal_date,
                            attempt_date=ts.date(),
                            reason=reason,
                            open_price=open_px,
                        )
                    )
                    rem -= 1
                    continue

                cost = shares * fill_px + comm
                ledger.debit_purchase(cost)
                total_buy_notional += shares * fill_px
                total_commissions += comm
                total_slippage += shares * (fill_px - open_px)

                init_stop = fill_px * (1.0 - cfg.trailing_stop_pct)
                sig = _candidate_to_signal(cand, buy_regime)
                positions[cand.symbol] = Position(
                    symbol=cand.symbol,
                    shares=shares,
                    entry_date=ts.date(),
                    entry_price=fill_px,
                    initial_stop=init_stop,
                    stop_price=init_stop,
                    highest_high=fill_px,
                    bars_held=0,
                    entry_commission=comm,
                    risk_fraction=cfg.trailing_stop_pct / cfg.top_k,
                    regime_at_entry=buy_regime.state,
                    signal=sig,
                )
                marks[cand.symbol] = fill_px
                rem -= 1

        # 4. Intraday -15% trailing catastrophe stop checks & close marks
        for sym, p in list(positions.items()):
            row = get_bar(sym, ts)
            if row is None:
                continue
            if float(row["low"]) <= p.stop_price:
                reason = ExitReason.TRAIL_STOP if p.trail_armed else ExitReason.STOP
                close_pos(ts, p, p.stop_price, reason)
            else:
                marks[sym] = float(row["close"])

        # 5. Post-close trailing stop ratchet (strictly after close -> zero intraday look-ahead)
        for sym, p in list(positions.items()):
            row = get_bar(sym, ts)
            if row is None:
                continue
            hh = max(p.highest_high, float(row["high"]))
            new_stop = max(p.stop_price, hh * (1.0 - cfg.trailing_stop_pct))
            armed = p.trail_armed or (hh > p.entry_price)
            positions[sym] = p.evolve(
                highest_high=hh,
                stop_price=new_stop,
                bars_held=p.bars_held + 1,
                trail_armed=armed,
            )

        # 6. Bi-weekly rebalance decision at Close (every 10 trading days)
        risk_on = data.spy_risk_on.get(ts, False)
        if i % cfg.rebalance_every_days == 0:
            reb_total += 1
            if not risk_on:
                reb_risk_off += 1
                scheduled_buy_date = None
                scheduled_buy_cands = []
                scheduled_buy_regime = None
                for sym, p in list(positions.items()):
                    positions[sym] = p.evolve(pending_exit=ExitReason.REGIME_EXIT)
            else:
                reb_risk_on += 1
                targets = leaders_by_date.get(ts, [])
                target_syms = {c.symbol for c in targets}

                exiting = [sym for sym in positions if sym not in target_syms]
                kept = [sym for sym in positions if sym in target_syms]
                kept_total += len(kept)
                rotated_out_total += len(exiting)

                for sym in exiting:
                    positions[sym] = positions[sym].evolve(pending_exit=ExitReason.REBALANCE_EXIT)

                new_buys = [c for c in targets if c.symbol not in positions]
                if new_buys:
                    if exiting:
                        # Sell exiting names at t+1 Open; buy new names at t+2 Open once T+1 cash settles.
                        buy_idx = i + 1 + cfg.settlement_days
                    else:
                        # No sells at t+1 Open -> buy immediately at t+1 Open using settled cash.
                        buy_idx = i + 1
                    if buy_idx < len(cal):
                        scheduled_buy_date = cal[buy_idx]
                        scheduled_buy_cands = new_buys
                        scheduled_buy_regime = data.regimes[ts]
                else:
                    scheduled_buy_date = None
                    scheduled_buy_cands = []
                    scheduled_buy_regime = None

        eq_close, mv_close = current_equity()
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
                "risk_on": risk_on,
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
    return MomentumBacktestResult(
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
        rebalance_days_total=reb_total,
        rebalance_days_risk_on=reb_risk_on,
        rebalance_days_risk_off=reb_risk_off,
        kept_positions_total=kept_total,
        rotated_out_total=rotated_out_total,
    )
