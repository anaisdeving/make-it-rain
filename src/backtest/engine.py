"""Event-driven daily backtest loop with zero look-ahead.

Timeline of session t (all decisions use data available at that moment):

  open t   : settle T+1 cash -> queued exits / gap-through stops at the open ->
             entries queued at close t-1 (skip if open > limit cap)
  intraday : stop touches (fill at stop, or at the open if it gapped through)
  close t  : marks; stop management (breakeven / trail / time stop -> next open);
             regime from SPY <= t; screen events whose *reaction session* is t;
             size + submit bracket orders for open t+1.

Reaction session of an earnings event announced on calendar day D:
  BMO / DMH -> first session >= D;   AMC / UNKNOWN -> first session > D.
UNKNOWN is mapped like AMC because assuming BMO for a release that was actually
after the close would evaluate the surprise on a bar that closed before it existed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.config import Settings
from src.data.provider_base import MarketDataProvider
from src.domain.models import (
    AnnouncementTiming,
    BarSeries,
    ClosedTrade,
    EarningsEvent,
    ExitReason,
    FeatureSnapshot,
    MarketRegime,
    Position,
    SkippedEntry,
    UniverseMember,
)
from src.execution.backtest_broker import BacktestBroker
from src.risk.risk_manager import build_order_intent
from src.strategy.indicators import add_sue, compute_features
from src.strategy.pead_screener import screen
from src.strategy.regime_filter import compute_regimes

log = logging.getLogger(__name__)


class PreparedData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calendar: pd.DatetimeIndex                     # benchmark sessions within [start, end]
    benchmark: BarSeries
    regimes: dict[pd.Timestamp, MarketRegime]
    features: dict[str, pd.DataFrame]
    events_by_session: dict[pd.Timestamp, list[EarningsEvent]]
    universe: list[UniverseMember]
    data_stats: dict[str, int | float | str]
    provenance: dict[str, str]
    is_synthetic: bool


class BacktestResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    initial_capital_usd: float
    equity_curve: pd.DataFrame
    trades: list[ClosedTrade]
    skipped: list[SkippedEntry]
    rejections: dict[str, int]
    sizing_rejections: dict[str, int]
    fees_paid: float
    vps_enabled: bool
    is_synthetic: bool


def reaction_session(event: EarningsEvent, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    d = pd.Timestamp(event.announced_at.date())
    side = "left" if event.timing in (AnnouncementTiming.BMO, AnnouncementTiming.DMH) else "right"
    i = calendar.searchsorted(d, side=side)
    return calendar[i] if i < len(calendar) else None


def snapshot(symbol: str, ts: pd.Timestamp, row: pd.Series) -> FeatureSnapshot:
    mc = row["market_cap"]
    return FeatureSnapshot(
        symbol=symbol, date=ts.date(),
        **{k: float(row[k]) for k in ("open", "high", "low", "close", "prev_close", "raw_close", "volume",
                                      "avg_volume_20", "avg_dollar_volume_20", "sma50", "high_52w", "atr14")},
        market_cap=None if pd.isna(mc) else float(mc),
    )


def prepare_data(settings: Settings, provider: MarketDataProvider, max_symbols: int | None = None) -> PreparedData:
    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    universe = [m for m in provider.get_universe(bt.end_date) if not m.is_etf]
    if max_symbols:
        universe = universe[:max_symbols]
    symbols = [m.symbol for m in universe]
    log.info("universe candidates: %d", len(symbols))

    benchmark = provider.get_benchmark_bars(fetch_start, bt.end_date, bt.benchmark_symbol)
    bars = provider.get_daily_bars(symbols, fetch_start, bt.end_date)
    shares = provider.get_shares_outstanding(list(bars), fetch_start, bt.end_date)
    mcaps = provider.get_market_cap_history(list(bars), fetch_start, bt.end_date)
    events = provider.get_earnings_calendar(list(bars), fetch_start - timedelta(days=800), bt.end_date)
    events = add_sue(events, settings.screener.sue_lookback_quarters, settings.screener.sue_min_quarters)

    features = {
        s: compute_features(b, settings.screener, settings.risk, shares.get(s), mcaps.get(s))
        for s, b in bars.items()
    }
    full_cal = benchmark.frame.index
    calendar = full_cal[(full_cal >= pd.Timestamp(bt.start_date)) & (full_cal <= pd.Timestamp(bt.end_date))]
    regimes = compute_regimes(benchmark, settings.regime)

    by_session: dict[pd.Timestamp, list[EarningsEvent]] = defaultdict(list)
    timing_counts: dict[str, int] = defaultdict(int)
    in_window = 0
    for e in events:
        ts = reaction_session(e, full_cal)
        if ts is None or ts < calendar[0] or ts > calendar[-1]:
            continue
        in_window += 1
        timing_counts[e.timing.value] += 1
        if e.symbol in features and ts in features[e.symbol].index:
            by_session[ts].append(e)

    with_mcap = sum(1 for f in features.values() if f["market_cap"].notna().any())
    removed = [m for m in universe if m.removed_on is not None]
    stats: dict[str, int | float | str] = {
        "universe_candidates": len(universe),
        "removed_index_members_in_candidates": len(removed),
        "symbols_with_bars": len(bars),
        "removed_members_with_bars": sum(1 for m in removed if m.symbol in bars),
        "symbols_missing_bars": len(universe) - len(bars),
        "symbols_with_market_cap": with_mcap,
        "symbols_with_earnings": len({e.symbol for e in events}),
        "earnings_events_in_window": in_window,
        "events_with_revenue_surprise": sum(1 for e in events if e.revenue_surprise_pct is not None),
        "events_with_sue": sum(1 for e in events if e.sue is not None),
        **{f"timing_{k}": v for k, v in sorted(timing_counts.items())},
        "sessions": len(calendar),
    }
    return PreparedData(
        calendar=calendar, benchmark=benchmark, regimes=regimes, features=features,
        events_by_session=dict(by_session), universe=universe, data_stats=stats,
        provenance=provider.describe(), is_synthetic=provider.is_synthetic,
    )


def manage_position(p: Position, row: pd.Series, settings: Settings) -> tuple[Position, ExitReason | None]:
    """Post-close stop management. Returns the updated position and an exit to queue (or None)."""
    x = settings.execution
    hh = max(p.highest_high, float(row["high"]))
    p = p.evolve(highest_high=hh, bars_held=p.bars_held + 1)
    r = p.r_distance
    stop = p.stop_price
    if hh >= p.entry_price + x.breakeven_trigger_r * r:
        stop = max(stop, p.entry_price)
        p = p.evolve(breakeven_armed=True)
    if hh >= p.entry_price + x.trail_trigger_r * r and not np.isnan(row["atr14"]):
        stop = max(stop, hh - x.trail_atr_multiple * float(row["atr14"]))
        p = p.evolve(trail_armed=True)
    p = p.evolve(stop_price=stop)
    if p.bars_held >= x.time_stop_days and p.r_multiple(float(row["close"])) < x.time_stop_min_r:
        return p, ExitReason.TIME_STOP
    return p, None


def run_backtest(
    data: PreparedData,
    settings: Settings,
    initial_capital_usd: float | None = None,
    vps_enabled: bool | None = None,
) -> BacktestResult:
    cash0 = initial_capital_usd if initial_capital_usd is not None else settings.backtest.initial_capital_usd
    vps = settings.execution.vps_enabled if vps_enabled is None else vps_enabled
    cal = data.calendar
    broker = BacktestBroker(data.benchmark.frame.index, cash0, settings.execution)
    rejections: dict[str, int] = defaultdict(int)
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
            broker.charge_fee(settings.execution.vps_monthly_eur * settings.backtest.eur_usd_rate)
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
            newp, exit_reason = manage_position(p, row, settings)
            broker.positions[sym] = newp
            if exit_reason is not None:
                broker.close_position(sym, exit_reason)

        regime = data.regimes[ts]
        next_ts = cal[i + 1] if i + 1 < len(cal) else None
        acct = broker.get_account_state()
        candidates = []
        for e in data.events_by_session.get(ts, []):
            row = bar(e.symbol, ts)
            if row is not None:
                candidates.append((e, snapshot(e.symbol, ts, row)))
        if candidates:
            signals, why = screen(candidates, regime, settings.screener)
            for k, v in why.items():
                rejections[k] += v
            slots = regime.max_positions - len(broker.positions) - len(broker.pending_orders)
            if next_ts is not None and signals:
                avail = broker.ledger.settled_by(next_ts)
                for sig in signals:
                    if slots <= 0:
                        sizing_rej["no_free_slot"] += 1
                        continue
                    if sig.symbol in broker.positions or sig.symbol in broker.pending_orders:
                        sizing_rej["already_held"] += 1
                        continue
                    intent, reason = build_order_intent(
                        sig, regime, acct.equity, avail, settings.risk, settings.execution
                    )
                    if intent is None:
                        sizing_rej[reason] += 1
                        continue
                    broker.submit_bracket_order(intent)
                    avail -= intent.shares * intent.limit_price + settings.execution.commission_per_side_usd
                    slots -= 1

        acct = broker.get_account_state()
        rows.append({
            "date": ts, "equity": acct.equity, "settled_cash": acct.settled_cash,
            "unsettled_cash": acct.unsettled_cash, "market_value": acct.market_value,
            "open_positions": acct.open_positions, "regime": regime.state.value,
            "benchmark_close": float(bench_close.loc[ts]),
        })

    last = cal[-1]
    if settings.backtest.force_close_at_end:
        for sym in list(broker.positions):
            f = data.features[sym]
            px = float(f["close"].loc[:last].iloc[-1])
            broker.force_exit(last, sym, px, ExitReason.END_OF_TEST)
        acct = broker.get_account_state()
        rows[-1].update(equity=acct.equity, settled_cash=acct.settled_cash, unsettled_cash=acct.unsettled_cash,
                        market_value=acct.market_value, open_positions=acct.open_positions)

    curve = pd.DataFrame(rows).set_index("date")
    return BacktestResult(
        initial_capital_usd=cash0, equity_curve=curve, trades=broker.closed_trades, skipped=broker.skipped,
        rejections=dict(rejections), sizing_rejections=dict(sizing_rej), fees_paid=broker.fees_paid,
        vps_enabled=vps, is_synthetic=data.is_synthetic,
    )
