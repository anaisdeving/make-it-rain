"""Experimental Variant v2 PEAD screener and backtest runner.

Leaves ``src/strategy/pead_screener.py`` and ``src/risk/risk_manager.py`` untouched:
- Delegates raw event screening to ``pead_screener.evaluate_event``.
- Applies stop-width (``2.5 * ATR14``) and entry-cap (``+2.0%``) overrides strictly via
  Pydantic ``Settings`` dependency injection into ``risk_manager.build_order_intent``
  and ``BacktestBroker``.
- Adds a strictly causal small-cap trend gate (``IWM > SMA200`` in addition to
  ``SPY > SMA200``) and an optional sub-variant ``v2b`` (2-to-5 day post-earnings
  consolidation breakout above event-day high).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.backtest.engine import BacktestResult, PreparedData, manage_position, snapshot
from src.config import ScreenerConfig, Settings
from src.domain.models import (
    BarSeries,
    EarningsEvent,
    ExitReason,
    FeatureSnapshot,
    MarketRegime,
    RegimeState,
    TradeSignal,
)
from src.execution.backtest_broker import BacktestBroker
from src.risk.risk_manager import build_order_intent
from src.strategy.pead_screener import evaluate_event as evaluate_event_v1


class V2Config(BaseModel):
    """Configuration for PEAD Variant v2 and single-lever ablations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    require_spy_above_sma200: bool = True
    require_iwm_above_sma200: bool = True
    iwm_sma_window: int = Field(200, ge=1)
    entry_limit_cap_pct: float = Field(0.02, ge=0.0)
    atr_stop_multiple: float = Field(2.5, gt=0.0)
    max_stop_pct: float | None = Field(None, gt=0.0, lt=1.0)
    consolidation_breakout: bool = False
    consolidation_min_days: int = Field(2, ge=1)
    consolidation_max_days: int = Field(5, ge=1)
    invalidate_on_close_below_event_low: bool = True
    invalidate_on_early_breakout: bool = True

    @model_validator(mode="after")
    def _check_window(self) -> "V2Config":
        if self.consolidation_max_days < self.consolidation_min_days:
            raise ValueError("consolidation_max_days must be >= consolidation_min_days")
        return self


class TrendSnapshot(BaseModel):
    """Causal benchmark close vs trailing SMA snapshot as of ``date``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: date
    close: float
    sma: float

    @property
    def is_bullish(self) -> bool:
        return not math.isnan(self.close) and not math.isnan(self.sma) and self.close > self.sma


class ConsolidationWatch(BaseModel):
    """Post-earnings candidate waiting for a 2-to-5 day breakout above event-day high."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    event_date: date
    event: EarningsEvent
    event_day_high: float
    event_day_low: float
    event_day_return: float
    volume_ratio: float
    close_location: float
    rank_score: float
    sessions_elapsed: int = 0

    def advance(self) -> "ConsolidationWatch":
        return self.model_copy(update={"sessions_elapsed": self.sessions_elapsed + 1})


def apply_v2_settings(base: Settings, v2_cfg: V2Config) -> Settings:
    """Return a new Settings object with v2 risk and execution overrides applied via DI."""
    risk_updates: dict[str, float] = {"atr_stop_multiple": v2_cfg.atr_stop_multiple}
    if v2_cfg.max_stop_pct is not None:
        risk_updates["max_stop_pct"] = v2_cfg.max_stop_pct
    return base.model_copy(
        update={
            "risk": base.risk.model_copy(update=risk_updates),
            "execution": base.execution.model_copy(
                update={"entry_limit_cap_pct": v2_cfg.entry_limit_cap_pct}
            ),
        }
    )


def compute_benchmark_trend(bars: BarSeries, sma_window: int = 200) -> dict[pd.Timestamp, TrendSnapshot]:
    """Strictly causal trailing SMA trend series: row t uses closes <= t only."""
    close = bars.frame["close"]
    sma = close.rolling(sma_window, min_periods=sma_window).mean()
    return {
        ts: TrendSnapshot(date=ts.date(), close=float(close.loc[ts]), sma=float(sma.loc[ts]))
        for ts in close.index
    }


def trend_gate_passes(
    regime: MarketRegime,
    iwm_trend: TrendSnapshot | None,
    v2_cfg: V2Config,
) -> tuple[bool, str]:
    """Check dual-benchmark trend filter (SPY > SMA200 and IWM > SMA200)."""
    if v2_cfg.require_spy_above_sma200:
        if math.isnan(regime.spy_close) or math.isnan(regime.sma200) or not (regime.spy_close > regime.sma200):
            return False, "spy_below_sma200"
    if v2_cfg.require_iwm_above_sma200:
        if iwm_trend is None or not iwm_trend.is_bullish:
            return False, "iwm_below_sma200"
    return True, "ok"


def evaluate_event_v2(
    event: EarningsEvent,
    snap: FeatureSnapshot,
    regime: MarketRegime,
    scfg: ScreenerConfig,
    v2_cfg: V2Config,
    iwm_trend: TrendSnapshot | None = None,
) -> tuple[TradeSignal | None, str]:
    """Evaluate one earnings candidate under v2 rules (zero look-ahead)."""
    if regime.state is RegimeState.PANIC:
        return None, "regime_panic"
    trend_ok, trend_reason = trend_gate_passes(regime, iwm_trend, v2_cfg)
    if not trend_ok:
        return None, trend_reason
    return evaluate_event_v1(event, snap, regime, scfg)


def screen_v2(
    candidates: list[tuple[EarningsEvent, FeatureSnapshot]],
    regime: MarketRegime,
    scfg: ScreenerConfig,
    v2_cfg: V2Config,
    iwm_trend: TrendSnapshot | None = None,
) -> tuple[list[TradeSignal], dict[str, int]]:
    """Screen candidates under v2 rules; returns deterministically ranked signals + counts."""
    signals: list[TradeSignal] = []
    reasons: dict[str, int] = {}
    for event, snap in candidates:
        sig, why = evaluate_event_v2(event, snap, regime, scfg, v2_cfg, iwm_trend)
        reasons[why] = reasons.get(why, 0) + 1
        if sig is not None:
            signals.append(sig)
    signals.sort(key=lambda s: (-s.rank_score, s.symbol))
    return signals, reasons


class ConsolidationTracker:
    """Causal state machine for sub-variant v2b (2-to-5 day consolidation breakout).

    On reaction session t_0, events passing the base PEAD filter are registered as
    watches rather than entered at t_0+1. On each subsequent trading session t_k:
    - If close < event_day_low (optional): watch is invalidated.
    - For k < consolidation_min_days: if close > event_day_high (optional), watch is
      invalidated as an immediate continuation rather than a consolidation.
    - For consolidation_min_days <= k <= consolidation_max_days: the first close
      strictly above event_day_high triggers a TradeSignal on t_k (for entry at t_k+1
      Open), provided regime, dual trend gate, SMA50, and ATR14 are valid on t_k.
    - After consolidation_max_days without breakout, the watch expires.
    """

    def __init__(self, v2_cfg: V2Config) -> None:
        self.v2_cfg = v2_cfg
        self.watches: dict[str, ConsolidationWatch] = {}

    def step_existing(
        self,
        ts: pd.Timestamp,
        snapshots: dict[str, FeatureSnapshot],
        regime: MarketRegime,
        iwm_trend: TrendSnapshot | None,
    ) -> tuple[list[TradeSignal], dict[str, int]]:
        """Advance watches by one session and emit any confirmed breakout signals."""
        signals: list[TradeSignal] = []
        reasons: dict[str, int] = {}

        def _count(key: str) -> None:
            reasons[key] = reasons.get(key, 0) + 1

        for sym in list(self.watches):
            watch = self.watches[sym].advance()
            if watch.sessions_elapsed > self.v2_cfg.consolidation_max_days:
                del self.watches[sym]
                _count("consolidation_expired")
                continue

            snap = snapshots.get(sym)
            if snap is None:
                self.watches[sym] = watch
                continue

            if self.v2_cfg.invalidate_on_close_below_event_low and snap.close < watch.event_day_low:
                del self.watches[sym]
                _count("consolidation_broke_low")
                continue

            if watch.sessions_elapsed < self.v2_cfg.consolidation_min_days:
                if self.v2_cfg.invalidate_on_early_breakout and snap.close > watch.event_day_high:
                    del self.watches[sym]
                    _count("consolidation_early_breakout")
                else:
                    self.watches[sym] = watch
                continue

            if snap.close > watch.event_day_high:
                del self.watches[sym]
                if regime.state is RegimeState.PANIC:
                    _count("breakout_regime_panic")
                    continue
                trend_ok, trend_reason = trend_gate_passes(regime, iwm_trend, self.v2_cfg)
                if not trend_ok:
                    _count(f"breakout_{trend_reason}")
                    continue
                if math.isnan(snap.atr14) or math.isnan(snap.sma50):
                    _count("breakout_insufficient_history")
                    continue
                if not (snap.close > snap.sma50):
                    _count("breakout_below_sma50")
                    continue
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        signal_date=ts.date(),
                        event=watch.event,
                        regime=regime.state,
                        event_day_return=watch.event_day_return,
                        volume_ratio=watch.volume_ratio,
                        close_location=watch.close_location,
                        event_day_low=watch.event_day_low,
                        signal_close=snap.close,
                        atr14=snap.atr14,
                        market_cap=snap.market_cap,
                        rank_score=watch.rank_score,
                    )
                )
                _count("breakout_ok")
            else:
                if watch.sessions_elapsed >= self.v2_cfg.consolidation_max_days:
                    del self.watches[sym]
                    _count("consolidation_expired")
                else:
                    self.watches[sym] = watch

        signals.sort(key=lambda s: (-s.rank_score, s.symbol))
        return signals, reasons

    def register_reaction_day(
        self,
        candidates: list[tuple[EarningsEvent, FeatureSnapshot]],
        regime: MarketRegime,
        scfg: ScreenerConfig,
    ) -> dict[str, int]:
        """Screen reaction-day events with untouched v1 screener and queue passing ones."""
        reasons: dict[str, int] = {}
        for event, snap in candidates:
            sig, why = evaluate_event_v1(event, snap, regime, scfg)
            reasons[why] = reasons.get(why, 0) + 1
            if sig is not None:
                self.watches[sig.symbol] = ConsolidationWatch(
                    symbol=sig.symbol,
                    event_date=sig.signal_date,
                    event=sig.event,
                    event_day_high=snap.high,
                    event_day_low=snap.low,
                    event_day_return=sig.event_day_return,
                    volume_ratio=sig.volume_ratio,
                    close_location=sig.close_location,
                    rank_score=sig.rank_score,
                    sessions_elapsed=0,
                )
        return reasons


def run_backtest_v2(
    data: PreparedData,
    settings: Settings,
    v2_cfg: V2Config,
    iwm: BarSeries | None = None,
    initial_capital_usd: float | None = None,
    vps_enabled: bool | None = None,
) -> BacktestResult:
    """Event-driven daily backtest loop for v2 variants (zero look-ahead).

    Reuses untouched ``BacktestBroker``, ``manage_position``, and
    ``risk_manager.build_order_intent`` with ``Settings`` configured via DI.
    """
    if v2_cfg.require_iwm_above_sma200 and iwm is None:
        raise ValueError("iwm BarSeries is required when require_iwm_above_sma200=True")

    eff_settings = apply_v2_settings(settings, v2_cfg)
    iwm_trends = (
        compute_benchmark_trend(iwm, v2_cfg.iwm_sma_window)
        if iwm is not None
        else {}
    )

    cash0 = initial_capital_usd if initial_capital_usd is not None else eff_settings.backtest.initial_capital_usd
    vps = eff_settings.execution.vps_enabled if vps_enabled is None else vps_enabled
    cal = data.calendar
    broker = BacktestBroker(data.benchmark.frame.index, cash0, eff_settings.execution)
    rejections: dict[str, int] = defaultdict(int)
    sizing_rej: dict[str, int] = defaultdict(int)
    rows = []
    last_month: tuple[int, int] | None = None
    bench_close = data.benchmark.frame["close"]
    tracker = ConsolidationTracker(v2_cfg) if v2_cfg.consolidation_breakout else None

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
        iwm_trend = iwm_trends.get(ts)
        next_ts = cal[i + 1] if i + 1 < len(cal) else None
        acct = broker.get_account_state()

        candidates = []
        for e in data.events_by_session.get(ts, []):
            row = bar(e.symbol, ts)
            if row is not None:
                candidates.append((e, snapshot(e.symbol, ts, row)))

        signals: list[TradeSignal] = []
        if tracker is not None:
            watched_snaps: dict[str, FeatureSnapshot] = {}
            for sym in tracker.watches:
                wrow = bar(sym, ts)
                if wrow is not None:
                    watched_snaps[sym] = snapshot(sym, ts, wrow)
            signals, b_why = tracker.step_existing(ts, watched_snaps, regime, iwm_trend)
            for k, v in b_why.items():
                rejections[k] += v
            if candidates:
                r_why = tracker.register_reaction_day(candidates, regime, eff_settings.screener)
                for k, v in r_why.items():
                    rejections[k] += v
        elif candidates:
            signals, why = screen_v2(candidates, regime, eff_settings.screener, v2_cfg, iwm_trend)
            for k, v in why.items():
                rejections[k] += v

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
        rejections=dict(rejections),
        sizing_rejections=dict(sizing_rej),
        fees_paid=broker.fees_paid,
        vps_enabled=vps,
        is_synthetic=data.is_synthetic,
    )
