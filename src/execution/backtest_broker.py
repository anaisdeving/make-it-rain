"""Simulated IBKR-style cash account with bracket orders, T+1 settlement and frictions.

Fill rules (no magical fills):
* Entry: limit = signal close x (1 + cap). If the next open > limit the order is
  skipped (counted as ``gap_above_limit``). Otherwise fill = min(open x (1+slip), limit).
* Stop: if a session opens at/below the stop the position exits at the open (gap-through);
  otherwise, if the low touches the stop, it exits at the stop. Slippage applies to both.
* Market exits (time stop / manual) fill at the next open less slippage.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from src.config import ExecutionConfig
from src.domain.models import (
    AccountState,
    ClosedTrade,
    ExitReason,
    OrderIntent,
    Position,
    SkippedEntry,
)
from src.execution.broker_base import ExecutionBroker
from src.risk.risk_manager import SettlementLedger


class BacktestBroker(ExecutionBroker):
    def __init__(self, calendar: pd.DatetimeIndex, initial_cash: float, cfg: ExecutionConfig) -> None:
        self.cfg = cfg
        self.ledger = SettlementLedger(calendar=calendar, settled=initial_cash, settlement_days=cfg.settlement_days)
        self.positions: dict[str, Position] = {}
        self.pending_orders: dict[str, OrderIntent] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.skipped: list[SkippedEntry] = []
        self.fees_paid = 0.0
        self._marks: dict[str, float] = {}
        self._today: pd.Timestamp = calendar[0]
        self._ids = itertools.count(1)

    # ---- ExecutionBroker interface -------------------------------------------------
    def submit_bracket_order(self, intent: OrderIntent) -> str:
        if intent.symbol in self.positions or intent.symbol in self.pending_orders:
            raise ValueError(f"duplicate order for {intent.symbol}")
        self.pending_orders[intent.symbol] = intent
        return f"BT-{next(self._ids)}"

    def update_stop(self, symbol: str, new_stop: float) -> None:
        p = self.positions[symbol]
        if new_stop > p.stop_price:
            self.positions[symbol] = p.evolve(stop_price=new_stop)

    def close_position(self, symbol: str, reason: ExitReason) -> None:
        self.positions[symbol] = self.positions[symbol].evolve(pending_exit=reason)

    def get_account_state(self) -> AccountState:
        mv = sum(p.shares * self._marks.get(s, p.entry_price) for s, p in self.positions.items())
        return AccountState(
            date=self._today.date(), settled_cash=self.ledger.settled, unsettled_cash=self.ledger.unsettled,
            market_value=mv, open_positions=len(self.positions),
        )

    # ---- simulation ---------------------------------------------------------------
    def charge_fee(self, amount: float) -> None:
        self.ledger.debit_fee(amount)
        self.fees_paid += amount

    def _sell_px(self, px: float) -> float:
        return px * (1.0 - self.cfg.slippage_per_side_pct)

    def _exit(self, ts: pd.Timestamp, p: Position, raw_px: float, reason: ExitReason) -> None:
        px = self._sell_px(raw_px)
        comm = self.cfg.commission_per_side_usd
        self.ledger.credit_sale(ts, p.shares * px - comm)
        gross = p.shares * (px - p.entry_price)
        commissions = p.entry_commission + comm
        risk_usd = p.shares * (p.entry_price - p.initial_stop)
        net = gross - commissions
        sig = p.signal
        self.closed_trades.append(ClosedTrade(
            symbol=p.symbol, signal_date=sig.signal_date, entry_date=p.entry_date, exit_date=ts.date(),
            shares=p.shares, entry_price=p.entry_price, exit_price=px, initial_stop=p.initial_stop,
            exit_reason=reason, bars_held=p.bars_held, gross_pnl=gross, commissions=commissions, net_pnl=net,
            initial_risk_usd=risk_usd, r_multiple=net / risk_usd if risk_usd > 0 else 0.0,
            regime_at_entry=p.regime_at_entry, event_timing=sig.event.timing,
            eps_surprise_pct=sig.event.eps_surprise_pct, event_day_return=sig.event_day_return,
        ))
        del self.positions[p.symbol]
        self._marks.pop(p.symbol, None)

    def _stop_reason(self, p: Position, gapped: bool) -> ExitReason:
        if gapped:
            return ExitReason.GAP_STOP
        if p.trail_armed:
            return ExitReason.TRAIL_STOP
        if p.breakeven_armed:
            return ExitReason.BREAKEVEN_STOP
        return ExitReason.STOP

    def process_session(self, ts: pd.Timestamp, bars: dict[str, pd.Series]) -> None:
        """Run one session: settle cash, open exits, entries, intraday stops, marks."""
        self._today = ts
        self.ledger.roll(ts)

        for sym, p in list(self.positions.items()):
            bar = bars.get(sym)
            if bar is None:
                continue
            if p.pending_exit is not None:
                self._exit(ts, p, float(bar["open"]), p.pending_exit)
            elif bar["open"] <= p.stop_price:
                self._exit(ts, p, float(bar["open"]), self._stop_reason(p, gapped=True))

        for sym, intent in list(self.pending_orders.items()):
            del self.pending_orders[sym]
            self._try_fill(ts, intent, bars.get(sym))

        for sym, p in list(self.positions.items()):
            bar = bars.get(sym)
            if bar is None:
                continue
            if bar["low"] <= p.stop_price:
                gapped = bar["open"] <= p.stop_price
                self._exit(ts, p, float(min(bar["open"], p.stop_price)), self._stop_reason(p, gapped))
            else:
                self._marks[sym] = float(bar["close"])

    def _try_fill(self, ts: pd.Timestamp, intent: OrderIntent, bar: pd.Series | None) -> None:
        def skip(reason: str, open_px: float | None = None) -> None:
            self.skipped.append(SkippedEntry(
                symbol=intent.symbol, signal_date=intent.signal_date, attempt_date=ts.date(),
                reason=reason, open_price=open_px, limit_price=intent.limit_price,
            ))

        if bar is None:
            skip("no_bar")
            return
        open_px = float(bar["open"])
        if open_px > intent.limit_price:
            skip("gap_above_limit", open_px)
            return
        fill = min(open_px * (1.0 + self.cfg.slippage_per_side_pct), intent.limit_price)
        comm = self.cfg.commission_per_side_usd
        shares = intent.shares
        if shares * fill + comm > self.ledger.settled:
            shares = math.floor(max(self.ledger.settled - comm, 0.0) / fill)
        if shares < 1:
            skip("insufficient_settled_cash", open_px)
            return
        self.ledger.debit_purchase(shares * fill + comm)
        self.positions[intent.symbol] = Position(
            symbol=intent.symbol, shares=shares, entry_date=ts.date(), entry_price=fill,
            initial_stop=intent.stop_price, stop_price=intent.stop_price, highest_high=fill,
            entry_commission=comm, risk_fraction=intent.risk_fraction, regime_at_entry=intent.regime,
            signal=intent.signal,
        )
        self._marks[intent.symbol] = fill

    def force_exit(self, ts: pd.Timestamp, symbol: str, price: float, reason: ExitReason) -> None:
        self._exit(ts, self.positions[symbol], price, reason)

    def mark(self, symbol: str, price: float) -> None:
        self._marks[symbol] = price
