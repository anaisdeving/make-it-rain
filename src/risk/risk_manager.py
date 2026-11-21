"""Deterministic position sizing and cash-account guard.

THE ONLY PLACE RISK MATH LIVES. Stop distance, share count, risk fraction and the
T+1 settled-cash rules must not be re-implemented elsewhere.

    d     = min(max(atr_mult * ATR14, Entry - EventDayLow), max_stop_pct * Entry)
    f     = base_risk_per_trade (<= max_risk_per_trade) x regime multiplier
    shares= floor(min(Equity*f/d, Equity*max_pos_pct/Entry, (SettledCash - commission)/Entry))
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field

import pandas as pd

from src.config import ExecutionConfig, RiskConfig
from src.domain.models import MarketRegime, OrderIntent, TradeSignal


class GoodFaithViolation(RuntimeError):
    """Raised when a purchase would use unsettled funds in a cash account."""


def stop_distance(entry: float, atr14: float, event_day_low: float, cfg: RiskConfig) -> float:
    return min(max(cfg.atr_stop_multiple * atr14, entry - event_day_low), cfg.max_stop_pct * entry)


def risk_fraction(regime: MarketRegime, cfg: RiskConfig) -> float:
    return min(cfg.base_risk_per_trade, cfg.max_risk_per_trade) * regime.risk_multiplier


def position_size(
    equity: float,
    entry: float,
    distance: float,
    fraction: float,
    settled_cash: float,
    commission: float,
    cfg: RiskConfig,
) -> int:
    """Whole shares only (floor); never more than settled cash can pay for."""
    if equity <= 0 or entry <= 0 or distance <= 0 or fraction <= 0:
        return 0
    by_risk = equity * fraction / distance
    by_notional = equity * cfg.max_position_pct_equity / entry
    by_cash = max(settled_cash - commission, 0.0) / entry
    shares = math.floor(min(by_risk, by_notional, by_cash) + 1e-9)
    return max(shares, 0)


def build_order_intent(
    signal: TradeSignal,
    regime: MarketRegime,
    equity: float,
    settled_cash: float,
    rcfg: RiskConfig,
    xcfg: ExecutionConfig,
) -> tuple[OrderIntent | None, str]:
    """Size a bracket order using the worst-case entry (the limit price)."""
    limit = signal.signal_close * (1.0 + xcfg.entry_limit_cap_pct)
    f = risk_fraction(regime, rcfg)
    if f <= 0:
        return None, "zero_risk_budget"
    d = stop_distance(limit, signal.atr14, signal.event_day_low, rcfg)
    if not d > 0:
        return None, "invalid_stop"
    worst_fill = limit
    shares = position_size(equity, worst_fill, d, f, settled_cash, xcfg.commission_per_side_usd, rcfg)
    if shares < 1:
        return None, "size_below_one_share"
    return (
        OrderIntent(
            symbol=signal.symbol, signal_date=signal.signal_date, shares=shares,
            limit_price=limit, stop_price=limit - d, stop_distance=d, risk_fraction=f,
            regime=regime.state, signal=signal,
        ),
        "ok",
    )


@dataclass
class SettlementLedger:
    """Cash-account ledger: sale proceeds settle ``settlement_days`` sessions after trade date."""

    calendar: pd.DatetimeIndex
    settled: float
    settlement_days: int = 1
    pending: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def settle_date(self, trade_ts: pd.Timestamp) -> pd.Timestamp:
        i = self.calendar.get_loc(trade_ts) if trade_ts in self.calendar else bisect_right(self.calendar, trade_ts) - 1
        j = i + self.settlement_days
        if j < len(self.calendar):
            return self.calendar[j]
        return trade_ts + pd.offsets.BDay(self.settlement_days)

    def roll(self, ts: pd.Timestamp) -> None:
        """Move every amount whose settlement date is <= ts into settled cash."""
        keep = []
        for when, amt in self.pending:
            if when <= ts:
                self.settled += amt
            else:
                keep.append((when, amt))
        self.pending = keep

    def settled_by(self, ts: pd.Timestamp) -> float:
        """Settled cash that will be available at the start of session ``ts``."""
        return self.settled + sum(a for when, a in self.pending if when <= ts)

    @property
    def unsettled(self) -> float:
        return sum(a for _, a in self.pending)

    def credit_sale(self, trade_ts: pd.Timestamp, amount: float) -> None:
        if self.settlement_days == 0:
            self.settled += amount
        else:
            self.pending.append((self.settle_date(trade_ts), amount))

    def debit_purchase(self, amount: float) -> None:
        if amount > self.settled + 1e-9:
            raise GoodFaithViolation(f"purchase {amount:.2f} exceeds settled cash {self.settled:.2f}")
        self.settled -= amount

    def debit_fee(self, amount: float) -> None:
        """Fees (commission on sales, VPS) are charged even if that takes settled cash negative."""
        self.settled -= amount
