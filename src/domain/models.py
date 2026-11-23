"""Immutable data contracts shared by every module.

Every object crossing a module boundary is one of these frozen pydantic models
(or a :class:`BarSeries`, which wraps a schema-validated DataFrame for speed).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnnouncementTiming(str, Enum):
    BMO = "BMO"          # before market open -> reaction session is the same day
    DMH = "DMH"          # during market hours -> reaction session is the same day
    AMC = "AMC"          # after market close -> reaction session is the next day
    UNKNOWN = "UNKNOWN"  # treated like AMC (the only look-ahead-safe assumption)


class RegimeState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    PANIC = "PANIC"


class ExitReason(str, Enum):
    STOP = "STOP"
    GAP_STOP = "GAP_STOP"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    TRAIL_STOP = "TRAIL_STOP"
    TIME_STOP = "TIME_STOP"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    REBALANCE_EXIT = "REBALANCE_EXIT"
    REGIME_EXIT = "REGIME_EXIT"
    END_OF_TEST = "END_OF_TEST"
    DATA_END = "DATA_END"      # symbol's price history ended (e.g. delisted) while held
    MANUAL = "MANUAL"


class DailyBar(_Frozen):
    symbol: str
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    raw_close: float | None = Field(None, gt=0)  # unadjusted close (for price / mcap filters)

    @model_validator(mode="after")
    def _ohlc_consistent(self) -> "DailyBar":
        tol = 1e-6 * self.high
        if self.low > min(self.open, self.close) + tol or self.high < max(self.open, self.close) - tol:
            raise ValueError(f"inconsistent OHLC for {self.symbol} {self.date}")
        return self


BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "raw_close")


class BarSeries(BaseModel):
    """Daily bars for one symbol: DatetimeIndex (tz-naive, ascending, unique) x BAR_COLUMNS.

    Prices are split-adjusted; ``raw_close`` is the as-traded close reconstructed from
    split events, used only for the point-in-time price and market-cap filters.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbol: str
    frame: pd.DataFrame

    @field_validator("frame")
    @classmethod
    def _validate_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in BAR_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"bar frame missing columns {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("bar frame index must be a DatetimeIndex")
        if df.index.tz is not None:
            raise ValueError("bar frame index must be tz-naive (exchange-local dates)")
        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError("bar frame index must be ascending and unique")
        out = df.loc[:, list(BAR_COLUMNS)].astype(float)
        if out[["open", "high", "low", "close"]].le(0).any().any():
            raise ValueError("non-positive prices in bar frame")
        return out

    def bar(self, ts: pd.Timestamp) -> DailyBar:
        r = self.frame.loc[ts]
        return DailyBar(symbol=self.symbol, date=ts.date(), **{k: float(r[k]) for k in BAR_COLUMNS})

    def __len__(self) -> int:
        return len(self.frame)


class UniverseMember(_Frozen):
    symbol: str
    name: str | None = None
    exchange: str | None = None
    is_etf: bool = False
    source: str
    removed_on: date | None = None  # date removed from the source index, if known


class EarningsEvent(_Frozen):
    symbol: str
    announced_at: datetime  # exchange-local (America/New_York) naive datetime
    timing: AnnouncementTiming
    fiscal_period: str | None = None
    eps_estimate: float | None = None
    eps_actual: float | None = None
    eps_surprise_pct: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    revenue_surprise_pct: float | None = None
    sue: float | None = None
    analyst_count: int | None = None
    source: str


class MarketRegime(_Frozen):
    date: date
    state: RegimeState
    spy_close: float
    sma200: float
    sigma_20: float
    risk_multiplier: float = Field(ge=0, le=1)
    min_event_gap: float
    max_positions: int = Field(ge=0)


class FeatureSnapshot(_Frozen):
    """Everything the screener may know about a symbol at the close of ``date``."""

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
    high_52w: float
    atr14: float
    market_cap: float | None


class TradeSignal(_Frozen):
    symbol: str
    signal_date: date          # close of the reaction session; entry is the next open
    event: EarningsEvent
    regime: RegimeState
    event_day_return: float
    volume_ratio: float
    close_location: float
    event_day_low: float
    signal_close: float
    atr14: float
    market_cap: float | None
    rank_score: float


class OrderIntent(_Frozen):
    symbol: str
    signal_date: date
    shares: int = Field(gt=0)
    limit_price: float = Field(gt=0)   # entry limit = signal close x (1 + cap)
    stop_price: float = Field(gt=0)
    stop_distance: float = Field(gt=0)
    risk_fraction: float = Field(gt=0)
    regime: RegimeState
    signal: TradeSignal

    @model_validator(mode="after")
    def _stop_below_limit(self) -> "OrderIntent":
        if self.stop_price >= self.limit_price:
            raise ValueError("stop must be below entry limit")
        return self


class Position(_Frozen):
    symbol: str
    shares: int = Field(gt=0)
    entry_date: date
    entry_price: float
    initial_stop: float
    stop_price: float
    highest_high: float
    bars_held: int = 0
    entry_commission: float
    risk_fraction: float
    regime_at_entry: RegimeState
    signal: TradeSignal
    breakeven_armed: bool = False
    trail_armed: bool = False
    pending_exit: ExitReason | None = None

    @property
    def r_distance(self) -> float:
        return self.entry_price - self.initial_stop

    def r_multiple(self, price: float) -> float:
        return (price - self.entry_price) / self.r_distance

    def evolve(self, **changes: Any) -> "Position":
        return self.model_copy(update=changes)


class ClosedTrade(_Frozen):
    symbol: str
    signal_date: date
    entry_date: date
    exit_date: date
    shares: int
    entry_price: float
    exit_price: float
    initial_stop: float
    exit_reason: ExitReason
    bars_held: int
    gross_pnl: float
    commissions: float
    net_pnl: float
    initial_risk_usd: float
    r_multiple: float          # net P&L / initial risk (shares x (entry - initial stop))
    regime_at_entry: RegimeState
    event_timing: AnnouncementTiming
    eps_surprise_pct: float | None
    event_day_return: float


class AccountState(_Frozen):
    date: date
    settled_cash: float
    unsettled_cash: float
    market_value: float
    open_positions: int

    @property
    def total_cash(self) -> float:
        return self.settled_cash + self.unsettled_cash

    @property
    def equity(self) -> float:
        return self.total_cash + self.market_value


class SkippedEntry(_Frozen):
    symbol: str
    signal_date: date
    attempt_date: date
    reason: str
    open_price: float | None = None
    limit_price: float | None = None
