"""Abstract market-data contract. Strategy/backtest code depends only on this ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from src.domain.models import BarSeries, EarningsEvent, UniverseMember


class MarketDataProvider(ABC):
    """Source of bars, earnings and universe membership.

    Implementations must return exchange-local, tz-naive daily data and must never
    silently substitute synthetic data for real data (set ``is_synthetic`` instead).
    """

    name: str = "base"
    is_synthetic: bool = False

    @abstractmethod
    def get_universe(self, as_of: date | None = None) -> list[UniverseMember]:
        """Candidate symbols. Point-in-time filtering happens in the screener."""

    @abstractmethod
    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> dict[str, BarSeries]:
        """Split-adjusted daily OHLCV plus reconstructed ``raw_close``; missing symbols omitted."""

    @abstractmethod
    def get_earnings_calendar(self, symbols: list[str], start: date, end: date) -> list[EarningsEvent]:
        """Historical earnings announcements with timing and surprise data."""

    @abstractmethod
    def get_benchmark_bars(self, start: date, end: date, symbol: str = "SPY") -> BarSeries:
        """Benchmark series used by the regime filter."""

    def get_shares_outstanding(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        """Reported shares outstanding history (date-indexed) for point-in-time market cap."""
        return {}

    def get_market_cap_history(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        """Vendor point-in-time market cap history; preferred over shares x price when present."""
        return {}

    def describe(self) -> dict[str, str]:
        """Human-readable provenance, surfaced in backtest reports."""
        return {"provider": self.name, "synthetic": str(self.is_synthetic)}
