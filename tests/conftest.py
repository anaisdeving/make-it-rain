from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, load_settings  # noqa: E402
from src.domain.models import (  # noqa: E402
    AnnouncementTiming,
    BarSeries,
    EarningsEvent,
    FeatureSnapshot,
    MarketRegime,
    RegimeState,
    TradeSignal,
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


def make_bars(closes: list[float] | np.ndarray, start: str = "2024-01-02", symbol: str = "TST",
              spread: float = 0.01, volume: float = 1e6) -> BarSeries:
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) * (1 + spread),
        "low": np.minimum(opens, closes) * (1 - spread),
        "close": closes,
        "volume": np.full(len(closes), volume),
        "raw_close": closes,
    }, index=idx)
    return BarSeries(symbol=symbol, frame=df)


def make_event(symbol: str = "TST", when: datetime = datetime(2024, 5, 1, 7, 0),
               timing: AnnouncementTiming = AnnouncementTiming.BMO, eps_surprise: float | None = 10.0,
               revenue_surprise: float | None = None, sue: float | None = None) -> EarningsEvent:
    return EarningsEvent(symbol=symbol, announced_at=when, timing=timing, eps_estimate=1.0,
                         eps_actual=None if eps_surprise is None else 1 + eps_surprise / 100,
                         eps_surprise_pct=eps_surprise, revenue_surprise_pct=revenue_surprise,
                         sue=sue, source="test")


def make_snapshot(**kw) -> FeatureSnapshot:
    base = dict(symbol="TST", date=date(2024, 5, 1), open=52.0, high=55.5, low=51.5, close=55.0,
                prev_close=50.0, raw_close=55.0, volume=3e6, avg_volume_20=1e6,
                avg_dollar_volume_20=50e6, sma50=48.0, high_52w=58.0, atr14=1.5, market_cap=2e9)
    base.update(kw)
    return FeatureSnapshot(**base)


def make_regime(state: RegimeState = RegimeState.NORMAL, d: date = date(2024, 5, 1)) -> MarketRegime:
    params = {RegimeState.NORMAL: (1.0, 0.05, 3), RegimeState.CAUTION: (0.5, 0.08, 2),
              RegimeState.PANIC: (0.0, float("inf"), 0)}[state]
    return MarketRegime(date=d, state=state, spy_close=500, sma200=480, sigma_20=0.15,
                        risk_multiplier=params[0], min_event_gap=params[1], max_positions=params[2])


def make_signal(symbol: str = "TST", close: float = 55.0, low: float = 51.5, atr: float = 1.5,
                d: date = date(2024, 5, 1), regime: RegimeState = RegimeState.NORMAL) -> TradeSignal:
    return TradeSignal(symbol=symbol, signal_date=d, event=make_event(symbol), regime=regime,
                       event_day_return=0.1, volume_ratio=3.0, close_location=0.9, event_day_low=low,
                       signal_close=close, atr14=atr, market_cap=2e9, rank_score=0.1)
