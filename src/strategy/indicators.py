"""Causal (trailing-window) indicators. Every value at row t uses rows <= t only.

``tests/test_lookahead.py`` asserts truncation invariance: features computed on the
full history equal features computed on history truncated at t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RiskConfig, ScreenerConfig
from src.domain.models import BarSeries, EarningsEvent


def wilder_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def asof_series(values: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    """Latest observation dated <= each index date (never back-filled)."""
    if values is None or len(values) == 0:
        return pd.Series(np.nan, index=index)
    v = values.sort_index()
    v = v[~v.index.duplicated(keep="last")]
    return v.reindex(v.index.union(index)).ffill().reindex(index)


def compute_features(
    bars: BarSeries,
    scfg: ScreenerConfig,
    rcfg: RiskConfig,
    shares: pd.Series | None = None,
    market_cap: pd.Series | None = None,
) -> pd.DataFrame:
    df = bars.frame
    w = scfg.avg_window
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"] = df["open"], df["high"], df["low"], df["close"]
    out["volume"], out["raw_close"] = df["volume"], df["raw_close"]
    out["prev_close"] = df["close"].shift(1)
    # Averages use the w sessions *before* t so the event-day spike does not inflate them.
    out["avg_volume_20"] = df["volume"].shift(1).rolling(w, min_periods=w).mean()
    out["avg_dollar_volume_20"] = (df["close"] * df["volume"]).shift(1).rolling(w, min_periods=w).mean()
    out["sma50"] = df["close"].rolling(scfg.trend_sma_window, min_periods=scfg.trend_sma_window).mean()
    out["high_52w"] = df["high"].rolling(scfg.high_52w_window, min_periods=scfg.high_52w_window).max()
    out["atr14"] = wilder_atr(df, rcfg.atr_window)
    if market_cap is not None and len(market_cap):
        out["market_cap"] = asof_series(market_cap, df.index)
    else:
        out["market_cap"] = asof_series(shares, df.index) * df["raw_close"]
    return out


def add_sue(events: list[EarningsEvent], lookback: int, min_quarters: int) -> list[EarningsEvent]:
    """Standardised unexpected earnings using only *prior* quarters' surprise dispersion."""
    by_sym: dict[str, list[EarningsEvent]] = {}
    for e in events:
        by_sym.setdefault(e.symbol, []).append(e)
    out: list[EarningsEvent] = []
    for evs in by_sym.values():
        evs.sort(key=lambda e: e.announced_at)
        history: list[float] = []
        for e in evs:
            sue = None
            if e.eps_actual is not None and e.eps_estimate is not None:
                surprise = e.eps_actual - e.eps_estimate
                past = history[-lookback:]
                if len(past) >= min_quarters:
                    sd = float(np.std(past, ddof=1))
                    sue = surprise / sd if sd > 1e-9 else None
                history.append(surprise)
            out.append(e.model_copy(update={"sue": sue}) if sue is not None else e)
    return out
