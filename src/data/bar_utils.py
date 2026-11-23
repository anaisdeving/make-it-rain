"""Normalisation helpers shared by real-data providers."""

from __future__ import annotations

import pandas as pd

from src.domain.models import BarSeries


def raw_close_from_splits(close: pd.Series, splits: pd.Series | None) -> pd.Series:
    """Reconstruct the as-traded close from a split-adjusted close.

    ``splits`` holds the split ratio on its effective date (2.0 for 2-for-1, 0 or NaN
    otherwise). The as-traded price at t is the adjusted price times every split
    ratio that became effective *after* t.
    """
    if splits is None or splits.empty:
        return close.copy()
    ratio = splits.reindex(close.index).fillna(0.0)
    ratio = ratio.where(ratio > 0, 1.0)
    factor_from_t = ratio[::-1].cumprod()[::-1]
    factor_after_t = factor_from_t.shift(-1).fillna(1.0)
    return close * factor_after_t


def normalise_ohlcv(symbol: str, df: pd.DataFrame, splits: pd.Series | None = None) -> BarSeries | None:
    """Lower-case columns, tz-naive dates, drop bad rows, repair minor OHLC glitches."""
    if df is None or df.empty:
        return None
    out = df.rename(columns=str.lower)
    out = out[["open", "high", "low", "close", "volume"]].copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([float("inf"), float("-inf")], float("nan")).dropna()
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if out.empty:
        return None
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    if splits is not None and not splits.empty:
        s_idx = pd.DatetimeIndex(splits.index)
        if s_idx.tz is not None:
            s_idx = s_idx.tz_localize(None)
        splits = pd.Series(splits.values, index=s_idx.normalize())
        splits = splits.groupby(level=0).max()
    out["raw_close"] = raw_close_from_splits(out["close"], splits)
    return BarSeries(symbol=symbol, frame=out)
