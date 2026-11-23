"""Market regime from SPY vs SMA(200) and 20-day annualised realised volatility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import RegimeConfig
from src.domain.models import BarSeries, MarketRegime, RegimeState


def classify_regime(spy_close: float, sma200: float, sigma_20: float, cfg: RegimeConfig) -> RegimeState:
    """Pure rule. PANIC is checked first, NORMAL needs both conditions, else CAUTION."""
    if any(math.isnan(x) for x in (spy_close, sma200, sigma_20)):
        return RegimeState.PANIC  # insufficient history: no new entries
    if spy_close < sma200 and sigma_20 > cfg.panic_vol_min:
        return RegimeState.PANIC
    if spy_close > sma200 and sigma_20 < cfg.normal_vol_max:
        return RegimeState.NORMAL
    return RegimeState.CAUTION


def regime_params(state: RegimeState, cfg: RegimeConfig) -> tuple[float, float, int]:
    """(risk multiplier, min event gap, max positions)."""
    if state is RegimeState.NORMAL:
        return 1.0, cfg.normal_min_gap, cfg.max_positions_normal
    if state is RegimeState.CAUTION:
        return cfg.caution_risk_multiplier, cfg.caution_min_gap, cfg.max_positions_caution
    return 0.0, float("inf"), 0


def make_regime(ts: pd.Timestamp, spy_close: float, sma200: float, sigma_20: float, cfg: RegimeConfig) -> MarketRegime:
    state = classify_regime(spy_close, sma200, sigma_20, cfg)
    mult, gap, maxpos = regime_params(state, cfg)
    return MarketRegime(
        date=ts.date(), state=state, spy_close=spy_close, sma200=sma200, sigma_20=sigma_20,
        risk_multiplier=mult, min_event_gap=gap, max_positions=maxpos,
    )


def compute_regimes(spy: BarSeries, cfg: RegimeConfig) -> dict[pd.Timestamp, MarketRegime]:
    close = spy.frame["close"]
    sma = close.rolling(cfg.sma_window, min_periods=cfg.sma_window).mean()
    sigma = np.log(close).diff().rolling(cfg.vol_window, min_periods=cfg.vol_window).std() * math.sqrt(252)
    return {
        ts: make_regime(ts, float(close.loc[ts]), float(sma.loc[ts]), float(sigma.loc[ts]), cfg)
        for ts in close.index
    }
