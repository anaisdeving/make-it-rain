"""Deterministic PEAD signal generator (pure functions, no I/O, no state).

Input is only what is knowable at the close of the reaction session: the event,
a :class:`FeatureSnapshot` for that close, and the regime at that close.
"""

from __future__ import annotations

import math

from src.config import ScreenerConfig
from src.domain.models import EarningsEvent, FeatureSnapshot, MarketRegime, RegimeState, TradeSignal


def surprise_passes(event: EarningsEvent, cfg: ScreenerConfig) -> bool:
    """EPS (and revenue, when known or required) surprise > threshold, or SUE > threshold."""
    eps = event.eps_surprise_pct
    if eps is None and event.eps_actual is not None and event.eps_estimate not in (None, 0):
        eps = (event.eps_actual - event.eps_estimate) / abs(event.eps_estimate) * 100  # type: ignore[operator]
    eps_ok = eps is not None and eps > cfg.min_eps_surprise_pct
    rev = event.revenue_surprise_pct
    rev_ok = (not cfg.require_revenue_surprise) if rev is None else rev > cfg.min_revenue_surprise_pct
    sue_ok = event.sue is not None and event.sue > cfg.sue_threshold
    return (eps_ok and rev_ok) or sue_ok


def universe_passes(s: FeatureSnapshot, cfg: ScreenerConfig) -> tuple[bool, str]:
    if not (cfg.min_price <= s.raw_close <= cfg.max_price):
        return False, "price"
    if s.market_cap is None or math.isnan(s.market_cap):
        return False, "no_market_cap"
    if not (cfg.min_market_cap <= s.market_cap <= cfg.max_market_cap):
        return False, "market_cap"
    if not s.avg_dollar_volume_20 >= cfg.min_avg_dollar_volume:
        return False, "liquidity"
    return True, "ok"


def evaluate_event(
    event: EarningsEvent, snap: FeatureSnapshot, regime: MarketRegime, cfg: ScreenerConfig
) -> tuple[TradeSignal | None, str]:
    """Return (signal, "ok") or (None, rejection reason)."""
    if regime.state is RegimeState.PANIC:
        return None, "regime_panic"
    needed = (snap.prev_close, snap.avg_volume_20, snap.avg_dollar_volume_20, snap.sma50, snap.high_52w, snap.atr14)
    if any(math.isnan(x) for x in needed):
        return None, "insufficient_history"
    ok, why = universe_passes(snap, cfg)
    if not ok:
        return None, why
    if not surprise_passes(event, cfg):
        return None, "surprise"
    ret = snap.close / snap.prev_close - 1.0
    if ret < regime.min_event_gap:
        return None, "event_gap"
    vol_ratio = snap.volume / snap.avg_volume_20 if snap.avg_volume_20 > 0 else 0.0
    if vol_ratio < cfg.min_volume_ratio:
        return None, "volume"
    rng = snap.high - snap.low
    close_loc = (snap.close - snap.low) / rng if rng > 0 else 0.0
    if close_loc < cfg.min_close_location:
        return None, "close_location"
    if not snap.close > snap.sma50:
        return None, "below_sma50"
    if snap.close < (1.0 - cfg.max_distance_from_52w_high) * snap.high_52w:
        return None, "far_from_52w_high"
    score = ret
    if cfg.analyst_tiebreak_enabled and event.analyst_count is not None and event.analyst_count < cfg.analyst_tiebreak_max:
        score += 1.0
    return (
        TradeSignal(
            symbol=event.symbol, signal_date=snap.date, event=event, regime=regime.state,
            event_day_return=ret, volume_ratio=vol_ratio, close_location=close_loc,
            event_day_low=snap.low, signal_close=snap.close, atr14=snap.atr14,
            market_cap=snap.market_cap, rank_score=score,
        ),
        "ok",
    )


def screen(
    candidates: list[tuple[EarningsEvent, FeatureSnapshot]], regime: MarketRegime, cfg: ScreenerConfig
) -> tuple[list[TradeSignal], dict[str, int]]:
    """Signals ranked best-first (score desc, symbol asc for determinism) + rejection counts."""
    signals: list[TradeSignal] = []
    reasons: dict[str, int] = {}
    for event, snap in candidates:
        sig, why = evaluate_event(event, snap, regime, cfg)
        reasons[why] = reasons.get(why, 0) + 1
        if sig is not None:
            signals.append(sig)
    signals.sort(key=lambda s: (-s.rank_score, s.symbol))
    return signals, reasons
