"""Provider fallback chain: FMP/Massive (keys) -> yfinance (network) -> bundled sample."""

from __future__ import annotations

import logging
from datetime import timedelta

import requests

from src.config import Settings
from src.data.provider_base import MarketDataProvider
from src.data.sample_provider import SampleProvider

log = logging.getLogger(__name__)

YAHOO_PROBE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d"


def yahoo_reachable(timeout: float = 8.0) -> bool:
    try:
        r = requests.get(YAHOO_PROBE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        return r.status_code == 200 and "chart" in r.text
    except requests.RequestException:
        return False


def _sample(settings: Settings) -> SampleProvider:
    bt = settings.backtest
    return SampleProvider(
        start=bt.start_date - timedelta(days=bt.warmup_calendar_days),
        end=bt.end_date,
        n_symbols=settings.data.sample_n_symbols,
        seed=settings.data.sample_seed,
    )


def _yfinance(settings: Settings) -> MarketDataProvider:
    from src.data.yfinance_provider import YFinanceProvider

    bt, d = settings.backtest, settings.data
    since = bt.start_date - timedelta(days=bt.warmup_calendar_days) if d.include_removed_index_members else None
    return YFinanceProvider(d.cache_path, d.yfinance_threads, d.universe_indexes, since)


def build_provider(settings: Settings) -> tuple[MarketDataProvider, list[str]]:
    """Return the provider plus a log of the fallback decisions taken."""
    choice = settings.data.provider
    notes: list[str] = []
    fmp_key = settings.fmp_api_key.get_secret_value() if settings.fmp_api_key else None
    massive_key = settings.massive_api_key.get_secret_value() if settings.massive_api_key else None

    if choice == "sample":
        notes.append("provider=sample requested: using bundled SYNTHETIC dataset")
        return _sample(settings), notes

    online = yahoo_reachable()
    notes.append(f"yahoo reachable: {online}")

    if choice in ("auto", "fmp_massive") and (fmp_key or massive_key):
        from src.data.fmp_massive_provider import FMPMassiveProvider, FMPProvider, MassiveProvider

        d = settings.data
        fmp = FMPProvider(fmp_key, d.fmp_base_url, d.cache_path, d.fmp_calls_per_minute) if fmp_key else None
        massive = (
            MassiveProvider(massive_key, d.massive_base_url, d.cache_path, d.massive_calls_per_minute)
            if massive_key else None
        )
        fallback = _yfinance(settings) if online else _sample(settings)
        notes.append(f"FMP key: {bool(fmp_key)}, MASSIVE key: {bool(massive_key)}; "
                     f"uncovered capabilities fall back to {fallback.name}")
        if fallback.is_synthetic:
            notes.append("WARNING: keyed data mixed with SYNTHETIC fallback; results are not real-world")
        return FMPMassiveProvider(fallback=fallback, fmp=fmp, massive=massive), notes
    if choice == "fmp_massive":
        raise RuntimeError("provider=fmp_massive but neither FMP_API_KEY nor MASSIVE_API_KEY is set")

    if online and choice in ("auto", "yfinance"):
        notes.append("no API keys: using yfinance + Wikipedia index universe")
        return _yfinance(settings), notes
    if choice == "yfinance":
        raise RuntimeError("provider=yfinance but Yahoo Finance is unreachable")

    notes.append("no keys and no network: using bundled SYNTHETIC dataset")
    return _sample(settings), notes
