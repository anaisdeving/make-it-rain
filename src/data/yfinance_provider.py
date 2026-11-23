"""Free fallback provider: Yahoo Finance via ``yfinance`` + Wikipedia index universe.

Known limits (surfaced in reports): no delisted tickers, no revenue surprises,
announcement times are Yahoo's best guess, shares outstanding are as-reported but
sparse, and earnings estimates are Yahoo's consensus snapshot.
"""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from src.data.bar_utils import normalise_ohlcv
from src.data.cache import DiskCache
from src.data.provider_base import MarketDataProvider
from src.data.universe import load_index_universe
from src.domain.models import AnnouncementTiming, BarSeries, EarningsEvent, UniverseMember

log = logging.getLogger(__name__)

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class _Transient(Exception):
    """Fetch failed for a retryable reason (rate limit / network): never cache as empty."""


def _is_missing(exc: Exception) -> bool:
    from yfinance import exceptions as yfe

    return isinstance(exc, (yfe.YFPricesMissingError, yfe.YFTickerMissingError, yfe.YFTzMissingError)) or (
        "delisted" in str(exc).lower() or "no data found" in str(exc).lower()
    )


def _with_retry(fn, what: str, attempts: int = 7):
    """Call ``fn``; back off on rate limits / transient errors. Returns None for genuinely missing data."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if _is_missing(exc):
                return None
            transient = "too many requests" in str(exc).lower() or "rate limit" in str(exc).lower()
            if k == attempts - 1:
                raise _Transient(f"{what}: {exc}") from exc
            _time.sleep(min(5 * 2**k, 120) if transient else 2)
    return None


def classify_timing(ts: datetime) -> AnnouncementTiming:
    """Map an exchange-local announcement time to a session timing.

    Midnight timestamps carry no time-of-day information and map to UNKNOWN.
    """
    t = ts.time()
    if t == time(0, 0):
        return AnnouncementTiming.UNKNOWN
    if t < MARKET_OPEN:
        return AnnouncementTiming.BMO
    if t >= MARKET_CLOSE:
        return AnnouncementTiming.AMC
    return AnnouncementTiming.DMH


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def __init__(
        self,
        cache_root: Path,
        threads: int = 8,
        universe_indexes: list[str] | None = None,
        include_removed_since: date | None = None,
    ) -> None:
        import yfinance as yf

        yf.config.debug.hide_exceptions = False  # surface rate limits instead of empty frames
        self.cache = DiskCache(cache_root, "yfinance")
        self.threads = threads
        self.universe_indexes = universe_indexes or ["sp400", "sp600"]
        self.include_removed_since = include_removed_since

    def get_universe(self, as_of: date | None = None) -> list[UniverseMember]:
        return load_index_universe(self.cache, self.universe_indexes, self.include_removed_since)

    # ---- bars ---------------------------------------------------------------------
    def _bars_one(self, symbol: str, start: date, end: date) -> BarSeries | None:
        key = f"{symbol}_{start}_{end}"
        status = self.cache.status("bars", key)
        if status == "ok":
            df = self.cache.read("bars", key)
            return BarSeries(symbol=symbol, frame=df) if df is not None else None
        if status == "empty":
            return None
        import yfinance as yf

        try:
            hist = _with_retry(lambda: yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
            ), f"bars {symbol}")
        except _Transient as exc:
            log.warning("bars fetch failed (not cached): %s", exc)
            return None
        if hist is None or hist.empty:
            self.cache.write("bars", key, None, status="empty")
            return None
        splits = hist["Stock Splits"] if "Stock Splits" in hist.columns else None
        series = normalise_ohlcv(symbol, hist, splits)
        if series is None:
            self.cache.write("bars", key, None, status="empty")
            return None
        self.cache.write("bars", key, series.frame)
        return series

    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> dict[str, BarSeries]:
        with ThreadPoolExecutor(self.threads) as ex:
            results = list(ex.map(lambda s: (s, self._bars_one(s, start, end)), symbols))
        return {s: b for s, b in results if b is not None}

    def get_benchmark_bars(self, start: date, end: date, symbol: str = "SPY") -> BarSeries:
        series = self._bars_one(symbol, start, end)
        if series is None:
            raise RuntimeError(f"benchmark {symbol} unavailable from yfinance")
        return series

    # ---- earnings -----------------------------------------------------------------
    def _earnings_one(self, symbol: str) -> pd.DataFrame | None:
        status = self.cache.status("earnings", symbol)
        if status == "ok":
            return self.cache.read("earnings", symbol)
        if status == "empty":
            return None
        import yfinance as yf

        try:
            df = _with_retry(lambda: yf.Ticker(symbol).get_earnings_dates(limit=32), f"earnings {symbol}")
        except _Transient as exc:
            log.warning("earnings fetch failed (not cached): %s", exc)
            return None
        if df is None or df.empty:
            self.cache.write("earnings", symbol, None, status="empty")
            return None
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        out = pd.DataFrame(
            {
                "announced_at": idx,
                "eps_estimate": pd.to_numeric(df.get("EPS Estimate"), errors="coerce").values,
                "eps_actual": pd.to_numeric(df.get("Reported EPS"), errors="coerce").values,
                "eps_surprise_pct": pd.to_numeric(df.get("Surprise(%)"), errors="coerce").values,
            }
        )
        self.cache.write("earnings", symbol, out)
        return out

    def get_earnings_calendar(self, symbols: list[str], start: date, end: date) -> list[EarningsEvent]:
        with ThreadPoolExecutor(self.threads) as ex:
            frames = list(ex.map(lambda s: (s, self._earnings_one(s)), symbols))
        events: list[EarningsEvent] = []
        for sym, df in frames:
            if df is None:
                continue
            for r in df.itertuples(index=False):
                ts = pd.Timestamp(r.announced_at).to_pydatetime()
                if not (start <= ts.date() <= end) or pd.isna(r.eps_actual):
                    continue
                events.append(
                    EarningsEvent(
                        symbol=sym,
                        announced_at=ts,
                        timing=classify_timing(ts),
                        eps_estimate=_f(r.eps_estimate),
                        eps_actual=_f(r.eps_actual),
                        eps_surprise_pct=_f(r.eps_surprise_pct),
                        source="yfinance",
                    )
                )
        return events

    # ---- shares -------------------------------------------------------------------
    def _shares_one(self, symbol: str, start: date, end: date) -> pd.Series | None:
        key = f"{symbol}_{start}_{end}"
        status = self.cache.status("shares", key)
        if status == "ok":
            df = self.cache.read("shares", key)
            return df["shares"] if df is not None else None
        if status == "empty":
            return None
        import yfinance as yf

        try:
            s = _with_retry(lambda: yf.Ticker(symbol).get_shares_full(
                start=start.isoformat(), end=(end + timedelta(days=1)).isoformat()), f"shares {symbol}")
        except _Transient as exc:
            log.warning("shares fetch failed (not cached): %s", exc)
            return None
        if s is None or len(s) == 0:
            self.cache.write("shares", key, None, status="empty")
            return None
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        s = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx.normalize()).dropna()
        s = s[s > 0]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        self.cache.write("shares", key, s.to_frame("shares"))
        return s

    def get_shares_outstanding(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        with ThreadPoolExecutor(self.threads) as ex:
            results = list(ex.map(lambda s: (s, self._shares_one(s, start, end)), symbols))
        return {s: v for s, v in results if v is not None and len(v)}

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.name,
            "synthetic": "False",
            "bars": "Yahoo Finance daily OHLCV (split-adjusted; raw close rebuilt from split events)",
            "earnings": "Yahoo Finance earnings dates (EPS estimate/actual/surprise; time-of-day -> BMO/AMC)",
            "universe": "Wikipedia S&P 400/600 current constituents + removals since warm-up start",
            "market_cap": "Yahoo reported shares outstanding (as-of) x as-traded close",
        }


def _f(v: object) -> float | None:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x
