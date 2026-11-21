"""Keyed providers: Financial Modeling Prep (earnings, market cap, universe incl. delisted)
and Massive, formerly Polygon.io (daily bars + splits).

Both are optional. :func:`src.data.factory.build_provider` composes them with the
yfinance fallback per capability, so a single key upgrades only what it covers.
Endpoint schemas follow the vendors' public docs; they were not exercised live in
Phase 0 because no keys were available (see docs/report).
"""

from __future__ import annotations

import logging
import threading
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.bar_utils import normalise_ohlcv
from src.data.cache import DiskCache
from src.data.provider_base import MarketDataProvider
from src.domain.models import AnnouncementTiming, BarSeries, EarningsEvent, UniverseMember

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, calls_per_minute: int) -> None:
        self.interval = 60.0 / calls_per_minute
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = _time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            _time.sleep(delay)


class _HttpClient:
    def __init__(self, base_url: str, key_param: str, api_key: str, calls_per_minute: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.key_param = key_param
        self._key = api_key
        self.limiter = RateLimiter(calls_per_minute)
        self.session = requests.Session()

    def get(self, path: str, **params: Any) -> Any:
        params[self.key_param] = self._key
        for attempt in range(4):
            self.limiter.wait()
            resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
            if resp.status_code == 429:
                _time.sleep(2 ** (attempt + 2))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"rate limited: {path}")


def _fmp_timing(raw: str | None) -> AnnouncementTiming:
    v = (raw or "").strip().lower()
    return {"bmo": AnnouncementTiming.BMO, "amc": AnnouncementTiming.AMC, "dmh": AnnouncementTiming.DMH}.get(
        v, AnnouncementTiming.UNKNOWN
    )


def _pct_surprise(actual: float | None, estimate: float | None) -> float | None:
    if actual is None or estimate is None or estimate == 0:
        return None
    return (actual - estimate) / abs(estimate) * 100.0


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x


class FMPProvider:
    """Earnings (with BMO/AMC + revenue), historical market cap, universe incl. delisted."""

    name = "fmp"

    def __init__(self, api_key: str, base_url: str, cache_root: Path, calls_per_minute: int = 250) -> None:
        self.http = _HttpClient(base_url, "apikey", api_key, calls_per_minute)
        self.cache = DiskCache(cache_root, "fmp")

    def universe(self) -> list[UniverseMember]:
        cached = self.cache.read("universe", "fmp_universe")
        if cached is None:
            rows: list[dict[str, Any]] = []
            for exch in ("NYSE", "NASDAQ"):
                data = self.http.get(
                    "/stable/company-screener",
                    exchange=exch, isEtf="false", isFund="false", isActivelyTrading="true",
                    marketCapMoreThan=150_000_000, marketCapLowerThan=20_000_000_000, limit=10000,
                )
                rows += [{"symbol": d["symbol"], "name": d.get("companyName"), "exchange": exch,
                          "removed_on": None, "source": "fmp:screener"} for d in data]
            page = 0
            while True:
                data = self.http.get("/stable/delisted-companies", page=page, limit=100)
                if not data:
                    break
                for d in data:
                    if d.get("exchange") in ("NYSE", "NASDAQ"):
                        rows.append({"symbol": d["symbol"], "name": d.get("companyName"), "exchange": d["exchange"],
                                     "removed_on": d.get("delistedDate"), "source": "fmp:delisted"})
                page += 1
                if page > 200:
                    break
            cached = pd.DataFrame(rows)
            self.cache.write("universe", "fmp_universe", cached)
        out = []
        for r in cached.to_dict("records"):
            rem = pd.to_datetime(r.get("removed_on"), errors="coerce")
            out.append(UniverseMember(symbol=r["symbol"], name=r.get("name"), exchange=r.get("exchange"),
                                      source=r["source"], removed_on=None if pd.isna(rem) else rem.date()))
        return out

    def earnings(self, symbol: str) -> list[EarningsEvent]:
        df = self.cache.read("earnings", symbol)
        if df is None and self.cache.status("earnings", symbol) != "empty":
            try:
                data = self.http.get(f"/api/v3/historical/earning_calendar/{symbol}")
            except requests.HTTPError:
                data = self.http.get("/stable/earnings", symbol=symbol, limit=100)
            df = pd.DataFrame(data or [])
            self.cache.write("earnings", symbol, df if not df.empty else None, "ok" if not df.empty else "empty")
        if df is None or df.empty:
            return []
        events = []
        for r in df.to_dict("records"):
            eps_a = _num(r.get("eps", r.get("epsActual")))
            if eps_a is None:
                continue
            eps_e = _num(r.get("epsEstimated"))
            rev_a = _num(r.get("revenue", r.get("revenueActual")))
            rev_e = _num(r.get("revenueEstimated"))
            d = pd.Timestamp(r["date"]).to_pydatetime()
            timing = _fmp_timing(r.get("time"))
            if timing == AnnouncementTiming.BMO:
                d = datetime.combine(d.date(), datetime.min.time()).replace(hour=7)
            elif timing == AnnouncementTiming.AMC:
                d = datetime.combine(d.date(), datetime.min.time()).replace(hour=16, minute=5)
            events.append(EarningsEvent(
                symbol=symbol, announced_at=d, timing=timing, fiscal_period=r.get("fiscalDateEnding"),
                eps_estimate=eps_e, eps_actual=eps_a, eps_surprise_pct=_pct_surprise(eps_a, eps_e),
                revenue_estimate=rev_e, revenue_actual=rev_a, revenue_surprise_pct=_pct_surprise(rev_a, rev_e),
                source="fmp",
            ))
        return events

    def market_cap(self, symbol: str, start: date, end: date) -> pd.Series | None:
        key = f"{symbol}_{start}_{end}"
        df = self.cache.read("mcap", key)
        if df is None and self.cache.status("mcap", key) != "empty":
            data = self.http.get("/stable/historical-market-capitalization", symbol=symbol,
                                 **{"from": start.isoformat(), "to": end.isoformat()}, limit=5000)
            df = pd.DataFrame(data or [])
            if df.empty:
                self.cache.write("mcap", key, None, "empty")
                return None
            df = df[["date", "marketCap"]]
            self.cache.write("mcap", key, df)
        if df is None or df.empty:
            return None
        s = pd.Series(df["marketCap"].astype(float).values, index=pd.to_datetime(df["date"]).dt.normalize())
        return s.sort_index()


class MassiveProvider:
    """Daily aggregates and split history from Massive (formerly Polygon.io)."""

    name = "massive"

    def __init__(self, api_key: str, base_url: str, cache_root: Path, calls_per_minute: int = 5) -> None:
        self.http = _HttpClient(base_url, "apiKey", api_key, calls_per_minute)
        self.cache = DiskCache(cache_root, "massive")

    def bars(self, symbol: str, start: date, end: date) -> BarSeries | None:
        key = f"{symbol}_{start}_{end}"
        if self.cache.status("bars", key) == "empty":
            return None
        df = self.cache.read("bars", key)
        if df is not None:
            return BarSeries(symbol=symbol, frame=df)
        data = self.http.get(f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
                             adjusted="true", sort="asc", limit=50000)
        results = data.get("results") or []
        if not results:
            self.cache.write("bars", key, None, "empty")
            return None
        raw = pd.DataFrame(results)
        raw.index = pd.to_datetime(raw["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        raw = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        splits_data = self.http.get("/v3/reference/splits", ticker=symbol, limit=1000).get("results") or []
        splits = pd.Series(
            {pd.Timestamp(s["execution_date"]): float(s["split_to"]) / float(s["split_from"]) for s in splits_data},
            dtype=float,
        )
        series = normalise_ohlcv(symbol, raw, splits if len(splits) else None)
        self.cache.write("bars", key, series.frame if series else None, "ok" if series else "empty")
        return series


class FMPMassiveProvider(MarketDataProvider):
    """Composite: each capability uses its keyed source when available, else ``fallback``."""

    name = "fmp_massive"

    def __init__(
        self,
        fallback: MarketDataProvider,
        fmp: FMPProvider | None = None,
        massive: MassiveProvider | None = None,
    ) -> None:
        if fmp is None and massive is None:
            raise ValueError("FMPMassiveProvider needs at least one keyed source")
        self.fmp, self.massive, self.fallback = fmp, massive, fallback

    def get_universe(self, as_of: date | None = None) -> list[UniverseMember]:
        return self.fmp.universe() if self.fmp else self.fallback.get_universe(as_of)

    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> dict[str, BarSeries]:
        if not self.massive:
            return self.fallback.get_daily_bars(symbols, start, end)
        out: dict[str, BarSeries] = {}
        for s in symbols:
            try:
                b = self.massive.bars(s, start, end)
            except requests.RequestException as exc:
                log.warning("massive bars failed for %s: %s", s, exc)
                b = None
            if b is not None:
                out[s] = b
        return out

    def get_benchmark_bars(self, start: date, end: date, symbol: str = "SPY") -> BarSeries:
        if self.massive:
            b = self.massive.bars(symbol, start, end)
            if b is not None:
                return b
        return self.fallback.get_benchmark_bars(start, end, symbol)

    def get_earnings_calendar(self, symbols: list[str], start: date, end: date) -> list[EarningsEvent]:
        if not self.fmp:
            return self.fallback.get_earnings_calendar(symbols, start, end)
        out: list[EarningsEvent] = []
        for s in symbols:
            out += [e for e in self.fmp.earnings(s) if start <= e.announced_at.date() <= end]
        return out

    def get_shares_outstanding(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        return {} if self.fmp else self.fallback.get_shares_outstanding(symbols, start, end)

    def get_market_cap_history(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        if not self.fmp:
            return {}
        out = {}
        for s in symbols:
            m = self.fmp.market_cap(s, start - timedelta(days=10), end)
            if m is not None and len(m):
                out[s] = m
        return out

    def describe(self) -> dict[str, str]:
        fb = self.fallback.describe()
        return {
            "provider": self.name,
            "synthetic": "False",
            "bars": "Massive (Polygon) daily aggregates" if self.massive else fb.get("bars", ""),
            "earnings": "FMP earnings calendar (BMO/AMC, EPS + revenue)" if self.fmp else fb.get("earnings", ""),
            "universe": "FMP screener + delisted companies" if self.fmp else fb.get("universe", ""),
            "market_cap": "FMP historical market cap" if self.fmp else fb.get("market_cap", ""),
        }
