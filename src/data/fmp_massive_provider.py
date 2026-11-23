"""Keyed providers: Financial Modeling Prep (earnings, market cap, universe incl. delisted)
and Massive, formerly Polygon.io (daily bars + splits).

Both are optional. :func:`src.data.factory.build_provider` composes them with the
yfinance fallback per capability and per symbol, so a key upgrades only what its plan
actually serves. FMP plans gate endpoints and symbols (HTTP 402/403); denied requests
fall back instead of failing the run, and are never cached so a plan upgrade takes
effect on the next run.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from collections import Counter
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

# Consecutive denials after which an endpoint is skipped for the rest of the process.
# Plans whitelist a few large caps, so a handful of denials is conclusive.
DENIAL_TRIP = 5
TIMING_MATCH_DAYS = 2


class FMPAccessDenied(Exception):
    """The key is valid but the plan does not cover this endpoint, parameter or symbol."""

    def __init__(self, path: str, status: int, message: str) -> None:
        super().__init__(f"{path}: HTTP {status}: {message}")
        self.path, self.status, self.message = path, status, message


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
        self.calls = 0

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "<redacted>") if self._key else text

    def get(self, path: str, **params: Any) -> Any:
        params[self.key_param] = self._key
        for attempt in range(4):
            self.limiter.wait()
            self.calls += 1
            resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
            if resp.status_code == 429:
                _time.sleep(2 ** (attempt + 2))
                continue
            if resp.status_code in (401, 402, 403):
                raise FMPAccessDenied(path, resp.status_code, self._redact(_error_text(resp))[:200])
            if resp.status_code >= 400:
                # requests' own HTTPError message embeds the full URL, key included.
                raise requests.HTTPError(f"{path}: HTTP {resp.status_code}", response=resp)
            return resp.json()
        raise RuntimeError(f"rate limited: {path}")


def _error_text(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text.strip()
    if isinstance(body, dict):
        return str(body.get("Error Message") or body.get("message") or body)
    return str(body)


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
    """Earnings (EPS + revenue), historical market cap, universe incl. delisted.

    Methods return ``None`` when the plan denies the request, distinct from an empty result.
    """

    name = "fmp"

    def __init__(self, api_key: str, base_url: str, cache_root: Path, calls_per_minute: int = 250) -> None:
        self.http = _HttpClient(base_url, "apikey", api_key, calls_per_minute)
        self.cache = DiskCache(cache_root, "fmp")
        self._denials: Counter[str] = Counter()
        self.denied: dict[str, str] = {}

    def _get(self, endpoint: str, path: str, **params: Any) -> Any:
        """Fetch, or raise FMPAccessDenied; trips a per-endpoint breaker after repeated denials."""
        if self._denials[endpoint] >= DENIAL_TRIP:
            raise FMPAccessDenied(path, 0, "skipped: endpoint repeatedly denied for this plan")
        try:
            data = self.http.get(path, **params)
        except FMPAccessDenied as exc:
            self._denials[endpoint] += 1
            if self._denials[endpoint] == DENIAL_TRIP or endpoint not in self.denied:
                log.warning("FMP denied %s (%d so far): %s", endpoint, self._denials[endpoint], exc.message)
            self.denied[endpoint] = exc.message
            raise
        self._denials[endpoint] = 0
        return data

    def universe(self) -> list[UniverseMember] | None:
        cached = self.cache.read("universe", "fmp_universe")
        if cached is None:
            rows: list[dict[str, Any]] = []
            try:
                for exch in ("NYSE", "NASDAQ"):
                    data = self._get(
                        "company-screener", "/stable/company-screener",
                        exchange=exch, isEtf="false", isFund="false", isActivelyTrading="true",
                        marketCapMoreThan=150_000_000, marketCapLowerThan=20_000_000_000, limit=10000,
                    )
                    rows += [{"symbol": d["symbol"], "name": d.get("companyName"), "exchange": exch,
                              "removed_on": None, "source": "fmp:screener"} for d in data]
            except FMPAccessDenied:
                return None
            page = 0
            while page <= 200:
                try:
                    data = self._get("delisted-companies", "/stable/delisted-companies", page=page, limit=100)
                except FMPAccessDenied:
                    log.warning("FMP delisted list truncated at page %d by plan; survivorship gap remains", page)
                    break
                if not data:
                    break
                for d in data:
                    if d.get("exchange") in ("NYSE", "NASDAQ"):
                        rows.append({"symbol": d["symbol"], "name": d.get("companyName"), "exchange": d["exchange"],
                                     "removed_on": d.get("delistedDate"), "source": "fmp:delisted"})
                page += 1
            cached = pd.DataFrame(rows)
            self.cache.write("universe", "fmp_universe", cached)
        out = []
        for r in cached.to_dict("records"):
            rem = pd.to_datetime(r.get("removed_on"), errors="coerce")
            out.append(UniverseMember(symbol=r["symbol"], name=r.get("name"), exchange=r.get("exchange"),
                                      source=r["source"], removed_on=None if pd.isna(rem) else rem.date()))
        return out

    def earnings(self, symbol: str) -> list[EarningsEvent] | None:
        df = self.cache.read("earnings", symbol)
        if df is None and self.cache.status("earnings", symbol) != "empty":
            try:
                data = self._get("earnings", "/stable/earnings", symbol=symbol, limit=100)
            except FMPAccessDenied:
                return None
            df = pd.DataFrame(data or [])
            self.cache.write("earnings", symbol, df if not df.empty else None, "ok" if not df.empty else "empty")
        if df is None or df.empty:
            return []
        events = []
        for r in df.to_dict("records"):
            eps_a = _num(r.get("epsActual", r.get("eps")))
            if eps_a is None:
                continue
            eps_e = _num(r.get("epsEstimated"))
            rev_a = _num(r.get("revenueActual", r.get("revenue")))
            rev_e = _num(r.get("revenueEstimated"))
            d = pd.Timestamp(r["date"]).to_pydatetime()
            # /stable/earnings carries no time-of-day; only legacy payloads had "time".
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
            try:
                data = self._get("historical-market-capitalization", "/stable/historical-market-capitalization",
                                 symbol=symbol, **{"from": start.isoformat(), "to": end.isoformat()}, limit=5000)
            except FMPAccessDenied:
                return None
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


def merge_fallback_timing(events: list[EarningsEvent], fallback: list[EarningsEvent]) -> list[EarningsEvent]:
    """Fill UNKNOWN timing on keyed events from the nearest fallback event of the same symbol.

    Only timing and the announcement timestamp are taken; EPS/revenue stay from the keyed source.
    """
    by_symbol: dict[str, list[EarningsEvent]] = {}
    for f in fallback:
        if f.timing != AnnouncementTiming.UNKNOWN:
            by_symbol.setdefault(f.symbol, []).append(f)
    out = []
    for e in events:
        if e.timing == AnnouncementTiming.UNKNOWN:
            cands = [f for f in by_symbol.get(e.symbol, [])
                     if abs((f.announced_at.date() - e.announced_at.date()).days) <= TIMING_MATCH_DAYS]
            if cands:
                best = min(cands, key=lambda f: abs((f.announced_at.date() - e.announced_at.date()).days))
                e = e.model_copy(update={"timing": best.timing, "announced_at": best.announced_at,
                                         "source": f"{e.source}+{best.source}:timing"})
        out.append(e)
    return out


class FMPMassiveProvider(MarketDataProvider):
    """Composite: each capability uses its keyed source when the plan serves it, else ``fallback``."""

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
        self.is_synthetic = fallback.is_synthetic
        self.field_sources: dict[str, Counter[str]] = {
            k: Counter() for k in ("universe", "bars", "earnings", "timing", "revenue", "market_cap")
        }
        self._mcap_memo: dict[tuple[tuple[str, ...], date, date], dict[str, pd.Series]] = {}

    def get_universe(self, as_of: date | None = None) -> list[UniverseMember]:
        members = self.fmp.universe() if self.fmp else None
        if members is None:
            members = self.fallback.get_universe(as_of)
            self.field_sources["universe"][self.fallback.name] += len(members)
        else:
            self.field_sources["universe"]["fmp"] += len(members)
        return members

    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> dict[str, BarSeries]:
        if not self.massive:
            out = self.fallback.get_daily_bars(symbols, start, end)
            self.field_sources["bars"][self.fallback.name] += len(out)
            return out
        out: dict[str, BarSeries] = {}
        for s in symbols:
            try:
                b = self.massive.bars(s, start, end)
            except requests.RequestException as exc:
                log.warning("massive bars failed for %s: %s", s, exc)
                b = None
            if b is not None:
                out[s] = b
        self.field_sources["bars"]["massive"] += len(out)
        return out

    def get_benchmark_bars(self, start: date, end: date, symbol: str = "SPY") -> BarSeries:
        if self.massive:
            b = self.massive.bars(symbol, start, end)
            if b is not None:
                return b
        return self.fallback.get_benchmark_bars(start, end, symbol)

    def get_earnings_calendar(self, symbols: list[str], start: date, end: date) -> list[EarningsEvent]:
        if not self.fmp:
            out = self.fallback.get_earnings_calendar(symbols, start, end)
            self._count_events(out)
            return out
        keyed: list[EarningsEvent] = []
        uncovered: list[str] = []
        for s in symbols:
            ev = self.fmp.earnings(s)
            if ev is None:
                uncovered.append(s)
            else:
                keyed += [e for e in ev if start <= e.announced_at.date() <= end]
        fb_symbols = sorted(set(uncovered) | {e.symbol for e in keyed
                                              if e.timing == AnnouncementTiming.UNKNOWN})
        fb = self.fallback.get_earnings_calendar(fb_symbols, start, end) if fb_symbols else []
        uncovered_set = set(uncovered)
        out = merge_fallback_timing(keyed, [e for e in fb if e.symbol not in uncovered_set])
        out += [e for e in fb if e.symbol in uncovered_set]
        self._count_events(out)
        return out

    def _count_events(self, events: list[EarningsEvent]) -> None:
        fs = self.field_sources
        for e in events:
            base, _, timing_src = e.source.partition("+")
            fs["earnings"][base] += 1
            if e.timing == AnnouncementTiming.UNKNOWN:
                fs["timing"]["unknown"] += 1
            else:
                fs["timing"][timing_src.split(":")[0] if timing_src else base] += 1
            fs["revenue"][base if e.revenue_surprise_pct is not None else "missing"] += 1

    def _market_caps(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        key = (tuple(symbols), start, end)
        if key not in self._mcap_memo:
            out = {}
            for s in symbols:
                m = self.fmp.market_cap(s, start - timedelta(days=10), end) if self.fmp else None
                if m is not None and len(m):
                    out[s] = m
            self._mcap_memo[key] = out
        return self._mcap_memo[key]

    def get_shares_outstanding(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        covered = self._market_caps(symbols, start, end)
        rest = [s for s in symbols if s not in covered]
        shares = self.fallback.get_shares_outstanding(rest, start, end) if rest else {}
        self.field_sources["market_cap"]["fmp"] += len(covered)
        self.field_sources["market_cap"][f"{self.fallback.name}:shares_x_price"] += len(shares)
        return shares

    def get_market_cap_history(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        return self._market_caps(symbols, start, end)

    def describe(self) -> dict[str, str]:
        fb = self.fallback.describe()
        out = {
            "provider": self.name,
            "synthetic": str(self.fallback.is_synthetic),
            "bars": "Massive (Polygon) daily aggregates" if self.massive else fb.get("bars", ""),
            "earnings": "FMP /stable/earnings (EPS + revenue) per symbol where the plan allows, else "
                        + fb.get("earnings", "") if self.fmp else fb.get("earnings", ""),
            "universe": "FMP screener + delisted companies, else " + fb.get("universe", "")
                        if self.fmp else fb.get("universe", ""),
            "market_cap": "FMP historical market cap where the plan allows, else " + fb.get("market_cap", "")
                          if self.fmp else fb.get("market_cap", ""),
        }
        for field, counts in self.field_sources.items():
            if +counts:
                out[f"source_counts_{field}"] = ", ".join(f"{k}={v}" for k, v in (+counts).most_common())
        if self.fmp:
            out["fmp_calls"] = str(self.fmp.http.calls)
            for endpoint, msg in self.fmp.denied.items():
                out[f"fmp_denied_{endpoint}"] = msg
        return out
