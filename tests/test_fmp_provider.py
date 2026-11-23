"""FMP adapter behaviour against plan-gated responses (shapes taken from the live /stable API)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
import requests

from src.data import fmp_massive_provider as fm
from src.data.fmp_massive_provider import (
    FMPAccessDenied,
    FMPMassiveProvider,
    FMPProvider,
    _HttpClient,
    merge_fallback_timing,
)
from src.data.provider_base import MarketDataProvider
from src.domain.models import AnnouncementTiming, EarningsEvent, UniverseMember

from tests.conftest import make_bars

KEY = "secret-test-key-123"
ALLOWED = {"AAPL"}


class _Resp:
    def __init__(self, status: int, body):
        self.status_code, self._body = status, body
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    """Mimics the Basic plan: per-symbol endpoints only serve whitelisted large caps."""

    def __init__(self):
        self.paths: list[str] = []

    def get(self, url, params, timeout):
        path = url.split("financialmodelingprep.com")[1]
        self.paths.append(path)
        sym = params.get("symbol")
        if path == "/stable/company-screener":
            return _Resp(402, "Restricted Endpoint: This endpoint is not available under your current subscription")
        if path == "/stable/earnings":
            if sym not in ALLOWED:
                return _Resp(402, "Premium Query Parameter: 'symbol' is not available")
            return _Resp(200, [
                {"symbol": sym, "date": "2024-05-02", "epsActual": 1.53, "epsEstimated": 1.5,
                 "revenueActual": 90.8e9, "revenueEstimated": 90.0e9, "lastUpdated": "2024-05-03"},
                {"symbol": sym, "date": "2027-02-04", "epsActual": None, "epsEstimated": 2.94,
                 "revenueActual": None, "revenueEstimated": 1.5e11, "lastUpdated": "2026-11-20"},
            ])
        if path == "/stable/historical-market-capitalization":
            return _Resp(402, "Premium Query Parameter: 'from' is not available")
        if path == "/stable/boom":
            return _Resp(500, "oops")
        raise AssertionError(path)


class StubFallback(MarketDataProvider):
    name = "yfinance"

    def get_universe(self, as_of=None):
        return [UniverseMember(symbol=s, source="wiki") for s in ("AAPL", "SAIA", "CROX")]

    def get_daily_bars(self, symbols, start, end):
        return {s: make_bars([10.0] * 5, symbol=s) for s in symbols}

    def get_benchmark_bars(self, start, end, symbol="SPY"):
        return make_bars([400.0] * 5, symbol=symbol)

    def get_earnings_calendar(self, symbols, start, end):
        out = []
        for s in symbols:
            out.append(EarningsEvent(symbol=s, announced_at=datetime(2024, 5, 2, 16, 30),
                                     timing=AnnouncementTiming.AMC, eps_actual=1.0, eps_estimate=0.9,
                                     eps_surprise_pct=11.1, source="yfinance"))
        return out

    def get_shares_outstanding(self, symbols, start, end):
        return {s: pd.Series([1e8], index=[pd.Timestamp("2023-01-02")]) for s in symbols}


@pytest.fixture
def fmp(tmp_path, monkeypatch):
    monkeypatch.setattr(fm.RateLimiter, "wait", lambda self: None)
    p = FMPProvider(KEY, "https://financialmodelingprep.com", tmp_path)
    p.http.session = FakeSession()
    return p


def test_denied_raises_typed_error_without_key(fmp):
    with pytest.raises(FMPAccessDenied) as ei:
        fmp.http.get("/stable/company-screener")
    assert ei.value.status == 402 and KEY not in str(ei.value)
    with pytest.raises(requests.HTTPError) as ei2:
        fmp.http.get("/stable/boom")
    assert KEY not in str(ei2.value)


def test_redacts_key_echoed_in_error_body(monkeypatch):
    monkeypatch.setattr(fm.RateLimiter, "wait", lambda self: None)
    c = _HttpClient("https://x", "apikey", KEY, 60)
    c.session.get = lambda url, params, timeout: _Resp(401, {"Error Message": f"Invalid API KEY {KEY}"})
    with pytest.raises(FMPAccessDenied) as ei:
        c.get("/stable/profile")
    assert KEY not in str(ei.value) and "<redacted>" in str(ei.value)


def test_universe_denied_returns_none_and_is_not_cached(fmp):
    assert fmp.universe() is None
    assert fmp.cache.status("universe", "fmp_universe") is None


def test_earnings_parses_stable_schema_and_denials_return_none(fmp):
    ev = fmp.earnings("AAPL")
    assert len(ev) == 1  # future row without an actual is dropped
    e = ev[0]
    assert e.revenue_surprise_pct == pytest.approx((90.8 - 90.0) / 90.0 * 100)
    assert e.timing is AnnouncementTiming.UNKNOWN and e.source == "fmp"
    assert fmp.earnings("SAIA") is None
    assert fmp.cache.status("earnings", "SAIA") is None


def test_circuit_breaker_stops_calling_denied_endpoint(fmp):
    for i in range(20):
        assert fmp.earnings(f"X{i}") is None
    assert fmp.http.session.paths.count("/stable/earnings") == fm.DENIAL_TRIP
    assert "earnings" in fmp.denied


def test_merge_fallback_timing_only_fills_unknown():
    keyed = EarningsEvent(symbol="A", announced_at=datetime(2024, 5, 2), timing=AnnouncementTiming.UNKNOWN,
                          eps_actual=1.0, revenue_surprise_pct=2.0, source="fmp")
    known = keyed.model_copy(update={"symbol": "B", "timing": AnnouncementTiming.BMO})
    fb = [
        EarningsEvent(symbol="A", announced_at=datetime(2024, 5, 1, 16, 5), timing=AnnouncementTiming.AMC,
                      eps_actual=9.0, source="yfinance"),
        EarningsEvent(symbol="A", announced_at=datetime(2024, 8, 1, 7), timing=AnnouncementTiming.BMO,
                      eps_actual=9.0, source="yfinance"),
        EarningsEvent(symbol="B", announced_at=datetime(2024, 5, 2, 16, 5), timing=AnnouncementTiming.AMC,
                      eps_actual=9.0, source="yfinance"),
    ]
    a, b = merge_fallback_timing([keyed, known], fb)
    assert a.timing is AnnouncementTiming.AMC and a.announced_at == datetime(2024, 5, 1, 16, 5)
    assert a.eps_actual == 1.0 and a.revenue_surprise_pct == 2.0 and a.source == "fmp+yfinance:timing"
    assert b == known
    lone = keyed.model_copy(update={"announced_at": datetime(2024, 6, 20)})
    assert merge_fallback_timing([lone], fb)[0].timing is AnnouncementTiming.UNKNOWN


def test_composite_falls_back_per_symbol_and_records_sources(fmp):
    comp = FMPMassiveProvider(fallback=StubFallback(), fmp=fmp)
    uni = comp.get_universe()
    assert [m.source for m in uni] == ["wiki"] * 3
    syms = [m.symbol for m in uni]
    start, end = date(2023, 1, 1), date(2024, 12, 31)
    shares = comp.get_shares_outstanding(syms, start, end)
    assert set(shares) == set(syms) and comp.get_market_cap_history(syms, start, end) == {}
    events = comp.get_earnings_calendar(syms, start, end)
    by = {e.symbol: e for e in events}
    assert by["AAPL"].source == "fmp+yfinance:timing" and by["AAPL"].revenue_surprise_pct is not None
    assert by["AAPL"].timing is AnnouncementTiming.AMC
    assert by["SAIA"].source == "yfinance" and by["CROX"].source == "yfinance"
    d = comp.describe()
    assert d["source_counts_universe"] == "yfinance=3"
    assert d["source_counts_earnings"] == "yfinance=2, fmp=1"
    assert d["source_counts_timing"] == "yfinance=3"
    assert d["source_counts_revenue"] == "missing=2, fmp=1"
    assert d["source_counts_market_cap"] == "yfinance:shares_x_price=3"
    assert "fmp_denied_company-screener" in d and "fmp_denied_historical-market-capitalization" in d
    assert all(KEY not in v for v in d.values())
