from datetime import date, datetime

import pandas as pd
import pytest
from pydantic import SecretStr

from src.data import factory
from src.data.bar_utils import normalise_ohlcv, raw_close_from_splits
from src.data.cache import DiskCache
from src.data.fmp_massive_provider import _fmp_timing, _pct_surprise
from src.data.universe import parse_removals, to_yahoo_symbol
from src.data.yfinance_provider import classify_timing
from src.domain.models import AnnouncementTiming, BarSeries


def test_raw_close_reconstructed_from_splits():
    idx = pd.bdate_range("2024-01-01", periods=5)
    adj = pd.Series([50.0, 51, 52, 53, 54], index=idx)
    splits = pd.Series([0, 0, 2.0, 0, 0], index=idx)  # 2:1 effective on day 3
    raw = raw_close_from_splits(adj, splits)
    assert list(raw) == [100.0, 102.0, 52.0, 53.0, 54.0]


def test_normalise_repairs_and_validates():
    idx = pd.date_range("2024-01-02", periods=3, tz="America/New_York")
    df = pd.DataFrame({"Open": [10, 11, 0], "High": [9.5, 12, 13], "Low": [9, 10, 11],
                       "Close": [10, 11.5, 12], "Volume": [1, 2, 3]}, index=idx)
    b = normalise_ohlcv("X", df)
    assert len(b) == 2 and b.frame.index.tz is None
    assert b.frame["high"].iloc[0] == 10  # high repaired to >= open/close


def test_bar_series_rejects_bad_frames():
    idx = pd.bdate_range("2024-01-01", periods=2)
    df = pd.DataFrame({"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [1, 1],
                       "raw_close": [1, 1]}, index=idx[::-1])
    with pytest.raises(ValueError):
        BarSeries(symbol="X", frame=df)


@pytest.mark.parametrize("t,exp", [
    ((7, 0), AnnouncementTiming.BMO), ((9, 29), AnnouncementTiming.BMO), ((9, 30), AnnouncementTiming.DMH),
    ((16, 0), AnnouncementTiming.AMC), ((16, 5), AnnouncementTiming.AMC), ((0, 0), AnnouncementTiming.UNKNOWN),
])
def test_timing_classification(t, exp):
    assert classify_timing(datetime(2024, 5, 1, *t)) is exp


def test_fmp_helpers():
    assert _fmp_timing("bmo") is AnnouncementTiming.BMO
    assert _fmp_timing("--") is AnnouncementTiming.UNKNOWN
    assert _pct_surprise(1.1, 1.0) == pytest.approx(10.0)
    assert _pct_surprise(-0.5, -1.0) == pytest.approx(50.0)
    assert _pct_surprise(1.0, 0) is None


def test_universe_removals_parsing():
    cols = pd.MultiIndex.from_tuples([("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
                                      ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason")])
    df = pd.DataFrame([["March 3, 2023", "NEW", "New Co", "OLD.B", "Old Co", "x"],
                       ["May 1, 2019", "A", "A", "ANCIENT", "Anc", "y"]], columns=cols)
    out = parse_removals([pd.DataFrame(), df], "sp600", date(2020, 1, 1))
    assert [m.symbol for m in out] == ["OLD-B"] and out[0].removed_on == date(2023, 3, 3)
    assert to_yahoo_symbol("brk.b") == "BRK-B"


def test_cache_roundtrip_and_negative(tmp_path):
    c = DiskCache(tmp_path, "t")
    c.write("bars", "AAA", pd.DataFrame({"x": [1]}))
    c.write("bars", "BBB", None, status="empty")
    assert c.read("bars", "AAA")["x"].iloc[0] == 1
    assert c.status("bars", "BBB") == "empty" and c.read("bars", "BBB") is None


def test_fallback_chain_offline_uses_sample(settings, monkeypatch):
    monkeypatch.setattr(factory, "yahoo_reachable", lambda timeout=8.0: False)
    s = settings.model_copy(update={"fmp_api_key": None, "massive_api_key": None})
    p, notes = factory.build_provider(s)
    assert p.is_synthetic and p.name == "sample"
    assert any("SYNTHETIC" in n for n in notes)


def test_fallback_chain_online_uses_yfinance(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "yahoo_reachable", lambda timeout=8.0: True)
    s = settings.model_copy(update={"fmp_api_key": None, "massive_api_key": None,
                                    "data": settings.data.model_copy(update={"cache_dir": str(tmp_path)})})
    p, _ = factory.build_provider(s)
    assert p.name == "yfinance" and not p.is_synthetic


def test_keys_select_fmp_massive(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "yahoo_reachable", lambda timeout=8.0: True)
    s = settings.model_copy(update={"fmp_api_key": SecretStr("k"), "massive_api_key": None,
                                    "data": settings.data.model_copy(update={"cache_dir": str(tmp_path)})})
    p, _ = factory.build_provider(s)
    assert p.name == "fmp_massive" and p.massive is None and p.fmp is not None
