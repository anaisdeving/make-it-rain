import pytest

from src.strategy.pead_screener import evaluate_event, screen, surprise_passes
from tests.conftest import make_event, make_regime, make_snapshot


def test_base_case_passes(settings):
    sig, why = evaluate_event(make_event(), make_snapshot(), make_regime(), settings.screener)
    assert why == "ok" and sig is not None
    assert sig.event_day_return == pytest.approx(0.10)
    assert sig.volume_ratio == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"raw_close": 4.0}, "price"),
        ({"raw_close": 151.0}, "price"),
        ({"market_cap": 2e8}, "market_cap"),
        ({"market_cap": 1.1e10}, "market_cap"),
        ({"market_cap": None}, "no_market_cap"),
        ({"avg_dollar_volume_20": 4e6}, "liquidity"),
        ({"close": 52.0, "high": 52.2}, "event_gap"),
        ({"volume": 2.4e6}, "volume"),
        ({"high": 60.0}, "close_location"),
        ({"sma50": 56.0}, "below_sma50"),
        ({"high_52w": 70.0}, "far_from_52w_high"),
        ({"atr14": float("nan")}, "insufficient_history"),
    ],
)
def test_each_filter_rejects(settings, kw, reason):
    _, why = evaluate_event(make_event(), make_snapshot(**kw), make_regime(), settings.screener)
    assert why == reason


def test_surprise_rules(settings):
    c = settings.screener
    assert surprise_passes(make_event(eps_surprise=5.0), c)
    assert not surprise_passes(make_event(eps_surprise=-1.0), c)
    assert not surprise_passes(make_event(eps_surprise=5.0, revenue_surprise=-2.0), c)
    assert surprise_passes(make_event(eps_surprise=-1.0, sue=2.0), c)          # SUE alternative
    strict = c.model_copy(update={"require_revenue_surprise": True})
    assert not surprise_passes(make_event(eps_surprise=5.0), strict)
    assert surprise_passes(make_event(eps_surprise=5.0, revenue_surprise=1.0), strict)


def test_ranking_deterministic(settings):
    a = (make_event("AAA"), make_snapshot(symbol="AAA", close=56.0, high=56.2))
    b = (make_event("BBB"), make_snapshot(symbol="BBB", close=55.0))
    c = (make_event("CCC"), make_snapshot(symbol="CCC", close=55.0))
    sigs, reasons = screen([c, b, a], make_regime(), settings.screener)
    assert [s.symbol for s in sigs] == ["AAA", "BBB", "CCC"]
    assert reasons == {"ok": 3}
