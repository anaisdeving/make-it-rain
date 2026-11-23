import math

import pytest

from src.config import RiskConfig
from src.domain.models import RegimeState
from src.risk.risk_manager import build_order_intent, position_size, risk_fraction, stop_distance
from tests.conftest import make_regime, make_signal


def test_stop_distance_uses_max_of_atr_and_event_low():
    cfg = RiskConfig()
    assert stop_distance(100, atr14=2.0, event_day_low=99.0, cfg=cfg) == pytest.approx(3.0)   # 1.5*ATR
    assert stop_distance(100, atr14=2.0, event_day_low=95.0, cfg=cfg) == pytest.approx(5.0)   # entry - low


def test_stop_distance_capped_at_15pct():
    assert stop_distance(100, atr14=20.0, event_day_low=60.0, cfg=RiskConfig()) == pytest.approx(15.0)


def test_whole_share_rounding_is_floor():
    cfg = RiskConfig()
    # risk: 1060*0.05/3.0 = 17.67 shares; notional: 1060*0.5/40 = 13.25 -> 13
    n = position_size(1060, 40.0, 3.0, 0.05, settled_cash=1060, commission=0.38, cfg=cfg)
    assert n == 13 and isinstance(n, int)
    # risk bound binds: 1060*0.05/5 = 10.6 -> 10
    assert position_size(1060, 20.0, 5.0, 0.05, 1060, 0.38, cfg) == 10


def test_never_fractional_and_zero_when_unaffordable():
    cfg = RiskConfig()
    assert position_size(1060, 600.0, 10.0, 0.05, 1060, 0.38, cfg) == 0  # 50% cap < 1 share
    for eq in (999.99, 1060, 5300.5):
        n = position_size(eq, 37.37, 2.11, 0.05, eq, 0.38, cfg)
        assert n == math.floor(n)


def test_size_limited_by_settled_cash():
    cfg = RiskConfig()
    assert position_size(10_000, 50.0, 2.0, 0.05, settled_cash=1000.38, commission=0.38, cfg=cfg) == 20


def test_caution_halves_risk_and_panic_zero():
    cfg = RiskConfig()
    assert risk_fraction(make_regime(RegimeState.NORMAL), cfg) == pytest.approx(0.05)
    assert risk_fraction(make_regime(RegimeState.CAUTION), cfg) == pytest.approx(0.025)
    assert risk_fraction(make_regime(RegimeState.PANIC), cfg) == 0.0


def test_risk_cap_validation():
    with pytest.raises(ValueError):
        RiskConfig(base_risk_per_trade=0.12)


def test_order_intent_sized_on_limit_price(settings):
    sig = make_signal(close=50.0, low=47.0, atr=1.0)
    intent, why = build_order_intent(sig, make_regime(), 1060, 1060, settings.risk, settings.execution)
    assert why == "ok"
    assert intent.limit_price == pytest.approx(50.15)
    assert intent.stop_distance == pytest.approx(50.15 - 47.0)
    assert intent.stop_price == pytest.approx(47.0)
    assert intent.shares == math.floor(min(1060 * 0.05 / 3.15, 530 / 50.15))
