"""Fill realism: limit-cap gap skips, gap-through stops, T+1 settled cash."""

from datetime import date

import pandas as pd
import pytest

from src.domain.models import ExitReason, OrderIntent, RegimeState
from src.execution.backtest_broker import BacktestBroker
from src.risk.risk_manager import GoodFaithViolation, SettlementLedger
from tests.conftest import make_signal

CAL = pd.bdate_range("2024-05-01", periods=10)


def bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def intent(symbol="TST", shares=10, limit=50.15, stop=47.0):
    return OrderIntent(symbol=symbol, signal_date=date(2024, 5, 1), shares=shares, limit_price=limit,
                       stop_price=stop, stop_distance=limit - stop, risk_fraction=0.05,
                       regime=RegimeState.NORMAL, signal=make_signal(symbol, close=50.0))


@pytest.fixture
def broker(settings):
    return BacktestBroker(CAL, 1060.0, settings.execution)


def test_entry_skipped_when_open_gaps_above_limit(broker):
    broker.submit_bracket_order(intent())
    broker.process_session(CAL[1], {"TST": bar(50.20, 51, 49.5, 50.5)})
    assert not broker.positions
    assert broker.skipped[0].reason == "gap_above_limit"


def test_entry_fill_capped_at_limit_with_slippage(broker, settings):
    broker.submit_bracket_order(intent())
    broker.process_session(CAL[1], {"TST": bar(50.10, 51, 49.5, 50.5)})
    p = broker.positions["TST"]
    assert p.entry_price == pytest.approx(50.15)  # 50.10*1.0025=50.225 capped at 50.15
    broker2 = BacktestBroker(CAL, 1060.0, settings.execution)
    broker2.submit_bracket_order(intent())
    broker2.process_session(CAL[1], {"TST": bar(49.0, 51, 48.5, 50.5)})
    assert broker2.positions["TST"].entry_price == pytest.approx(49.0 * 1.0025)


def test_gap_through_stop_fills_at_open_not_stop(broker, settings):
    broker.submit_bracket_order(intent(stop=47.0))
    broker.process_session(CAL[1], {"TST": bar(50.0, 51, 49.5, 50.5)})
    broker.process_session(CAL[2], {"TST": bar(44.0, 45, 43.0, 44.5)})  # gaps far below 47
    t = broker.closed_trades[0]
    assert t.exit_reason is ExitReason.GAP_STOP
    assert t.exit_price == pytest.approx(44.0 * (1 - settings.execution.slippage_per_side_pct))
    assert t.r_multiple < -1.0  # loss worse than 1R is honoured


def test_intraday_stop_fills_at_stop(broker, settings):
    broker.submit_bracket_order(intent(stop=47.0))
    broker.process_session(CAL[1], {"TST": bar(50.0, 51, 49.5, 50.5)})
    broker.process_session(CAL[2], {"TST": bar(49.0, 49.5, 46.0, 48.0)})
    t = broker.closed_trades[0]
    assert t.exit_reason is ExitReason.STOP
    assert t.exit_price == pytest.approx(47.0 * (1 - settings.execution.slippage_per_side_pct))


def test_entry_day_stop_touch_is_honoured(broker):
    broker.submit_bracket_order(intent(stop=47.0))
    broker.process_session(CAL[1], {"TST": bar(50.0, 50.2, 46.5, 47.5)})
    assert broker.closed_trades and broker.closed_trades[0].exit_reason is ExitReason.STOP


def test_commissions_charged_both_sides(broker, settings):
    broker.submit_bracket_order(intent(stop=47.0))
    broker.process_session(CAL[1], {"TST": bar(50.0, 51, 49.5, 50.5)})
    broker.process_session(CAL[2], {"TST": bar(49.0, 49.5, 46.0, 48.0)})
    assert broker.closed_trades[0].commissions == pytest.approx(2 * settings.execution.commission_per_side_usd)


# ---- T+1 settled cash ------------------------------------------------------------

def test_ledger_t_plus_one():
    led = SettlementLedger(calendar=CAL, settled=0.0, settlement_days=1)
    led.credit_sale(CAL[2], 500.0)
    assert led.settled == 0 and led.unsettled == 500
    assert led.settled_by(CAL[2]) == 0
    assert led.settled_by(CAL[3]) == 500
    led.roll(CAL[3])
    assert led.settled == 500 and led.unsettled == 0


def test_purchase_with_unsettled_funds_raises():
    led = SettlementLedger(calendar=CAL, settled=100.0)
    led.credit_sale(CAL[0], 1000.0)
    with pytest.raises(GoodFaithViolation):
        led.debit_purchase(500.0)


def test_sale_proceeds_not_reusable_same_day(settings):
    b = BacktestBroker(CAL, 600.0, settings.execution)
    b.submit_bracket_order(intent("AAA", shares=11, limit=50.15, stop=47.0))
    b.process_session(CAL[1], {"AAA": bar(50.0, 51, 49.5, 50.5)})
    assert b.ledger.settled < 50
    b.close_position("AAA", ExitReason.MANUAL)
    b.submit_bracket_order(intent("BBB", shares=10, limit=50.15, stop=47.0))
    # AAA sells at the open of CAL[2]; its proceeds settle CAL[3], so BBB cannot fill on CAL[2].
    b.process_session(CAL[2], {"AAA": bar(51.0, 52, 50.5, 51.5), "BBB": bar(50.0, 51, 49.5, 50.5)})
    assert "BBB" not in b.positions
    assert b.skipped[-1].reason == "insufficient_settled_cash"
    b.submit_bracket_order(intent("BBB", shares=10, limit=50.15, stop=47.0))
    b.process_session(CAL[3], {"BBB": bar(50.0, 51, 49.5, 50.5)})
    assert "BBB" in b.positions


def test_fill_reduces_shares_to_settled_cash(settings):
    b = BacktestBroker(CAL, 300.0, settings.execution)
    b.submit_bracket_order(intent(shares=10))
    b.process_session(CAL[1], {"TST": bar(50.0, 51, 49.5, 50.5)})
    p = b.positions["TST"]
    assert p.shares == 5 and p.shares * p.entry_price + 0.38 <= 300.0
    assert b.ledger.settled >= 0
