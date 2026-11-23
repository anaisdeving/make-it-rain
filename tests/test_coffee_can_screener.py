"""Unit tests for Strategy v10_coffee_can (XBRL loader, screener, thesis-watch, T+1).

Verifies all strict anti-bias and accounting invariants BEFORE running Blind OOS:
1. Zero filing-date look-ahead (``filed > rebalance_date`` never influences selection;
   truncation and row-scrambling invariance).
2. XBRL tag fallback chains (Revenue, GrossProfit/COGS, OCF/CapEx, Cash+STI,
   Debt, Shares), strict ``coreg`` & ``segments`` exclusion (both 9-col and 10-col
   2026 schemas), ``qtrs`` & ``uom`` enforcement, and ticker resolution.
3. SIC 6000-6999 exclusion, 400-day filing staleness filter, survival guardrail
   (FCF > 0 OR runway >= 24m), D/E <= 1.0, YoY share dilution <= 5%, and
   Rule-of-40 ranking.
4. Point-in-time market cap & AS_TRADED whole-share invariance to future stock splits.
5. ``v10a_annual_top5`` T+1 annual rebalancing and ``v10b_coffee_can_thesis_watch``
   fundamental thesis breaks (2 consecutive negative YoY rev filings OR runway < 12m)
   checked strictly on ``filed_date``.
6. Survivorship-Penalized (-50%) sensitivity in Part A and Part B.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.sec_xbrl_loader import (
    enrich_annual_comparatives,
    extract_ticker_from_instance,
    parse_xbrl_quarter_frames,
    resolve_cik_ticker,
)
from src.domain.models import BarSeries
from src.strategy.coffee_can_screener import (
    CoffeeCanConfig,
    build_thesis_watch_events,
    evaluate_vintage_candidates,
    first_trading_day_of_april,
    run_coffee_can_event_study,
    run_coffee_can_portfolio_backtest,
    screen_fundamentals_asof,
    screen_replacement_candidates,
)


def _make_bars(
    symbol: str,
    dates: pd.DatetimeIndex,
    opens: np.ndarray | list[float],
    closes: np.ndarray | list[float] | None = None,
    raw_closes: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
) -> BarSeries:
    o = np.asarray(opens, dtype=float)
    c = o.copy() if closes is None else np.asarray(closes, dtype=float)
    rc = c.copy() if raw_closes is None else np.asarray(raw_closes, dtype=float)
    v = np.full(len(dates), 200_000.0) if volumes is None else np.asarray(volumes, dtype=float)
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) * 1.01,
            "low": np.minimum(o, c) * 0.99,
            "close": c,
            "volume": v,
            "raw_close": rc,
        },
        index=dates,
    )
    return BarSeries(symbol=symbol, frame=df)


def _make_spy(start: str = "2021-10-01", end: str = "2026-11-20") -> BarSeries:
    idx = pd.bdate_range(start=start, end=end)
    px = np.linspace(400.0, 560.0, len(idx))
    return _make_bars("SPY", idx, opens=px, closes=px, volumes=np.full(len(idx), 50_000_000.0))


def _make_annual_row(
    *,
    cik: str,
    ticker: str,
    name: str = "Acme Growth Corp",
    sic: int = 7372,
    adsh: str = "0001-22-000001",
    filed_date: str = "2022-02-28",
    period_date: str = "2021-12-31",
    revenue: float = 200e6,
    revenue_prior: float = 120e6,
    revenue_t2: float | None = None,
    gross_profit: float = 150e6,
    operating_income: float | None = None,
    ocf: float = 50e6,
    capex: float = 10e6,
    capex_reported: bool | None = None,
    cash_total: float = 120e6,
    total_debt: float = 20e6,
    equity: float = 150e6,
    shares_curr: float = 50e6,
    shares_prior: float = 49e6,
) -> dict:
    t2_val = (revenue_prior * 0.75) if revenue_t2 is None else revenue_t2
    opinc_val = (0.20 * revenue) if operating_income is None else operating_income
    capex_missing = capex is None or (isinstance(capex, float) and pd.isna(capex))
    if capex_reported is False:
        capex_missing = True
    reported = not capex_missing
    fcf_val = (ocf - capex) if reported else np.nan
    return {
        "adsh": adsh,
        "quarter": "2022q1",
        "cik": cik,
        "ticker": ticker,
        "ticker_source": "both_match",
        "name": name,
        "sic": sic,
        "form": "10-K",
        "filed_date": pd.Timestamp(filed_date),
        "period_date": pd.Timestamp(period_date),
        "rev_tag": "Revenues",
        "revenue": revenue,
        "revenue_prior_in_filing": revenue_prior,
        "revenue_t2_in_filing": t2_val,
        "gp_tag": "GrossProfit",
        "gross_profit": gross_profit,
        "operating_income": opinc_val,
        "ocf": ocf,
        "capex": capex if reported else np.nan,
        "capex_reported": reported,
        "fcf": fcf_val,
        "cash_total": cash_total,
        "total_debt": total_debt,
        "stockholders_equity": equity,
        "shares_tag": "CommonStockSharesOutstanding",
        "shares_date": pd.Timestamp(period_date),
        "shares_outstanding": shares_curr,
        "shares_comp_tag": "CommonStockSharesOutstanding",
        "shares_comp_curr": shares_curr,
        "shares_comp_prior_in_filing": shares_prior,
    }


# ---------------------------------------------------------------------------
# 1. XBRL tag fallback chains, coreg/segments exclusion, 9-col & 10-col schemas
# ---------------------------------------------------------------------------
def test_xbrl_parsing_chains_coreg_segments_and_schemas() -> None:
    assert extract_ticker_from_instance("nvda-20240128_htm.xml") == "NVDA"
    assert extract_ticker_from_instance("brk.b-20231231_htm.xml") == "BRK-B"
    assert extract_ticker_from_instance("form10k_htm.xml") == ""
    assert resolve_cik_ticker("0001045810", "nvda-20240128_htm.xml", {"0001045810": "NVDA"}) == ("NVDA", "both_match")
    assert resolve_cik_ticker("0000999999", "dead-20221231_htm.xml", {}) == ("DEAD", "instance_only_delisted")

    sub = pd.DataFrame([
        {
            "adsh": "A1", "cik": "1001", "name": "Cloud Co", "sic": "7372",
            "form": "10-K", "period": "20251231", "fp": "FY", "filed": "20260220",
            "instance": "cldc-20251231_htm.xml",
        },
        {
            "adsh": "A2", "cik": "1002", "name": "Fallback Co", "sic": "3674",
            "form": "10-K", "period": "20251231", "fp": "FY", "filed": "20260225",
            "instance": "flbk-20251231_htm.xml",
        },
    ])

    # 10-column schema (2026 update with 'segments')
    num_10 = pd.DataFrame([
        # A1: consolidated revenue = $200M (curr) / $125M (prior)
        {"adsh": "A1", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "200000000"},
        {"adsh": "A1", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "version": "us-gaap/2025",
         "ddate": "20241231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "125000000"},
        # Segment row & coreg row & non-USD row MUST be ignored!
        {"adsh": "A1", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": "ProductA=US", "coreg": np.nan, "value": "999000000"},
        {"adsh": "A1", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": "SubLLC", "value": "888000000"},
        {"adsh": "A1", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "CAD", "segments": np.nan, "coreg": np.nan, "value": "777000000"},
        # Quarterly qtrs==1 row inside 10-K must NOT overwrite qtrs==4 revenue
        {"adsh": "A1", "tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "1", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "55000000"},
        # GrossProfit direct
        {"adsh": "A1", "tag": "GrossProfit", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "140000000"},
        # OCF + CapEx (reported negative -> abs still subtracts!)
        {"adsh": "A1", "tag": "NetCashProvidedByUsedInOperatingActivities", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "60000000"},
        {"adsh": "A1", "tag": "PaymentsToAcquirePropertyPlantAndEquipment", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "-10000000"},
        # Cash + ShortTermInvestments
        {"adsh": "A1", "tag": "CashAndCashEquivalentsAtCarryingValue", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "80000000"},
        {"adsh": "A1", "tag": "ShortTermInvestments", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "20000000"},
        # Debt & Equity
        {"adsh": "A1", "tag": "LongTermDebtNoncurrent", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "30000000"},
        {"adsh": "A1", "tag": "LongTermDebtCurrent", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "5000000"},
        {"adsh": "A1", "tag": "StockholdersEquity", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "100000000"},
        # Shares (curr & prior in same filing)
        {"adsh": "A1", "tag": "CommonStockSharesOutstanding", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "shares", "segments": np.nan, "coreg": np.nan, "value": "51000000"},
        {"adsh": "A1", "tag": "CommonStockSharesOutstanding", "version": "us-gaap/2025",
         "ddate": "20241231", "qtrs": "0", "uom": "shares", "segments": np.nan, "coreg": np.nan, "value": "50000000"},
        # A2: fallback chain (SalesRevenueNet, CostOfRevenue -> GP, LongTermDebt fallback, Diluted shares fallback)
        {"adsh": "A2", "tag": "SalesRevenueNet", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "100000000"},
        {"adsh": "A2", "tag": "SalesRevenueNet", "version": "us-gaap/2025",
         "ddate": "20241231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "70000000"},
        {"adsh": "A2", "tag": "CostOfRevenue", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "40000000"},
        {"adsh": "A2", "tag": "NetCashProvidedByUsedInOperatingActivities", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "20000000"},
        {"adsh": "A2", "tag": "LongTermDebt", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "15000000"},
        {"adsh": "A2", "tag": "StockholdersEquity", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "0", "uom": "USD", "segments": np.nan, "coreg": np.nan, "value": "60000000"},
        {"adsh": "A2", "tag": "WeightedAverageNumberOfDilutedSharesOutstanding", "version": "us-gaap/2025",
         "ddate": "20251231", "qtrs": "4", "uom": "shares", "segments": np.nan, "coreg": np.nan, "value": "20400000"},
        {"adsh": "A2", "tag": "WeightedAverageNumberOfDilutedSharesOutstanding", "version": "us-gaap/2025",
         "ddate": "20241231", "qtrs": "4", "uom": "shares", "segments": np.nan, "coreg": np.nan, "value": "20000000"},
    ])

    df_10k, df_10q, aud = parse_xbrl_quarter_frames(sub, num_10, {"0000001001": "CLDC"}, quarter="2026q1")
    assert aud["has_segments_col"] is True
    assert aud["excluded_segments_rows"] == 1
    assert aud["excluded_coreg_rows"] == 1
    assert aud["excluded_non_usd_or_bad_uom_rows"] == 1
    assert df_10q.empty

    enriched = enrich_annual_comparatives(df_10k)
    r1 = enriched[enriched["adsh"] == "A1"].iloc[0]
    assert r1["revenue"] == pytest.approx(200e6)
    assert r1["yoy_rev_growth"] == pytest.approx(0.60)
    assert r1["gross_margin"] == pytest.approx(0.70)
    assert r1["fcf"] == pytest.approx(50e6)  # 60M - abs(-10M)
    assert r1["cash_total"] == pytest.approx(100e6)  # 80M + 20M
    assert r1["total_debt"] == pytest.approx(35e6)  # 30M + 5M
    assert r1["debt_to_equity"] == pytest.approx(0.35)
    assert r1["yoy_share_dilution"] == pytest.approx(0.02)
    assert r1["rule_of_40"] == pytest.approx(85.0)  # 60% + 25%

    r2 = enriched[enriched["adsh"] == "A2"].iloc[0]
    assert r2["gross_profit"] == pytest.approx(60e6)  # 100M - 40M CostOfRevenue
    assert r2["gross_margin"] == pytest.approx(0.60)
    assert r2["total_debt"] == pytest.approx(15e6)  # LongTermDebt fallback
    assert r2["shares_outstanding"] == pytest.approx(20.4e6)
    assert r2["yoy_share_dilution"] == pytest.approx(0.02)

    # Also verify 9-column legacy schema (no 'segments' column) parses identically on consolidated rows
    num_9 = num_10[num_10["segments"].isna()].drop(columns=["segments"]).copy()
    df_10k_legacy, _, aud9 = parse_xbrl_quarter_frames(sub, num_9, {"0000001001": "CLDC"}, quarter="2021q1")
    assert aud9["has_segments_col"] is False
    assert len(df_10k_legacy) == 2


# ---------------------------------------------------------------------------
# 2. Zero filing-date look-ahead + truncation & scrambling invariance
# ---------------------------------------------------------------------------
def test_zero_lookahead_truncation_and_scrambling_invariance() -> None:
    rebalance_dt = date(2023, 4, 3)
    raw = pd.DataFrame([
        # Filed BEFORE rebalance_dt -> eligible
        _make_annual_row(cik="0000000001", ticker="GOOD", adsh="G-2023", filed_date="2023-02-20", period_date="2022-12-31", revenue=200e6, revenue_prior=125e6),
        # Filed AFTER rebalance_dt (2023-04-04) -> MUST NEVER be seen at 2023-04-03!
        _make_annual_row(cik="0000000002", ticker="FUTR", adsh="F-2023", filed_date="2023-04-04", period_date="2022-12-31", revenue=300e6, revenue_prior=100e6),
        # Newer filing for GOOD filed AFTER rebalance_dt with terrible growth -> MUST NOT leak back into 2023-04-03!
        _make_annual_row(cik="0000000001", ticker="GOOD", adsh="G-2024", filed_date="2024-02-20", period_date="2023-12-31", revenue=150e6, revenue_prior=200e6),
        # Older filing for GOOD filed 2022-02-20 (superseded by G-2023 at rebalance_dt)
        _make_annual_row(cik="0000000001", ticker="GOOD", adsh="G-2022", filed_date="2022-02-20", period_date="2021-12-31", revenue=125e6, revenue_prior=90e6),
    ])

    cfg = CoffeeCanConfig()
    full_enriched = enrich_annual_comparatives(raw)
    cands_full, _ = screen_fundamentals_asof(full_enriched, rebalance_dt, cfg)

    # Truncate strictly at rebalance_dt and scramble row order
    trunc_raw = raw[raw["filed_date"] <= pd.Timestamp(rebalance_dt)].sample(frac=1.0, random_state=42)
    trunc_enriched = enrich_annual_comparatives(trunc_raw)
    cands_trunc, _ = screen_fundamentals_asof(trunc_enriched, rebalance_dt, cfg)

    assert [c.ticker for c in cands_full] == ["GOOD"]
    assert [c.adsh for c in cands_full] == ["G-2023"]
    assert cands_full == cands_trunc


# ---------------------------------------------------------------------------
# 3. SIC 6000-6999 exclusion, 400-day staleness, survival & Rule-of-40 ranking
# ---------------------------------------------------------------------------
def test_sic_staleness_guardrails_and_rule_of_40_ranking() -> None:
    reb = date(2023, 4, 3)
    raw = pd.DataFrame([
        # Valid #1: Rule of 40 = 80% + 25% = 105
        _make_annual_row(cik="0001", ticker="TOP1", sic=7372, adsh="1", filed_date="2023-03-01", revenue=180e6, revenue_prior=100e6, ocf=55e6, capex=10e6),
        # Valid #2 (FCF negative, cash runway = 3.0 yrs >= 2.0 yrs -> passes survival guardrail!): R40 = 60% - 10% = 50
        _make_annual_row(cik="0002", ticker="TOP2", sic=3841, adsh="2", filed_date="2023-03-01", revenue=160e6, revenue_prior=100e6, ocf=-10e6, capex=6e6, cash_total=48e6),
        # Excluded: SIC 6021 (Financial / Bank)
        _make_annual_row(cik="0003", ticker="BANK", sic=6021, adsh="3", filed_date="2023-03-01", revenue=500e6, revenue_prior=200e6),
        # Excluded: SIC 6798 (REIT)
        _make_annual_row(cik="0004", ticker="REIT", sic=6798, adsh="4", filed_date="2023-03-01", revenue=500e6, revenue_prior=200e6),
        # Excluded: Stale (> 400 days old: filed 2022-02-20 is 407 days before 2023-04-03!)
        _make_annual_row(cik="0005", ticker="STAL", sic=7372, adsh="5", filed_date="2022-02-20", period_date="2021-12-31"),
        # Excluded: Cash runway < 24 months (FCF = -20M, Cash = 30M -> 1.5 yrs < 2.0 yrs)
        _make_annual_row(cik="0006", ticker="BURN", sic=7372, adsh="6", filed_date="2023-03-01", ocf=-15e6, capex=5e6, cash_total=30e6),
        # Excluded: Share dilution > 5% (55M / 50M - 1 = +10%)
        _make_annual_row(cik="0007", ticker="DILU", sic=7372, adsh="7", filed_date="2023-03-01", shares_curr=55e6, shares_prior=50e6),
        # Excluded: Debt/Equity > 1.0 (200M / 150M = 1.33)
        _make_annual_row(cik="0008", ticker="DEBT", sic=7372, adsh="8", filed_date="2023-03-01", total_debt=200e6, equity=150e6),
    ])

    cfg = CoffeeCanConfig()
    cands, counts = screen_fundamentals_asof(enrich_annual_comparatives(raw), reb, cfg)
    assert [c.ticker for c in cands] == ["TOP1", "TOP2"]
    assert cands[0].rule_of_40 == pytest.approx(105.0)
    assert cands[1].rule_of_40 == pytest.approx(50.0)
    assert counts["excluded_sic_financial_reit_spac"] == 2
    assert counts["excluded_survival_fcf_runway"] == 1
    assert counts["excluded_share_dilution_gt_5pct"] == 1
    assert counts["excluded_equity_or_debt_to_equity"] == 1


# ---------------------------------------------------------------------------
# 4. Point-in-time market cap & AS_TRADED whole-share invariance to future splits
# ---------------------------------------------------------------------------
def test_pit_market_cap_and_future_split_invariance() -> None:
    spy = _make_spy("2022-01-03", "2023-06-01")
    idx = spy.frame.index
    reb = first_trading_day_of_april(idx, 2022)

    # Stock traded at $50 as-traded in April 2022 with 50M shares -> TRUE PIT market cap = $2.5B (in $300M-$10B!).
    # Later it did a 10-for-1 split, so Yahoo split-adjusted close in April 2022 is $5.00!
    # If we wrongly used split-adjusted close * historical shares, mcap would be $250M (< $300M, WRONG!).
    adj_px = np.full(len(idx), 5.0)
    raw_px = np.full(len(idx), 50.0)
    bar_splt = _make_bars("SPLT", idx, opens=adj_px, closes=adj_px, raw_closes=raw_px, volumes=np.full(len(idx), 1_000_000.0))

    raw = pd.DataFrame([
        _make_annual_row(cik="0010", ticker="SPLT", adsh="S1", filed_date="2022-02-25", period_date="2021-12-31", shares_curr=50e6, shares_prior=49e6),
    ])
    cfg = CoffeeCanConfig()
    cands, _ = screen_fundamentals_asof(enrich_annual_comparatives(raw), reb, cfg)
    ev, _ = evaluate_vintage_candidates(cands, {"SPLT": bar_splt}, {}, spy, cfg)
    assert ev[0].status == "qualified_matched"
    assert ev[0].asof_raw_price == pytest.approx(50.0)
    assert ev[0].pit_market_cap_usd == pytest.approx(2.5e9)

    # Verify AS_TRADED whole-share sizing: with $1,060 account -> 20% slot = $212 -> floor(($212-$0.38)/(50*1.0025)) = 4 raw shares!
    res = run_coffee_can_portfolio_backtest(
        {"2022-04": ev}, {"2022-04": reb}, {"SPLT": bar_splt}, spy, {}, cfg, reb, date(2022, 6, 30), 1060.0, 1000.0
    )
    assert len(res.open_positions_at_end) == 1
    tr = res.open_positions_at_end[0]
    assert tr.raw_qty == 4
    assert tr.adj_qty == pytest.approx(40.0)  # 4 raw shares * 10 future split factor


# ---------------------------------------------------------------------------
# 5. v10b Thesis-Break Monitoring (2 consec neg YoY rev OR runway < 12m) & v10a
# ---------------------------------------------------------------------------
def test_v10b_thesis_breaks_and_v10a_rebalance_t_plus_1() -> None:
    spy = _make_spy("2022-01-03", "2025-06-01")
    idx = spy.frame.index
    reb22 = first_trading_day_of_april(idx, 2022)
    reb23 = first_trading_day_of_april(idx, 2023)

    bar_rev = _make_bars("REVD", idx, opens=np.full(len(idx), 20.0), closes=np.full(len(idx), 20.0))
    bar_run = _make_bars("RUNW", idx, opens=np.full(len(idx), 20.0), closes=np.full(len(idx), 20.0))
    bar_win = _make_bars("WINR", idx, opens=np.linspace(20.0, 60.0, len(idx)), closes=np.linspace(20.0, 60.0, len(idx)))

    raw22 = pd.DataFrame([
        _make_annual_row(cik="0001", ticker="REVD", adsh="K1", filed_date="2022-02-28", period_date="2021-12-31", revenue=200e6, revenue_prior=100e6),
        _make_annual_row(cik="0002", ticker="RUNW", adsh="K2", filed_date="2022-02-28", period_date="2021-12-31", revenue=180e6, revenue_prior=100e6),
        _make_annual_row(cik="0003", ticker="WINR", adsh="K3", filed_date="2022-02-28", period_date="2021-12-31", revenue=160e6, revenue_prior=100e6),
    ])
    cfg_b = CoffeeCanConfig(rule="v10b_coffee_can_thesis_watch")
    cands22, _ = screen_fundamentals_asof(enrich_annual_comparatives(raw22), reb22, cfg_b)
    bars_map = {"REVD": bar_rev, "RUNW": bar_run, "WINR": bar_win}
    ev22, _ = evaluate_vintage_candidates(cands22, bars_map, {}, spy, cfg_b)

    # Quarterly thesis-watch filings after entry (2022-04-04):
    # - REVD: 2022-05-10 (-5%), 2022-08-10 (+3% -> resets!), 2022-11-10 (-4%), 2023-02-15 (-8% -> 2nd consecutive -> EXIT next open!)
    # - RUNW: 2022-08-05 cash runway 9.0 months (< 12m -> EXIT next open!)
    # - WINR: positive growth throughout -> HELD all the way to 3Y maturity (2025-04)!
    q_df = pd.DataFrame([
        {"adsh": "Q1", "cik": "0001", "filed_date": pd.Timestamp("2022-05-10"), "yoy_rev_growth": -0.05, "cash_runway_months": float("inf")},
        {"adsh": "Q2", "cik": "0001", "filed_date": pd.Timestamp("2022-08-10"), "yoy_rev_growth": +0.03, "cash_runway_months": float("inf")},
        {"adsh": "Q3", "cik": "0001", "filed_date": pd.Timestamp("2022-11-10"), "yoy_rev_growth": -0.04, "cash_runway_months": float("inf")},
        {"adsh": "Q4", "cik": "0001", "filed_date": pd.Timestamp("2023-02-15"), "yoy_rev_growth": -0.08, "cash_runway_months": float("inf")},
        {"adsh": "Q5", "cik": "0002", "filed_date": pd.Timestamp("2022-08-05"), "yoy_rev_growth": +0.30, "cash_runway_months": 9.0},
        {"adsh": "Q6", "cik": "0003", "filed_date": pd.Timestamp("2022-08-05"), "yoy_rev_growth": +0.40, "cash_runway_months": float("inf")},
    ])
    thesis_events = build_thesis_watch_events(pd.DataFrame(), q_df)

    reb25 = first_trading_day_of_april(idx, 2025)
    res_b = run_coffee_can_portfolio_backtest(
        {"2022-04": ev22, "2023-04": [], "2024-04": [], "2025-04": []},
        {"2022-04": reb22, "2023-04": reb23, "2024-04": first_trading_day_of_april(idx, 2024), "2025-04": reb25},
        bars_map, spy, thesis_events, cfg_b, reb22, date(2025, 5, 1), 5300.0, 5000.0,
    )
    by_sym = {t.ticker: t for t in res_b.trades}
    assert by_sym["REVD"].exit_reason == "THESIS_BREAK_REV_DECLINE"
    assert by_sym["REVD"].exit_date == date(2023, 2, 16)  # Strictly next trading day after 2023-02-15!
    assert by_sym["RUNW"].exit_reason == "THESIS_BREAK_RUNWAY"
    assert by_sym["RUNW"].exit_date == date(2022, 8, 8)   # Strictly next trading day (Mon) after Fri 2022-08-05!
    assert by_sym["WINR"].exit_reason == "COFFEE_CAN_3Y_MATURITY"

    # Also verify v10a exits at 2023-04 rebalance (next open) and settles T+1 before buying new cohort
    cfg_a = CoffeeCanConfig(rule="v10a_annual_top5")
    res_a = run_coffee_can_portfolio_backtest(
        {"2022-04": ev22, "2023-04": ev22},
        {"2022-04": reb22, "2023-04": reb23},
        bars_map, spy, {}, cfg_a, reb22, date(2023, 5, 1), 5300.0, 5000.0,
    )
    assert all(t.exit_reason == "ANNUAL_REBALANCE_EXIT" for t in res_a.trades)
    # Re-entered positions in 2023-04 MUST enter strictly on reb23 + 2 trading days (T+1 after reb23 + 1 sell!)
    reb23_idx = int(idx.searchsorted(pd.Timestamp(reb23)))
    expected_rebuy_date = idx[reb23_idx + 2].date()
    assert all(t.entry_date == expected_rebuy_date for t in res_a.open_positions_at_end)


# ---------------------------------------------------------------------------
# 6. Survivorship-Penalized (-50%) Sensitivity in Part A and Part B
# ---------------------------------------------------------------------------
def test_survivorship_penalized_part_a_and_part_b() -> None:
    spy = _make_spy("2022-01-03", "2024-06-01")
    idx = spy.frame.index
    reb22 = first_trading_day_of_april(idx, 2022)

    bar_ok = _make_bars("LIVE", idx, opens=np.linspace(20.0, 40.0, len(idx)), closes=np.linspace(20.0, 40.0, len(idx)))
    raw = pd.DataFrame([
        _make_annual_row(cik="0001", ticker="DEAD", adsh="D1", filed_date="2022-02-28", period_date="2021-12-31", revenue=300e6, revenue_prior=100e6),
        _make_annual_row(cik="0002", ticker="LIVE", adsh="L1", filed_date="2022-02-28", period_date="2021-12-31", revenue=200e6, revenue_prior=120e6),
    ])
    cfg = CoffeeCanConfig()
    cands, _ = screen_fundamentals_asof(enrich_annual_comparatives(raw), reb22, cfg)
    ev, audit = evaluate_vintage_candidates(cands, {"LIVE": bar_ok}, {}, spy, cfg)
    assert audit["qualified_matched"] == 1
    assert audit["unmatched_yfinance"] == 1

    es = run_coffee_can_event_study({"2022-04": ev}, {"LIVE": bar_ok}, spy, date(2024, 5, 31), cfg, is_vintages=("2022-04",), oos_vintages=())
    assert es["per_vintage"]["2022-04"]["raw_matched"]["horizons"]["1Y"]["n"] == 1
    assert es["per_vintage"]["2022-04"]["survivorship_penalized"]["horizons"]["1Y"]["n"] == 2
    assert es["per_vintage"]["2022-04"]["survivorship_penalized"]["horizons"]["1Y"]["mean_stock_ret"] < es["per_vintage"]["2022-04"]["raw_matched"]["horizons"]["1Y"]["mean_stock_ret"]

    res_raw = run_coffee_can_portfolio_backtest({"2022-04": ev}, {"2022-04": reb22}, {"LIVE": bar_ok}, spy, {}, cfg, reb22, date(2023, 6, 1), 5300.0, 5000.0, survivorship_penalized=False)
    res_pen = run_coffee_can_portfolio_backtest({"2022-04": ev}, {"2022-04": reb22}, {"LIVE": bar_ok}, spy, {}, cfg, reb22, date(2023, 6, 1), 5300.0, 5000.0, survivorship_penalized=True)
    assert any(t.exit_reason == "UNMATCHED_PENALIZED_50" for t in res_pen.trades)
    assert res_pen.equity_curve["equity"].iloc[-1] < res_raw.equity_curve["equity"].iloc[-1]


# ---------------------------------------------------------------------------
# 7. Hygiene: tiny base / one-year spikes, extractive margins, Rule-of-40 caps
# ---------------------------------------------------------------------------
def test_reject_tiny_prior_revenue_and_single_year_spikes() -> None:
    reb = date(2023, 4, 3)
    raw = pd.DataFrame([
        # Prior-year revenue $40M < $50M (launch base). Current year is large.
        _make_annual_row(cik="0001", ticker="TINY", adsh="T1", filed_date="2023-02-20", revenue=200e6, revenue_prior=40e6, revenue_t2=10e6),
        # Both years clear $50M, but Rev_t-1 ($80M) is not above Rev_t-2 ($90M).
        _make_annual_row(cik="0002", ticker="SPIKE", adsh="S1", filed_date="2023-02-20", revenue=200e6, revenue_prior=80e6, revenue_t2=90e6),
        # Two consecutive growing years, both bases >= $50M.
        _make_annual_row(cik="0003", ticker="STEADY", adsh="G1", filed_date="2023-02-20", revenue=200e6, revenue_prior=120e6, revenue_t2=80e6),
        # t-2 omitted on the current 10-K; the prior 10-K's comparative supplies it.
        _make_annual_row(
            cik="0004", ticker="FALL", adsh="F-NEW", filed_date="2023-02-20", period_date="2022-12-31",
            revenue=180e6, revenue_prior=110e6, revenue_t2=np.nan,
        ),
        _make_annual_row(
            cik="0004", ticker="FALL", adsh="F-OLD", filed_date="2022-02-20", period_date="2021-12-31",
            revenue=110e6, revenue_prior=70e6, revenue_t2=40e6,
        ),
    ])
    cands, counts = screen_fundamentals_asof(enrich_annual_comparatives(raw), reb, CoffeeCanConfig())
    tickers = [c.ticker for c in cands]
    assert "TINY" not in tickers
    assert "SPIKE" not in tickers
    assert "STEADY" in tickers
    assert "FALL" in tickers
    fall = next(c for c in cands if c.ticker == "FALL")
    assert fall.revenue_t2_usd == pytest.approx(70e6)
    assert fall.revenue_prior_usd > fall.revenue_t2_usd
    assert counts["excluded_revenue_lt_50m"] >= 1
    assert counts["excluded_single_year_revenue_spike"] >= 1


def test_reject_extractive_sic_weak_operating_margin_fcf_above_sales_and_missing_capex() -> None:
    # Loader: missing capex stays NaN, a reported zero stays zero, an alternate tag is used.
    sub = pd.DataFrame([
        {"adsh": "B1", "cik": "2001", "name": "No Capex Co", "sic": "7372", "form": "10-K", "period": "20231231", "fp": "FY", "filed": "20240220", "instance": "ncap-20231231_htm.xml"},
        {"adsh": "B2", "cik": "2002", "name": "Zero Capex Co", "sic": "7372", "form": "10-K", "period": "20231231", "fp": "FY", "filed": "20240220", "instance": "zcap-20231231_htm.xml"},
        {"adsh": "B3", "cik": "2003", "name": "Alt Capex Co", "sic": "7372", "form": "10-K", "period": "20231231", "fp": "FY", "filed": "20240220", "instance": "acap-20231231_htm.xml"},
        {"adsh": "B4", "cik": "2004", "name": "Opex Fallback Co", "sic": "7372", "form": "10-K", "period": "20231231", "fp": "FY", "filed": "20240220", "instance": "opex-20231231_htm.xml"},
    ])

    def _fact(adsh: str, tag: str, ddate: str, qtrs: str, value: str, uom: str = "USD") -> dict:
        return {"adsh": adsh, "tag": tag, "version": "us-gaap/2023", "ddate": ddate, "qtrs": qtrs, "uom": uom, "coreg": np.nan, "value": value}

    num = pd.DataFrame([
        _fact("B1", "Revenues", "20231231", "4", "100000000"),
        _fact("B1", "Revenues", "20221231", "4", "80000000"),
        _fact("B1", "NetCashProvidedByUsedInOperatingActivities", "20231231", "4", "20000000"),
        _fact("B2", "Revenues", "20231231", "4", "100000000"),
        _fact("B2", "NetCashProvidedByUsedInOperatingActivities", "20231231", "4", "20000000"),
        _fact("B2", "PaymentsToAcquirePropertyPlantAndEquipment", "20231231", "4", "0"),
        _fact("B3", "Revenues", "20231231", "4", "100000000"),
        _fact("B3", "NetCashProvidedByUsedInOperatingActivities", "20231231", "4", "20000000"),
        _fact("B3", "PaymentsToAcquireProductiveAssets", "20231231", "4", "-4000000"),
        _fact("B4", "Revenues", "20231231", "4", "100000000"),
        _fact("B4", "OperatingExpenses", "20231231", "4", "70000000"),
        _fact("B4", "NetCashProvidedByUsedInOperatingActivities", "20231231", "4", "20000000"),
        _fact("B4", "PaymentsToAcquirePropertyPlantAndEquipment", "20231231", "4", "-5000000"),
        _fact("B1", "OperatingIncomeLoss", "20231231", "4", "30000000"),
    ])
    parsed, _, _ = parse_xbrl_quarter_frames(sub, num, {}, quarter="2024q1")
    enriched = enrich_annual_comparatives(parsed).set_index("adsh")
    assert pd.isna(enriched.loc["B1", "capex"])
    assert bool(enriched.loc["B1", "capex_reported"]) is False
    assert pd.isna(enriched.loc["B1", "fcf"])
    assert enriched.loc["B1", "fcf"] != pytest.approx(20e6)
    assert enriched.loc["B2", "capex"] == pytest.approx(0.0)
    assert enriched.loc["B2", "fcf"] == pytest.approx(20e6)
    assert enriched.loc["B3", "capex"] == pytest.approx(4e6)
    assert enriched.loc["B3", "fcf"] == pytest.approx(16e6)
    assert enriched.loc["B1", "operating_income"] == pytest.approx(30e6)
    assert enriched.loc["B4", "operating_income"] == pytest.approx(30e6)  # 100 - 70

    reb = date(2023, 4, 3)
    raw = pd.DataFrame([
        _make_annual_row(cik="3001", ticker="OIL", sic=1311, adsh="O1", filed_date="2023-02-20"),
        _make_annual_row(cik="3002", ticker="COAL", sic=1221, adsh="C1", filed_date="2023-02-20"),
        _make_annual_row(cik="3003", ticker="THIN", adsh="M1", filed_date="2023-02-20", operating_income=8e6),  # 8/200 = 4%
        _make_annual_row(cik="3004", ticker="WIND", adsh="W1", filed_date="2023-02-20", ocf=260e6, capex=10e6),  # FCF 250 > revenue 200
        _make_annual_row(cik="3005", ticker="NOCAP", adsh="N1", filed_date="2023-02-20", capex=np.nan, capex_reported=False),
        _make_annual_row(cik="3006", ticker="KEEP", adsh="K1", filed_date="2023-02-20"),
    ])
    cands, counts = screen_fundamentals_asof(enrich_annual_comparatives(raw), reb, CoffeeCanConfig())
    tickers = [c.ticker for c in cands]
    assert tickers == ["KEEP"]
    assert counts["excluded_sic_extractive"] == 2
    assert counts["excluded_operating_margin_lt_10pct"] == 1
    assert counts["excluded_fcf_gt_revenue"] == 1
    assert counts["excluded_missing_capex"] == 1
    nocap = enrich_annual_comparatives(raw)
    nocap_row = nocap[nocap["ticker"] == "NOCAP"].iloc[0]
    assert pd.isna(nocap_row["fcf"])
    assert nocap_row["fcf"] != pytest.approx(float(nocap_row["ocf"]))


def test_rule_of_40_caps_growth_at_100_and_fcf_margin_at_50() -> None:
    reb = date(2023, 4, 3)
    # YoY = 200/80 - 1 = 150%, FCF margin = 160/200 = 80%. Uncapped score would be 230.
    raw = pd.DataFrame([
        _make_annual_row(
            cik="4001", ticker="HOT", adsh="H1", filed_date="2023-02-20",
            revenue=200e6, revenue_prior=80e6, revenue_t2=50e6,
            ocf=170e6, capex=10e6,
        ),
    ])
    enriched = enrich_annual_comparatives(raw)
    row = enriched.iloc[0]
    assert row["yoy_rev_growth"] == pytest.approx(1.50)
    assert row["fcf_margin"] == pytest.approx(0.80)
    assert row["rule_of_40"] == pytest.approx(150.0)

    cands, _ = screen_fundamentals_asof(enriched, reb, CoffeeCanConfig())
    assert len(cands) == 1
    assert cands[0].yoy_rev_growth == pytest.approx(1.50)
    assert cands[0].fcf_margin == pytest.approx(0.80)
    assert cands[0].rule_of_40 == pytest.approx(150.0)
    assert cands[0].rule_of_40 == pytest.approx(100.0 + 50.0)


def test_v10b_ttm_runway_redeploys_without_lookahead() -> None:
    spy = _make_spy("2021-10-01", "2022-09-30")
    idx = spy.frame.index
    reb22 = first_trading_day_of_april(idx, 2022)
    flat = np.full(len(idx), 20.0)
    bars = {
        "HELD": _make_bars("HELD", idx, opens=flat, closes=flat),
        "ONLY": _make_bars("ONLY", idx, opens=flat, closes=flat),
        "REPL": _make_bars("REPL", idx, opens=flat, closes=flat),
        "FUTR": _make_bars("FUTR", idx, opens=flat, closes=flat),
    }
    raw = pd.DataFrame([
        _make_annual_row(cik="0001", ticker="HELD", adsh="K-HELD", filed_date="2022-02-28", period_date="2021-12-31", ocf=50e6, capex=10e6),
        _make_annual_row(cik="0002", ticker="ONLY", adsh="K-ONLY", filed_date="2022-02-28", period_date="2021-12-31", ocf=50e6, capex=10e6),
        _make_annual_row(
            cik="0003", ticker="REPL", adsh="K-REPL", filed_date="2022-06-15", period_date="2022-03-31",
            revenue=180e6, revenue_prior=100e6, revenue_t2=70e6,
        ),
        # Filed on the exit morning. Visible only if the rank peeks past the decision date.
        _make_annual_row(
            cik="0004", ticker="FUTR", adsh="K-FUTR", filed_date="2022-08-08", period_date="2022-06-30",
            revenue=300e6, revenue_prior=100e6, revenue_t2=60e6, ocf=80e6, capex=10e6,
        ),
    ])
    annual = enrich_annual_comparatives(raw)
    cfg = CoffeeCanConfig(rule="v10b_coffee_can_thesis_watch")
    cands, _ = screen_fundamentals_asof(annual, reb22, cfg, vintage_label="2022-04")
    assert sorted(c.ticker for c in cands) == ["HELD", "ONLY"]
    ev, _ = evaluate_vintage_candidates(cands, bars, {}, spy, cfg)

    def _q(adsh: str, cik: str, filed: str, period: str, fp: str, ytd: float, qtrs: int, cash: float) -> dict:
        return {
            "adsh": adsh,
            "cik": cik,
            "filed_date": pd.Timestamp(filed),
            "period_date": pd.Timestamp(period),
            "fp": fp,
            "yoy_rev_growth": 0.30,
            "cash_total": cash,
            "ytd_fcf": ytd,
            "ytd_qtrs": qtrs,
            "q_fcf": np.nan,
        }

    quarterly = pd.DataFrame([
        _q("Q1A", "0001", "2021-05-10", "2021-03-31", "Q1", 10e6, 1, 40e6),
        _q("Q2A", "0001", "2021-08-10", "2021-06-30", "Q2", 20e6, 2, 40e6),
        _q("Q3A", "0001", "2021-11-10", "2021-09-30", "Q3", 30e6, 3, 40e6),
        _q("Q1B", "0001", "2022-05-10", "2022-03-31", "Q1", -5e6, 1, 20e6),
        _q("Q2B", "0001", "2022-08-05", "2022-06-30", "Q2", -45e6, 2, 10e6),
        # One ugly quarter. Annualizing it would fake a sub-12-month runway.
        _q("Q1C", "0002", "2022-05-10", "2022-03-31", "Q1", -40e6, 1, 10e6),
    ])
    thesis = build_thesis_watch_events(annual, quarterly)
    held_q = {e["adsh"]: e for e in thesis["0001"] if e["form"] == "10-Q"}
    assert held_q["Q1B"]["runway_source"] == "ttm_4q"
    assert held_q["Q1B"]["ttm_fcf"] == pytest.approx(25e6)
    assert held_q["Q1B"]["cash_runway_months"] == float("inf")
    assert held_q["Q2B"]["runway_source"] == "ttm_4q"
    assert held_q["Q2B"]["ttm_fcf"] == pytest.approx(-25e6)
    assert held_q["Q2B"]["cash_runway_months"] == pytest.approx(4.8)
    only_q = next(e for e in thesis["0002"] if e["form"] == "10-Q")
    assert only_q["runway_source"] == "annual_10k"
    assert only_q["cash_runway_months"] == float("inf")
    assert only_q["ttm_fcf"] == pytest.approx(40e6)

    seen: list[date] = []

    def _screen(decision: date):
        seen.append(decision)
        return screen_replacement_candidates(annual, decision, bars, {}, spy, cfg)

    res = run_coffee_can_portfolio_backtest(
        {"2022-04": ev},
        {"2022-04": reb22},
        bars,
        spy,
        thesis,
        cfg,
        reb22,
        date(2022, 9, 30),
        5300.0,
        5000.0,
        replacement_screen=_screen,
    )
    closed = {t.ticker: t for t in res.trades}
    opened = {t.ticker: t for t in res.open_positions_at_end}
    assert closed["HELD"].exit_reason == "THESIS_BREAK_RUNWAY"
    assert closed["HELD"].exit_date == date(2022, 8, 8)
    assert "ONLY" not in closed
    assert opened["ONLY"].exit_reason == "OPEN_MTM"
    assert opened["REPL"].entry_date == date(2022, 8, 8)
    assert "FUTR" not in closed and "FUTR" not in opened
    assert seen == [date(2022, 8, 5)]
    assert all(d <= date(2022, 8, 5) for d in seen)
