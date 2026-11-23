"""Unit tests for Strategy v9_form4 (SEC DERA loader, cluster detection, exits, T+1).

Verifies all strict anti-bias and execution invariants BEFORE running Blind OOS:
1. Zero ``TRANS_DATE`` look-ahead bias (triggers strictly on ``FILING_DATE``,
   executes strictly on ``FILING_DATE + 1 trading day`` Open, causal liquidity
   evaluated strictly on trading days ``<= FILING_DATE``).
2. ``AFF10B5ONE`` Rule 10b5-1 filtering (both numeric ``1`` and string ``true``/``Y``)
   and graceful handling of pre-2023 quarters lacking ``AFF10B5ONE``.
3. ``DOCUMENT_TYPE == "4"`` strict filtering (excludes ``4/A``, ``3``, ``5``) and
   critical multi-owner accession deduplication (no dollar-spend multiplication
   across co-filers, no fake 2-insider clusters from a single joint accession).
4. Pure passive 10% owner exclusion vs Officer/Director inclusion, plus C-suite
   title matching (CEO/CFO/President included; Vice President excluded).
5. ``v9a_standard`` and ``v9b_high_conviction`` thresholds & 60-day per-issuer cooldown.
6. ``-25%`` gap-through catastrophe stop (fills at Open, NEVER inside gap) and
   intraday stop fill.
7. ``+30%``-activated ``HighestClose - 3.5 * ATR14`` trailing stop (inactive below
   ``+30%``, active once ``>= +30%`` close reached, executes next Open).
8. 90-trading-day hold time exit.
9. IBKR Cash Account T+1 settlement (sale proceeds unavailable same day, available
   next trading day Open) and whole-share sizing with ``$0.38`` + ``0.25%`` friction.
10. Survivorship-penalized sensitivity (``-50%`` on unmatched & early-terminated).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.sec_form4_loader import (
    is_c_suite_title,
    is_excluded_non_common_issuer,
    normalize_sec_ticker,
    parse_quarter_frames,
)
from src.domain.models import BarSeries
from src.strategy.sec_form4_cluster import (
    detect_form4_clusters,
    evaluate_clusters_and_survivorship,
    make_v9a_config,
    make_v9b_config,
    run_event_study,
    run_form4_portfolio_backtest,
)


def _make_bars(
    symbol: str,
    dates: pd.DatetimeIndex,
    opens: list[float] | np.ndarray,
    highs: list[float] | np.ndarray | None = None,
    lows: list[float] | np.ndarray | None = None,
    closes: list[float] | np.ndarray | None = None,
    volumes: list[float] | np.ndarray | None = None,
) -> BarSeries:
    o = np.asarray(opens, dtype=float)
    c = o.copy() if closes is None else np.asarray(closes, dtype=float)
    h = np.maximum(o, c) * 1.01 if highs is None else np.asarray(highs, dtype=float)
    l = np.minimum(o, c) * 0.99 if lows is None else np.asarray(lows, dtype=float)
    v = np.full(len(dates), 500_000.0) if volumes is None else np.asarray(volumes, dtype=float)
    df = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "raw_close": c,
        },
        index=dates,
    )
    return BarSeries(symbol=symbol, frame=df)


def _make_spy(start: str = "2023-01-03", periods: int = 260) -> BarSeries:
    idx = pd.bdate_range(start=start, periods=periods)
    px = np.linspace(400.0, 440.0, periods)
    return _make_bars("SPY", idx, opens=px, closes=px, volumes=np.full(periods, 50_000_000.0))


# ---------------------------------------------------------------------------
# 1. Ticker normalization, C-suite titles, and non-common SEC name exclusion
# ---------------------------------------------------------------------------
def test_ticker_normalization_and_csuite_matching() -> None:
    assert normalize_sec_ticker("NYSE: VTEX") == "VTEX"
    assert normalize_sec_ticker("NASDAQ:SVC") == "SVC"
    assert normalize_sec_ticker("(CALX)") == "CALX"
    assert normalize_sec_ticker("[ENTX]") == "ENTX"
    assert normalize_sec_ticker("BRK.B") == "BRK-B"
    assert normalize_sec_ticker("WLY, WLYB") == "WLY"
    assert normalize_sec_ticker("NONE") == ""
    assert normalize_sec_ticker("[NONE]") == ""
    assert normalize_sec_ticker("N/A") == ""
    assert normalize_sec_ticker("ASX:LNW") == ""
    assert normalize_sec_ticker("1314152") == ""
    assert normalize_sec_ticker("PCG-PREF") == ""

    assert is_c_suite_title("Chief Executive Officer") is True
    assert is_c_suite_title("CFO & EVP") is True
    assert is_c_suite_title("President and CEO") is True
    assert is_c_suite_title("Co-President") is True
    assert is_c_suite_title("President") is True
    assert is_c_suite_title("Vice President") is False
    assert is_c_suite_title("Executive Vice President") is False
    assert is_c_suite_title("Senior Vice-President") is False
    assert is_c_suite_title("VP Finance") is False
    assert is_c_suite_title("Director") is False

    assert is_excluded_non_common_issuer("Nuveen Municipal Income Fund", "NMI") is True
    assert is_excluded_non_common_issuer("Apex Acquisition Corp.", "APAC") is True
    assert is_excluded_non_common_issuer("Acme Industrial Corp", "ACME") is False


# ---------------------------------------------------------------------------
# 2. DOCUMENT_TYPE=="4", AFF10B5ONE, and multi-owner accession deduplication
# ---------------------------------------------------------------------------
def test_dera_parsing_dedup_and_10b51_filtering() -> None:
    sub = pd.DataFrame([
        # Valid Form 4 with 2 joint co-filers (insider + family trust)
        {"ACCESSION_NUMBER": "ACC-1", "FILING_DATE": "10-MAY-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "12345", "ISSUERNAME": "Acme Corp", "ISSUERTRADINGSYMBOL": "ACME", "AFF10B5ONE": "0"},
        # Form 4/A amendment -> MUST be excluded
        {"ACCESSION_NUMBER": "ACC-2", "FILING_DATE": "10-MAY-2024", "DOCUMENT_TYPE": "4/A",
         "ISSUERCIK": "12345", "ISSUERNAME": "Acme Corp", "ISSUERTRADINGSYMBOL": "ACME", "AFF10B5ONE": "0"},
        # Form 4 with AFF10B5ONE == "1" -> MUST be excluded
        {"ACCESSION_NUMBER": "ACC-3", "FILING_DATE": "11-MAY-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "12345", "ISSUERNAME": "Acme Corp", "ISSUERTRADINGSYMBOL": "ACME", "AFF10B5ONE": "1"},
        # Form 4 with AFF10B5ONE == "true" -> MUST be excluded
        {"ACCESSION_NUMBER": "ACC-4", "FILING_DATE": "12-MAY-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "12345", "ISSUERNAME": "Acme Corp", "ISSUERTRADINGSYMBOL": "ACME", "AFF10B5ONE": "true"},
        # Pure 10% passive owner -> MUST be excluded
        {"ACCESSION_NUMBER": "ACC-5", "FILING_DATE": "12-MAY-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "12345", "ISSUERNAME": "Acme Corp", "ISSUERTRADINGSYMBOL": "ACME", "AFF10B5ONE": "false"},
        # Second genuine independent insider (Director)
        {"ACCESSION_NUMBER": "ACC-6", "FILING_DATE": "13-MAY-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "12345", "ISSUERNAME": "Acme Corp", "ISSUERTRADINGSYMBOL": "ACME", "AFF10B5ONE": "0"},
    ])
    own = pd.DataFrame([
        # Joint co-filers on ACC-1: CEO + their Family Trust LLC
        {"ACCESSION_NUMBER": "ACC-1", "RPTOWNERCIK": "9001", "RPTOWNERNAME": "Jane CEO",
         "RPTOWNER_RELATIONSHIP": "Officer,Director", "RPTOWNER_TITLE": "Chief Executive Officer"},
        {"ACCESSION_NUMBER": "ACC-1", "RPTOWNERCIK": "9002", "RPTOWNERNAME": "Jane Family Trust LLC",
         "RPTOWNER_RELATIONSHIP": "TenPercentOwner", "RPTOWNER_TITLE": ""},
        {"ACCESSION_NUMBER": "ACC-2", "RPTOWNERCIK": "9003", "RPTOWNERNAME": "Bob Amend",
         "RPTOWNER_RELATIONSHIP": "Officer", "RPTOWNER_TITLE": "CFO"},
        {"ACCESSION_NUMBER": "ACC-3", "RPTOWNERCIK": "9004", "RPTOWNERNAME": "Plan Insider",
         "RPTOWNER_RELATIONSHIP": "Officer", "RPTOWNER_TITLE": "CFO"},
        {"ACCESSION_NUMBER": "ACC-4", "RPTOWNERCIK": "9005", "RPTOWNERNAME": "Plan Insider 2",
         "RPTOWNER_RELATIONSHIP": "Director", "RPTOWNER_TITLE": ""},
        {"ACCESSION_NUMBER": "ACC-5", "RPTOWNERCIK": "9006", "RPTOWNERNAME": "Mega Fund LP",
         "RPTOWNER_RELATIONSHIP": "TenPercentOwner", "RPTOWNER_TITLE": ""},
        {"ACCESSION_NUMBER": "ACC-6", "RPTOWNERCIK": "9007", "RPTOWNERNAME": "John Director",
         "RPTOWNER_RELATIONSHIP": "Director", "RPTOWNER_TITLE": ""},
    ])
    trx = pd.DataFrame([
        {"ACCESSION_NUMBER": "ACC-1", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "08-MAY-2024",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "10000", "TRANS_PRICEPERSHARE": "15.0"},
        {"ACCESSION_NUMBER": "ACC-2", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "08-MAY-2024",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "5000", "TRANS_PRICEPERSHARE": "15.0"},
        {"ACCESSION_NUMBER": "ACC-3", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "09-MAY-2024",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "10000", "TRANS_PRICEPERSHARE": "15.0"},
        {"ACCESSION_NUMBER": "ACC-4", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "10-MAY-2024",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "10000", "TRANS_PRICEPERSHARE": "15.0"},
        {"ACCESSION_NUMBER": "ACC-5", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "10-MAY-2024",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "50000", "TRANS_PRICEPERSHARE": "15.0"},
        {"ACCESSION_NUMBER": "ACC-6", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "11-MAY-2024",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "10000", "TRANS_PRICEPERSHARE": "12.0"},
    ])

    acc_df, audit = parse_quarter_frames(sub, own, trx, quarter="2024q2")

    assert list(acc_df["accession_number"]) == ["ACC-1", "ACC-6"]
    assert audit["doc_type_non_4_excluded"] == 1
    assert audit["aff10b5one_excluded"] == 2
    assert audit["multi_owner_accessions_collapsed"] == 1
    assert audit["pure_ten_percent_accessions_excluded"] == 1

    # Critical check: ACC-1 spend is $150,000 ONCE (NOT $300,000 across 2 co-filers!)
    row1 = acc_df[acc_df["accession_number"] == "ACC-1"].iloc[0]
    assert row1["total_dollar_spend"] == pytest.approx(150_000.0)
    assert row1["num_co_filers"] == 2
    assert row1["is_c_suite"] == True

    # Single joint accession ACC-1 alone MUST NOT trigger a 2-insider cluster!
    only_joint = acc_df[acc_df["accession_number"] == "ACC-1"].copy()
    sigs_joint, _ = detect_form4_clusters(only_joint, make_v9a_config())
    assert len(sigs_joint) == 0

    # Together ACC-1 ($150k, CEO) + ACC-6 ($120k, Director) = $270k, 2 distinct insiders incl C-suite -> triggers BOTH v9a and v9b
    sigs_v9a, _ = detect_form4_clusters(acc_df, make_v9a_config())
    sigs_v9b, _ = detect_form4_clusters(acc_df, make_v9b_config())
    assert len(sigs_v9a) == 1
    assert sigs_v9a[0].combined_dollar_spend == pytest.approx(270_000.0)
    assert sigs_v9a[0].distinct_insiders == 2
    assert len(sigs_v9b) == 1
    assert sigs_v9b[0].has_c_suite is True


def test_pre2023_quarter_without_aff10b5one_column() -> None:
    sub = pd.DataFrame([
        {"ACCESSION_NUMBER": "OLD-1", "FILING_DATE": "15-NOV-2021", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "777", "ISSUERNAME": "Legacy Co", "ISSUERTRADINGSYMBOL": "LEGC"},
    ])
    own = pd.DataFrame([
        {"ACCESSION_NUMBER": "OLD-1", "RPTOWNERCIK": "111", "RPTOWNERNAME": "Alice",
         "RPTOWNER_RELATIONSHIP": "Officer", "RPTOWNER_TITLE": "CEO"},
    ])
    trx = pd.DataFrame([
        {"ACCESSION_NUMBER": "OLD-1", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "12-NOV-2021",
         "TRANS_CODE": "P", "TRANS_ACQUIRED_DISP_CD": "A", "TRANS_SHARES": "5000", "TRANS_PRICEPERSHARE": "20.0"},
    ])
    acc_df, audit = parse_quarter_frames(sub, own, trx, quarter="2021q4")
    assert audit["has_aff10b5one_col"] is False
    assert audit["aff10b5one_excluded"] == 0
    assert len(acc_df) == 1


# ---------------------------------------------------------------------------
# 3. v9a vs v9b thresholds & 60-calendar-day per-issuer cooldown
# ---------------------------------------------------------------------------
def test_v9a_v9b_thresholds_and_60d_cooldown() -> None:
    # 2 Directors spending $150k total -> qualifies for v9a, NOT v9b (needs >=$250k and >=3 or C-suite)
    rows = [
        {
            "accession_number": "A1", "quarter": "2024q1", "filing_date": pd.Timestamp("2024-02-01"),
            "earliest_trans_date": pd.Timestamp("2024-01-30"), "latest_trans_date": pd.Timestamp("2024-01-30"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000001", "primary_owner_name": "Dir A",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 7500.0, "total_dollar_spend": 75_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
        {
            "accession_number": "A2", "quarter": "2024q1", "filing_date": pd.Timestamp("2024-02-05"),
            "earliest_trans_date": pd.Timestamp("2024-02-02"), "latest_trans_date": pd.Timestamp("2024-02-02"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000002", "primary_owner_name": "Dir B",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 7500.0, "total_dollar_spend": 75_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
        # Duplicate cluster 30 days later -> MUST be suppressed by 60-day cooldown
        {
            "accession_number": "A3", "quarter": "2024q1", "filing_date": pd.Timestamp("2024-03-05"),
            "earliest_trans_date": pd.Timestamp("2024-03-04"), "latest_trans_date": pd.Timestamp("2024-03-04"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000003", "primary_owner_name": "Dir C",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 10000.0, "total_dollar_spend": 100_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
        {
            "accession_number": "A4", "quarter": "2024q1", "filing_date": pd.Timestamp("2024-03-06"),
            "earliest_trans_date": pd.Timestamp("2024-03-05"), "latest_trans_date": pd.Timestamp("2024-03-05"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000004", "primary_owner_name": "Dir D",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 10000.0, "total_dollar_spend": 100_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
        # Fresh cluster 65 days after 2024-02-05 (2024-04-10) -> allowed!
        {
            "accession_number": "A5", "quarter": "2024q2", "filing_date": pd.Timestamp("2024-04-08"),
            "earliest_trans_date": pd.Timestamp("2024-04-05"), "latest_trans_date": pd.Timestamp("2024-04-05"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000001", "primary_owner_name": "Dir A",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 15000.0, "total_dollar_spend": 150_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
        {
            "accession_number": "A6", "quarter": "2024q2", "filing_date": pd.Timestamp("2024-04-10"),
            "earliest_trans_date": pd.Timestamp("2024-04-09"), "latest_trans_date": pd.Timestamp("2024-04-09"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000002", "primary_owner_name": "Dir B",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 10000.0, "total_dollar_spend": 100_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
        {
            "accession_number": "A7", "quarter": "2024q2", "filing_date": pd.Timestamp("2024-04-11"),
            "earliest_trans_date": pd.Timestamp("2024-04-10"), "latest_trans_date": pd.Timestamp("2024-04-10"),
            "issuer_cik": "0000001000", "ticker": "TEST", "issuer_name": "Test Inc",
            "is_non_common_name": False, "primary_owner_cik": "0000000003", "primary_owner_name": "Dir C",
            "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
            "is_c_suite": False, "num_co_filers": 1, "total_shares": 5000.0, "total_dollar_spend": 50_000.0,
            "weighted_avg_price": 10.0, "trans_row_count": 1,
        },
    ]
    df = pd.DataFrame(rows)

    sigs_a, st_a = detect_form4_clusters(df, make_v9a_config())
    assert [s.filing_date for s in sigs_a] == [date(2024, 2, 5), date(2024, 4, 10)]
    assert st_a["suppressed_by_cooldown"] >= 1

    # v9b: 2024-04-11 has 3 distinct insiders + $300k spend -> qualifies for v9b!
    sigs_b, _ = detect_form4_clusters(df, make_v9b_config())
    assert [s.filing_date for s in sigs_b] == [date(2024, 4, 11)]
    assert sigs_b[0].distinct_insiders == 3
    assert sigs_b[0].combined_dollar_spend == pytest.approx(300_000.0)


# ---------------------------------------------------------------------------
# 4. Zero TRANS_DATE look-ahead bias & strictly causal liquidity filter
# ---------------------------------------------------------------------------
def test_zero_trans_date_lookahead_and_causal_liquidity() -> None:
    spy = _make_spy("2023-01-03", 150)
    idx = spy.frame.index
    # TRANS_DATE is day 25, but FILING_DATE is day 35 (delayed filing!)
    trans_dt = idx[25]
    filing_dt = idx[35]
    expected_entry_dt = idx[36]  # Strictly FILING_DATE + 1 trading day Open!

    # Stock surges on day 26 (between TRANS_DATE and FILING_DATE) from $20 to $30
    opens = np.full(len(idx), 20.0)
    closes = np.full(len(idx), 20.0)
    opens[26:] = 30.0
    closes[26:] = 30.0
    # Make ADV below $1M before filing_dt, then huge AFTER filing_dt (must NOT pass!)
    vols_fail = np.full(len(idx), 10_000.0)  # $200k-$300k/day <= filing_dt
    vols_fail[36:] = 5_000_000.0  # $150M/day AFTER filing_dt (future!)

    vols_pass = np.full(len(idx), 200_000.0)  # $4M-$6M/day <= filing_dt

    acc = pd.DataFrame([
        {
            "accession_number": "L1", "quarter": "2023q1", "filing_date": filing_dt,
            "earliest_trans_date": trans_dt, "latest_trans_date": trans_dt,
            "issuer_cik": "0000002000", "ticker": "LATE", "issuer_name": "Late File Corp",
            "is_non_common_name": False, "primary_owner_cik": "0000000010", "primary_owner_name": "CEO",
            "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
            "is_c_suite": True, "num_co_filers": 1, "total_shares": 5000.0, "total_dollar_spend": 100_000.0,
            "weighted_avg_price": 20.0, "trans_row_count": 1,
        },
        {
            "accession_number": "L2", "quarter": "2023q1", "filing_date": filing_dt,
            "earliest_trans_date": trans_dt, "latest_trans_date": trans_dt,
            "issuer_cik": "0000002000", "ticker": "LATE", "issuer_name": "Late File Corp",
            "is_non_common_name": False, "primary_owner_cik": "0000000011", "primary_owner_name": "CFO",
            "owner_relationship": "Officer", "owner_title": "CFO", "is_officer": True, "is_director": False,
            "is_c_suite": True, "num_co_filers": 1, "total_shares": 5000.0, "total_dollar_spend": 100_000.0,
            "weighted_avg_price": 20.0, "trans_row_count": 1,
        },
    ])
    cfg = make_v9a_config()
    sigs, _ = detect_form4_clusters(acc, cfg)
    assert len(sigs) == 1
    assert sigs[0].filing_date == filing_dt.date()

    # Future volume spike after FILING_DATE MUST NOT leak into liquidity filter
    bars_fail = {"LATE": _make_bars("LATE", idx, opens=opens, closes=closes, volumes=vols_fail)}
    ev_fail, _ = evaluate_clusters_and_survivorship(
        sigs, bars_fail, {}, spy, idx[-1].date(), cfg
    )
    assert ev_fail[0].status == "rejected_adv"

    # With causal ADV >= $1M, entry MUST be strictly on FILING_DATE + 1 trading day Open ($30, NOT $20 at TRANS_DATE)
    bars_pass = {"LATE": _make_bars("LATE", idx, opens=opens, closes=closes, volumes=vols_pass)}
    ev_pass, _ = evaluate_clusters_and_survivorship(
        sigs, bars_pass, {}, spy, idx[-1].date(), cfg
    )
    assert ev_pass[0].status == "qualified_matched"
    assert ev_pass[0].entry_date == expected_entry_dt.date()
    assert ev_pass[0].entry_raw_open == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 5. -25% Gap-Through Hard Stop (never fills inside gap!)
# ---------------------------------------------------------------------------
def test_hard_stop_gap_through_fills_at_open() -> None:
    spy = _make_spy("2023-01-03", 120)
    idx = spy.frame.index
    filing_dt = idx[25]
    entry_idx = 26

    opens = np.full(len(idx), 40.0)
    highs = np.full(len(idx), 40.5)
    lows = np.full(len(idx), 39.5)
    closes = np.full(len(idx), 40.0)

    # On day entry_idx + 5, stock gaps down overnight to $22.00 (far below -25% stop ~$30.075)
    gap_i = entry_idx + 5
    opens[gap_i:] = 22.0
    highs[gap_i:] = 23.0
    lows[gap_i:] = 21.0
    closes[gap_i:] = 22.5

    bar = _make_bars("GAP", idx, opens=opens, highs=highs, lows=lows, closes=closes)
    acc = pd.DataFrame([
        {"accession_number": "G1", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0000003000", "ticker": "GAP", "issuer_name": "Gap Corp",
         "is_non_common_name": False, "primary_owner_cik": "1", "primary_owner_name": "A",
         "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
         "is_c_suite": True, "num_co_filers": 1, "total_shares": 2000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 40.0, "trans_row_count": 1},
        {"accession_number": "G2", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0000003000", "ticker": "GAP", "issuer_name": "Gap Corp",
         "is_non_common_name": False, "primary_owner_cik": "2", "primary_owner_name": "B",
         "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
         "is_c_suite": False, "num_co_filers": 1, "total_shares": 2000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 40.0, "trans_row_count": 1},
    ])
    cfg = make_v9a_config()
    sigs, _ = detect_form4_clusters(acc, cfg)
    ev, _ = evaluate_clusters_and_survivorship(sigs, {"GAP": bar}, {}, spy, idx[-1].date(), cfg)
    res = run_form4_portfolio_backtest(
        ev, {"GAP": bar}, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0
    )
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.exit_reason == "HARD_STOP_GAP"
    assert tr.exit_date == idx[gap_i].date()
    # MUST fill at gap open $22.00 (minus 0.25% slippage), NEVER at the ~$30.08 stop!
    assert tr.exit_raw_price == pytest.approx(22.0)
    assert tr.exit_fill_price == pytest.approx(22.0 * (1.0 - 0.0025))


# ---------------------------------------------------------------------------
# 6. +30%-Activated 3.5 * ATR14 Trailing Stop & 90-Day Hold Exit
# ---------------------------------------------------------------------------
def test_trailing_stop_activation_and_90d_time_exit() -> None:
    spy = _make_spy("2023-01-03", 180)
    idx = spy.frame.index
    filing_dt = idx[25]
    entry_i = 26

    # Case A: Stock rises to +25% (below +30% activation), pulls back to +5% (well > 3.5*ATR),
    # MUST NOT trigger trailing stop -> holds all the way to 90-trading-day TIME_EXIT_90D!
    opens_a = np.full(len(idx), 20.0)
    highs_a = np.full(len(idx), 20.2)
    lows_a = np.full(len(idx), 19.8)
    closes_a = np.full(len(idx), 20.0)
    opens_a[entry_i + 10 : entry_i + 20] = 24.5
    highs_a[entry_i + 10 : entry_i + 20] = 24.8
    lows_a[entry_i + 10 : entry_i + 20] = 24.2
    closes_a[entry_i + 10 : entry_i + 20] = 24.5  # +22.5% (< +30%)
    opens_a[entry_i + 20 :] = 21.0
    highs_a[entry_i + 20 :] = 21.2
    lows_a[entry_i + 20 :] = 20.8
    closes_a[entry_i + 20 :] = 21.0

    # Case B: Stock rises to +40% ($28 -> activates trailing stop!), then drops sharply to $22 ->
    # triggers TRAILING_STOP_EXIT at next Open!
    opens_b = np.full(len(idx), 20.0)
    highs_b = np.full(len(idx), 20.2)
    lows_b = np.full(len(idx), 19.8)
    closes_b = np.full(len(idx), 20.0)
    opens_b[entry_i + 10 : entry_i + 20] = 28.0
    highs_b[entry_i + 10 : entry_i + 20] = 28.3
    lows_b[entry_i + 10 : entry_i + 20] = 27.7
    closes_b[entry_i + 10 : entry_i + 20] = 28.0  # +40% (>= +30%!)
    drop_i = entry_i + 20
    opens_b[drop_i:] = 22.0
    highs_b[drop_i:] = 22.2
    lows_b[drop_i:] = 21.8
    closes_b[drop_i:] = 22.0  # Drop of $6 > 3.5 * ATR14 (~$1.0)

    def _run_one(sym: str, bar: BarSeries):
        acc = pd.DataFrame([
            {"accession_number": f"{sym}1", "quarter": "2023q1", "filing_date": filing_dt,
             "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
             "issuer_cik": f"0000004{sym}", "ticker": sym, "issuer_name": f"{sym} Inc",
             "is_non_common_name": False, "primary_owner_cik": "1", "primary_owner_name": "A",
             "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
             "is_c_suite": True, "num_co_filers": 1, "total_shares": 4000.0, "total_dollar_spend": 80_000.0,
             "weighted_avg_price": 20.0, "trans_row_count": 1},
            {"accession_number": f"{sym}2", "quarter": "2023q1", "filing_date": filing_dt,
             "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
             "issuer_cik": f"0000004{sym}", "ticker": sym, "issuer_name": f"{sym} Inc",
             "is_non_common_name": False, "primary_owner_cik": "2", "primary_owner_name": "B",
             "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
             "is_c_suite": False, "num_co_filers": 1, "total_shares": 4000.0, "total_dollar_spend": 80_000.0,
             "weighted_avg_price": 20.0, "trans_row_count": 1},
        ])
        cfg = make_v9a_config()
        sigs, _ = detect_form4_clusters(acc, cfg)
        ev, _ = evaluate_clusters_and_survivorship(sigs, {sym: bar}, {}, spy, idx[-1].date(), cfg)
        return run_form4_portfolio_backtest(
            ev, {sym: bar}, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0
        )

    res_a = _run_one("HOLD", _make_bars("HOLD", idx, opens_a, highs_a, lows_a, closes_a))
    assert len(res_a.trades) == 1
    assert res_a.trades[0].exit_reason == "TIME_EXIT_90D"
    assert res_a.trades[0].trailing_activated is False
    assert res_a.trades[0].bars_held == 90
    assert res_a.trades[0].exit_date == idx[entry_i + 90].date()

    res_b = _run_one("TRAIL", _make_bars("TRAIL", idx, opens_b, highs_b, lows_b, closes_b))
    assert len(res_b.trades) == 1
    assert res_b.trades[0].exit_reason == "TRAILING_STOP_EXIT"
    assert res_b.trades[0].trailing_activated is True
    assert res_b.trades[0].exit_date == idx[drop_i + 1].date()  # Queued at drop_i close -> next Open!


# ---------------------------------------------------------------------------
# 7. IBKR Cash Account T+1 Settlement Enforcement
# ---------------------------------------------------------------------------
def test_t_plus_1_cash_account_settlement() -> None:
    spy = _make_spy("2023-01-03", 120)
    idx = spy.frame.index

    # Use max_slots=1, target_slot_pct=1.0 so 100% of cash is deployed in Trade 1.
    # Trade 1 enters on idx[26], gaps down below -25% hard stop on idx[30] Open -> sells on idx[30].
    # Trade 2 triggers on idx[29] (so entry_date == idx[30] Open, SAME day as Trade 1 sale!).
    # Under T+1 settlement, proceeds from Trade 1 settle on idx[31] Open -> Trade 2 MUST be skipped!
    # Trade 3 triggers on idx[30] (entry_date == idx[31] Open, AFTER T+1 settlement!) -> Trade 3 MUST execute!
    o1 = np.full(len(idx), 20.0)
    c1 = np.full(len(idx), 20.0)
    o1[30:] = 10.0
    c1[30:] = 10.0
    bar1 = _make_bars("ONE", idx, opens=o1, closes=c1)
    bar2 = _make_bars("TWO", idx, opens=np.full(len(idx), 20.0), closes=np.full(len(idx), 20.0))
    bar3 = _make_bars("THR", idx, opens=np.full(len(idx), 20.0), closes=np.full(len(idx), 20.0))

    def _pair(tag: str, sym: str, cik: str, f_ts: pd.Timestamp) -> list[dict]:
        return [
            {"accession_number": f"{tag}A", "quarter": "2023q1", "filing_date": f_ts,
             "earliest_trans_date": f_ts, "latest_trans_date": f_ts,
             "issuer_cik": cik, "ticker": sym, "issuer_name": f"{sym} Corp",
             "is_non_common_name": False, "primary_owner_cik": "1", "primary_owner_name": "A",
             "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
             "is_c_suite": True, "num_co_filers": 1, "total_shares": 3000.0, "total_dollar_spend": 60_000.0,
             "weighted_avg_price": 20.0, "trans_row_count": 1},
            {"accession_number": f"{tag}B", "quarter": "2023q1", "filing_date": f_ts,
             "earliest_trans_date": f_ts, "latest_trans_date": f_ts,
             "issuer_cik": cik, "ticker": sym, "issuer_name": f"{sym} Corp",
             "is_non_common_name": False, "primary_owner_cik": "2", "primary_owner_name": "B",
             "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
             "is_c_suite": False, "num_co_filers": 1, "total_shares": 3000.0, "total_dollar_spend": 60_000.0,
             "weighted_avg_price": 20.0, "trans_row_count": 1},
        ]

    acc = pd.DataFrame(
        _pair("T1", "ONE", "0001", idx[25])
        + _pair("T2", "TWO", "0002", idx[29])
        + _pair("T3", "THR", "0003", idx[30])
    )
    cfg = make_v9a_config(max_slots=1, target_slot_pct=1.0, settlement_days=1)
    sigs, _ = detect_form4_clusters(acc, cfg)
    bars = {"ONE": bar1, "TWO": bar2, "THR": bar3}
    ev, _ = evaluate_clusters_and_survivorship(sigs, bars, {}, spy, idx[-1].date(), cfg)
    res = run_form4_portfolio_backtest(
        ev, bars, spy, cfg, idx[0].date(), idx[-1].date(), 1060.0, 1000.0
    )
    entered_tickers = [t.ticker for t in res.trades] + [t.ticker for t in res.open_positions_at_end]
    assert "ONE" in entered_tickers
    assert "TWO" not in entered_tickers  # Blocked by T+1 unsettled cash on idx[30]!
    assert "THR" in entered_tickers      # Settled on idx[31] Open -> executes!
    assert res.skipped_counts["insufficient_settled_cash"] >= 1


# ---------------------------------------------------------------------------
# 8. Survivorship Audit & Penalized Sensitivity (-50% on unmatched)
# ---------------------------------------------------------------------------
def test_survivorship_penalized_event_study_and_portfolio() -> None:
    spy = _make_spy("2023-01-03", 180)
    idx = spy.frame.index
    filing_dt = idx[25]

    # Matched winner (+20%) vs Unmatched delisted ticker (absent from bars, Form 4 price $15)
    bar_win = _make_bars("WIN", idx, opens=np.linspace(20.0, 26.0, len(idx)), closes=np.linspace(20.0, 26.0, len(idx)))
    acc = pd.DataFrame([
        {"accession_number": "W1", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0001", "ticker": "WIN", "issuer_name": "Winner Co",
         "is_non_common_name": False, "primary_owner_cik": "1", "primary_owner_name": "A",
         "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
         "is_c_suite": True, "num_co_filers": 1, "total_shares": 4000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 20.0, "trans_row_count": 1},
        {"accession_number": "W2", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0001", "ticker": "WIN", "issuer_name": "Winner Co",
         "is_non_common_name": False, "primary_owner_cik": "2", "primary_owner_name": "B",
         "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
         "is_c_suite": False, "num_co_filers": 1, "total_shares": 4000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 20.0, "trans_row_count": 1},
        {"accession_number": "D1", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0002", "ticker": "DEAD", "issuer_name": "Delisted Bank Corp",
         "is_non_common_name": False, "primary_owner_cik": "3", "primary_owner_name": "C",
         "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
         "is_c_suite": True, "num_co_filers": 1, "total_shares": 10000.0, "total_dollar_spend": 150_000.0,
         "weighted_avg_price": 15.0, "trans_row_count": 1},
        {"accession_number": "D2", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0002", "ticker": "DEAD", "issuer_name": "Delisted Bank Corp",
         "is_non_common_name": False, "primary_owner_cik": "4", "primary_owner_name": "D",
         "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
         "is_c_suite": False, "num_co_filers": 1, "total_shares": 10000.0, "total_dollar_spend": 150_000.0,
         "weighted_avg_price": 15.0, "trans_row_count": 1},
    ])
    cfg = make_v9a_config()
    sigs, _ = detect_form4_clusters(acc, cfg)
    ev, audit = evaluate_clusters_and_survivorship(sigs, {"WIN": bar_win}, {}, spy, idx[-1].date(), cfg)

    assert audit["qualified_matched"] == 1
    assert audit["unmatched_yfinance"] == 1
    assert audit["unmatched_price_eligible"] == 1

    es = run_event_study(ev, {"WIN": bar_win}, spy, idx[0].date(), idx[-1].date(), idx[-1].date(), cfg)
    assert es["raw_matched"]["horizons"]["+90d"]["n"] == 1
    assert es["survivorship_penalized"]["horizons"]["+90d"]["n"] == 2
    assert es["survivorship_penalized"]["horizons"]["+90d"]["mean_stock_ret"] < es["raw_matched"]["horizons"]["+90d"]["mean_stock_ret"]

    res_raw = run_form4_portfolio_backtest(
        ev, {"WIN": bar_win}, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0, survivorship_penalized=False
    )
    res_pen = run_form4_portfolio_backtest(
        ev, {"WIN": bar_win}, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0, survivorship_penalized=True
    )
    assert any(t.exit_reason == "UNMATCHED_PENALIZED_50" for t in res_pen.trades)
    assert res_pen.equity_curve["equity"].iloc[-1] < res_raw.equity_curve["equity"].iloc[-1]


# ---------------------------------------------------------------------------
# 9. Intraday hard stop, causal tie-breaker, and early-terminated delisting
# ---------------------------------------------------------------------------
def test_intraday_hard_stop_and_causal_tiebreaker() -> None:
    spy = _make_spy("2023-01-03", 120)
    idx = spy.frame.index
    filing_dt = idx[25]
    entry_i = 26

    # Stock BIG ($400k spend) vs Stock SML ($150k spend) on the SAME filing_date with max_slots=1.
    # BIG must win the tie-breaker! Then on entry_i + 3, Open > stop and Low <= stop -> HARD_STOP_INTRADAY.
    o_big = np.full(len(idx), 40.0)
    h_big = np.full(len(idx), 40.5)
    l_big = np.full(len(idx), 39.5)
    c_big = np.full(len(idx), 40.0)
    stop_i = entry_i + 3
    o_big[stop_i] = 35.0  # Above -25% stop (~30.075)
    l_big[stop_i] = 28.0  # Breaches -25% stop intraday!
    c_big[stop_i] = 31.0

    bar_big = _make_bars("BIG", idx, opens=o_big, highs=h_big, lows=l_big, closes=c_big)
    bar_sml = _make_bars("SML", idx, opens=np.full(len(idx), 40.0), closes=np.full(len(idx), 40.0))

    acc = pd.DataFrame([
        {"accession_number": "S1", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0010", "ticker": "SML", "issuer_name": "Small Spend Corp",
         "is_non_common_name": False, "primary_owner_cik": "1", "primary_owner_name": "A",
         "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
         "is_c_suite": True, "num_co_filers": 1, "total_shares": 2000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 40.0, "trans_row_count": 1},
        {"accession_number": "S2", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0010", "ticker": "SML", "issuer_name": "Small Spend Corp",
         "is_non_common_name": False, "primary_owner_cik": "2", "primary_owner_name": "B",
         "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
         "is_c_suite": False, "num_co_filers": 1, "total_shares": 1750.0, "total_dollar_spend": 70_000.0,
         "weighted_avg_price": 40.0, "trans_row_count": 1},
        {"accession_number": "B1", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0011", "ticker": "BIG", "issuer_name": "Big Spend Corp",
         "is_non_common_name": False, "primary_owner_cik": "3", "primary_owner_name": "C",
         "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
         "is_c_suite": True, "num_co_filers": 1, "total_shares": 5000.0, "total_dollar_spend": 200_000.0,
         "weighted_avg_price": 40.0, "trans_row_count": 1},
        {"accession_number": "B2", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0011", "ticker": "BIG", "issuer_name": "Big Spend Corp",
         "is_non_common_name": False, "primary_owner_cik": "4", "primary_owner_name": "D",
         "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
         "is_c_suite": False, "num_co_filers": 1, "total_shares": 5000.0, "total_dollar_spend": 200_000.0,
         "weighted_avg_price": 40.0, "trans_row_count": 1},
    ])
    cfg = make_v9a_config(max_slots=1, target_slot_pct=1.0)
    sigs, _ = detect_form4_clusters(acc, cfg)
    bars = {"BIG": bar_big, "SML": bar_sml}
    ev, _ = evaluate_clusters_and_survivorship(sigs, bars, {}, spy, idx[-1].date(), cfg)
    res = run_form4_portfolio_backtest(
        ev, bars, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0
    )
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.ticker == "BIG"
    assert tr.exit_reason == "HARD_STOP_INTRADAY"
    assert tr.exit_date == idx[stop_i].date()
    expected_stop = 40.0 * 1.0025 * 0.75
    assert tr.exit_raw_price == pytest.approx(expected_stop)
    assert res.skipped_counts["max_slots_full"] == 1


def test_early_terminated_delisting_raw_vs_penalized() -> None:
    spy = _make_spy("2023-01-03", 180)
    idx = spy.frame.index
    filing_dt = idx[25]
    # Stock bars end abruptly at idx[45] (only 20 bars after entry, well before +90d and sample end!)
    short_idx = idx[:46]
    bar_early = _make_bars("TERM", short_idx, opens=np.full(len(short_idx), 20.0), closes=np.full(len(short_idx), 19.0))
    acc = pd.DataFrame([
        {"accession_number": "E1", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0099", "ticker": "TERM", "issuer_name": "Early Term Corp",
         "is_non_common_name": False, "primary_owner_cik": "1", "primary_owner_name": "A",
         "owner_relationship": "Officer", "owner_title": "CEO", "is_officer": True, "is_director": False,
         "is_c_suite": True, "num_co_filers": 1, "total_shares": 4000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 20.0, "trans_row_count": 1},
        {"accession_number": "E2", "quarter": "2023q1", "filing_date": filing_dt,
         "earliest_trans_date": filing_dt, "latest_trans_date": filing_dt,
         "issuer_cik": "0099", "ticker": "TERM", "issuer_name": "Early Term Corp",
         "is_non_common_name": False, "primary_owner_cik": "2", "primary_owner_name": "B",
         "owner_relationship": "Director", "owner_title": "", "is_officer": False, "is_director": True,
         "is_c_suite": False, "num_co_filers": 1, "total_shares": 4000.0, "total_dollar_spend": 80_000.0,
         "weighted_avg_price": 20.0, "trans_row_count": 1},
    ])
    cfg = make_v9a_config()
    sigs, _ = detect_form4_clusters(acc, cfg)
    ev, audit = evaluate_clusters_and_survivorship(sigs, {"TERM": bar_early}, {}, spy, idx[-1].date(), cfg)
    assert audit["early_terminated_before_90d"] == 1
    assert ev[0].early_terminated_before_90d is True

    res_raw = run_form4_portfolio_backtest(
        ev, {"TERM": bar_early}, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0, survivorship_penalized=False
    )
    res_pen = run_form4_portfolio_backtest(
        ev, {"TERM": bar_early}, spy, cfg, idx[0].date(), idx[-1].date(), 5300.0, 5000.0, survivorship_penalized=True
    )
    assert res_raw.trades[0].exit_reason == "DELISTED_LAST_BAR"
    assert res_pen.trades[0].exit_reason == "DELISTED_PENALIZED_50"
    assert res_pen.trades[0].exit_raw_price == pytest.approx(10.0)

