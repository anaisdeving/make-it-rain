"""Official SEC DERA Financial Statement Data Sets (XBRL) Bulk Loader.

Downloads, caches, and parses quarterly SEC DERA Financial Statement Data Sets
(``2020q4`` through ``2026q3``) from:
  https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

Strict anti-bias & XBRL data-integrity guarantees:
- Dynamically resolves official ZIP URLs (handles path migrations cleanly),
  sends compliant ``SEC_USER_AGENT``, and enforces ``<= 5 req/sec`` rate limiting.
- Gracefully handles both 9-column (pre-2026 without ``segments``) and 10-column
  (with ``segments``) ``num.txt`` schemas.
- Extracts **strictly consolidated** facts (``coreg`` empty AND ``segments`` empty
  where present), strictly ``uom == "USD"`` for monetary tags and ``uom == "shares"``
  for share counts.
- Pulls current-year AND prior-year comparative ``qtrs == 4`` values directly from
  each ``10-K`` (falling back to the prior ``10-K`` for the same ``cik`` when absent)
  so YoY revenue growth and YoY share dilution remain split-consistent and
  strictly point-in-time as of ``sub.txt`` ``filed``.
- Extracts quarterly ``10-Q`` comparative ``qtrs == 1`` YoY revenue growth and
  YTD-annualized FCF / cash runway for causal thesis-break monitoring.
- Maps ``cik -> historical ticker`` via ``sub.txt`` ``instance`` prefix
  cross-checked against ``https://www.sec.gov/files/company_tickers.json``,
  logging exact resolution sources and unresolved mappings.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from src.data.sec_form4_loader import (
    MIN_SEC_REQUEST_INTERVAL_SEC,
    SEC_BASE_URL,
    get_sec_user_agent,
    normalize_cik,
    normalize_sec_ticker,
)

log = logging.getLogger(__name__)

SEC_XBRL_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets"
SEC_COMPANY_TICKERS_URL = f"{SEC_BASE_URL}/files/company_tickers.json"

DEFAULT_XBRL_QUARTERS: tuple[str, ...] = (
    "2020q4",
    "2021q1", "2021q2", "2021q3", "2021q4",
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
    "2026q1", "2026q2", "2026q3",
)

# Canonical US-GAAP / DEI tag fallback chains
REVENUE_TAGS: tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
GROSS_PROFIT_TAGS: tuple[str, ...] = ("GrossProfit",)
COGS_TAGS: tuple[str, ...] = ("CostOfGoodsAndServicesSold", "CostOfRevenue")
OCF_TAGS: tuple[str, ...] = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX_TAGS: tuple[str, ...] = ("PaymentsToAcquirePropertyPlantAndEquipment",)
CASH_TAGS: tuple[str, ...] = ("CashAndCashEquivalentsAtCarryingValue",)
STI_TAGS: tuple[str, ...] = ("ShortTermInvestments", "MarketableSecuritiesCurrent")
DEBT_NONCURR_TAGS: tuple[str, ...] = ("LongTermDebtNoncurrent",)
DEBT_CURR_TAGS: tuple[str, ...] = ("LongTermDebtCurrent", "DebtCurrent")
DEBT_FALLBACK_TAGS: tuple[str, ...] = ("LongTermDebt",)
EQUITY_TAGS: tuple[str, ...] = ("StockholdersEquity",)
SHARES_BS_TAGS: tuple[str, ...] = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
)
SHARES_DILUTED_TAGS: tuple[str, ...] = ("WeightedAverageNumberOfDilutedSharesOutstanding",)

ALL_TARGET_TAGS: frozenset[str] = frozenset(
    REVENUE_TAGS
    + GROSS_PROFIT_TAGS
    + COGS_TAGS
    + OCF_TAGS
    + CAPEX_TAGS
    + CASH_TAGS
    + STI_TAGS
    + DEBT_NONCURR_TAGS
    + DEBT_CURR_TAGS
    + DEBT_FALLBACK_TAGS
    + EQUITY_TAGS
    + SHARES_BS_TAGS
    + SHARES_DILUTED_TAGS
)

SHARES_TAGS_SET: frozenset[str] = frozenset(SHARES_BS_TAGS + SHARES_DILUTED_TAGS)

_INSTANCE_BAD_PREFIXES = frozenset({
    "FORM", "FY", "DOC", "XML", "HTM", "EX", "CK", "AR", "SR", "K10", "Q10",
    "F10K", "F10Q", "T10K", "T10Q", "DEF", "PRE", "FILE", "RPT", "SEC", "CORP",
    "INC", "LTD", "LLC", "NONE", "NULL", "TEST",
})


def extract_ticker_from_instance(instance: Any) -> str:
    """Extract historical ticker prefix from SEC ``sub.txt`` ``instance`` filename.

    Examples:
    - ``nvda-20240128_htm.xml`` -> ``NVDA``
    - ``brk.b-20231231_htm.xml`` -> ``BRK-B``
    - ``form10k_htm.xml`` / ``d25678d10k_htm.xml`` -> ``""``
    """
    if instance is None or (isinstance(instance, float) and np.isnan(instance)):
        return ""
    s = str(instance).strip()
    if not s:
        return ""
    # Primary standard iXBRL format: "{ticker}-{YYYYMMDD}..."
    m = re.match(r"^([a-z]{1,5}(?:[.-][a-z])?)-(?:20\d{6}|19\d{6})", s, re.IGNORECASE)
    if m:
        raw = m.group(1).upper()
        if raw not in _INSTANCE_BAD_PREFIXES:
            return normalize_sec_ticker(raw)
    # Secondary format: "{ticker}10-k2020..." or "{ticker}20201031..." (avoiding d12345d10k)
    m2 = re.match(r"^([a-z]{2,5})(?:10-?[kq]|20\d{2})", s, re.IGNORECASE)
    if m2:
        raw = m2.group(1).upper()
        if raw not in _INSTANCE_BAD_PREFIXES and not raw.startswith("FORM"):
            return normalize_sec_ticker(raw)
    return ""


def load_sec_company_tickers_map(
    data_root: Path = Path("data/sec_xbrl"),
    user_agent: str | None = None,
) -> dict[str, str]:
    """Fetch & cache official SEC ``company_tickers.json`` mapped as ``cik_10 -> ticker``."""
    dest = Path(data_root) / "company_tickers.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        payload = json.loads(dest.read_text())
    else:
        ua = user_agent or get_sec_user_agent()
        resp = requests.get(
            SEC_COMPANY_TICKERS_URL,
            headers={"User-Agent": ua},
            timeout=30,
        )
        resp.raise_for_status()
        dest.write_text(resp.text)
        payload = resp.json()

    out: dict[str, str] = {}
    items = payload.values() if isinstance(payload, dict) else payload
    for entry in items:
        if not isinstance(entry, dict):
            continue
        cik = normalize_cik(entry.get("cik_str"))
        ticker = normalize_sec_ticker(entry.get("ticker"))
        if cik and ticker and cik not in out:
            out[cik] = ticker
    return out


def resolve_cik_ticker(
    cik: str,
    instance: Any,
    company_tickers_by_cik: dict[str, str],
) -> tuple[str, str]:
    """Resolve ``(ticker, resolution_source)`` for an XBRL filing.

    Prefers active SEC ``company_tickers.json`` mapping when available (so ticker
    renames like FB->META or weird instance filenames like ``fico10-k2020_htm.xml``
    resolve cleanly to their active Yahoo Finance symbol), and falls back to the
    historical ``sub.txt`` ``instance`` prefix for delisted/acquired companies.
    """
    inst_t = extract_ticker_from_instance(instance)
    ct_t = company_tickers_by_cik.get(cik, "")
    if ct_t and inst_t:
        if ct_t == inst_t:
            return ct_t, "both_match"
        return ct_t, "company_tickers_override_instance"
    if ct_t:
        return ct_t, "company_tickers_only"
    if inst_t:
        return inst_t, "instance_only_delisted"
    return "", "unresolved"


def _parse_yyyymmdd(val: Any) -> pd.Timestamp:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    s = re.sub(r"\D", "", str(val).strip())[:8]
    if len(s) != 8:
        return pd.NaT
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def _select_fact_by_ddate(
    facts: list[tuple[pd.Timestamp, float]],
    target_ddate: pd.Timestamp,
    max_day_diff: int = 45,
) -> tuple[pd.Timestamp | None, float | None]:
    """Pick the fact whose ``ddate`` is closest to ``target_ddate`` within ``max_day_diff``."""
    best_dt: pd.Timestamp | None = None
    best_val: float | None = None
    best_diff = max_day_diff + 1
    for dt, val in facts:
        if pd.isna(dt) or math.isnan(val):
            continue
        diff = abs((dt - target_ddate).days)
        if diff < best_diff or (diff == best_diff and best_dt is not None and dt > best_dt):
            best_diff = diff
            best_dt = dt
            best_val = val
    return best_dt, best_val


def _select_prior_year_fact(
    facts: list[tuple[pd.Timestamp, float]],
    curr_ddate: pd.Timestamp,
    min_days: int = 300,
    max_days: int = 430,
) -> tuple[pd.Timestamp | None, float | None]:
    """Pick comparative prior-year fact (~365 calendar days prior to ``curr_ddate``)."""
    best_dt: pd.Timestamp | None = None
    best_val: float | None = None
    best_dist = 10_000
    for dt, val in facts:
        if pd.isna(dt) or math.isnan(val):
            continue
        delta = (curr_ddate - dt).days
        if min_days <= delta <= max_days:
            dist = abs(delta - 365)
            if dist < best_dist:
                best_dist = dist
                best_dt = dt
                best_val = val
    return best_dt, best_val


def _pick_chain_current_and_prior(
    tag_map: dict[tuple[str, int], list[tuple[pd.Timestamp, float]]],
    tags: tuple[str, ...],
    qtrs: int,
    period_ts: pd.Timestamp,
) -> tuple[str | None, pd.Timestamp | None, float | None, float | None]:
    """From a priority tag chain, pick ``(tag, curr_ddate, curr_val, prior_yr_val)``."""
    for tag in tags:
        entries = tag_map.get((tag, qtrs))
        if not entries:
            continue
        c_dt, c_val = _select_fact_by_ddate(entries, period_ts, max_day_diff=45)
        if c_dt is not None and c_val is not None:
            _, p_val = _select_prior_year_fact(entries, c_dt)
            return tag, c_dt, c_val, p_val
    return None, None, None, None


def _pick_chain_single(
    tag_map: dict[tuple[str, int], list[tuple[pd.Timestamp, float]]],
    tags: tuple[str, ...],
    qtrs: int,
    period_ts: pd.Timestamp,
    max_day_diff: int = 65,
) -> tuple[str | None, pd.Timestamp | None, float | None]:
    for tag in tags:
        entries = tag_map.get((tag, qtrs))
        if not entries:
            continue
        c_dt, c_val = _select_fact_by_ddate(entries, period_ts, max_day_diff=max_day_diff)
        if c_dt is not None and c_val is not None:
            return tag, c_dt, c_val
    return None, None, None


def _is_empty_str_series(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")


def parse_xbrl_quarter_frames(
    sub_df: pd.DataFrame,
    num_df: pd.DataFrame,
    company_tickers_by_cik: dict[str, str],
    quarter: str = "unknown",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract clean consolidated 10-K annual and 10-Q quarterly facts for one quarter.

    Handles both 9-column (no ``segments``) and 10-column (with ``segments``)
    ``num.txt`` schemas. Pure and directly unit-testable without network access.

    Returns ``(annual_10k_df, quarterly_10q_df, audit_stats)``.
    """
    audit: dict[str, Any] = {
        "quarter": quarter,
        "raw_sub_rows": int(len(sub_df)),
        "raw_num_rows_passed": int(len(num_df)),
        "has_segments_col": "segments" in num_df.columns,
        "ticker_resolution_counts": {},
    }

    sub = sub_df.copy()
    sub["adsh"] = sub["adsh"].astype(str).str.strip()
    sub["form_clean"] = sub["form"].fillna("").astype(str).str.strip().str.upper()
    sub = sub[sub["form_clean"].isin(["10-K", "10-Q"])].copy()
    sub = sub.drop_duplicates(subset=["adsh"], keep="first")

    sub["cik_10"] = sub["cik"].map(normalize_cik)
    sub["filed_dt"] = sub["filed"].map(_parse_yyyymmdd)
    sub["period_dt"] = sub["period"].map(_parse_yyyymmdd)
    sub["sic_int"] = pd.to_numeric(sub["sic"], errors="coerce").fillna(-1).astype(int)
    sub = sub[(sub["cik_10"] != "") & sub["filed_dt"].notna() & sub["period_dt"].notna()].copy()

    audit["sub_10k_count"] = int((sub["form_clean"] == "10-K").sum())
    audit["sub_10q_count"] = int((sub["form_clean"] == "10-Q").sum())

    valid_adsh = set(sub["adsh"])
    if not valid_adsh or num_df.empty:
        return pd.DataFrame(), pd.DataFrame(), audit

    # Filter num_df to strictly consolidated target facts (coreg empty, segments empty)
    num = num_df[num_df["adsh"].isin(valid_adsh) & num_df["tag"].isin(ALL_TARGET_TAGS)].copy()
    if num.empty:
        return pd.DataFrame(), pd.DataFrame(), audit

    coreg_empty = _is_empty_str_series(num["coreg"]) if "coreg" in num.columns else pd.Series(True, index=num.index)
    seg_empty = _is_empty_str_series(num["segments"]) if "segments" in num.columns else pd.Series(True, index=num.index)
    audit["excluded_coreg_rows"] = int((~coreg_empty).sum())
    audit["excluded_segments_rows"] = int((~seg_empty).sum())
    num = num[coreg_empty & seg_empty].copy()

    if num.empty:
        return pd.DataFrame(), pd.DataFrame(), audit

    # Enforce strict unit of measure: USD for monetary tags, shares for share tags
    uom_u = num["uom"].fillna("").astype(str).str.strip()
    is_share_tag = num["tag"].isin(SHARES_TAGS_SET)
    uom_ok = (is_share_tag & (uom_u.str.lower() == "shares")) | ((~is_share_tag) & (uom_u.str.upper() == "USD"))
    audit["excluded_non_usd_or_bad_uom_rows"] = int((~uom_ok).sum())
    num = num[uom_ok].copy()

    num["qtrs_int"] = pd.to_numeric(num["qtrs"], errors="coerce")
    num["val_num"] = pd.to_numeric(num["value"], errors="coerce")
    num["ddate_dt"] = num["ddate"].map(_parse_yyyymmdd)
    num = num[num["qtrs_int"].notna() & num["val_num"].notna() & num["ddate_dt"].notna()].copy()
    num["qtrs_int"] = num["qtrs_int"].astype(int)

    # Index facts by adsh -> (tag, qtrs) -> list[(ddate, val)]
    facts_by_adsh: dict[str, dict[tuple[str, int], list[tuple[pd.Timestamp, float]]]] = {}
    for adsh, tag, q_i, ddt, val in zip(
        num["adsh"], num["tag"], num["qtrs_int"], num["ddate_dt"], num["val_num"]
    ):
        d = facts_by_adsh.setdefault(adsh, {})
        d.setdefault((tag, q_i), []).append((ddt, float(val)))

    annual_rows: list[dict[str, Any]] = []
    quarterly_rows: list[dict[str, Any]] = []
    res_counts: dict[str, int] = {}

    for row in sub.itertuples(index=False):
        adsh = row.adsh
        tmap = facts_by_adsh.get(adsh)
        if not tmap:
            continue

        cik = row.cik_10
        ticker, res_src = resolve_cik_ticker(cik, getattr(row, "instance", None), company_tickers_by_cik)
        res_counts[res_src] = res_counts.get(res_src, 0) + 1

        period_ts = row.period_dt
        filed_ts = row.filed_dt
        form = row.form_clean
        name = str(getattr(row, "name", "") or "").strip()
        sic = int(row.sic_int)
        fp = str(getattr(row, "fp", "") or "").strip().upper()

        if form == "10-K":
            # 1. Revenue (qtrs == 4) + comparative prior-year from same 10-K
            rev_tag, rev_dt, rev_curr, rev_prior = _pick_chain_current_and_prior(
                tmap, REVENUE_TAGS, 4, period_ts
            )
            anchor_dt = rev_dt if rev_dt is not None else period_ts

            # 2. Gross Profit (direct GrossProfit, else Revenue - COGS)
            gp_tag, _, gp_val = _pick_chain_single(tmap, GROSS_PROFIT_TAGS, 4, anchor_dt)
            if gp_val is None and rev_curr is not None:
                cogs_tag, _, cogs_val = _pick_chain_single(tmap, COGS_TAGS, 4, anchor_dt)
                if cogs_val is not None:
                    gp_val = rev_curr - abs(cogs_val)
                    gp_tag = f"computed_from_{cogs_tag}"

            # 3. OCF & CapEx -> FCF (qtrs == 4)
            _, _, ocf_val = _pick_chain_single(tmap, OCF_TAGS, 4, anchor_dt)
            _, _, capex_val = _pick_chain_single(tmap, CAPEX_TAGS, 4, anchor_dt)
            capex_reported = capex_val is not None
            fcf_val = (ocf_val - abs(capex_val if capex_val is not None else 0.0)) if ocf_val is not None else None

            # 4. Cash + optional ShortTermInvestments / MarketableSecuritiesCurrent (qtrs == 0)
            _, _, cash_val = _pick_chain_single(tmap, CASH_TAGS, 0, anchor_dt)
            _, _, sti_val = _pick_chain_single(tmap, STI_TAGS, 0, anchor_dt)
            if cash_val is not None:
                cash_total = cash_val + (max(0.0, sti_val) if sti_val is not None else 0.0)
            elif sti_val is not None:
                cash_total = max(0.0, sti_val)
            else:
                cash_total = None

            # 5. Total Debt & Stockholders Equity (qtrs == 0)
            _, _, eq_val = _pick_chain_single(tmap, EQUITY_TAGS, 0, anchor_dt)
            _, _, debt_nc = _pick_chain_single(tmap, DEBT_NONCURR_TAGS, 0, anchor_dt)
            _, _, debt_c = _pick_chain_single(tmap, DEBT_CURR_TAGS, 0, anchor_dt)
            if debt_nc is not None or debt_c is not None:
                total_debt = max(0.0, debt_nc or 0.0) + max(0.0, debt_c or 0.0)
            else:
                _, _, debt_fb = _pick_chain_single(tmap, DEBT_FALLBACK_TAGS, 0, anchor_dt)
                total_debt = max(0.0, debt_fb) if debt_fb is not None else (0.0 if eq_val is not None else None)

            # 6. Shares Outstanding (point-in-time market cap) + same-tag comparative shares (YoY dilution)
            # EntityCommonStockSharesOutstanding can be dated at filing/cover date (allow up to 90d after period)
            sh_tag, sh_dt, sh_out = None, None, None
            ent_entries = tmap.get(("EntityCommonStockSharesOutstanding", 0))
            if ent_entries:
                # Latest positive cover-page share count in filing
                valid_ent = [(dt, v) for dt, v in ent_entries if v > 0 and not pd.isna(dt)]
                if valid_ent:
                    sh_dt, sh_out = max(valid_ent, key=lambda x: x[0])
                    sh_tag = "EntityCommonStockSharesOutstanding"

            _, cs_dt, cs_curr, cs_prior = _pick_chain_current_and_prior(
                tmap, ("CommonStockSharesOutstanding",), 0, anchor_dt
            )
            _, dil_dt, dil_curr, dil_prior = _pick_chain_current_and_prior(
                tmap, SHARES_DILUTED_TAGS, 4, anchor_dt
            )

            if sh_out is None or sh_out <= 0:
                if cs_curr is not None and cs_curr > 0:
                    sh_tag, sh_dt, sh_out = "CommonStockSharesOutstanding", cs_dt, cs_curr
                elif dil_curr is not None and dil_curr > 0:
                    sh_tag, sh_dt, sh_out = "WeightedAverageNumberOfDilutedSharesOutstanding", dil_dt, dil_curr

            # Split-consistent same-tag comparative share pair from the SAME 10-K
            sh_comp_tag, sh_comp_curr, sh_comp_prior = None, None, None
            if cs_curr is not None and cs_prior is not None and cs_curr > 0 and cs_prior > 0:
                sh_comp_tag, sh_comp_curr, sh_comp_prior = "CommonStockSharesOutstanding", cs_curr, cs_prior
            elif dil_curr is not None and dil_prior is not None and dil_curr > 0 and dil_prior > 0:
                sh_comp_tag, sh_comp_curr, sh_comp_prior = (
                    "WeightedAverageNumberOfDilutedSharesOutstanding",
                    dil_curr,
                    dil_prior,
                )

            annual_rows.append({
                "adsh": adsh,
                "quarter": quarter,
                "cik": cik,
                "ticker": ticker,
                "ticker_source": res_src,
                "name": name,
                "sic": sic,
                "form": "10-K",
                "filed_date": filed_ts,
                "period_date": anchor_dt,
                "rev_tag": rev_tag or "",
                "revenue": rev_curr,
                "revenue_prior_in_filing": rev_prior,
                "gp_tag": gp_tag or "",
                "gross_profit": gp_val,
                "ocf": ocf_val,
                "capex": abs(capex_val) if capex_val is not None else np.nan,
                "capex_reported": capex_reported,
                "fcf": fcf_val,
                "cash_total": cash_total,
                "total_debt": total_debt,
                "stockholders_equity": eq_val,
                "shares_tag": sh_tag or "",
                "shares_date": sh_dt if sh_dt is not None else pd.NaT,
                "shares_outstanding": sh_out,
                "shares_comp_tag": sh_comp_tag or "",
                "shares_comp_curr": sh_comp_curr,
                "shares_comp_prior_in_filing": sh_comp_prior,
            })

        else:  # 10-Q (quarterly thesis-break monitoring)
            rev_tag, rev_dt, rev_q_curr, rev_q_prior = _pick_chain_current_and_prior(
                tmap, REVENUE_TAGS, 1, period_ts
            )
            anchor_dt = rev_dt if rev_dt is not None else period_ts

            _, _, cash_val = _pick_chain_single(tmap, CASH_TAGS, 0, anchor_dt)
            _, _, sti_val = _pick_chain_single(tmap, STI_TAGS, 0, anchor_dt)
            cash_total = (
                cash_val + (max(0.0, sti_val) if sti_val is not None else 0.0)
                if cash_val is not None
                else (max(0.0, sti_val) if sti_val is not None else None)
            )

            # 10-Q cash flow is YTD (qtrs == 1 in Q1, 2 in Q2, 3 in Q3)
            preferred_q = {"Q1": (1,), "Q2": (2, 1), "Q3": (3, 1, 2)}.get(fp, (3, 2, 1))
            ytd_ocf, ytd_capex, ytd_qtrs = None, None, None
            for qk in preferred_q:
                _, _, o_v = _pick_chain_single(tmap, OCF_TAGS, qk, anchor_dt)
                if o_v is not None:
                    ytd_ocf = o_v
                    ytd_qtrs = qk
                    _, _, c_v = _pick_chain_single(tmap, CAPEX_TAGS, qk, anchor_dt)
                    ytd_capex = abs(c_v) if c_v is not None else 0.0
                    break

            ann_fcf = (
                (ytd_ocf - (ytd_capex or 0.0)) * (4.0 / float(ytd_qtrs))
                if (ytd_ocf is not None and ytd_qtrs)
                else None
            )
            q_yoy_rev = (
                (rev_q_curr / rev_q_prior - 1.0)
                if (rev_q_curr is not None and rev_q_prior is not None and rev_q_prior > 0)
                else None
            )
            runway_m = (
                (cash_total / abs(ann_fcf)) * 12.0
                if (ann_fcf is not None and ann_fcf < 0 and cash_total is not None)
                else (float("inf") if (ann_fcf is not None and ann_fcf >= 0) else None)
            )

            quarterly_rows.append({
                "adsh": adsh,
                "quarter": quarter,
                "cik": cik,
                "ticker": ticker,
                "name": name,
                "sic": sic,
                "form": "10-Q",
                "fp": fp,
                "filed_date": filed_ts,
                "period_date": anchor_dt,
                "rev_q_curr": rev_q_curr,
                "rev_q_prior": rev_q_prior,
                "yoy_rev_growth": q_yoy_rev,
                "cash_total": cash_total,
                "annualized_fcf": ann_fcf,
                "cash_runway_months": runway_m,
            })

    audit["ticker_resolution_counts"] = res_counts
    audit["extracted_10k_rows"] = len(annual_rows)
    audit["extracted_10q_rows"] = len(quarterly_rows)
    return pd.DataFrame(annual_rows), pd.DataFrame(quarterly_rows), audit


def enrich_annual_comparatives(annual_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing prior-year revenue/shares strictly causally from earlier 10-K of same ``cik``.

    Only uses earlier 10-K filings with ``prior.filed_date < curr.filed_date`` and
    ``300 <= (curr.period_date - prior.period_date).days <= 430``. Zero look-ahead!
    """
    if annual_df.empty:
        return annual_df.copy()

    df = annual_df.copy()
    df["filed_date"] = pd.to_datetime(df["filed_date"]).dt.normalize()
    df["period_date"] = pd.to_datetime(df["period_date"]).dt.normalize()
    df = (
        df.sort_values(["cik", "filed_date", "period_date", "adsh"])
        .drop_duplicates(subset=["adsh"], keep="first")
        .reset_index(drop=True)
    )

    rev_prior_final = df["revenue_prior_in_filing"].to_numpy(dtype=float, copy=True)
    sh_curr_final = df["shares_comp_curr"].to_numpy(dtype=float, copy=True)
    sh_prior_final = df["shares_comp_prior_in_filing"].to_numpy(dtype=float, copy=True)

    # Index by cik for causal prior-10-K fallback
    for _cik, idxs in df.groupby("cik", sort=False).groups.items():
        pos_list = list(idxs)
        if len(pos_list) < 2:
            continue
        for k in range(1, len(pos_list)):
            cur_i = pos_list[k]
            cur_filed = df.at[cur_i, "filed_date"]
            cur_period = df.at[cur_i, "period_date"]
            need_rev = math.isnan(rev_prior_final[cur_i])
            need_sh = math.isnan(sh_prior_final[cur_i])
            if not need_rev and not need_sh:
                continue

            # Search strictly earlier filings (filed_date < cur_filed) with period ~1y earlier
            best_prev = None
            best_dist = 10_000
            for j in range(k - 1, -1, -1):
                prev_i = pos_list[j]
                if df.at[prev_i, "filed_date"] >= cur_filed:
                    continue
                delta = (cur_period - df.at[prev_i, "period_date"]).days
                if 300 <= delta <= 430:
                    dist = abs(delta - 365)
                    if dist < best_dist:
                        best_dist = dist
                        best_prev = prev_i

            if best_prev is None:
                continue

            if need_rev:
                pv = df.at[best_prev, "revenue"]
                if pd.notna(pv) and float(pv) > 0:
                    rev_prior_final[cur_i] = float(pv)

            if need_sh:
                cur_sh = df.at[cur_i, "shares_outstanding"]
                prev_sh = df.at[best_prev, "shares_outstanding"]
                cur_tag = df.at[cur_i, "shares_tag"]
                prev_tag = df.at[best_prev, "shares_tag"]
                if pd.notna(cur_sh) and pd.notna(prev_sh) and float(cur_sh) > 0 and float(prev_sh) > 0:
                    # Prefer matching tag type; allow BS-to-BS (EntityCommon vs CommonStock)
                    same_family = (cur_tag == prev_tag) or (
                        cur_tag in SHARES_BS_TAGS and prev_tag in SHARES_BS_TAGS
                    )
                    if same_family:
                        sh_curr_final[cur_i] = float(cur_sh)
                        sh_prior_final[cur_i] = float(prev_sh)

    df["revenue_prior"] = rev_prior_final
    df["shares_dilution_curr"] = sh_curr_final
    df["shares_dilution_prior"] = sh_prior_final

    # Compute derived ratios cleanly
    df["yoy_rev_growth"] = np.where(
        (df["revenue"].notna()) & (df["revenue_prior"].notna()) & (df["revenue_prior"] > 0),
        df["revenue"] / df["revenue_prior"] - 1.0,
        np.nan,
    )
    df["gross_margin"] = np.where(
        (df["gross_profit"].notna()) & (df["revenue"].notna()) & (df["revenue"] > 0),
        df["gross_profit"] / df["revenue"],
        np.nan,
    )
    df["fcf_margin"] = np.where(
        (df["fcf"].notna()) & (df["revenue"].notna()) & (df["revenue"] > 0),
        df["fcf"] / df["revenue"],
        np.nan,
    )
    df["yoy_share_dilution"] = np.where(
        (df["shares_dilution_curr"].notna())
        & (df["shares_dilution_prior"].notna())
        & (df["shares_dilution_prior"] > 0),
        df["shares_dilution_curr"] / df["shares_dilution_prior"] - 1.0,
        np.nan,
    )
    df["debt_to_equity"] = np.where(
        (df["total_debt"].notna())
        & (df["stockholders_equity"].notna())
        & (df["stockholders_equity"] > 0),
        df["total_debt"] / df["stockholders_equity"],
        np.nan,
    )
    df["cash_runway_years"] = np.where(
        df["fcf"].notna() & (df["fcf"] > 0),
        np.inf,
        np.where(
            df["fcf"].notna() & (df["fcf"] < 0) & df["cash_total"].notna(),
            df["cash_total"] / df["fcf"].abs(),
            np.nan,
        ),
    )
    df["rule_of_40"] = (df["yoy_rev_growth"] + df["fcf_margin"]) * 100.0
    return df


class SecXbrlLoader:
    """Downloads, caches, and parses official SEC DERA Financial Statement Data Sets."""

    def __init__(
        self,
        data_root: Path = Path("data/sec_xbrl"),
        user_agent: str | None = None,
        min_request_interval_sec: float = MIN_SEC_REQUEST_INTERVAL_SEC,
    ) -> None:
        self.data_root = Path(data_root)
        self.zips_dir = self.data_root / "zips"
        self.parsed_dir = self.data_root / "parsed"
        self.zips_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or get_sec_user_agent()
        self.min_interval = max(0.20, float(min_request_interval_sec))
        self._last_req_ts = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    def _rate_limited_get(self, url: str, timeout: int = 180, stream: bool = False) -> requests.Response:
        for attempt in range(5):
            elapsed = time.monotonic() - self._last_req_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_req_ts = time.monotonic()
            resp = self._session.get(url, timeout=timeout, stream=stream)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** (attempt + 1), 30)
                log.warning("SEC XBRL HTTP %d for %s; retrying in %ds", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            return resp
        return resp  # type: ignore[possibly-undefined]

    def discover_quarter_links(self) -> dict[str, str]:
        resp = self._rate_limited_get(SEC_XBRL_INDEX_URL, timeout=45)
        resp.raise_for_status()
        links: dict[str, str] = {}
        for m in re.finditer(r'href=["\']([^"\']*?(\d{4}q[1-4])\.zip)["\']', resp.text, re.I):
            href, q = m.group(1), m.group(2).lower()
            links[q] = urljoin(SEC_BASE_URL, href)
        return links

    def download_quarter_zip(self, quarter: str, url: str) -> Path | None:
        q = quarter.lower()
        dest = self.zips_dir / f"{q}.zip"
        if dest.exists() and dest.stat().st_size > 100_000 and zipfile.is_zipfile(dest):
            return dest
        log.info("Downloading SEC XBRL %s from %s", q, url)
        resp = self._rate_limited_get(url, timeout=240, stream=True)
        if resp.status_code != 200:
            log.warning("SEC XBRL %s unavailable (HTTP %d): %s", q, resp.status_code, url)
            return None
        tmp = dest.with_suffix(".zip.tmp")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            log.warning("SEC XBRL %s returned non-zip payload", q)
            return None
        tmp.replace(dest)
        return dest

    def parse_zip_file(
        self,
        zip_path: Path,
        quarter: str,
        company_tickers_by_cik: dict[str, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        with zipfile.ZipFile(zip_path) as zf:
            names = {Path(n).name.lower(): n for n in zf.namelist()}
            with zf.open(names["sub.txt"]) as f:
                sub_df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
            valid_adsh = set(
                sub_df.loc[
                    sub_df["form"].fillna("").str.strip().str.upper().isin(["10-K", "10-Q"]),
                    "adsh",
                ].astype(str).str.strip()
            )
            chunks: list[pd.DataFrame] = []
            with zf.open(names["num.txt"]) as f:
                for ch in pd.read_csv(f, sep="\t", dtype=str, chunksize=500_000, low_memory=False):
                    m = ch["adsh"].isin(valid_adsh) & ch["tag"].isin(ALL_TARGET_TAGS)
                    if m.any():
                        chunks.append(ch.loc[m])
            num_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
                columns=["adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "value"]
            )
        return parse_xbrl_quarter_frames(sub_df, num_df, company_tickers_by_cik, quarter=quarter)

    def load_quarters(
        self,
        quarters: tuple[str, ...] = DEFAULT_XBRL_QUARTERS,
        force_reparse: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Load & cache all requested SEC XBRL quarters into enriched 10-K and 10-Q tables."""
        company_tickers = load_sec_company_tickers_map(self.data_root, self.user_agent)
        discovered = self.discover_quarter_links()

        annual_frames: list[pd.DataFrame] = []
        quarterly_frames: list[pd.DataFrame] = []
        quarter_audits: list[dict[str, Any]] = []
        downloaded_quarters: list[str] = []
        unavailable_quarters: list[str] = []

        for q in quarters:
            ql = q.lower()
            p_10k = self.parsed_dir / f"{ql}_10k.parquet"
            p_10q = self.parsed_dir / f"{ql}_10q.parquet"
            p_aud = self.parsed_dir / f"{ql}_audit.json"
            url = discovered.get(ql)

            if not force_reparse and p_10k.exists() and p_10q.exists() and p_aud.exists():
                df_k = pd.read_parquet(p_10k)
                df_q = pd.read_parquet(p_10q)
                aud_q = json.loads(p_aud.read_text())
                if not df_k.empty:
                    annual_frames.append(df_k)
                if not df_q.empty:
                    quarterly_frames.append(df_q)
                quarter_audits.append(aud_q)
                downloaded_quarters.append(ql)
                continue

            if url is None:
                fallback_urls = [
                    f"{SEC_BASE_URL}/files/dera/data/financial-statement-data-sets/{ql}.zip",
                    f"{SEC_BASE_URL}/files/datastandardsinnovation/data/financial-statement-data-sets/{ql}.zip",
                ]
                zip_path = None
                for fu in fallback_urls:
                    zip_path = self.download_quarter_zip(ql, fu)
                    if zip_path is not None:
                        url = fu
                        break
            else:
                zip_path = self.download_quarter_zip(ql, url)

            if zip_path is None:
                unavailable_quarters.append(ql)
                continue

            df_k, df_q, aud_q = self.parse_zip_file(zip_path, ql, company_tickers)
            aud_q["source_url"] = url
            aud_q["zip_bytes"] = int(zip_path.stat().st_size)
            df_k.to_parquet(p_10k, index=False)
            df_q.to_parquet(p_10q, index=False)
            p_aud.write_text(json.dumps(aud_q, indent=2))
            if not df_k.empty:
                annual_frames.append(df_k)
            if not df_q.empty:
                quarterly_frames.append(df_q)
            quarter_audits.append(aud_q)
            downloaded_quarters.append(ql)

        annual_raw = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
        annual_df = enrich_annual_comparatives(annual_raw)

        quarterly_df = pd.concat(quarterly_frames, ignore_index=True) if quarterly_frames else pd.DataFrame()
        if not quarterly_df.empty:
            quarterly_df["filed_date"] = pd.to_datetime(quarterly_df["filed_date"]).dt.normalize()
            quarterly_df["period_date"] = pd.to_datetime(quarterly_df["period_date"]).dt.normalize()
            quarterly_df = (
                quarterly_df.sort_values(["cik", "filed_date", "period_date", "adsh"])
                .drop_duplicates(subset=["adsh"], keep="first")
                .reset_index(drop=True)
            )

        # Aggregate resolution & schema audit
        res_totals: dict[str, int] = {}
        for a in quarter_audits:
            for k, v in a.get("ticker_resolution_counts", {}).items():
                res_totals[k] = res_totals.get(k, 0) + int(v)

        manifest = {
            "requested_quarters": list(quarters),
            "downloaded_quarters": downloaded_quarters,
            "unavailable_quarters": unavailable_quarters,
            "company_tickers_count": len(company_tickers),
            "total_10k_rows": int(len(annual_df)),
            "total_10q_rows": int(len(quarterly_df)),
            "filed_date_min_10k": str(annual_df["filed_date"].min().date()) if not annual_df.empty else None,
            "filed_date_max_10k": str(annual_df["filed_date"].max().date()) if not annual_df.empty else None,
            "quarters_with_segments_col": [a["quarter"] for a in quarter_audits if a.get("has_segments_col")],
            "ticker_resolution_totals": res_totals,
            "unresolved_filings_count": int(res_totals.get("unresolved", 0)),
            "per_quarter_audit": quarter_audits,
        }
        (self.data_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return annual_df, quarterly_df, manifest
