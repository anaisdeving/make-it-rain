"""Official SEC DERA Form 3/4/5 Bulk Loader & yfinance Market-Bar Batch Cacher.

Downloads and parses official quarterly SEC DERA Insider Transactions datasets
(``2021q4`` through ``2026q3``) from:
  https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

Critical anti-bias & data-integrity guarantees:
- Compliant SEC ``User-Agent`` header and strict rate limit (``<= 5 req/sec``).
- Keeps strictly ``DOCUMENT_TYPE == "4"`` (excludes Form 3/5 and ``4/A`` amendments).
- Excludes Rule 10b5-1 pre-scheduled trades wherever ``AFF10B5ONE`` is present and
  truthy (gracefully handles pre-2023 quarters before ``AFF10B5ONE`` existed).
- Strictly open-market purchases: ``TRANS_CODE == "P"`` and
  ``TRANS_ACQUIRED_DISP_CD == "A"``, positive ``TRANS_SHARES`` and
  ``TRANS_PRICEPERSHARE``, non-debt/preferred/warrant ``SECURITY_TITLE``, and
  explicitly logged outlier guarding.
- **Zero Multi-Owner Join Multiplication:** ``NONDERIV_TRANS.tsv`` is aggregated to
  1 row per ``ACCESSION_NUMBER`` BEFORE joining ``REPORTINGOWNER.tsv`` (also collapsed
  to 1 canonical insider record per ``ACCESSION_NUMBER``). A joint filing by an
  insider + family trust/LLC can NEVER inflate dollar spend or count as 2 independent
  cluster insiders.
- Excludes pure passive/institutional 10% owners unless at least one co-filer on the
  accession is an Officer or Director.
- Preserves ``FILING_DATE`` as the sole causal signal timestamp (``TRANS_DATE`` is
  retained strictly for audit logging and NEVER used for entry timing).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from src.data.bar_utils import normalise_ohlcv
from src.data.cache import DiskCache
from src.domain.models import BarSeries

log = logging.getLogger(__name__)

SEC_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
SEC_BASE_URL = "https://www.sec.gov"
DEFAULT_SEC_USER_AGENT = "make-it-rain-research/1.0 (contact@make-it-rain-research.local)"
MIN_SEC_REQUEST_INTERVAL_SEC = 0.22  # <= 4.5 req/sec (strictly <= 5 req/sec)

DEFAULT_QUARTERS: tuple[str, ...] = (
    "2021q4",
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
    "2026q1", "2026q2", "2026q3",
)

# Outlier caps on NONDERIV_TRANS open-market purchase rows (logged explicitly)
MAX_VALID_SHARE_PRICE_USD = 10_000.0
MAX_VALID_ROW_VALUE_USD = 250_000_000.0

_TRUTHY_10B51 = frozenset({"1", "true", "y", "yes", "t"})

_INVALID_TICKERS = frozenset({
    "", "NONE", "N/A", "NA", "NULL", "NIL", "N.A.", "N.A", "TBD", "PENDING",
    "PRIVATE", "UNLISTED", "NOTAPPLICABLE", "NOT APPLICABLE", "NON", "NO",
    "UNKNOWN", "SEE REMARKS", "SEEREMARKS", "N/A.", "NONE.",
})

_NON_US_EXCHANGE_PREFIXES = (
    "ASX:", "TSX:", "TSXV:", "TSX-V:", "LSE:", "LON:", "TSE:", "HKEX:",
    "EPA:", "FRA:", "SIX:", "CSE:", "CNSX:", "NEO:", "AIM:", "SGX:", "KRX:",
    "ASX/", "TSX/", "LSE/", "HKG:", "TYO:", "BIT:", "AMS:", "SWX:",
)

_US_EXCHANGE_PREFIX_RE = re.compile(
    r"^(?:NYSE\s+ARCA|NYSE\s+AMERICAN|NYSE\s+MKT|NYSE|NASDAQ\s+GM|NASDAQ\s+CM|"
    r"NASDAQ\s+GS|NASDAQ|AMEX|BATS|CBOE|OTCQX|OTCQB|OTC\s+PINK|OTC)\s*[:/\-]\s*",
    re.IGNORECASE,
)

_US_EXCHANGE_SUFFIX_RE = re.compile(
    r"\s*[-/\s](?:NYSE|NASDAQ|AMEX|OTC|OTCQX|OTCQB)$",
    re.IGNORECASE,
)

_NON_COMMON_SECURITY_RE = re.compile(
    r"\b(?:senior\s+notes?|unsecured\s+notes?|convertible\s+notes?|notes?\s+due|"
    r"promissory\s+notes?|subordinated\s+notes?|debentures?|bonds?|"
    r"preferred\s+stock|preferred\s+shares?|preferred\s+units?|series\s+[a-z0-9]+\s+preferred|"
    r"warrants?|subscription\s+rights?|depositary\s+receipts?\s+.*preferred)\b",
    re.IGNORECASE,
)

_NON_COMMON_ISSUER_NAME_RE = re.compile(
    r"\b(?:"
    r"closed[- ]end|etf\b|exchange[- ]traded|mutual\s+fund|"
    r"income\s+fund|bond\s+fund|municipal\s+.*(?:fund|trust)|"
    r"dividend\s+.*(?:fund|trust)|opportunity\s+fund|total\s+return\s+fund|"
    r"allocation\s+fund|equity\s+fund|credit\s+fund|strategy\s+fund|"
    r"growth\s+fund|value\s+fund|select\s+fund|index\s+fund|"
    r"enhanced\s+.*fund|high[- ]yield\s+.*fund|floating\s+rate\s+.*fund|"
    r"tax[- ]advantaged|term\s+trust|income\s+trust|royalty\s+trust|"
    r"balanced\s+fund|multi[- ]sector\s+.*(?:fund|trust)|"
    r"investment\s+grade\s+.*(?:fund|trust)|"
    r"nuveen\b|pimco\s+.*(?:fund|income|dynamic|municipal|corporate)|"
    r"blackrock\s+.*(?:fund|trust|income|muniyield|munivest|munienhanced|munihldgs)|"
    r"eaton\s+vance\s+.*(?:fund|trust)|gabelli\s+.*(?:fund|trust|multimedia|utility|dividend)|"
    r"cohen\s*&\s*steers\s+.*(?:fund|reit|infrastructure|quality|select|total)|"
    r"john\s+hancock\s+.*(?:fund|trust)|invesco\s+.*(?:fund|trust|municipal|bond|senior)|"
    r"first\s+trust\s+.*(?:fund|etf|income|specialty|energy|mlp)|"
    r"kayne\s+anderson\s+.*(?:fund|energy|bdc|nextgen)|"
    r"tortoise\s+.*(?:fund|energy|midstream|pipeline)|"
    r"calamos\s+.*(?:fund|trust)|royce\s+.*(?:fund|trust)|"
    r"flaherty\s*&\s*crumrine|guggenheim\s+.*(?:fund|trust)|"
    r"neuberger\s+berman\s+.*(?:fund|trust)|virtus\s+.*(?:fund|trust)|"
    r"western\s+asset\s+.*(?:fund|trust)|abrdn\s+.*(?:fund|trust|income)|"
    r"templeton\s+.*(?:fund|income)|mfs\s+.*(?:fund|trust|charter|intermediate|multimarket)|"
    r"putnam\s+.*(?:fund|trust)|dws\s+.*(?:fund|trust)|allspring\s+.*(?:fund|trust)|"
    r"barings\s+.*(?:fund|trust|participation|corporate\s+investors)|"
    r"liberty\s+all[- ]star|adams\s+diversified|adams\s+natural\s+resources|"
    r"general\s+american\s+investors|central\s+securities\s+corp|"
    r"source\s+capital|tri[- ]continental|saba\s+capital\s+income|"
    r"special\s+purpose\s+acquisition|blank\s+check|"
    r"acquisition\s+corp\.?|acquisition\s+co\.?\b|acquisition\s+holdings|merger\s+corp\.?"
    r")\b",
    re.IGNORECASE,
)

_CSUITE_CEO_CFO_RE = re.compile(
    r"(?:"
    r"\b(?:co[- ]?)?c\.?e\.?o\.?\b|"
    r"\b(?:co[- ]?)?chief\s+executive\s+officer\b|"
    r"\b(?:co[- ]?)?c\.?f\.?o\.?\b|"
    r"\b(?:co[- ]?)?chief\s+financial\s+officer\b|"
    r"\bprincipal\s+(?:executive|financial)\s+officer\b"
    r")",
    re.IGNORECASE,
)

_CSUITE_PRESIDENT_RE = re.compile(
    r"(?<!vice\s)(?<!vice-)(?<!v\.p\.\s)(?<!vp\s)\b(?:co[- ]?)?president\b",
    re.IGNORECASE,
)

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:-[A-Z])?$")

_DISALLOWED_QUOTE_TYPES = frozenset({
    "ETF", "MUTUALFUND", "CLOSED_END_FUND", "INDEX", "CURRENCY",
    "CRYPTOCURRENCY", "FUTURE", "OPTION", "MONEYMARKET",
})

_US_MAJOR_EXCHANGES = frozenset({
    "NMS", "NGM", "NCM", "NYQ", "ASE", "BTS", "PCX", "NYS", "NAS", "YHD", "BATS",
    "NASDAQ", "NYSE", "CBOE", "NYSE AMERICAN", "NYSE ARCA", "AMEX", "TXSE",
})


def get_sec_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT).strip() or DEFAULT_SEC_USER_AGENT


def normalize_cik(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = re.sub(r"\D", "", str(raw).strip())
    return s.zfill(10) if s else ""


def normalize_sec_ticker(raw: Any) -> str:
    """Normalize SEC ``ISSUERTRADINGSYMBOL`` cleanly for US equity lookup.

    Strips exchange prefixes (``NYSE:``, ``NASDAQ:``), outer quotes/brackets,
    handles multi-ticker lists (taking primary ticker), converts single-letter
    share-class dot notation (``BRK.B`` -> ``BRK-B``), and returns ``""`` for
    empty, sentinel (``NONE``, ``N/A``), numeric, non-US, or warrant/preferred strings.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Strip outer wrappers/backslashes/quotes/brackets repeatedly
    s = s.strip(" \t\r\n[](){}\"'\\")
    if not s:
        return ""
    u = re.sub(r"\s+", " ", s).upper()
    if u in _INVALID_TICKERS:
        return ""
    for prefix in _NON_US_EXCHANGE_PREFIXES:
        if u.startswith(prefix):
            return ""
    u = _US_EXCHANGE_PREFIX_RE.sub("", u)
    u = _US_EXCHANGE_SUFFIX_RE.sub("", u)
    u = u.strip(" \t\r\n[](){}\"'\\")
    if not u or u in _INVALID_TICKERS:
        return ""

    # Reject explicit OTC pink/bulletin-board or preferred/warrant suffixes
    if re.search(r"(?:\.(?:PK|OB|WS|WT|U|R)|-PREF|-P[A-Z])$", u):
        return ""

    # Multi-ticker strings such as "WLY, WLYB", "HEI, HEI.A", "SMX; SMXWW", "Z AND ZG"
    if any(sep in u for sep in (",", ";", "/", " AND ", " & ")):
        tokens = re.split(r"[,;/]|\s+AND\s+|\s*&\s*", u)
        for tok in tokens:
            cand = normalize_sec_ticker(tok)
            if cand and not (len(cand) == 5 and cand.endswith(("W", "R", "U"))):
                return cand
        return ""

    # Space-separated pair of short tickers such as "BIO BIO.B" or "CRDA CRDB"
    if " " in u:
        parts = u.split()
        if len(parts) == 2 and all(1 <= len(p) <= 6 for p in parts):
            c0 = normalize_sec_ticker(parts[0])
            c1 = normalize_sec_ticker(parts[1])
            if c0 and c1 and c1.startswith(c0[:2]):
                return c0
        return ""

    # Convert share-class dot notation (e.g. BRK.B -> BRK-B, NYT.A -> NYT-A)
    if re.match(r"^[A-Z]{1,4}\.[A-Z]$", u):
        u = u.replace(".", "-")

    if not _VALID_TICKER_RE.match(u):
        return ""
    if u in _INVALID_TICKERS:
        return ""
    return u


def is_c_suite_title(title: str | None) -> bool:
    """Return True if ``RPTOWNER_TITLE`` indicates CEO, CFO, or President (not VP)."""
    if not title or (isinstance(title, float) and np.isnan(title)):
        return False
    norm = re.sub(r"\s+", " ", str(title).strip())
    if not norm:
        return False
    if _CSUITE_CEO_CFO_RE.search(norm):
        return True
    if _CSUITE_PRESIDENT_RE.search(norm):
        return True
    return False


def is_excluded_non_common_issuer(issuer_name: str | None, ticker: str | None = None) -> bool:
    """Return True if SEC issuer name / ticker indicates an ETF, CEF, fund, or SPAC."""
    if issuer_name and not (isinstance(issuer_name, float) and np.isnan(issuer_name)):
        if _NON_COMMON_ISSUER_NAME_RE.search(str(issuer_name)):
            return True
    if ticker:
        t = str(ticker).upper().strip()
        # 5-letter Nasdaq suffixes: W=warrant, R=right, U=unit
        if len(t) == 5 and t[-1] in {"W", "R", "U"}:
            return True
    return False


def _parse_sec_dates(series: pd.Series) -> pd.Series:
    """Parse SEC DERA date strings (``DD-MON-YYYY`` or ISO ``YYYY-MM-DD``)."""
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    missing = dt.isna() & s.notna() & (s != "") & (s.str.lower() != "nan")
    if missing.any():
        dt.loc[missing] = pd.to_datetime(s.loc[missing], errors="coerce")
    return dt.dt.normalize()


def parse_quarter_frames(
    sub_df: pd.DataFrame,
    owner_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    quarter: str = "unknown",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean, filter, and join one quarter's SEC DERA Form 3/4/5 TSV frames.

    Returns ``(accessions_df, audit_stats)`` with strictly ONE row per qualifying
    open-market Form 4 purchase accession.
    """
    audit: dict[str, Any] = {
        "quarter": quarter,
        "raw_submissions": int(len(sub_df)),
        "raw_owners": int(len(owner_df)),
        "raw_nonderiv_trans": int(len(trans_df)),
        "has_aff10b5one_col": "AFF10B5ONE" in sub_df.columns,
    }

    # --- 1. SUBMISSION.tsv ----------------------------------------------------
    sub = sub_df.copy()
    sub["ACCESSION_NUMBER"] = sub["ACCESSION_NUMBER"].astype(str).str.strip()
    sub["DOCUMENT_TYPE"] = sub["DOCUMENT_TYPE"].astype(str).str.strip()
    audit["doc_type_non_4_excluded"] = int((sub["DOCUMENT_TYPE"] != "4").sum())
    sub = sub[sub["DOCUMENT_TYPE"] == "4"].copy()
    sub = sub.drop_duplicates(subset=["ACCESSION_NUMBER"], keep="first")

    if "AFF10B5ONE" in sub.columns:
        aff_flag = sub["AFF10B5ONE"].fillna("").astype(str).str.strip().str.lower().isin(_TRUTHY_10B51)
        audit["aff10b5one_excluded"] = int(aff_flag.sum())
        sub = sub[~aff_flag].copy()
    else:
        audit["aff10b5one_excluded"] = 0

    sub["filing_date"] = _parse_sec_dates(sub["FILING_DATE"])
    invalid_fdate = sub["filing_date"].isna()
    audit["invalid_filing_date_excluded"] = int(invalid_fdate.sum())
    sub = sub[~invalid_fdate].copy()

    sub["issuer_cik"] = sub["ISSUERCIK"].map(normalize_cik)
    sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].map(normalize_sec_ticker)
    sub["issuer_name"] = sub["ISSUERNAME"].fillna("").astype(str).str.strip()
    valid_issuer = (sub["issuer_cik"] != "") & (sub["ticker"] != "")
    audit["invalid_ticker_or_cik_excluded"] = int((~valid_issuer).sum())
    sub = sub[valid_issuer].copy()

    sub["is_non_common_name"] = [
        is_excluded_non_common_issuer(nm, tk)
        for nm, tk in zip(sub["issuer_name"], sub["ticker"])
    ]
    valid_acc_sub = set(sub["ACCESSION_NUMBER"])
    audit["submissions_after_filter"] = int(len(sub))

    # --- 2. NONDERIV_TRANS.tsv (aggregated 1 row per ACCESSION_NUMBER) -------
    trx = trans_df.copy()
    trx["ACCESSION_NUMBER"] = trx["ACCESSION_NUMBER"].astype(str).str.strip()
    trx = trx[trx["ACCESSION_NUMBER"].isin(valid_acc_sub)].copy()

    t_code = trx["TRANS_CODE"].fillna("").astype(str).str.strip().str.upper()
    t_ad = trx["TRANS_ACQUIRED_DISP_CD"].fillna("").astype(str).str.strip().str.upper()
    trx = trx[(t_code == "P") & (t_ad == "A")].copy()
    audit["p_and_a_rows"] = int(len(trx))

    if "SECURITY_TITLE" in trx.columns and not trx.empty:
        non_common_sec = trx["SECURITY_TITLE"].fillna("").astype(str).str.contains(
            _NON_COMMON_SECURITY_RE, regex=True
        )
        audit["non_common_security_rows_excluded"] = int(non_common_sec.sum())
        trx = trx[~non_common_sec].copy()
    else:
        audit["non_common_security_rows_excluded"] = 0

    trx["shares"] = pd.to_numeric(trx["TRANS_SHARES"], errors="coerce")
    trx["price"] = pd.to_numeric(trx["TRANS_PRICEPERSHARE"], errors="coerce")
    pos_mask = (trx["shares"] > 0) & (trx["price"] > 0)
    audit["non_positive_shares_or_price_excluded"] = int((~pos_mask).sum())
    trx = trx[pos_mask].copy()

    trx["row_value"] = trx["shares"] * trx["price"]
    outlier_mask = (trx["price"] > MAX_VALID_SHARE_PRICE_USD) | (trx["row_value"] > MAX_VALID_ROW_VALUE_USD)
    audit["outlier_price_or_value_rows_excluded"] = int(outlier_mask.sum())
    trx = trx[~outlier_mask].copy()

    if trx.empty:
        empty_cols = [
            "accession_number", "quarter", "filing_date", "earliest_trans_date",
            "latest_trans_date", "issuer_cik", "ticker", "issuer_name",
            "is_non_common_name", "primary_owner_cik", "primary_owner_name",
            "owner_relationship", "owner_title", "is_officer", "is_director",
            "is_c_suite", "num_co_filers", "total_shares", "total_dollar_spend",
            "weighted_avg_price", "trans_row_count",
        ]
        audit["qualifying_accessions"] = 0
        audit["multi_owner_accessions_collapsed"] = 0
        audit["pure_ten_percent_accessions_excluded"] = 0
        return pd.DataFrame(columns=empty_cols), audit

    trx["trans_date"] = _parse_sec_dates(trx["TRANS_DATE"])
    trx_agg = (
        trx.groupby("ACCESSION_NUMBER", as_index=False)
        .agg(
            total_shares=("shares", "sum"),
            total_dollar_spend=("row_value", "sum"),
            trans_row_count=("row_value", "size"),
            earliest_trans_date=("trans_date", "min"),
            latest_trans_date=("trans_date", "max"),
        )
    )
    trx_agg["weighted_avg_price"] = trx_agg["total_dollar_spend"] / trx_agg["total_shares"]
    valid_acc_trx = set(trx_agg["ACCESSION_NUMBER"])

    # --- 3. REPORTINGOWNER.tsv (collapsed 1 canonical insider per accession) --
    own = owner_df.copy()
    own["ACCESSION_NUMBER"] = own["ACCESSION_NUMBER"].astype(str).str.strip()
    own = own[own["ACCESSION_NUMBER"].isin(valid_acc_trx)].copy()

    own["owner_cik"] = own["RPTOWNERCIK"].map(normalize_cik)
    own["owner_name"] = own["RPTOWNERNAME"].fillna("").astype(str).str.strip()
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").astype(str)
    rel_l = rel.str.lower()
    own["is_officer"] = rel_l.str.contains("officer", regex=False)
    own["is_director"] = rel_l.str.contains("director", regex=False)
    own["is_officer_or_director"] = own["is_officer"] | own["is_director"]
    own["owner_title"] = own["RPTOWNER_TITLE"].fillna("").astype(str).str.strip()
    own["owner_rel"] = rel.str.strip()

    # Deterministic priority per accession: Officer first, then Director, then lowest CIK
    own["_sort_off"] = (~own["is_officer"]).astype(int)
    own["_sort_dir"] = (~own["is_director"]).astype(int)
    own = own.sort_values(
        ["ACCESSION_NUMBER", "_sort_off", "_sort_dir", "owner_cik", "owner_name"],
        kind="mergesort",
    )

    owner_rows: list[dict[str, Any]] = []
    multi_owner_collapsed = 0
    pure_ten_pct_excluded = 0

    for acc, grp in own.groupby("ACCESSION_NUMBER", sort=False):
        distinct_ciks = [c for c in grp["owner_cik"].unique() if c]
        n_co = max(1, len(distinct_ciks))
        od_grp = grp[grp["is_officer_or_director"]]
        if od_grp.empty:
            pure_ten_pct_excluded += 1
            continue
        if n_co > 1:
            multi_owner_collapsed += 1
        primary = od_grp.iloc[0]
        titles = sorted({t for t in od_grp["owner_title"] if t and t.lower() != "nan"})
        combined_title = " | ".join(titles)
        rels = sorted({r for r in od_grp["owner_rel"] if r and r.lower() != "nan"})
        combined_rel = " | ".join(rels)
        owner_rows.append({
            "ACCESSION_NUMBER": acc,
            "primary_owner_cik": primary["owner_cik"],
            "primary_owner_name": primary["owner_name"],
            "owner_relationship": combined_rel,
            "owner_title": combined_title,
            "is_officer": bool(od_grp["is_officer"].any()),
            "is_director": bool(od_grp["is_director"].any()),
            "is_c_suite": is_c_suite_title(combined_title),
            "num_co_filers": int(n_co),
        })

    audit["multi_owner_accessions_collapsed"] = int(multi_owner_collapsed)
    audit["pure_ten_percent_accessions_excluded"] = int(pure_ten_pct_excluded)

    if not owner_rows:
        empty_cols = [
            "accession_number", "quarter", "filing_date", "earliest_trans_date",
            "latest_trans_date", "issuer_cik", "ticker", "issuer_name",
            "is_non_common_name", "primary_owner_cik", "primary_owner_name",
            "owner_relationship", "owner_title", "is_officer", "is_director",
            "is_c_suite", "num_co_filers", "total_shares", "total_dollar_spend",
            "weighted_avg_price", "trans_row_count",
        ]
        audit["qualifying_accessions"] = 0
        return pd.DataFrame(columns=empty_cols), audit

    own_agg = pd.DataFrame(owner_rows)

    # --- 4. Strict 1:1:1 Join on ACCESSION_NUMBER -----------------------------
    merged = (
        sub[["ACCESSION_NUMBER", "filing_date", "issuer_cik", "ticker", "issuer_name", "is_non_common_name"]]
        .merge(trx_agg, on="ACCESSION_NUMBER", how="inner", validate="one_to_one")
        .merge(own_agg, on="ACCESSION_NUMBER", how="inner", validate="one_to_one")
    )
    merged = merged.rename(columns={"ACCESSION_NUMBER": "accession_number"})
    merged["quarter"] = quarter
    # Drop rows with missing primary_owner_cik
    merged = merged[merged["primary_owner_cik"] != ""].copy()
    merged = merged.sort_values(["filing_date", "issuer_cik", "accession_number"]).reset_index(drop=True)
    audit["qualifying_accessions"] = int(len(merged))
    return merged, audit


class SecForm4Loader:
    """Downloads, caches, and parses official SEC DERA quarterly Form 3/4/5 ZIPs."""

    def __init__(
        self,
        data_root: Path = Path("data/sec_form4"),
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

    def _rate_limited_get(self, url: str, timeout: int = 90) -> requests.Response:
        for attempt in range(5):
            elapsed = time.monotonic() - self._last_req_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_req_ts = time.monotonic()
            resp = self._session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** (attempt + 1), 30)
                log.warning("SEC HTTP %d for %s; retrying in %ds", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            return resp
        return resp  # type: ignore[possibly-undefined]

    def discover_quarter_links(self) -> dict[str, str]:
        """Scrape official SEC DERA index page for ``{year}q{q}_form345.zip`` URLs."""
        resp = self._rate_limited_get(SEC_INDEX_URL, timeout=45)
        resp.raise_for_status()
        links: dict[str, str] = {}
        for m in re.finditer(r'href=["\']([^"\']*?(\d{4}q[1-4])_form345\.zip)["\']', resp.text, re.I):
            href, q = m.group(1), m.group(2).lower()
            full_url = urljoin(SEC_BASE_URL, href)
            links[q] = full_url
        return links

    def download_quarter_zip(self, quarter: str, url: str) -> Path | None:
        q = quarter.lower()
        dest = self.zips_dir / f"{q}_form345.zip"
        if dest.exists() and dest.stat().st_size > 10_000 and zipfile.is_zipfile(dest):
            return dest
        log.info("Downloading SEC DERA %s from %s", q, url)
        resp = self._rate_limited_get(url, timeout=120)
        if resp.status_code != 200:
            log.warning("SEC DERA %s unavailable (HTTP %d): %s", q, resp.status_code, url)
            return None
        tmp = dest.with_suffix(".zip.tmp")
        tmp.write_bytes(resp.content)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            log.warning("SEC DERA %s returned non-zip payload", q)
            return None
        tmp.replace(dest)
        return dest

    def parse_zip_file(self, zip_path: Path, quarter: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        with zipfile.ZipFile(zip_path) as zf:
            names = {Path(n).name.upper(): n for n in zf.namelist()}
            with zf.open(names["SUBMISSION.TSV"]) as f:
                sub_df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
            with zf.open(names["REPORTINGOWNER.TSV"]) as f:
                owner_df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
            with zf.open(names["NONDERIV_TRANS.TSV"]) as f:
                trans_df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
        return parse_quarter_frames(sub_df, owner_df, trans_df, quarter=quarter)

    def load_quarters(
        self,
        quarters: tuple[str, ...] = DEFAULT_QUARTERS,
        force_reparse: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load & cache all requested SEC DERA quarters into one deduplicated DataFrame."""
        discovered = self.discover_quarter_links()
        frames: list[pd.DataFrame] = []
        quarter_audits: list[dict[str, Any]] = []
        downloaded_quarters: list[str] = []
        unavailable_quarters: list[str] = []

        for q in quarters:
            ql = q.lower()
            parquet_path = self.parsed_dir / f"{ql}_accessions.parquet"
            audit_path = self.parsed_dir / f"{ql}_audit.json"
            url = discovered.get(ql)

            if not force_reparse and parquet_path.exists() and audit_path.exists():
                df_q = pd.read_parquet(parquet_path)
                aud_q = json.loads(audit_path.read_text())
                frames.append(df_q)
                quarter_audits.append(aud_q)
                downloaded_quarters.append(ql)
                continue

            if url is None:
                # Fallback check on known SEC paths
                fallback_urls = [
                    f"{SEC_BASE_URL}/files/datastandardsinnovation/data/insider-transactions-data-sets/{ql}_form345.zip",
                    f"{SEC_BASE_URL}/files/structureddata/data/insider-transactions-data-sets/{ql}_form345.zip",
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

            df_q, aud_q = self.parse_zip_file(zip_path, quarter=ql)
            aud_q["source_url"] = url
            aud_q["zip_bytes"] = int(zip_path.stat().st_size)
            df_q.to_parquet(parquet_path, index=False)
            audit_path.write_text(json.dumps(aud_q, indent=2))
            frames.append(df_q)
            quarter_audits.append(aud_q)
            downloaded_quarters.append(ql)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            before_cross_dedup = len(combined)
            combined = (
                combined.sort_values(["filing_date", "quarter", "accession_number"])
                .drop_duplicates(subset=["accession_number"], keep="first")
                .reset_index(drop=True)
            )
            cross_quarter_dups = before_cross_dedup - len(combined)
        else:
            combined = pd.DataFrame()
            cross_quarter_dups = 0

        manifest = {
            "requested_quarters": list(quarters),
            "downloaded_quarters": downloaded_quarters,
            "unavailable_quarters": unavailable_quarters,
            "total_qualifying_accessions": int(len(combined)),
            "cross_quarter_duplicate_accessions_dropped": int(cross_quarter_dups),
            "filing_date_min": str(combined["filing_date"].min().date()) if not combined.empty else None,
            "filing_date_max": str(combined["filing_date"].max().date()) if not combined.empty else None,
            "quarters_with_aff10b5one": [a["quarter"] for a in quarter_audits if a.get("has_aff10b5one_col")],
            "quarters_without_aff10b5one": [a["quarter"] for a in quarter_audits if not a.get("has_aff10b5one_col")],
            "total_aff10b5one_excluded": int(sum(a.get("aff10b5one_excluded", 0) for a in quarter_audits)),
            "total_doc_type_non_4_excluded": int(sum(a.get("doc_type_non_4_excluded", 0) for a in quarter_audits)),
            "total_multi_owner_accessions_collapsed": int(sum(a.get("multi_owner_accessions_collapsed", 0) for a in quarter_audits)),
            "total_pure_ten_percent_excluded": int(sum(a.get("pure_ten_percent_accessions_excluded", 0) for a in quarter_audits)),
            "total_outlier_rows_excluded": int(sum(a.get("outlier_price_or_value_rows_excluded", 0) for a in quarter_audits)),
            "per_quarter_audit": quarter_audits,
        }
        (self.data_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return combined, manifest


def batch_load_yfinance_bars(
    symbols: list[str],
    cache_root: Path = Path("data/cache"),
    start: date = date(2020, 10, 18),
    end: date = date(2026, 11, 20),
    chunk_size: int = 60,
    max_workers: int = 6,
) -> tuple[dict[str, BarSeries], dict[str, str]]:
    """Batch-download and cache daily OHLCV bars under ``data/cache/yfinance/``.

    Reuses existing cached ``{SYMBOL}_{start}_{end}.parquet`` bars first, then
    downloads uncached tickers in batches via ``yf.download`` and writes both
    valid frames (``status="ok"``) and missing/delisted tickers (``status="empty"``)
    into ``DiskCache(cache_root, "yfinance")``.

    Returns ``(bars_map, status_map)`` where ``status_map[sym]`` is one of
    ``"cached_ok"``, ``"cached_empty"``, ``"fetched_ok"``, ``"fetched_empty"``.
    """
    cache = DiskCache(cache_root, "yfinance")
    unique_syms = sorted(set(symbols))
    bars: dict[str, BarSeries] = {}
    statuses: dict[str, str] = {}
    to_fetch: list[str] = []

    for sym in unique_syms:
        key = f"{sym}_{start}_{end}"
        st = cache.status("bars", key)
        if st == "ok":
            df = cache.read("bars", key)
            if df is not None and not df.empty:
                bars[sym] = BarSeries(symbol=sym, frame=df)
                statuses[sym] = "cached_ok"
                continue
        if st == "empty":
            statuses[sym] = "cached_empty"
            continue
        to_fetch.append(sym)

    if not to_fetch:
        return bars, statuses

    import yfinance as yf
    from yfinance import shared as yf_shared

    log.info(
        "Batch-fetching %d uncached yfinance tickers (%d already cached)",
        len(to_fetch),
        len(unique_syms) - len(to_fetch),
    )
    end_excl = (end + timedelta(days=1)).isoformat()
    start_iso = start.isoformat()

    pending = list(to_fetch)
    for round_idx in range(5):
        if not pending:
            break
        if round_idx > 0:
            wait = min(20 * round_idx, 90)
            log.info(
                "Retrying %d rate-limited yfinance tickers (round %d) after %ds cooldown...",
                len(pending),
                round_idx + 1,
                wait,
            )
            time.sleep(wait)
        next_pending: list[str] = []

        for i in range(0, len(pending), chunk_size):
            chunk = pending[i : i + chunk_size]
            if i > 0:
                time.sleep(2.0)
            raw = None
            err_map: dict[str, str] = {}
            try:
                yf_shared._ERRORS.clear()
                raw = yf.download(
                    tickers=chunk,
                    start=start_iso,
                    end=end_excl,
                    auto_adjust=False,
                    actions=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                err_map = {str(k).upper(): str(v) for k, v in yf_shared._ERRORS.items()}
            except Exception as exc:
                msg = str(exc).lower()
                if "rate limit" in msg or "too many requests" in msg:
                    next_pending.extend(chunk)
                    time.sleep(15)
                    continue
                log.warning("yf.download chunk %d failed: %s", i, exc)

            is_multi = raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex)
            for sym in chunk:
                err_msg = err_map.get(sym.upper(), "").lower()
                if "ratelimit" in err_msg or "too many requests" in err_msg:
                    next_pending.append(sym)
                    continue

                key = f"{sym}_{start}_{end}"
                sub_df = None
                if raw is not None and not raw.empty:
                    if is_multi:
                        if sym in raw.columns.get_level_values(0):
                            sub_df = raw[sym].dropna(how="all")
                    elif len(chunk) == 1:
                        sub_df = raw.dropna(how="all")

                if sub_df is None or sub_df.empty or "Close" not in sub_df.columns:
                    cache.write("bars", key, None, status="empty", detail="yf_batch_empty")
                    statuses[sym] = "fetched_empty"
                    continue

                splits = sub_df["Stock Splits"] if "Stock Splits" in sub_df.columns else None
                series = normalise_ohlcv(sym, sub_df, splits)
                if series is None or series.frame.empty:
                    cache.write("bars", key, None, status="empty", detail="normalise_empty")
                    statuses[sym] = "fetched_empty"
                else:
                    cache.write("bars", key, series.frame, status="ok")
                    bars[sym] = series
                    statuses[sym] = "fetched_ok"

        pending = next_pending

    for sym in pending:
        log.warning("yfinance ticker %s remained rate-limited after retries (not cached as empty)", sym)
        statuses[sym] = "rate_limited"

    return bars, statuses


def load_sec_company_tickers_exchange(
    data_root: Path = Path("data/sec_form4"),
    user_agent: str | None = None,
) -> dict[str, dict[str, str]]:
    """Fetch & cache official SEC ``company_tickers_exchange.json`` keyed by normalized ticker."""
    dest = Path(data_root) / "company_tickers_exchange.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        payload = json.loads(dest.read_text())
    else:
        ua = user_agent or get_sec_user_agent()
        resp = requests.get(
            f"{SEC_BASE_URL}/files/company_tickers_exchange.json",
            headers={"User-Agent": ua},
            timeout=30,
        )
        resp.raise_for_status()
        dest.write_text(resp.text)
        payload = resp.json()

    out: dict[str, dict[str, str]] = {}
    for row in payload.get("data", []):
        if len(row) < 4:
            continue
        _cik, name, ticker, exchange = row[0], row[1], row[2], row[3]
        norm_t = normalize_sec_ticker(ticker)
        if not norm_t:
            continue
        out[norm_t] = {
            "quoteType": "EQUITY",
            "exchange": str(exchange or "").upper().strip(),
            "shortName": str(name or "").strip(),
        }
    return out


def _fetch_single_fallback(
    sym: str,
    start: date,
    end: date,
    cache: DiskCache,
    bars: dict[str, BarSeries],
    statuses: dict[str, str],
) -> None:
    import yfinance as yf

    key = f"{sym}_{start}_{end}"
    try:
        hist = yf.Ticker(sym).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=True,
        )
    except Exception:
        hist = None
    if hist is None or hist.empty:
        cache.write("bars", key, None, status="empty", detail="single_empty")
        statuses[sym] = "fetched_empty"
        return
    splits = hist["Stock Splits"] if "Stock Splits" in hist.columns else None
    series = normalise_ohlcv(sym, hist, splits)
    if series is None:
        cache.write("bars", key, None, status="empty", detail="single_norm_empty")
        statuses[sym] = "fetched_empty"
    else:
        cache.write("bars", key, series.frame, status="ok")
        bars[sym] = series
        statuses[sym] = "fetched_ok"


def batch_load_ticker_metadata(
    symbols: list[str],
    cache_root: Path = Path("data/cache"),
    max_workers: int = 8,
) -> dict[str, dict[str, str]]:
    """Fetch & cache minimal yfinance metadata (``quoteType``, ``exchange``, ``shortName``)."""
    cache = DiskCache(cache_root, "yfinance")
    unique_syms = sorted(set(symbols))
    out: dict[str, dict[str, str]] = {}
    to_fetch: list[str] = []

    for sym in unique_syms:
        st = cache.status("ticker_meta", sym)
        if st == "ok":
            df = cache.read("ticker_meta", sym)
            if df is not None and not df.empty:
                row = df.iloc[0].to_dict()
                out[sym] = {
                    "quoteType": str(row.get("quoteType", "") or ""),
                    "exchange": str(row.get("exchange", "") or ""),
                    "shortName": str(row.get("shortName", "") or ""),
                }
                continue
        if st == "empty":
            out[sym] = {"quoteType": "", "exchange": "", "shortName": ""}
            continue
        to_fetch.append(sym)

    if not to_fetch:
        return out

    import yfinance as yf

    def _one(sym: str) -> tuple[str, dict[str, str]]:
        try:
            fi = yf.Ticker(sym).fast_info
            qt = str(getattr(fi, "quote_type", "") or "").upper()
            ex = str(getattr(fi, "exchange", "") or "").upper()
            sn = ""
            if not qt:
                info = yf.Ticker(sym).info or {}
                qt = str(info.get("quoteType", "") or "").upper()
                ex = str(info.get("exchange", "") or "").upper()
                sn = str(info.get("shortName", "") or "")
            meta = {"quoteType": qt, "exchange": ex, "shortName": sn}
            if qt or ex:
                cache.write("ticker_meta", sym, pd.DataFrame([meta]), status="ok")
            else:
                cache.write("ticker_meta", sym, None, status="empty")
            return sym, meta
        except Exception:
            cache.write("ticker_meta", sym, None, status="empty")
            return sym, {"quoteType": "", "exchange": "", "shortName": ""}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for sym, meta in pool.map(_one, to_fetch):
            out[sym] = meta
    return out


def is_allowed_us_common_stock(
    issuer_name: str,
    ticker: str,
    meta: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Return ``(allowed, rejection_reason)`` for US common stock classification."""
    if is_excluded_non_common_issuer(issuer_name, ticker):
        return False, "sec_non_common_name"
    if meta:
        qt = str(meta.get("quoteType", "")).upper().strip()
        ex = str(meta.get("exchange", "")).upper().strip()
        sn = str(meta.get("shortName", ""))
        if qt in _DISALLOWED_QUOTE_TYPES:
            return False, f"yf_quote_type_{qt.lower()}"
        if sn and is_excluded_non_common_issuer(sn, ticker):
            return False, "yf_non_common_name"
        if ex and ex not in _US_MAJOR_EXCHANGES:
            # Exclude OTC / foreign exchanges explicitly when known
            if ex in {"PNK", "OQX", "OQB", "OTC", "GREY", "OTCMKTS"}:
                return False, "yf_otc_exchange"
            return False, f"yf_non_us_exchange_{ex.lower()}"
    return True, "ok"
