"""Strategy v10_coffee_can (Layer 1): Quantitative Asymmetric Compounder Screener.

Tests whether a low-touch, concentrated "Coffee Can" basket of 5 high-growth,
high-margin, low-dilution US small/mid-cap compounders beats SPY, CSPX.AS, and
v5b QQQ3.MI over 2022-04 to 2026-11 without daily trading or tight price stops.

Strict Anti-Bias Guarantees:
- **Zero Financial Look-Ahead:** Every fundamental fact comes strictly from SEC
  XBRL filings with ``filed_date <= rebalance_date`` (10-K within prior 400
  calendar days). Yahoo Finance's current ``marketCap`` / ratios are NEVER used.
- **Point-in-Time Market Cap & Price:** Computed strictly from historical XBRL
  shares outstanding (adjusted only for causal splits in ``(shares_date, rebalance_date]``)
  times as-traded historical ``raw_close`` on trading days ``<= rebalance_date``.
- **Execution:** Orders execute strictly at next trading day's Open with IBKR
  Cash Account T+1 settlement, whole-share AS_TRADED sizing, ``$0.38``/order +
  ``0.25%`` slippage/side, and ``EUR 0/mo`` hosting.
- **Two Locked Portfolio Rules:**
  - ``v10a_annual_top5``: Rebalance once per year (first trading day of April)
    equally into the current Top 5 Rule-of-40 qualifiers. No intra-year price stops.
  - ``v10b_coffee_can_thesis_watch``: Buy Top 5 cohort, hold winners up to 3 years.
    Exit early ONLY on a fundamental thesis break (2 consecutive 10-Q/10-K filings
    with negative YoY revenue growth, OR TTM cash runway < 12 months). Runway uses
    trailing-twelve-month free cash flow, never a single quarter. A confirmed
    break exits at the next trading day's Open and that freed slot is redeployed
    the same morning into the highest-ranked qualifier whose filings are dated
    ``filed <=`` the exit decision date.
- **Honest Survivorship Audit & Penalized Sensitivity:** Logs exact unmatched /
  early-delisted rates per vintage (IS vs Blind OOS) and runs side-by-side
  Raw-Matched and Survivorship-Penalized (``-50%``) evaluations in Part A and Part B.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from src.backtest.metrics import (
    cagr,
    daily_returns,
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
)
from src.config import MetricsConfig
from src.data.sec_form4_loader import is_allowed_us_common_stock, is_excluded_non_common_issuer
from src.data.sec_xbrl_loader import RULE40_MAX_FCF_MARGIN, RULE40_MAX_YOY
from src.domain.models import BarSeries

US_PERIODS_PER_YEAR = 252

CoffeeCanRule = Literal["v10a_annual_top5", "v10b_coffee_can_thesis_watch"]
ExitReason = Literal[
    "ANNUAL_REBALANCE_EXIT",
    "COFFEE_CAN_3Y_MATURITY",
    "THESIS_BREAK_REV_DECLINE",
    "THESIS_BREAK_RUNWAY",
    "DELISTED_LAST_BAR",
    "DELISTED_PENALIZED_50",
    "UNMATCHED_PENALIZED_50",
    "OPEN_MTM",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CoffeeCanConfig(_Frozen):
    """Locked configuration for Strategy v10_coffee_can."""

    rule: CoffeeCanRule = "v10a_annual_top5"

    # Sector / universe exclusions
    sic_exclude_min: int = 6000
    sic_exclude_max: int = 6999
    sic_extractive_min: int = 1000
    sic_extractive_max: int = 1499

    # Fundamental filters (strictly from latest 10-K filed <= rebalance_date)
    max_filing_age_days: int = Field(400, ge=180, le=730)
    min_revenue_usd: float = Field(50_000_000.0, gt=0.0)
    min_yoy_rev_growth: float = Field(0.25, ge=-1.0)
    min_gross_margin: float = Field(0.50, ge=0.0, le=1.0)
    min_operating_margin: float = Field(0.10, ge=-1.0, le=5.0)
    max_gross_margin_sanity: float = Field(1.00, ge=0.5, le=1.05)
    min_cash_runway_years_if_fcf_neg: float = Field(2.0, gt=0.0)  # 24 months
    max_debt_to_equity: float = Field(1.0, ge=0.0)
    max_yoy_share_dilution: float = Field(0.05, ge=-0.5, le=2.0)
    min_yoy_share_dilution_sanity: float = Field(-0.50, ge=-0.99, le=0.0)

    # Causal tradability filters (strictly on bars <= rebalance_date)
    min_market_cap_usd: float = Field(300_000_000.0, gt=0.0)
    max_market_cap_usd: float = Field(10_000_000_000.0, gt=0.0)
    min_price_usd: float = Field(5.0, gt=0.0)
    max_price_usd: float = Field(150.0, gt=0.0)
    adv_window_days: int = Field(20, ge=5)
    min_adv_usd: float = Field(2_000_000.0, ge=0.0)
    max_entry_lag_calendar_days: int = Field(7, ge=1, le=30)

    # Portfolio & execution (IBKR Cash Account, whole shares, T+1, EUR 0/mo)
    top_n: int = Field(5, ge=1, le=25)
    target_slot_pct: float = Field(0.20, gt=0.0, le=1.0)
    slippage_pct: float = Field(0.0025, ge=0.0, lt=0.1)
    commission_per_side_usd: float = Field(0.38, ge=0.0)
    settlement_days: int = Field(1, ge=0, le=5)

    # v10b Thesis-Watch & 3-year hold
    max_hold_years: int = Field(3, ge=1, le=10)
    thesis_break_consec_neg_rev: int = Field(2, ge=1, le=8)
    thesis_break_min_runway_months: float = Field(12.0, gt=0.0)

    # Survivorship sensitivity
    survivorship_penalty_return: float = Field(-0.50, ge=-0.99, le=0.0)
    penalized_unmatched_hold_days: int = Field(126, ge=20)


class FundamentalCandidate(BaseModel):
    """One issuer passing all point-in-time XBRL fundamental filters on ``rebalance_date``."""

    model_config = ConfigDict(frozen=True)

    vintage: str
    rebalance_date: date
    cik: str
    ticker: str
    ticker_source: str
    name: str
    sic: int
    adsh: str
    filed_date: date
    period_date: date
    revenue_usd: float
    revenue_prior_usd: float
    revenue_t2_usd: float
    yoy_rev_growth: float
    yoy_rev_growth_prior: float
    gross_margin: float
    operating_income_usd: float
    operating_margin: float
    fcf_usd: float
    fcf_margin: float
    cash_total_usd: float | None
    cash_runway_years: float
    total_debt_usd: float
    stockholders_equity_usd: float
    debt_to_equity: float
    shares_outstanding: float
    shares_date: date | None
    shares_tag: str
    rev_tag: str = ""
    gp_tag: str = ""
    yoy_share_dilution: float
    rule_of_40: float


class EvaluatedCandidate(BaseModel):
    """Fundamental candidate enriched with strictly causal tradability & survivorship status."""

    model_config = ConfigDict(frozen=True)

    candidate: FundamentalCandidate
    status: Literal[
        "qualified_matched",
        "unmatched_yfinance",
        "rejected_non_common",
        "rejected_insufficient_history",
        "rejected_price",
        "rejected_market_cap",
        "rejected_adv",
        "rejected_no_entry_bar",
    ]
    rejection_detail: str = ""
    entry_date: date | None = None
    asof_raw_price: float | None = None
    pit_shares_adjusted: float | None = None
    pit_market_cap_usd: float | None = None
    asof_adv20_usd: float | None = None
    last_bar_date: date | None = None


class CoffeeCanTradeRecord(BaseModel):
    """Completed or open-at-end position record."""

    model_config = ConfigDict(frozen=True)

    rule: CoffeeCanRule
    vintage: str
    cik: str
    ticker: str
    name: str
    filed_date: date
    entry_date: date
    exit_date: date
    rule_of_40: float
    yoy_rev_growth: float
    gross_margin: float
    fcf_margin: float
    pit_market_cap_usd: float | None
    raw_qty: int
    adj_qty: float
    entry_raw_price: float
    entry_adj_price: float
    exit_adj_price: float
    bars_held: int
    cost_basis_usd: float
    gross_pnl_usd: float
    commission_usd: float
    slippage_usd: float
    net_pnl_usd: float
    return_pct: float
    exit_reason: ExitReason
    is_open_at_end: bool = False
    is_synthetic_penalty: bool = False


class CoffeeCanPortfolioResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CoffeeCanConfig
    initial_capital_usd: float
    initial_capital_eur: float
    window_label: str
    survivorship_penalized: bool
    equity_curve: pd.DataFrame
    trades: list[CoffeeCanTradeRecord]
    open_positions_at_end: list[CoffeeCanTradeRecord]
    skipped_counts: dict[str, int]
    total_orders: int
    total_commission_usd: float
    total_slippage_usd: float


def _is_missing(v: Any) -> bool:
    """True for None / NaN. False for 0, False, and ordinary numbers."""
    if v is None:
        return True
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (float, np.floating)):
        return bool(math.isnan(float(v)))
    if isinstance(v, (int, np.integer)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def first_trading_day_of_april(spy_cal: pd.DatetimeIndex, year: int) -> date:
    """Return the first US trading day in April of ``year`` on ``spy_cal``."""
    apr = spy_cal[(spy_cal.year == year) & (spy_cal.month == 4)]
    if apr.empty:
        raise ValueError(f"No April trading days for {year} in SPY calendar")
    return apr[0].date()


def screen_fundamentals_asof(
    annual_df: pd.DataFrame,
    rebalance_date: date,
    cfg: CoffeeCanConfig,
    vintage_label: str | None = None,
) -> tuple[list[FundamentalCandidate], dict[str, int]]:
    """Screen SEC XBRL 10-K filings strictly causally as of ``rebalance_date``.

    Only considers 10-K filings with ``filed_date <= rebalance_date`` and age
    ``<= cfg.max_filing_age_days`` (400 days). Picks the latest qualifying-date
    10-K per ``cik`` BEFORE applying financial thresholds so an older filing
    cannot mask a newer bad 10-K.
    """
    vlabel = vintage_label or f"{rebalance_date.year}-{rebalance_date.month:02d}"
    counts = {
        "total_10k_asof_fresh": 0,
        "excluded_sic_financial_reit_spac": 0,
        "excluded_sic_extractive": 0,
        "excluded_non_common_name": 0,
        "excluded_missing_capex": 0,
        "excluded_missing_core_fields": 0,
        "excluded_revenue_lt_50m": 0,
        "excluded_single_year_revenue_spike": 0,
        "excluded_rev_growth_lt_25pct": 0,
        "excluded_gross_margin_lt_50pct": 0,
        "excluded_operating_margin_lt_10pct": 0,
        "excluded_fcf_gt_revenue": 0,
        "excluded_survival_fcf_runway": 0,
        "excluded_equity_or_debt_to_equity": 0,
        "excluded_share_dilution_gt_5pct": 0,
        "fundamental_qualifiers": 0,
        "fundamental_qualifiers_unresolved_ticker": 0,
    }
    if annual_df.empty:
        return [], counts

    reb_ts = pd.Timestamp(rebalance_date)
    min_filed_ts = reb_ts - pd.Timedelta(days=cfg.max_filing_age_days)

    df = annual_df.copy()
    df["filed_date"] = pd.to_datetime(df["filed_date"]).dt.normalize()
    df["period_date"] = pd.to_datetime(df["period_date"]).dt.normalize()

    # STRICT causal filter: filed_date <= rebalance_date AND within prior 400 days
    asof = df[(df["filed_date"] <= reb_ts) & (df["filed_date"] >= min_filed_ts)].copy()
    if asof.empty:
        return [], counts

    # Pick strictly the latest 10-K per cik as of rebalance_date (by filed_date, then period_date)
    asof = (
        asof.sort_values(["cik", "filed_date", "period_date", "adsh"], kind="mergesort")
        .drop_duplicates(subset=["cik"], keep="last")
        .reset_index(drop=True)
    )
    counts["total_10k_asof_fresh"] = int(len(asof))

    candidates: list[FundamentalCandidate] = []
    for row in asof.itertuples(index=False):
        sic = int(row.sic) if pd.notna(row.sic) else -1
        if cfg.sic_exclude_min <= sic <= cfg.sic_exclude_max:
            counts["excluded_sic_financial_reit_spac"] += 1
            continue
        if cfg.sic_extractive_min <= sic <= cfg.sic_extractive_max:
            counts["excluded_sic_extractive"] += 1
            continue

        name = str(row.name or "").strip()
        ticker = str(row.ticker or "").strip()
        if is_excluded_non_common_issuer(name, ticker):
            counts["excluded_non_common_name"] += 1
            continue

        capex_flag = getattr(row, "capex_reported", False)
        capex_raw = getattr(row, "capex", None)
        capex_reported = (not _is_missing(capex_flag)) and bool(capex_flag)
        if (not capex_reported) or _is_missing(capex_raw):
            counts["excluded_missing_capex"] += 1
            continue

        rev = row.revenue
        rev_p = row.revenue_prior
        rev_t2 = getattr(row, "revenue_t2", None)
        yoy_g = row.yoy_rev_growth
        gm = row.gross_margin
        op_inc = getattr(row, "operating_income", None)
        op_m = getattr(row, "operating_margin", None)
        fcf = row.fcf
        fcf_m = row.fcf_margin
        cash = row.cash_total
        eq = row.stockholders_equity
        debt = row.total_debt
        dte = row.debt_to_equity
        sh_out = row.shares_outstanding
        dil = row.yoy_share_dilution

        req = (rev, rev_p, yoy_g, gm, fcf, fcf_m, eq, debt, dte, sh_out, dil)
        if any(_is_missing(v) for v in req):
            counts["excluded_missing_core_fields"] += 1
            continue

        rev_f = float(rev)
        rev_pf = float(rev_p)
        yoy_gf = float(yoy_g)
        gm_f = float(gm)
        fcf_f = float(fcf)
        fcf_mf = float(fcf_m)
        eq_f = float(eq)
        debt_f = float(debt)
        dte_f = float(dte)
        sh_f = float(sh_out)
        dil_f = float(dil)
        cash_f = None if _is_missing(cash) else float(cash)

        if sh_f <= 0 or rev_f < cfg.min_revenue_usd or rev_pf < cfg.min_revenue_usd:
            counts["excluded_revenue_lt_50m"] += 1
            continue
        if _is_missing(rev_t2):
            counts["excluded_single_year_revenue_spike"] += 1
            continue
        rev_t2f = float(rev_t2)
        # Two consecutive years of positive growth: Rev_t > Rev_t-1 > Rev_t-2.
        if not (rev_f > rev_pf > rev_t2f):
            counts["excluded_single_year_revenue_spike"] += 1
            continue
        if yoy_gf < cfg.min_yoy_rev_growth:
            counts["excluded_rev_growth_lt_25pct"] += 1
            continue
        if gm_f < cfg.min_gross_margin or gm_f > cfg.max_gross_margin_sanity:
            counts["excluded_gross_margin_lt_50pct"] += 1
            continue
        if _is_missing(op_m) or float(op_m) < cfg.min_operating_margin:
            counts["excluded_operating_margin_lt_10pct"] += 1
            continue
        if fcf_f > rev_f:
            counts["excluded_fcf_gt_revenue"] += 1
            continue

        op_mf = float(op_m)
        op_inc_f = float(op_inc) if not _is_missing(op_inc) else op_mf * rev_f
        yoy_prior = (rev_pf / rev_t2f - 1.0) if rev_t2f > 0 else float("nan")
        r40_f = (min(yoy_gf, RULE40_MAX_YOY) + min(fcf_mf, RULE40_MAX_FCF_MARGIN)) * 100.0

        # Survival guardrail: FCF > 0 OR cash runway >= 24 months (Cash / abs(FCF) >= 2.0)
        if fcf_f > 0.0:
            runway_yr = float("inf")
        else:
            if cash_f is None or cash_f <= 0.0 or fcf_f == 0.0:
                counts["excluded_survival_fcf_runway"] += 1
                continue
            runway_yr = cash_f / abs(fcf_f)
            if runway_yr < cfg.min_cash_runway_years_if_fcf_neg:
                counts["excluded_survival_fcf_runway"] += 1
                continue

        if eq_f <= 0.0 or dte_f > cfg.max_debt_to_equity or dte_f < 0.0:
            counts["excluded_equity_or_debt_to_equity"] += 1
            continue
        if dil_f > cfg.max_yoy_share_dilution or dil_f < cfg.min_yoy_share_dilution_sanity:
            counts["excluded_share_dilution_gt_5pct"] += 1
            continue

        if not ticker:
            counts["fundamental_qualifiers_unresolved_ticker"] += 1

        sh_dt_val = getattr(row, "shares_date", pd.NaT)
        sh_date = None if pd.isna(sh_dt_val) else pd.Timestamp(sh_dt_val).date()

        candidates.append(
            FundamentalCandidate(
                vintage=vlabel,
                rebalance_date=rebalance_date,
                cik=str(row.cik),
                ticker=ticker,
                ticker_source=str(getattr(row, "ticker_source", "") or ""),
                name=name,
                sic=sic,
                adsh=str(row.adsh),
                filed_date=pd.Timestamp(row.filed_date).date(),
                period_date=pd.Timestamp(row.period_date).date(),
                revenue_usd=rev_f,
                revenue_prior_usd=rev_pf,
                revenue_t2_usd=rev_t2f,
                yoy_rev_growth=yoy_gf,
                yoy_rev_growth_prior=yoy_prior,
                gross_margin=gm_f,
                operating_income_usd=op_inc_f,
                operating_margin=op_mf,
                fcf_usd=fcf_f,
                fcf_margin=fcf_mf,
                cash_total_usd=cash_f,
                cash_runway_years=runway_yr if math.isfinite(runway_yr) else 999.0,
                total_debt_usd=debt_f,
                stockholders_equity_usd=eq_f,
                debt_to_equity=dte_f,
                shares_outstanding=sh_f,
                shares_date=sh_date,
                shares_tag=str(getattr(row, "shares_tag", "") or ""),
                rev_tag=str(getattr(row, "rev_tag", "") or ""),
                gp_tag=str(getattr(row, "gp_tag", "") or ""),
                yoy_share_dilution=dil_f,
                rule_of_40=r40_f,
            )
        )

    # Sort deterministically by descending Rule of 40, then YoY growth, then ticker/cik
    candidates.sort(key=lambda c: (-c.rule_of_40, -c.yoy_rev_growth, c.ticker or "ZZZZZ", c.cik))
    counts["fundamental_qualifiers"] = len(candidates)
    return candidates, counts


def _causal_split_factor_between(
    hist_df: pd.DataFrame,
    ref_date: date | None,
) -> float:
    """Compute any split ratio occurring strictly in ``(ref_date, hist_df.index[-1]]``.

    Uses ONLY ``hist_df`` (already truncated to ``<= rebalance_date``). Any future
    splits after ``rebalance_date`` cancel out identically between numerator and
    denominator, guaranteeing zero look-ahead.
    """
    if ref_date is None or "raw_close" not in hist_df.columns or hist_df.empty:
        return 1.0
    end_close = float(hist_df["close"].iloc[-1])
    end_raw = float(hist_df["raw_close"].iloc[-1])
    if end_close <= 0 or end_raw <= 0:
        return 1.0
    ref_sub = hist_df.loc[hist_df.index <= pd.Timestamp(ref_date)]
    if ref_sub.empty:
        return 1.0
    ref_close = float(ref_sub["close"].iloc[-1])
    ref_raw = float(ref_sub["raw_close"].iloc[-1])
    if ref_close <= 0 or ref_raw <= 0:
        return 1.0
    end_mult = end_raw / end_close
    ref_mult = ref_raw / ref_close
    ratio = ref_mult / end_mult
    # Only apply if genuinely split (avoid floating point jitter)
    return float(ratio) if (0.01 <= ratio <= 100.0 and abs(ratio - 1.0) > 0.02) else 1.0


def evaluate_vintage_candidates(
    candidates: list[FundamentalCandidate],
    bars: dict[str, BarSeries],
    ticker_meta: dict[str, dict[str, str]],
    spy_bars: BarSeries,
    cfg: CoffeeCanConfig,
) -> tuple[list[EvaluatedCandidate], dict[str, Any]]:
    """Apply strictly causal tradability filters ($300M-$10B mcap, $5-$150 px, $2M ADV)."""
    spy_cal = spy_bars.frame.index.sort_values()
    evaluated: list[EvaluatedCandidate] = []
    counts: dict[str, int] = {
        "fundamental_qualifiers": len(candidates),
        "qualified_matched": 0,
        "unmatched_yfinance": 0,
        "rejected_non_common": 0,
        "rejected_insufficient_history": 0,
        "rejected_price": 0,
        "rejected_market_cap": 0,
        "rejected_adv": 0,
        "rejected_no_entry_bar": 0,
    }

    for cand in candidates:
        reb_ts = pd.Timestamp(cand.rebalance_date)
        spy_pos = int(spy_cal.searchsorted(reb_ts, side="right"))
        next_spy_ts = spy_cal[spy_pos] if spy_pos < len(spy_cal) else None

        if not cand.ticker:
            counts["unmatched_yfinance"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="unmatched_yfinance",
                    rejection_detail="unresolved_ticker",
                    entry_date=None if next_spy_ts is None else next_spy_ts.date(),
                )
            )
            continue

        meta = ticker_meta.get(cand.ticker)
        allowed, reason = is_allowed_us_common_stock(cand.name, cand.ticker, meta)
        if not allowed:
            counts["rejected_non_common"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="rejected_non_common",
                    rejection_detail=reason,
                )
            )
            continue

        series = bars.get(cand.ticker)
        if series is None or series.frame.empty:
            counts["unmatched_yfinance"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="unmatched_yfinance",
                    rejection_detail="no_yfinance_bars",
                    entry_date=None if next_spy_ts is None else next_spy_ts.date(),
                )
            )
            continue

        df = series.frame
        hist = df.loc[df.index <= reb_ts]
        future = df.loc[df.index > reb_ts]

        if hist.empty or (reb_ts - hist.index[-1]).days > 14:
            if future.empty or (future.index[0] - reb_ts).days > 14:
                counts["unmatched_yfinance"] += 1
                evaluated.append(
                    EvaluatedCandidate(
                        candidate=cand,
                        status="unmatched_yfinance",
                        rejection_detail="no_bars_around_rebalance_date",
                        entry_date=None if next_spy_ts is None else next_spy_ts.date(),
                    )
                )
                continue

        if len(hist) < cfg.adv_window_days:
            counts["rejected_insufficient_history"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="rejected_insufficient_history",
                    rejection_detail=f"bars_le_rebalance={len(hist)}",
                )
            )
            continue

        price_col = "raw_close" if "raw_close" in hist.columns else "close"
        asof_raw_px = float(hist[price_col].iloc[-1])
        tail_w = hist.iloc[-cfg.adv_window_days :]
        asof_adv20 = float((tail_w["close"] * tail_w["volume"]).mean())

        ref_sh_date = cand.shares_date or cand.period_date
        split_adj = _causal_split_factor_between(hist, ref_sh_date)
        pit_shares = cand.shares_outstanding * split_adj
        pit_mcap = pit_shares * asof_raw_px

        if not (cfg.min_price_usd <= asof_raw_px <= cfg.max_price_usd):
            counts["rejected_price"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="rejected_price",
                    rejection_detail=f"raw_px={asof_raw_px:.2f}",
                    asof_raw_price=asof_raw_px,
                    pit_shares_adjusted=pit_shares,
                    pit_market_cap_usd=pit_mcap,
                    asof_adv20_usd=asof_adv20,
                )
            )
            continue

        if not (cfg.min_market_cap_usd <= pit_mcap <= cfg.max_market_cap_usd):
            counts["rejected_market_cap"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="rejected_market_cap",
                    rejection_detail=f"pit_mcap={pit_mcap/1e6:.1f}M",
                    asof_raw_price=asof_raw_px,
                    pit_shares_adjusted=pit_shares,
                    pit_market_cap_usd=pit_mcap,
                    asof_adv20_usd=asof_adv20,
                )
            )
            continue

        if asof_adv20 < cfg.min_adv_usd:
            counts["rejected_adv"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="rejected_adv",
                    rejection_detail=f"adv20={asof_adv20/1e6:.2f}M",
                    asof_raw_price=asof_raw_px,
                    pit_shares_adjusted=pit_shares,
                    pit_market_cap_usd=pit_mcap,
                    asof_adv20_usd=asof_adv20,
                )
            )
            continue

        if future.empty or (future.index[0] - reb_ts).days > cfg.max_entry_lag_calendar_days:
            counts["rejected_no_entry_bar"] += 1
            evaluated.append(
                EvaluatedCandidate(
                    candidate=cand,
                    status="rejected_no_entry_bar",
                    rejection_detail="no_next_open",
                    asof_raw_price=asof_raw_px,
                    pit_shares_adjusted=pit_shares,
                    pit_market_cap_usd=pit_mcap,
                    asof_adv20_usd=asof_adv20,
                )
            )
            continue

        counts["qualified_matched"] += 1
        evaluated.append(
            EvaluatedCandidate(
                candidate=cand,
                status="qualified_matched",
                entry_date=future.index[0].date(),
                asof_raw_price=asof_raw_px,
                pit_shares_adjusted=pit_shares,
                pit_market_cap_usd=pit_mcap,
                asof_adv20_usd=asof_adv20,
                last_bar_date=df.index[-1].date(),
            )
        )

    # Maintain strict Rule-of-40 ranking
    evaluated.sort(
        key=lambda e: (-e.candidate.rule_of_40, -e.candidate.yoy_rev_growth, e.candidate.ticker or "ZZZZZ", e.candidate.cik)
    )

    denom_common = counts["fundamental_qualifiers"] - counts["rejected_non_common"]
    unmatched_rate = counts["unmatched_yfinance"] / denom_common if denom_common > 0 else 0.0
    audit = {
        **counts,
        "common_fundamental_qualifiers": denom_common,
        "unmatched_pct_of_fundamental_qualifiers": unmatched_rate,
    }
    return evaluated, audit


def _summarize_returns(stock_rets: list[float], excess_rets: list[float]) -> dict[str, Any]:
    n = len(excess_rets)
    if n == 0:
        return {
            "n": 0,
            "mean_stock_ret": float("nan"),
            "median_stock_ret": float("nan"),
            "mean_excess_ret": float("nan"),
            "median_excess_ret": float("nan"),
            "p_value_ttest": float("nan"),
            "p_value_wilcoxon": float("nan"),
            "pct_beat_spy": float("nan"),
        }
    s_arr = np.asarray(stock_rets, dtype=float)
    e_arr = np.asarray(excess_rets, dtype=float)
    if n >= 2 and np.std(e_arr, ddof=1) > 0:
        p_t = float(stats.ttest_1samp(e_arr, 0.0).pvalue)
    else:
        p_t = float("nan")
    try:
        nz = e_arr[e_arr != 0.0]
        p_w = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    except Exception:
        p_w = float("nan")
    return {
        "n": int(n),
        "mean_stock_ret": float(np.mean(s_arr)),
        "median_stock_ret": float(np.median(s_arr)),
        "mean_excess_ret": float(np.mean(e_arr)),
        "median_excess_ret": float(np.median(e_arr)),
        "p_value_ttest": p_t,
        "p_value_wilcoxon": p_w,
        "pct_beat_spy": float(np.mean(e_arr > 0.0)),
    }


def run_coffee_can_event_study(
    evaluated_by_vintage: dict[str, list[EvaluatedCandidate]],
    bars: dict[str, BarSeries],
    spy_bars: BarSeries,
    sample_end_date: date,
    cfg: CoffeeCanConfig,
    is_vintages: tuple[str, ...] = ("2022-04", "2023-04", "2024-04"),
    oos_vintages: tuple[str, ...] = ("2025-04", "2026-04"),
) -> dict[str, Any]:
    """Part A: Qualifying-pool event study (1Y/2Y/3Y vs SPY, multi-baggers, blow-ups) + Top-5 table."""
    spy_df = spy_bars.frame.loc[spy_bars.frame.index <= pd.Timestamp(sample_end_date)]
    spy_cal = spy_df.index.sort_values()
    end_ts = pd.Timestamp(sample_end_date)
    penalty = cfg.survivorship_penalty_return
    horizons: tuple[tuple[str, int], ...] = (("1Y", 252), ("2Y", 504), ("3Y", 756))

    def _eval_subset(items: list[EvaluatedCandidate]) -> dict[str, Any]:
        matched = [e for e in items if e.status == "qualified_matched"]
        unmatched = [e for e in items if e.status == "unmatched_yfinance"]

        raw_h: dict[str, tuple[list[float], list[float]]] = {lbl: ([], []) for lbl, _ in horizons}
        pen_h: dict[str, tuple[list[float], list[float]]] = {lbl: ([], []) for lbl, _ in horizons}
        raw_to_end: tuple[list[float], list[float]] = ([], [])
        pen_to_end: tuple[list[float], list[float]] = ([], [])

        mb100_raw: list[int] = []
        mb200_raw: list[int] = []
        bu50_raw: list[int] = []
        mb100_pen: list[int] = []
        mb200_pen: list[int] = []
        bu50_pen: list[int] = []

        early_term_1y = 0
        early_term_3y = 0

        for ev in matched:
            if ev.entry_date is None:
                continue
            entry_ts = pd.Timestamp(ev.entry_date)
            if entry_ts > end_ts:
                continue
            df = bars[ev.candidate.ticker].frame
            df = df.loc[(df.index >= entry_ts) & (df.index <= end_ts)]
            if df.empty:
                continue
            p0 = float(df["open"].iloc[0])
            if p0 <= 0:
                continue

            spy_pos = int(spy_cal.searchsorted(entry_ts, side="left"))
            if spy_pos >= len(spy_cal):
                continue
            spy_p0 = float(spy_df["open"].iloc[spy_pos])
            last_stock_ts = df.index[-1]
            is_delisted_early_overall = (end_ts - last_stock_ts).days >= 15

            # Horizon returns
            for lbl, h_bars in horizons:
                tgt_pos = spy_pos + h_bars - 1
                if tgt_pos >= len(spy_cal):
                    continue  # Incomplete horizon (right-censored by 2026-11-20)
                tgt_ts = spy_cal[tgt_pos]
                spy_ret = float(spy_df["close"].iloc[tgt_pos]) / spy_p0 - 1.0

                if len(df) >= h_bars:
                    s_ret = float(df["close"].iloc[h_bars - 1]) / p0 - 1.0
                    exit_ts = df.index[h_bars - 1]
                    s_spy_idx = int(spy_cal.searchsorted(exit_ts, side="right")) - 1
                    s_spy_ret = float(spy_df["close"].iloc[s_spy_idx]) / spy_p0 - 1.0 if s_spy_idx >= spy_pos else spy_ret
                    raw_h[lbl][0].append(s_ret)
                    raw_h[lbl][1].append(s_ret - s_spy_ret)
                    pen_h[lbl][0].append(s_ret)
                    pen_h[lbl][1].append(s_ret - s_spy_ret)
                elif (tgt_ts - last_stock_ts).days >= 15:
                    if lbl == "1Y":
                        early_term_1y += 1
                    if lbl == "3Y":
                        early_term_3y += 1
                    last_ret = float(df["close"].iloc[-1]) / p0 - 1.0
                    l_spy_idx = int(spy_cal.searchsorted(last_stock_ts, side="right")) - 1
                    l_spy_ret = float(spy_df["close"].iloc[l_spy_idx]) / spy_p0 - 1.0 if l_spy_idx >= spy_pos else spy_ret
                    raw_h[lbl][0].append(last_ret)
                    raw_h[lbl][1].append(last_ret - l_spy_ret)
                    pen_h[lbl][0].append(penalty)
                    pen_h[lbl][1].append(penalty - spy_ret)

            # Return through sample_end_date (up to 3Y max)
            w_3y = df.iloc[:756]
            end_ret = float(w_3y["close"].iloc[-1]) / p0 - 1.0
            end_spy_pos = min(spy_pos + 755, len(spy_cal) - 1)
            end_spy_ret = float(spy_df["close"].iloc[end_spy_pos]) / spy_p0 - 1.0
            raw_to_end[0].append(end_ret)
            raw_to_end[1].append(end_ret - end_spy_ret)

            peak_g = float(w_3y["close"].max()) / p0 - 1.0
            trough_d = float(w_3y["close"].min()) / p0 - 1.0
            mb100_raw.append(1 if peak_g >= 1.00 else 0)
            mb200_raw.append(1 if peak_g >= 2.00 else 0)
            bu50_raw.append(1 if trough_d <= -0.50 else 0)

            if is_delisted_early_overall and len(df) < 756:
                pen_to_end[0].append(penalty)
                pen_to_end[1].append(penalty - end_spy_ret)
                mb100_pen.append(0)
                mb200_pen.append(0)
                bu50_pen.append(1)
            else:
                pen_to_end[0].append(end_ret)
                pen_to_end[1].append(end_ret - end_spy_ret)
                mb100_pen.append(1 if peak_g >= 1.00 else 0)
                mb200_pen.append(1 if peak_g >= 2.00 else 0)
                bu50_pen.append(1 if trough_d <= -0.50 else 0)

        # Add unmatched fundamental qualifiers to Survivorship-Penalized pool
        for ev in unmatched:
            if ev.entry_date is None:
                continue
            entry_ts = pd.Timestamp(ev.entry_date)
            if entry_ts > end_ts:
                continue
            spy_pos = int(spy_cal.searchsorted(entry_ts, side="left"))
            if spy_pos >= len(spy_cal):
                continue
            spy_p0 = float(spy_df["open"].iloc[spy_pos])
            for lbl, h_bars in horizons:
                tgt_pos = spy_pos + h_bars - 1
                if tgt_pos >= len(spy_cal):
                    continue
                spy_ret = float(spy_df["close"].iloc[tgt_pos]) / spy_p0 - 1.0
                pen_h[lbl][0].append(penalty)
                pen_h[lbl][1].append(penalty - spy_ret)
            end_spy_pos = min(spy_pos + 755, len(spy_cal) - 1)
            end_spy_ret = float(spy_df["close"].iloc[end_spy_pos]) / spy_p0 - 1.0
            pen_to_end[0].append(penalty)
            pen_to_end[1].append(penalty - end_spy_ret)
            mb100_pen.append(0)
            mb200_pen.append(0)
            bu50_pen.append(1)

        def _rt(xs: list[int]) -> float:
            return float(np.mean(xs)) if xs else float("nan")

        return {
            "n_qualified_matched": len(matched),
            "n_unmatched_fundamental": len(unmatched),
            "early_terminated_1y": early_term_1y,
            "early_terminated_3y": early_term_3y,
            "raw_matched": {
                "horizons": {lbl: _summarize_returns(*raw_h[lbl]) for lbl, _ in horizons},
                "to_end_or_3y": _summarize_returns(*raw_to_end),
                "multibagger_ge_100pct": _rt(mb100_raw),
                "multibagger_ge_200pct": _rt(mb200_raw),
                "blowup_le_neg50pct": _rt(bu50_raw),
            },
            "survivorship_penalized": {
                "horizons": {lbl: _summarize_returns(*pen_h[lbl]) for lbl, _ in horizons},
                "to_end_or_3y": _summarize_returns(*pen_to_end),
                "multibagger_ge_100pct": _rt(mb100_pen),
                "multibagger_ge_200pct": _rt(mb200_pen),
                "blowup_le_neg50pct": _rt(bu50_pen),
            },
        }

    per_vintage: dict[str, Any] = {}
    top5_per_vintage: dict[str, list[dict[str, Any]]] = {}
    top5_penalized_per_vintage: dict[str, list[dict[str, Any]]] = {}

    for vlabel, items in sorted(evaluated_by_vintage.items()):
        per_vintage[vlabel] = _eval_subset(items)

        # Build detailed Top-5 inspection table (Raw-Matched)
        matched = [e for e in items if e.status == "qualified_matched"][: cfg.top_n]
        t5_rows: list[dict[str, Any]] = []
        for rank, ev in enumerate(matched, 1):
            c = ev.candidate
            df = bars[c.ticker].frame
            e_ts = pd.Timestamp(ev.entry_date)
            sub = df.loc[(df.index >= e_ts) & (df.index <= end_ts)]
            p0 = float(sub["open"].iloc[0])
            spy_pos = int(spy_cal.searchsorted(e_ts, side="left"))
            spy_p0 = float(spy_df["open"].iloc[spy_pos])

            def _h_ret(h_bars: int) -> float | None:
                if spy_pos + h_bars - 1 >= len(spy_cal):
                    return None
                if len(sub) >= h_bars:
                    return float(sub["close"].iloc[h_bars - 1]) / p0 - 1.0
                return float(sub["close"].iloc[-1]) / p0 - 1.0

            ret_end = float(sub["close"].iloc[-1]) / p0 - 1.0
            spy_end = float(spy_df["close"].iloc[-1]) / spy_p0 - 1.0
            w3 = sub.iloc[:756]
            t5_rows.append({
                "vintage": vlabel,
                "rank": rank,
                "ticker": c.ticker,
                "cik": c.cik,
                "name": c.name,
                "sic": c.sic,
                "filed_date": str(c.filed_date),
                "period_date": str(c.period_date),
                "revenue_m": round(c.revenue_usd / 1e6, 1),
                "revenue_prior_m": round(c.revenue_prior_usd / 1e6, 1),
                "revenue_t2_m": round(c.revenue_t2_usd / 1e6, 1),
                "yoy_rev_growth_pct": round(c.yoy_rev_growth * 100.0, 1),
                "yoy_rev_growth_prior_pct": round(c.yoy_rev_growth_prior * 100.0, 1),
                "gross_margin_pct": round(c.gross_margin * 100.0, 1),
                "operating_margin_pct": round(c.operating_margin * 100.0, 1),
                "fcf_margin_pct": round(c.fcf_margin * 100.0, 1),
                "yoy_dilution_pct": round(c.yoy_share_dilution * 100.0, 2),
                "debt_to_equity": round(c.debt_to_equity, 2),
                "pit_market_cap_m": round((ev.pit_market_cap_usd or 0.0) / 1e6, 1),
                "asof_raw_price": round(ev.asof_raw_price or 0.0, 2),
                "rule_of_40": round(c.rule_of_40, 1),
                "entry_date": str(ev.entry_date),
                "ret_1y_pct": None if _h_ret(252) is None else round(_h_ret(252) * 100.0, 1),  # type: ignore[operator]
                "ret_2y_pct": None if _h_ret(504) is None else round(_h_ret(504) * 100.0, 1),  # type: ignore[operator]
                "ret_3y_pct": None if _h_ret(756) is None else round(_h_ret(756) * 100.0, 1),  # type: ignore[operator]
                "ret_to_end_pct": round(ret_end * 100.0, 1),
                "excess_to_end_pct": round((ret_end - spy_end) * 100.0, 1),
                "peak_gain_pct": round((float(w3["close"].max()) / p0 - 1.0) * 100.0, 1),
                "trough_dd_pct": round((float(w3["close"].min()) / p0 - 1.0) * 100.0, 1),
                "shares_tag": c.shares_tag,
                "rev_tag": c.rev_tag,
                "gp_tag": c.gp_tag,
            })
        top5_per_vintage[vlabel] = t5_rows

        # Also record which unmatched names would crack the Top 5 in Penalized mode
        pen_pool = [e for e in items if e.status in ("qualified_matched", "unmatched_yfinance")][: cfg.top_n]
        top5_penalized_per_vintage[vlabel] = [
            {
                "rank": idx,
                "ticker": e.candidate.ticker or f"CIK:{e.candidate.cik}",
                "name": e.candidate.name,
                "status": e.status,
                "rule_of_40": round(e.candidate.rule_of_40, 1),
                "yoy_rev_growth_pct": round(e.candidate.yoy_rev_growth * 100.0, 1),
                "fcf_margin_pct": round(e.candidate.fcf_margin * 100.0, 1),
            }
            for idx, e in enumerate(pen_pool, 1)
        ]

    is_items = [e for v in is_vintages for e in evaluated_by_vintage.get(v, [])]
    oos_items = [e for v in oos_vintages for e in evaluated_by_vintage.get(v, [])]
    all_items = [e for v in sorted(evaluated_by_vintage) for e in evaluated_by_vintage[v]]

    return {
        "per_vintage": per_vintage,
        "pooled": {
            "IS": _eval_subset(is_items),
            "OOS": _eval_subset(oos_items),
            "ALL": _eval_subset(all_items),
        },
        "top5_per_vintage": top5_per_vintage,
        "top5_penalized_per_vintage": top5_penalized_per_vintage,
    }


def screen_replacement_candidates(
    annual_df: pd.DataFrame,
    decision_date: date,
    bars: dict[str, BarSeries],
    ticker_meta: dict[str, dict[str, str]],
    spy_bars: BarSeries,
    cfg: CoffeeCanConfig,
    *,
    survivorship_penalized: bool = False,
) -> list[EvaluatedCandidate]:
    """Rank point-in-time qualifiers using only filings with ``filed <= decision_date``."""
    label = f"redeploy-{decision_date.isoformat()}"
    cands, _ = screen_fundamentals_asof(annual_df, decision_date, cfg, vintage_label=label)
    evaluated, _ = evaluate_vintage_candidates(cands, bars, ticker_meta, spy_bars, cfg)
    if survivorship_penalized:
        return [e for e in evaluated if e.status in ("qualified_matched", "unmatched_yfinance")]
    return [e for e in evaluated if e.status == "qualified_matched"]


def _opt_float(v: Any) -> float | None:
    if _is_missing(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _runway_months_from_fcf(fcf: float | None, cash: float | None) -> float | None:
    """Months of cash against a trailing-twelve-month free-cash-flow burn.

    Positive TTM FCF → infinite runway. Missing TTM FCF → ``None`` (no false break).
    A single quarter is never annualized here.
    """
    if fcf is None:
        return None
    if fcf > 0.0:
        return float("inf")
    if fcf < 0.0 and cash is not None and cash > 0.0:
        return (cash / abs(fcf)) * 12.0
    return None


def _has_ttm_flow_columns(quarterly_df: pd.DataFrame) -> bool:
    if quarterly_df is None or quarterly_df.empty:
        return False
    return "ytd_fcf" in quarterly_df.columns or "q_fcf" in quarterly_df.columns


def _qtrs_int(v: Any) -> int | None:
    f = _opt_float(v)
    if f is None:
        return None
    return int(round(f))


def _find_prev_ytd(
    flows: list[dict[str, Any]],
    period: pd.Timestamp,
    want_qtrs: int,
    know_by: pd.Timestamp,
) -> dict[str, Any] | None:
    """Earlier same-year YTD flow already filed by ``know_by``."""
    best: dict[str, Any] | None = None
    best_dist = 10_000
    for row in flows:
        if row["ytd_qtrs"] != want_qtrs or row["ytd_fcf"] is None or row["filed"] > know_by:
            continue
        gap = (period - row["period"]).days
        if 70 <= gap <= 120:
            dist = abs(gap - 91)
            if best is None or dist < best_dist or (dist == best_dist and row["filed"] > best["filed"]):
                best = row
                best_dist = dist
    return best


def _standalone_fcf_flows(
    q_flows: list[dict[str, Any]],
    annual_flows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Standalone quarterly FCF. Q4 is FY 10-K minus Q3 YTD when both are known."""
    standalones: list[dict[str, Any]] = []
    for row in q_flows:
        fcf: float | None = None
        if row["q_fcf"] is not None:
            fcf = row["q_fcf"]
        elif row["ytd_qtrs"] == 1 and row["ytd_fcf"] is not None:
            fcf = row["ytd_fcf"]
        elif row["ytd_qtrs"] in (2, 3) and row["ytd_fcf"] is not None:
            prev = _find_prev_ytd(q_flows, row["period"], row["ytd_qtrs"] - 1, row["filed"])
            if prev is not None:
                fcf = float(row["ytd_fcf"]) - float(prev["ytd_fcf"])
        if fcf is None:
            continue
        standalones.append({"period": row["period"], "filed": row["filed"], "fcf": fcf})

    for ann in annual_flows:
        if ann["fcf"] is None:
            continue
        prev_q3 = _find_prev_ytd(q_flows, ann["period"], 3, ann["filed"])
        if prev_q3 is None:
            continue
        standalones.append({
            "period": ann["period"],
            "filed": ann["filed"],
            "fcf": float(ann["fcf"]) - float(prev_q3["ytd_fcf"]),
        })
    return standalones


def _ttm_fcf_asof(
    standalones: list[dict[str, Any]],
    asof_period: pd.Timestamp,
    asof_filed: pd.Timestamp,
) -> float | None:
    """Sum of the last four contiguous standalone quarters known by ``asof_filed``."""
    by_period: dict[pd.Timestamp, dict[str, Any]] = {}
    for row in standalones:
        if row["filed"] > asof_filed or row["period"] > asof_period:
            continue
        prev = by_period.get(row["period"])
        if prev is None or row["filed"] >= prev["filed"]:
            by_period[row["period"]] = row
    ordered = [by_period[p] for p in sorted(by_period)]
    if len(ordered) < 4:
        return None
    last4 = ordered[-4:]
    if abs((asof_period - last4[-1]["period"]).days) > 20:
        return None
    gaps = [(last4[i]["period"] - last4[i - 1]["period"]).days for i in range(1, 4)]
    span = (last4[-1]["period"] - last4[0]["period"]).days
    if not (250 <= span <= 400 and all(60 <= g <= 130 for g in gaps)):
        return None
    return float(sum(row["fcf"] for row in last4))


def _latest_annual_fcf(
    annual_flows: list[dict[str, Any]],
    asof_filed: pd.Timestamp,
    asof_period: pd.Timestamp,
) -> float | None:
    best: dict[str, Any] | None = None
    horizon = asof_period + pd.Timedelta(days=20)
    for row in annual_flows:
        if row["fcf"] is None or row["filed"] > asof_filed or row["period"] > horizon:
            continue
        if best is None or (row["filed"], row["period"]) > (best["filed"], best["period"]):
            best = row
    return None if best is None else float(best["fcf"])


def _build_ttm_thesis_events(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Thesis events whose runway is TTM FCF, or the latest 10-K if quarters are incomplete."""
    q_groups: dict[str, list[dict[str, Any]]] = {}
    if not quarterly_df.empty:
        for cik, grp in quarterly_df.groupby(quarterly_df["cik"].astype(str), sort=False):
            flows: list[dict[str, Any]] = []
            for rec in grp.to_dict(orient="records"):
                flows.append({
                    "adsh": str(rec.get("adsh", "")),
                    "filed": pd.Timestamp(rec["filed_date"]),
                    "period": pd.Timestamp(rec["period_date"]),
                    "yoy": _opt_float(rec.get("yoy_rev_growth")),
                    "cash": _opt_float(rec.get("cash_total")),
                    "q_fcf": _opt_float(rec.get("q_fcf")),
                    "ytd_fcf": _opt_float(rec.get("ytd_fcf")),
                    "ytd_qtrs": _qtrs_int(rec.get("ytd_qtrs")),
                })
            flows.sort(key=lambda r: (r["filed"], r["period"], r["adsh"]))
            q_groups[str(cik)] = flows

    a_groups: dict[str, list[dict[str, Any]]] = {}
    if annual_df is not None and not annual_df.empty:
        for cik, grp in annual_df.groupby(annual_df["cik"].astype(str), sort=False):
            flows = []
            for rec in grp.to_dict(orient="records"):
                flows.append({
                    "adsh": str(rec.get("adsh", "")),
                    "filed": pd.Timestamp(rec["filed_date"]),
                    "period": pd.Timestamp(rec["period_date"]),
                    "yoy": _opt_float(rec.get("yoy_rev_growth")),
                    "cash": _opt_float(rec.get("cash_total")),
                    "fcf": _opt_float(rec.get("fcf")),
                })
            a_groups[str(cik)] = flows

    events_by_cik: dict[str, list[dict[str, Any]]] = {}
    for cik in set(q_groups) | set(a_groups):
        q_flows = q_groups.get(cik, [])
        a_flows = a_groups.get(cik, [])
        standalones = _standalone_fcf_flows(q_flows, a_flows)
        events: list[dict[str, Any]] = []
        for row in q_flows:
            ttm = _ttm_fcf_asof(standalones, row["period"], row["filed"])
            source = "ttm_4q"
            if ttm is None:
                ttm = _latest_annual_fcf(a_flows, row["filed"], row["period"])
                source = "annual_10k" if ttm is not None else "unavailable"
            events.append({
                "filed_date": row["filed"].date(),
                "form": "10-Q",
                "adsh": row["adsh"],
                "yoy_rev_growth": row["yoy"],
                "cash_runway_months": _runway_months_from_fcf(ttm, row["cash"]),
                "runway_source": source,
                "ttm_fcf": ttm,
            })
        for row in a_flows:
            events.append({
                "filed_date": row["filed"].date(),
                "form": "10-K",
                "adsh": row["adsh"],
                "yoy_rev_growth": row["yoy"],
                "cash_runway_months": _runway_months_from_fcf(row["fcf"], row["cash"]),
                "runway_source": "annual_10k" if row["fcf"] is not None else "unavailable",
                "ttm_fcf": row["fcf"],
            })
        events.sort(key=lambda x: (x["filed_date"], x["adsh"]))
        if events:
            events_by_cik[cik] = events
    return events_by_cik


def build_thesis_watch_events(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Build a ``filed_date`` stream of 10-Q/10-K thesis checks per ``cik``.

    When quarterly rows carry YTD / standalone cash-flow columns, cash runway is
    trailing-twelve-month FCF (last four standalone quarters, or the latest
    ``qtrs == 4`` 10-K if that breakdown is incomplete). A lone quarter is never
    annualized. Rows that already carry ``cash_runway_months`` and no flow
    columns keep that precomputed TTM figure (unit-test fixtures).
    """
    if _has_ttm_flow_columns(quarterly_df):
        return _build_ttm_thesis_events(annual_df, quarterly_df)

    events_by_cik: dict[str, list[dict[str, Any]]] = {}

    if quarterly_df is not None and not quarterly_df.empty:
        has_runway = "cash_runway_months" in quarterly_df.columns
        for row in quarterly_df.itertuples(index=False):
            f_dt = pd.Timestamp(row.filed_date).date()
            yoy = None if pd.isna(row.yoy_rev_growth) else float(row.yoy_rev_growth)
            rw_raw = getattr(row, "cash_runway_months", None) if has_runway else None
            rw = None if _is_missing(rw_raw) else float(rw_raw)
            events_by_cik.setdefault(str(row.cik), []).append({
                "filed_date": f_dt,
                "form": "10-Q",
                "adsh": str(row.adsh),
                "yoy_rev_growth": yoy,
                "cash_runway_months": rw,
                "runway_source": "provided",
            })

    if annual_df is not None and not annual_df.empty:
        for row in annual_df.itertuples(index=False):
            f_dt = pd.Timestamp(row.filed_date).date()
            yoy_raw = getattr(row, "yoy_rev_growth", None)
            yoy = None if _is_missing(yoy_raw) else float(yoy_raw)
            rw_yr = getattr(row, "cash_runway_years", np.nan)
            if _is_missing(rw_yr):
                rw = None
            else:
                rw_f = float(rw_yr)
                rw = float("inf") if math.isinf(rw_f) else rw_f * 12.0
            events_by_cik.setdefault(str(row.cik), []).append({
                "filed_date": f_dt,
                "form": "10-K",
                "adsh": str(row.adsh),
                "yoy_rev_growth": yoy,
                "cash_runway_months": rw,
                "runway_source": "annual_10k",
            })

    for cik in events_by_cik:
        events_by_cik[cik].sort(key=lambda x: (x["filed_date"], x["adsh"]))
    return events_by_cik


@dataclass
class _HeldPos:
    ev: EvaluatedCandidate
    vintage: str
    entry_year: int
    entry_date: date
    raw_qty: int
    adj_qty: float
    entry_raw_open: float
    entry_adj_open: float
    entry_comm: float
    entry_slip_usd: float
    cost_basis_usd: float
    last_adj_close: float
    bars_held: int = 0
    consec_neg_rev: int = 0
    processed_adsh: set[str] | None = None
    queued_exit_reason: ExitReason | None = None
    is_synthetic_unmatched: bool = False


def run_coffee_can_portfolio_backtest(
    evaluated_by_vintage: dict[str, list[EvaluatedCandidate]],
    rebalance_dates: dict[str, date],
    bars: dict[str, BarSeries],
    spy_bars: BarSeries,
    thesis_events_by_cik: dict[str, list[dict[str, Any]]],
    cfg: CoffeeCanConfig,
    window_start: date,
    window_end: date,
    initial_capital_usd: float,
    initial_capital_eur: float,
    window_label: str = "FULL",
    survivorship_penalized: bool = False,
    replacement_screen: Callable[[date], list[EvaluatedCandidate]] | None = None,
) -> CoffeeCanPortfolioResult:
    """Part B: Simulate ``v10a_annual_top5`` or ``v10b_coffee_can_thesis_watch`` on IBKR Cash Account.

    ``replacement_screen(decision_date)`` ranks current qualifiers using only
    information filed on or before the thesis-break decision date. v10b buys the
    top name not already held at the next trading day's Open.
    """
    spy_cal = spy_bars.frame.index.sort_values()
    cal = spy_cal[(spy_cal >= pd.Timestamp(window_start)) & (spy_cal <= pd.Timestamp(window_end))]
    if cal.empty:
        raise ValueError(f"Empty SPY calendar for {window_start}..{window_end}")

    # Map rebalance signal date (first trading day of April) -> vintage label
    reb_ts_to_vintage: dict[pd.Timestamp, str] = {}
    for vlabel, rdt in rebalance_dates.items():
        r_ts = pd.Timestamp(rdt)
        if cal[0] <= r_ts <= cal[-1]:
            reb_ts_to_vintage[r_ts] = vlabel

    settled_cash = float(initial_capital_usd)
    unsettled: list[tuple[pd.Timestamp, float]] = []
    open_positions: list[_HeldPos] = []
    closed_trades: list[CoffeeCanTradeRecord] = []
    skipped: dict[str, int] = {
        "insufficient_settled_cash": 0,
        "whole_share_too_expensive": 0,
        "no_bar_at_buy_open": 0,
    }
    total_orders = 0
    total_comm = 0.0
    total_slip = 0.0
    curve_rows: list[dict[str, Any]] = []

    slip = cfg.slippage_pct
    comm = cfg.commission_per_side_usd
    penalty_mult = 1.0 + cfg.survivorship_penalty_return

    # (buy_ts, label, targets, max_new_buys or None to fill to top_n, settlement retries left)
    pending_rebalance_buys: list[
        tuple[pd.Timestamp, str, list[EvaluatedCandidate], int | None, int]
    ] = []

    def _settle_dt(idx_i: int) -> pd.Timestamp:
        if cfg.settlement_days <= 0:
            return cal[idx_i]
        tgt = idx_i + cfg.settlement_days
        return cal[tgt] if tgt < len(cal) else cal[-1] + pd.Timedelta(days=cfg.settlement_days)

    def _close_pos(
        pos: _HeldPos,
        exit_ts: pd.Timestamp,
        raw_exit_adj_px: float,
        reason: ExitReason,
        idx_i: int,
    ) -> None:
        nonlocal settled_cash, total_orders, total_comm, total_slip
        sell_fill_adj = raw_exit_adj_px * (1.0 - slip)
        sell_slip_usd = pos.adj_qty * raw_exit_adj_px * slip
        gross_proc = pos.adj_qty * sell_fill_adj
        net_proc = max(0.0, gross_proc - comm)

        total_orders += 1
        total_comm += comm
        total_slip += sell_slip_usd

        s_dt = _settle_dt(idx_i)
        if cfg.settlement_days == 0:
            settled_cash += net_proc
        else:
            unsettled.append((s_dt, net_proc))

        gross_pnl = pos.adj_qty * (raw_exit_adj_px - pos.entry_adj_open)
        net_pnl = net_proc - pos.cost_basis_usd
        ret_pct = net_pnl / pos.cost_basis_usd if pos.cost_basis_usd > 0 else 0.0
        c = pos.ev.candidate
        closed_trades.append(
            CoffeeCanTradeRecord(
                rule=cfg.rule,
                vintage=pos.vintage,
                cik=c.cik,
                ticker=c.ticker or f"CIK:{c.cik}",
                name=c.name,
                filed_date=c.filed_date,
                entry_date=pos.entry_date,
                exit_date=exit_ts.date(),
                rule_of_40=c.rule_of_40,
                yoy_rev_growth=c.yoy_rev_growth,
                gross_margin=c.gross_margin,
                fcf_margin=c.fcf_margin,
                pit_market_cap_usd=pos.ev.pit_market_cap_usd,
                raw_qty=pos.raw_qty,
                adj_qty=pos.adj_qty,
                entry_raw_price=pos.entry_raw_open,
                entry_adj_price=pos.entry_adj_open,
                exit_adj_price=sell_fill_adj,
                bars_held=max(1, pos.bars_held),
                cost_basis_usd=pos.cost_basis_usd,
                gross_pnl_usd=gross_pnl,
                commission_usd=pos.entry_comm + comm,
                slippage_usd=pos.entry_slip_usd + sell_slip_usd,
                net_pnl_usd=net_pnl,
                return_pct=ret_pct,
                exit_reason=reason,
                is_open_at_end=False,
                is_synthetic_penalty=pos.is_synthetic_unmatched or (reason == "DELISTED_PENALIZED_50"),
            )
        )

    def _buy_pos(
        ev: EvaluatedCandidate,
        vlabel: str,
        buy_ts: pd.Timestamp,
        budget: float,
    ) -> _HeldPos | None:
        nonlocal settled_cash, total_orders, total_comm, total_slip
        is_synth = ev.status == "unmatched_yfinance"
        if is_synth:
            raw_open = 25.0  # Representative mid-band price for synthetic unmatched allocation
            adj_open = 25.0
            split_mult = 1.0
        else:
            df_s = bars[ev.candidate.ticker].frame
            if buy_ts not in df_s.index:
                skipped["no_bar_at_buy_open"] += 1
                return None
            row = df_s.loc[buy_ts]
            adj_open = float(row["open"])
            adj_close = float(row["close"])
            raw_close = float(row["raw_close"]) if "raw_close" in df_s.columns else adj_close
            split_mult = (raw_close / adj_close) if (adj_close > 0 and raw_close > 0) else 1.0
            raw_open = adj_open * split_mult

        if raw_open <= 0 or adj_open <= 0:
            return None
        buy_fill_raw = raw_open * (1.0 + slip)
        eff_budget = min(budget, settled_cash)
        if eff_budget <= comm:
            skipped["insufficient_settled_cash"] += 1
            return None
        raw_qty = int(math.floor((eff_budget - comm) / buy_fill_raw))
        if raw_qty < 1:
            if settled_cash < buy_fill_raw + comm:
                skipped["insufficient_settled_cash"] += 1
            else:
                skipped["whole_share_too_expensive"] += 1
            return None

        adj_qty = raw_qty * split_mult
        cost_basis = raw_qty * buy_fill_raw + comm
        buy_slip = raw_qty * raw_open * slip
        settled_cash -= cost_basis
        total_orders += 1
        total_comm += comm
        total_slip += buy_slip

        return _HeldPos(
            ev=ev,
            vintage=vlabel,
            entry_year=buy_ts.year,
            entry_date=buy_ts.date(),
            raw_qty=raw_qty,
            adj_qty=adj_qty,
            entry_raw_open=raw_open,
            entry_adj_open=adj_open,
            entry_comm=comm,
            entry_slip_usd=buy_slip,
            cost_basis_usd=cost_basis,
            last_adj_close=adj_open,
            processed_adsh=set(),
            is_synthetic_unmatched=is_synth,
        )

    for i, ts in enumerate(cal):
        # 1. T+1 settlement before Open
        if unsettled:
            still_pend: list[tuple[pd.Timestamp, float]] = []
            for s_dt, amt in unsettled:
                if s_dt <= ts:
                    settled_cash += amt
                else:
                    still_pend.append((s_dt, amt))
            unsettled = still_pend

        # 2. Execute queued exits at Open
        survivors: list[_HeldPos] = []
        for pos in open_positions:
            if pos.queued_exit_reason is None:
                survivors.append(pos)
                continue
            if pos.is_synthetic_unmatched:
                _close_pos(pos, ts, pos.last_adj_close, pos.queued_exit_reason, i)
                continue
            df_s = bars[pos.ev.candidate.ticker].frame
            if ts in df_s.index:
                _close_pos(pos, ts, float(df_s.loc[ts, "open"]), pos.queued_exit_reason, i)
            else:
                # Bar missing on exit day -> exit at last known close
                _close_pos(pos, ts, pos.last_adj_close, pos.queued_exit_reason, i)
        open_positions = survivors

        # 3. Execute queued rebalance / redeploy buys at Open.
        #    Sale proceeds settle T+1, so a same-morning redeploy spends cash that
        #    was already settled. Anything still unfilled retries once that cash settles.
        if pending_rebalance_buys:
            next_pend_buys: list[tuple[pd.Timestamp, str, list[EvaluatedCandidate], int | None, int]] = []
            for buy_ts, vlabel, targets, max_new, retries_left in pending_rebalance_buys:
                if buy_ts > ts:
                    next_pend_buys.append((buy_ts, vlabel, targets, max_new, retries_left))
                    continue
                held_ciks = {p.ev.candidate.cik for p in open_positions}
                held_tickers = {p.ev.candidate.ticker for p in open_positions if p.ev.candidate.ticker}
                open_mtm = sum(p.adj_qty * p.last_adj_close for p in open_positions)
                settled_equity = settled_cash + open_mtm
                room = max(0, cfg.top_n - len(open_positions))
                buy_cap = room if max_new is None else min(room, max_new)
                bought = 0
                if buy_cap > 0:
                    slot_budget = cfg.target_slot_pct * settled_equity
                    for cand_ev in targets:
                        if bought >= buy_cap or len(open_positions) >= cfg.top_n:
                            break
                        if cand_ev.candidate.cik in held_ciks or (
                            cand_ev.candidate.ticker and cand_ev.candidate.ticker in held_tickers
                        ):
                            continue
                        new_pos = _buy_pos(cand_ev, vlabel, ts, slot_budget)
                        if new_pos is not None:
                            open_positions.append(new_pos)
                            held_ciks.add(cand_ev.candidate.cik)
                            if cand_ev.candidate.ticker:
                                held_tickers.add(cand_ev.candidate.ticker)
                            bought += 1
                still_needed = 0 if max_new is None else max(0, max_new - bought)
                if still_needed > 0 and retries_left > 0 and cfg.settlement_days > 0 and i + cfg.settlement_days < len(cal):
                    next_pend_buys.append(
                        (cal[i + cfg.settlement_days], vlabel, targets, still_needed, retries_left - 1)
                    )
            pending_rebalance_buys = next_pend_buys

        # 4. Close MTM, early delisting check, and v10b causal thesis-break check
        survivors_close: list[_HeldPos] = []
        thesis_break_dates: list[date] = []
        for pos in open_positions:
            if pos.is_synthetic_unmatched:
                pos.bars_held += 1
                frac = min(1.0, pos.bars_held / float(cfg.penalized_unmatched_hold_days))
                pos.last_adj_close = pos.entry_adj_open * (1.0 + cfg.survivorship_penalty_return * frac)
                if pos.bars_held >= cfg.penalized_unmatched_hold_days:
                    _close_pos(pos, ts, pos.entry_adj_open * penalty_mult, "UNMATCHED_PENALIZED_50", i)
                else:
                    survivors_close.append(pos)
                continue

            df_s = bars[pos.ev.candidate.ticker].frame
            if ts in df_s.index:
                pos.bars_held += 1
                close_px = float(df_s.loc[ts, "close"])
                pos.last_adj_close = close_px

                # Check early delisting strictly before window end
                if ts == df_s.index[-1] and (cal[-1] - ts).days >= 15:
                    if survivorship_penalized:
                        _close_pos(pos, ts, pos.entry_adj_open * penalty_mult, "DELISTED_PENALIZED_50", i)
                    else:
                        _close_pos(pos, ts, close_px, "DELISTED_LAST_BAR", i)
                    continue

            # v10b: Fundamental thesis-break monitoring strictly on filing_date <= ts
            if cfg.rule == "v10b_coffee_can_thesis_watch" and pos.queued_exit_reason is None:
                t_events = thesis_events_by_cik.get(pos.ev.candidate.cik, [])
                if pos.processed_adsh is None:
                    pos.processed_adsh = set()
                for tev in t_events:
                    fdt = tev["filed_date"]
                    if fdt < pos.entry_date or fdt > ts.date():
                        continue
                    adsh = tev["adsh"]
                    if adsh in pos.processed_adsh:
                        continue
                    pos.processed_adsh.add(adsh)
                    yoy = tev["yoy_rev_growth"]
                    rw = tev["cash_runway_months"]
                    if yoy is not None:
                        if yoy < 0.0:
                            pos.consec_neg_rev += 1
                        else:
                            pos.consec_neg_rev = 0
                    if pos.consec_neg_rev >= cfg.thesis_break_consec_neg_rev:
                        pos.queued_exit_reason = "THESIS_BREAK_REV_DECLINE"
                        thesis_break_dates.append(fdt)
                        break
                    if rw is not None and rw < cfg.thesis_break_min_runway_months:
                        pos.queued_exit_reason = "THESIS_BREAK_RUNWAY"
                        thesis_break_dates.append(fdt)
                        break

            survivors_close.append(pos)
        open_positions = survivors_close

        # 4b. v10b: redeploy each thesis-break slot at the next open.
        #     Ranking is frozen on the filing date (decision date), so a filing
        #     that arrives on the exit morning cannot enter the list.
        if (
            thesis_break_dates
            and replacement_screen is not None
            and cfg.rule == "v10b_coffee_can_thesis_watch"
            and i + 1 < len(cal)
        ):
            decision = max(thesis_break_dates)
            held_ciks = {p.ev.candidate.cik for p in open_positions}
            held_tickers = {p.ev.candidate.ticker for p in open_positions if p.ev.candidate.ticker}
            ranked = [
                ev for ev in replacement_screen(decision)
                if ev.candidate.cik not in held_ciks
                and (not ev.candidate.ticker or ev.candidate.ticker not in held_tickers)
            ]
            if ranked:
                pending_rebalance_buys.append(
                    (cal[i + 1], f"redeploy-{decision.isoformat()}", ranked, len(thesis_break_dates), 1)
                )

        # 5. Annual April Rebalance Signal Check (evaluated at close of first trading day of April)
        if ts in reb_ts_to_vintage and i + 1 < len(cal):
            vlabel = reb_ts_to_vintage[ts]
            v_items = evaluated_by_vintage.get(vlabel, [])
            if survivorship_penalized:
                pool = [e for e in v_items if e.status in ("qualified_matched", "unmatched_yfinance")]
            else:
                pool = [e for e in v_items if e.status == "qualified_matched"]
            top_targets = pool[: cfg.top_n]
            top_ciks = {e.candidate.cik for e in top_targets}

            will_sell_any = False
            if cfg.rule == "v10a_annual_top5":
                # Rebalance cleanly into the current Top 5: exit all current holdings on next Open,
                # settle T+1, and buy fresh equal-weight Top 5 on t+2 Open (exact 20% equal weight!)
                for pos in open_positions:
                    if pos.queued_exit_reason is None:
                        pos.queued_exit_reason = "ANNUAL_REBALANCE_EXIT"
                    will_sell_any = True
            else:  # v10b_coffee_can_thesis_watch
                # Hold winners up to 3 years; exit ONLY positions reaching 3Y maturity
                for pos in open_positions:
                    if (ts.year - pos.entry_year) >= cfg.max_hold_years:
                        if pos.queued_exit_reason is None:
                            pos.queued_exit_reason = "COFFEE_CAN_3Y_MATURITY"
                        will_sell_any = True
                # Expand candidate list beyond top_n if needed to replace exited slots without duplicating held names
                top_targets = pool[: cfg.top_n + len(open_positions)]

            # Schedule buys: if sells occur on t+1 Open, settle T+1 and buy on t+2 Open;
            # else buy immediately on t+1 Open!
            if will_sell_any and cfg.settlement_days > 0:
                buy_idx = min(i + 1 + cfg.settlement_days, len(cal) - 1)
            else:
                buy_idx = i + 1
            pending_rebalance_buys.append((cal[buy_idx], vlabel, top_targets, None, 0))

        # 6. Record daily equity curve
        unsettled_tot = sum(amt for _, amt in unsettled)
        pos_val = sum(p.adj_qty * p.last_adj_close for p in open_positions)
        tot_eq = settled_cash + unsettled_tot + pos_val
        curve_rows.append({
            "date": ts,
            "equity": tot_eq,
            "settled_cash": settled_cash,
            "unsettled_cash": unsettled_tot,
            "positions_value": pos_val,
            "open_positions": len(open_positions),
            "exposure_pct": pos_val / tot_eq if tot_eq > 0 else 0.0,
        })

    # Record open positions at end (MTM)
    end_ts = cal[-1]
    end_mtm: list[CoffeeCanTradeRecord] = []
    for pos in open_positions:
        gross_pnl = pos.adj_qty * (pos.last_adj_close - pos.entry_adj_open)
        mtm_val = pos.adj_qty * pos.last_adj_close
        net_pnl = mtm_val - pos.cost_basis_usd
        ret_pct = net_pnl / pos.cost_basis_usd if pos.cost_basis_usd > 0 else 0.0
        c = pos.ev.candidate
        end_mtm.append(
            CoffeeCanTradeRecord(
                rule=cfg.rule,
                vintage=pos.vintage,
                cik=c.cik,
                ticker=c.ticker or f"CIK:{c.cik}",
                name=c.name,
                filed_date=c.filed_date,
                entry_date=pos.entry_date,
                exit_date=end_ts.date(),
                rule_of_40=c.rule_of_40,
                yoy_rev_growth=c.yoy_rev_growth,
                gross_margin=c.gross_margin,
                fcf_margin=c.fcf_margin,
                pit_market_cap_usd=pos.ev.pit_market_cap_usd,
                raw_qty=pos.raw_qty,
                adj_qty=pos.adj_qty,
                entry_raw_price=pos.entry_raw_open,
                entry_adj_price=pos.entry_adj_open,
                exit_adj_price=pos.last_adj_close,
                bars_held=max(1, pos.bars_held),
                cost_basis_usd=pos.cost_basis_usd,
                gross_pnl_usd=gross_pnl,
                commission_usd=pos.entry_comm,
                slippage_usd=pos.entry_slip_usd,
                net_pnl_usd=net_pnl,
                return_pct=ret_pct,
                exit_reason="OPEN_MTM",
                is_open_at_end=True,
                is_synthetic_penalty=pos.is_synthetic_unmatched,
            )
        )

    eq_df = pd.DataFrame(curve_rows).set_index("date")
    return CoffeeCanPortfolioResult(
        config=cfg,
        initial_capital_usd=initial_capital_usd,
        initial_capital_eur=initial_capital_eur,
        window_label=window_label,
        survivorship_penalized=survivorship_penalized,
        equity_curve=eq_df,
        trades=closed_trades,
        open_positions_at_end=end_mtm,
        skipped_counts=skipped,
        total_orders=total_orders,
        total_commission_usd=total_comm,
        total_slippage_usd=total_slip,
    )


def compute_coffee_can_metrics(
    res: CoffeeCanPortfolioResult,
    mcfg: MetricsConfig,
    standalone_trials_n: int = 10,
    cumulative_trials_n: int = 50,
    cumulative_trials_conservative_n: int = 70,
) -> dict[str, Any]:
    eq = res.equity_curve["equity"]
    init = res.initial_capital_usd
    r = daily_returns(eq, init)

    days = (eq.index[-1] - eq.index[0]).days + 1
    years = max(days / 365.25, 1e-6)
    cagr_val = cagr(eq, init)
    mdd_val = max_drawdown(eq, init)
    calmar_val = (cagr_val / abs(mdd_val)) if (not math.isnan(cagr_val) and mdd_val < 0) else float("nan")

    all_trips = list(res.trades) + list(res.open_positions_at_end)
    pnl_arr = np.array([t.net_pnl_usd for t in all_trips], dtype=float) if all_trips else np.array([], dtype=float)
    wins = pnl_arr > 0.0
    losses = pnl_arr <= 0.0
    gross_win = float(pnl_arr[wins].sum()) if wins.any() else 0.0
    gross_loss = float(-pnl_arr[losses].sum()) if losses.any() else 0.0
    pf = float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan"))

    exit_breakdown: dict[str, int] = {}
    for t in all_trips:
        exit_breakdown[t.exit_reason] = exit_breakdown.get(t.exit_reason, 0) + 1

    total_friction = res.total_commission_usd + res.total_slippage_usd

    return {
        "initial_capital_usd": init,
        "initial_capital_eur": res.initial_capital_eur,
        "final_equity_usd": float(eq.iloc[-1]),
        "final_equity_eur": float(eq.iloc[-1]) * (res.initial_capital_eur / init),
        "total_return": float(eq.iloc[-1] / init - 1.0),
        "cagr": cagr_val,
        "sharpe": sharpe(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "sortino": sortino(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "dsr_n10": deflated_sharpe(r, US_PERIODS_PER_YEAR, standalone_trials_n, mcfg.dsr_trial_sharpe_std),
        "dsr_n50": deflated_sharpe(r, US_PERIODS_PER_YEAR, cumulative_trials_n, mcfg.dsr_trial_sharpe_std),
        "dsr_n70": deflated_sharpe(r, US_PERIODS_PER_YEAR, cumulative_trials_conservative_n, mcfg.dsr_trial_sharpe_std),
        "max_drawdown": mdd_val,
        "calmar": calmar_val,
        "avg_exposure_pct": float(res.equity_curve["exposure_pct"].mean()),
        "total_orders": int(res.total_orders),
        "orders_per_year": float(res.total_orders / years),
        "closed_trades": len(res.trades),
        "open_mtm_trades": len(res.open_positions_at_end),
        "win_rate_incl_open": float(wins.mean()) if len(pnl_arr) > 0 else float("nan"),
        "profit_factor_incl_open": pf,
        "total_commission_usd": float(res.total_commission_usd),
        "total_slippage_usd": float(res.total_slippage_usd),
        "total_friction_usd": float(total_friction),
        "friction_pct_of_initial": float(total_friction / init),
        "skipped_counts": dict(res.skipped_counts),
        "exit_breakdown": exit_breakdown,
    }
