"""Strategy v9_form4 (Path 2, Stage 1): SEC Form 4 C-Suite Open-Market Cluster Buying.

Deterministic, zero-LLM event study and concentrated IBKR Cash Account portfolio
backtester with strict anti-bias controls:

- **Zero Look-Ahead Timestamp Rule:** Signals trigger strictly on ``FILING_DATE``
  (after-close 17:30 ET) and execute at the **next trading day's Open**
  (``FILING_DATE + 1 trading day``). ``TRANS_DATE`` is NEVER used for signal timing.
- **Strictly Causal Liquidity Filter:** US common stock (ETFs/CEFs/SPACs/funds
  excluded), as-of price in ``[$3, $150]``, 20-day avg dollar volume ``>= $1M``,
  all computed strictly on trading days ``<= FILING_DATE``.
- **Two Locked Signal Variants:**
  - ``v9a_standard``: ``>= 2`` distinct insiders (``primary_owner_cik`` across
    distinct filings) within 14 calendar days ending on ``FILING_DATE``, combined
    open-market spend ``>= $100,000``, at least one Officer or Director.
  - ``v9b_high_conviction``: ``>= 3`` distinct insiders (OR ``>= 2`` insiders
    including CEO/CFO/President in ``RPTOWNER_TITLE``) within 14 calendar days,
    combined spend ``>= $250,000``.
- **60-Calendar-Day Issuer Cooldown:** Duplicate cluster triggers on the same
  ``issuer_cik`` within 60 calendar days of an active trigger are ignored.
- **Honest Survivorship Audit & Penalized Sensitivity:** Explicitly tracks every
  qualifying cluster issuer missing from ``yfinance`` or terminating early before
  ``+90d``, and runs a Survivorship-Penalized Sensitivity assigning ``-50%`` loss
  to unmatched/early-delisted events (not ending in a verified cash buyout) in BOTH
  Part A (Event Study) and Part B (Portfolio Simulation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
from src.data.sec_form4_loader import is_allowed_us_common_stock
from src.domain.models import BarSeries
from src.strategy.indicators import wilder_atr

US_PERIODS_PER_YEAR = 252

VariantName = Literal["v9a_standard", "v9b_high_conviction"]
ExitReason = Literal[
    "HARD_STOP_GAP",
    "HARD_STOP_INTRADAY",
    "TRAILING_STOP_EXIT",
    "TIME_EXIT_90D",
    "DELISTED_LAST_BAR",
    "DELISTED_PENALIZED_50",
    "UNMATCHED_PENALIZED_50",
    "OPEN_MTM",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Form4StrategyConfig(_Frozen):
    """Locked configuration for Strategy v9_form4."""

    variant: VariantName = "v9a_standard"
    cluster_window_days: int = Field(14, ge=1, le=90)
    cooldown_days: int = Field(60, ge=0, le=365)

    # Variant thresholds (automatically validated / set via factory helpers)
    min_insiders: int = Field(2, ge=1)
    min_combined_spend_usd: float = Field(100_000.0, gt=0.0)
    high_conviction_mode: bool = False  # If True: >=3 insiders OR (>=2 incl C-suite)

    # Causal liquidity & tradability filter (evaluated strictly on bars <= FILING_DATE)
    min_price_usd: float = Field(3.0, gt=0.0)
    max_price_usd: float = Field(150.0, gt=0.0)
    adv_window_days: int = Field(20, ge=5)
    min_adv_usd: float = Field(1_000_000.0, ge=0.0)
    max_entry_lag_calendar_days: int = Field(7, ge=1, le=30)

    # Portfolio & execution (IBKR Cash Account, whole shares, T+1, EUR 0/mo hosting)
    max_slots: int = Field(4, ge=1, le=20)
    target_slot_pct: float = Field(0.25, gt=0.0, le=1.0)
    slippage_pct: float = Field(0.0025, ge=0.0, lt=0.1)
    commission_per_side_usd: float = Field(0.38, ge=0.0)
    settlement_days: int = Field(1, ge=0, le=5)

    # Exits
    hold_trading_days: int = Field(90, ge=1)
    hard_stop_pct: float = Field(0.25, gt=0.0, lt=1.0)
    trail_activation_gain_pct: float = Field(0.30, gt=0.0)
    trail_atr_mult: float = Field(3.5, gt=0.0)
    atr_window: int = Field(14, ge=2)

    # Survivorship sensitivity
    survivorship_penalty_return: float = Field(-0.50, ge=-0.99, le=0.0)
    penalized_unmatched_hold_days: int = Field(30, ge=1)


def make_v9a_config(**overrides: Any) -> Form4StrategyConfig:
    defaults = dict(
        variant="v9a_standard",
        min_insiders=2,
        min_combined_spend_usd=100_000.0,
        high_conviction_mode=False,
    )
    defaults.update(overrides)
    return Form4StrategyConfig(**defaults)


def make_v9b_config(**overrides: Any) -> Form4StrategyConfig:
    defaults = dict(
        variant="v9b_high_conviction",
        min_insiders=2,
        min_combined_spend_usd=250_000.0,
        high_conviction_mode=True,
    )
    defaults.update(overrides)
    return Form4StrategyConfig(**defaults)


class ClusterSignal(BaseModel):
    """One deduplicated Form 4 open-market cluster trigger on a specific ``FILING_DATE``."""

    model_config = ConfigDict(frozen=True)

    variant: VariantName
    issuer_cik: str
    ticker: str
    issuer_name: str
    filing_date: date
    earliest_trans_date: date | None
    latest_trans_date: date | None
    distinct_insiders: int
    distinct_accessions: int
    combined_dollar_spend: float
    weighted_avg_form4_price: float
    has_officer: bool
    has_director: bool
    has_c_suite: bool
    insiders_summary: str
    accession_numbers: tuple[str, ...]


class EvaluatedCluster(BaseModel):
    """Cluster signal enriched with strictly causal tradability & survivorship status."""

    model_config = ConfigDict(frozen=True)

    signal: ClusterSignal
    status: Literal[
        "qualified_matched",
        "unmatched_yfinance",
        "rejected_non_common",
        "rejected_insufficient_history",
        "rejected_price",
        "rejected_adv",
        "rejected_no_entry_bar",
    ]
    rejection_detail: str = ""
    entry_date: date | None = None
    entry_raw_open: float | None = None
    asof_price: float | None = None
    asof_adv20: float | None = None
    form4_price_eligible: bool = False
    early_terminated_before_90d: bool = False
    last_bar_date: date | None = None
    verified_cash_buyout: bool = False


class Form4TradeRecord(BaseModel):
    """Completed or end-of-window mark-to-market position record."""

    model_config = ConfigDict(frozen=True)

    variant: VariantName
    issuer_cik: str
    ticker: str
    issuer_name: str
    filing_date: date
    entry_date: date
    exit_date: date
    distinct_insiders: int
    has_c_suite: bool
    combined_dollar_spend: float
    insiders_summary: str
    qty: int
    entry_raw_price: float
    entry_fill_price: float
    hard_stop_price: float
    exit_raw_price: float
    exit_fill_price: float
    bars_held: int
    peak_gain_pct: float
    trailing_activated: bool
    cost_basis_usd: float
    gross_pnl_usd: float
    commission_usd: float
    slippage_usd: float
    net_pnl_usd: float
    return_pct: float
    exit_reason: ExitReason
    is_open_at_end: bool = False
    is_synthetic_penalty: bool = False


class Form4PortfolioResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Form4StrategyConfig
    initial_capital_usd: float
    initial_capital_eur: float
    window_label: str
    survivorship_penalized: bool
    equity_curve: pd.DataFrame
    trades: list[Form4TradeRecord]
    open_positions_at_end: list[Form4TradeRecord]
    skipped_counts: dict[str, int]
    total_orders: int
    total_commission_usd: float
    total_slippage_usd: float


def detect_form4_clusters(
    accessions_df: pd.DataFrame,
    cfg: Form4StrategyConfig,
    exclude_non_common_sec_names: bool = True,
) -> tuple[list[ClusterSignal], dict[str, int]]:
    """Detect strictly causal Form 4 open-market cluster buys per ``issuer_cik``.

    Uses ONLY ``filing_date`` (never ``trans_date``) for rolling 14-calendar-day
    windows and 60-calendar-day per-issuer cooldowns.
    """
    stats_out = {
        "input_accessions": int(len(accessions_df)),
        "excluded_non_common_sec_name_accessions": 0,
        "raw_threshold_hits": 0,
        "suppressed_by_cooldown": 0,
        "emitted_clusters": 0,
    }
    if accessions_df.empty:
        return [], stats_out

    df = accessions_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.normalize()
    if exclude_non_common_sec_names and "is_non_common_name" in df.columns:
        non_c = df["is_non_common_name"].astype(bool)
        stats_out["excluded_non_common_sec_name_accessions"] = int(non_c.sum())
        df = df[~non_c].copy()

    if df.empty:
        return [], stats_out

    df = df.sort_values(["issuer_cik", "filing_date", "accession_number"], kind="mergesort")
    signals: list[ClusterSignal] = []
    win_delta = np.timedelta64(cfg.cluster_window_days, "D")

    for issuer_cik, grp in df.groupby("issuer_cik", sort=False):
        if len(grp) < cfg.min_insiders:
            continue
        f_dates = grp["filing_date"].to_numpy()
        owners = grp["primary_owner_cik"].to_numpy()
        spends = grp["total_dollar_spend"].to_numpy(dtype=float)
        officers = grp["is_officer"].to_numpy(dtype=bool)
        directors = grp["is_director"].to_numpy(dtype=bool)
        csuites = grp["is_c_suite"].to_numpy(dtype=bool)
        n_rows = len(f_dates)
        last_trigger: pd.Timestamp | None = None

        left = 0
        right = 0
        while right < n_rows:
            d_np = f_dates[right]
            # Advance right to include all rows with exact filing_date == d_np
            while right + 1 < n_rows and f_dates[right + 1] == d_np:
                right += 1
            # Advance left so f_dates[left] >= d_np - win_delta
            cutoff = d_np - win_delta
            while left <= right and f_dates[left] < cutoff:
                left += 1

            # Fast pre-checks before slicing DataFrame
            sl = slice(left, right + 1)
            combined_spend = float(spends[sl].sum())
            if combined_spend >= cfg.min_combined_spend_usd:
                has_officer = bool(officers[sl].any())
                has_director = bool(directors[sl].any())
                if has_officer or has_director:
                    distinct_insiders = len(set(owners[sl]))
                    has_c_suite = bool(csuites[sl].any())
                    if cfg.high_conviction_mode:
                        qualifies = (distinct_insiders >= 3) or (distinct_insiders >= 2 and has_c_suite)
                    else:
                        qualifies = distinct_insiders >= cfg.min_insiders

                    if qualifies:
                        stats_out["raw_threshold_hits"] += 1
                        d_ts = pd.Timestamp(d_np)
                        if last_trigger is not None and (d_ts - last_trigger).days <= cfg.cooldown_days:
                            stats_out["suppressed_by_cooldown"] += 1
                        else:
                            last_trigger = d_ts
                            w = grp.iloc[sl]
                            latest_row = w.iloc[-1]
                            tot_shares = float(w["total_shares"].sum())
                            w_avg_px = (
                                combined_spend / tot_shares
                                if tot_shares > 0
                                else float(latest_row["weighted_avg_price"])
                            )

                            per_insider = (
                                w.groupby("primary_owner_cik", as_index=False)
                                .agg(
                                    name=("primary_owner_name", "first"),
                                    title=("owner_title", lambda s: " / ".join(sorted({x for x in s if x}))),
                                    rel=("owner_relationship", "first"),
                                    spend=("total_dollar_spend", "sum"),
                                )
                                .sort_values("spend", ascending=False)
                            )
                            parts = []
                            for _, r in per_insider.iterrows():
                                role = r["title"] if r["title"] else r["rel"]
                                parts.append(f"{r['name']} ({role}, ${r['spend']:,.0f})")
                            insiders_str = "; ".join(parts)

                            e_td = pd.to_datetime(w["earliest_trans_date"], errors="coerce").min()
                            l_td = pd.to_datetime(w["latest_trans_date"], errors="coerce").max()

                            signals.append(
                                ClusterSignal(
                                    variant=cfg.variant,
                                    issuer_cik=str(issuer_cik),
                                    ticker=str(latest_row["ticker"]),
                                    issuer_name=str(latest_row["issuer_name"]),
                                    filing_date=d_ts.date(),
                                    earliest_trans_date=None if pd.isna(e_td) else e_td.date(),
                                    latest_trans_date=None if pd.isna(l_td) else l_td.date(),
                                    distinct_insiders=distinct_insiders,
                                    distinct_accessions=int(w["accession_number"].nunique()),
                                    combined_dollar_spend=combined_spend,
                                    weighted_avg_form4_price=w_avg_px,
                                    has_officer=has_officer,
                                    has_director=has_director,
                                    has_c_suite=has_c_suite,
                                    insiders_summary=insiders_str,
                                    accession_numbers=tuple(sorted(w["accession_number"].astype(str).unique())),
                                )
                            )
            right += 1

    signals.sort(key=lambda s: (s.filing_date, -s.combined_dollar_spend, -s.distinct_insiders, s.ticker))
    stats_out["emitted_clusters"] = len(signals)
    return signals, stats_out


def evaluate_clusters_and_survivorship(
    clusters: list[ClusterSignal],
    bars: dict[str, BarSeries],
    ticker_meta: dict[str, dict[str, str]],
    spy_bars: BarSeries,
    sample_end_date: date,
    cfg: Form4StrategyConfig,
    verified_cash_buyout_ciks: set[str] | None = None,
) -> tuple[list[EvaluatedCluster], dict[str, Any]]:
    """Apply strictly causal tradability filters and produce a full survivorship audit.

    Every price and 20-day dollar volume check uses strictly bars with ``date <= FILING_DATE``.
    Entry is strictly ``FILING_DATE + 1 trading day`` Open.
    """
    buyouts = verified_cash_buyout_ciks or set()
    spy_cal = spy_bars.frame.index.sort_values()
    end_ts = pd.Timestamp(sample_end_date)

    evaluated: list[EvaluatedCluster] = []
    counts: dict[str, int] = {
        "total_clusters": len(clusters),
        "qualified_matched": 0,
        "unmatched_yfinance": 0,
        "unmatched_price_eligible": 0,
        "rejected_non_common": 0,
        "rejected_insufficient_history": 0,
        "rejected_price": 0,
        "rejected_adv": 0,
        "rejected_no_entry_bar": 0,
        "early_terminated_before_90d": 0,
        "early_terminated_verified_buyout": 0,
        "early_terminated_penalized": 0,
    }
    non_common_breakdown: dict[str, int] = {}

    for sig in clusters:
        f_ts = pd.Timestamp(sig.filing_date)
        meta = ticker_meta.get(sig.ticker)
        allowed, reason = is_allowed_us_common_stock(sig.issuer_name, sig.ticker, meta)
        form4_px_ok = cfg.min_price_usd <= sig.weighted_avg_form4_price <= cfg.max_price_usd

        if not allowed:
            counts["rejected_non_common"] += 1
            non_common_breakdown[reason] = non_common_breakdown.get(reason, 0) + 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="rejected_non_common",
                    rejection_detail=reason,
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        # Next SPY trading day strictly after FILING_DATE
        spy_pos = int(spy_cal.searchsorted(f_ts, side="right"))
        next_spy_ts = spy_cal[spy_pos] if spy_pos < len(spy_cal) else None

        series = bars.get(sig.ticker)
        if series is None or series.frame.empty:
            counts["unmatched_yfinance"] += 1
            if form4_px_ok:
                counts["unmatched_price_eligible"] += 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="unmatched_yfinance",
                    rejection_detail="no_yfinance_bars",
                    entry_date=None if next_spy_ts is None else next_spy_ts.date(),
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        df = series.frame
        hist = df.loc[df.index <= f_ts]
        future = df.loc[df.index > f_ts]

        # If ticker has bars ONLY far away from filing_date (e.g. reused symbol years later or delisted before)
        if hist.empty or (f_ts - hist.index[-1]).days > 14:
            # Check if ticker actually existed around filing_date at all
            if future.empty or (future.index[0] - f_ts).days > 14:
                counts["unmatched_yfinance"] += 1
                if form4_px_ok:
                    counts["unmatched_price_eligible"] += 1
                evaluated.append(
                    EvaluatedCluster(
                        signal=sig,
                        status="unmatched_yfinance",
                        rejection_detail="no_bars_around_filing_date",
                        entry_date=None if next_spy_ts is None else next_spy_ts.date(),
                        form4_price_eligible=form4_px_ok,
                    )
                )
                continue

        if len(hist) < cfg.adv_window_days:
            counts["rejected_insufficient_history"] += 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="rejected_insufficient_history",
                    rejection_detail=f"bars_le_filing={len(hist)}",
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        # Strictly causal price and 20-day avg dollar volume on trading days <= FILING_DATE
        price_col = "raw_close" if "raw_close" in hist.columns else "close"
        asof_price = float(hist[price_col].iloc[-1])
        tail_w = hist.iloc[-cfg.adv_window_days:]
        asof_adv20 = float((tail_w["close"] * tail_w["volume"]).mean())

        if not (cfg.min_price_usd <= asof_price <= cfg.max_price_usd):
            counts["rejected_price"] += 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="rejected_price",
                    rejection_detail=f"asof_price={asof_price:.2f}",
                    asof_price=asof_price,
                    asof_adv20=asof_adv20,
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        if asof_adv20 < cfg.min_adv_usd:
            counts["rejected_adv"] += 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="rejected_adv",
                    rejection_detail=f"adv20={asof_adv20:.0f}",
                    asof_price=asof_price,
                    asof_adv20=asof_adv20,
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        if future.empty:
            # Ticker disappeared right after filing_date before next trading day!
            counts["unmatched_yfinance"] += 1
            if form4_px_ok:
                counts["unmatched_price_eligible"] += 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="unmatched_yfinance",
                    rejection_detail="no_post_filing_bars",
                    entry_date=None if next_spy_ts is None else next_spy_ts.date(),
                    asof_price=asof_price,
                    asof_adv20=asof_adv20,
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        entry_ts = future.index[0]
        if (entry_ts - f_ts).days > cfg.max_entry_lag_calendar_days:
            counts["rejected_no_entry_bar"] += 1
            evaluated.append(
                EvaluatedCluster(
                    signal=sig,
                    status="rejected_no_entry_bar",
                    rejection_detail=f"entry_lag_days={(entry_ts - f_ts).days}",
                    asof_price=asof_price,
                    asof_adv20=asof_adv20,
                    form4_price_eligible=form4_px_ok,
                )
            )
            continue

        entry_open = float(future["open"].iloc[0])
        last_bar_ts = df.index[-1]

        # Determine if bars terminate early before +90 trading days (not merely right-censored by sample_end)
        e_spy_pos = int(spy_cal.searchsorted(entry_ts, side="left"))
        target_90_pos = min(e_spy_pos + cfg.hold_trading_days - 1, len(spy_cal) - 1)
        target_90_ts = min(spy_cal[target_90_pos], end_ts)
        early_term = (last_bar_ts < target_90_ts) and ((target_90_ts - last_bar_ts).days >= 10)
        is_buyout = sig.issuer_cik in buyouts

        counts["qualified_matched"] += 1
        if early_term:
            counts["early_terminated_before_90d"] += 1
            if is_buyout:
                counts["early_terminated_verified_buyout"] += 1
            else:
                counts["early_terminated_penalized"] += 1

        evaluated.append(
            EvaluatedCluster(
                signal=sig,
                status="qualified_matched",
                entry_date=entry_ts.date(),
                entry_raw_open=entry_open,
                asof_price=asof_price,
                asof_adv20=asof_adv20,
                form4_price_eligible=form4_px_ok,
                early_terminated_before_90d=early_term,
                last_bar_date=last_bar_ts.date(),
                verified_cash_buyout=is_buyout,
            )
        )

    # Survivorship summary ratios
    common_total = counts["total_clusters"] - counts["rejected_non_common"]
    unmatched_pct_of_common = (counts["unmatched_yfinance"] / common_total) if common_total > 0 else 0.0
    unmatched_elig_plus_qual = counts["qualified_matched"] + counts["unmatched_price_eligible"]
    unmatched_elig_share = (
        counts["unmatched_price_eligible"] / unmatched_elig_plus_qual
        if unmatched_elig_plus_qual > 0
        else 0.0
    )
    early_term_pct_of_qual = (
        counts["early_terminated_before_90d"] / counts["qualified_matched"]
        if counts["qualified_matched"] > 0
        else 0.0
    )
    # Empirical ADV pass rate among matched stocks in [$3, $150]
    matched_in_price = counts["qualified_matched"] + counts["rejected_adv"] + counts["rejected_no_entry_bar"]
    empirical_adv_pass_rate = counts["qualified_matched"] / matched_in_price if matched_in_price > 0 else 0.0

    audit = {
        **counts,
        "common_stock_clusters_after_name_filter": common_total,
        "unmatched_pct_of_common_clusters": unmatched_pct_of_common,
        "unmatched_price_eligible_share_of_penalized_pool": unmatched_elig_share,
        "early_terminated_pct_of_qualified_matched": early_term_pct_of_qual,
        "empirical_adv_pass_rate_in_price_band": empirical_adv_pass_rate,
        "non_common_rejection_breakdown": non_common_breakdown,
    }
    return evaluated, audit


def _summarize_excess_array(
    stock_rets: list[float],
    excess_rets: list[float],
) -> dict[str, Any]:
    n = len(excess_rets)
    if n == 0:
        return {
            "n": 0,
            "mean_stock_ret": float("nan"),
            "median_stock_ret": float("nan"),
            "mean_excess_ret": float("nan"),
            "median_excess_ret": float("nan"),
            "t_stat": float("nan"),
            "p_value_ttest": float("nan"),
            "p_value_wilcoxon": float("nan"),
            "pct_beat_spy": float("nan"),
        }
    s_arr = np.asarray(stock_rets, dtype=float)
    e_arr = np.asarray(excess_rets, dtype=float)
    if n >= 2 and np.std(e_arr, ddof=1) > 0:
        t_res = stats.ttest_1samp(e_arr, 0.0)
        t_stat = float(t_res.statistic)
        p_t = float(t_res.pvalue)
    else:
        t_stat, p_t = float("nan"), float("nan")
    try:
        nonzero = e_arr[e_arr != 0.0]
        p_w = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else float("nan")
    except Exception:
        p_w = float("nan")
    return {
        "n": int(n),
        "mean_stock_ret": float(np.mean(s_arr)),
        "median_stock_ret": float(np.median(s_arr)),
        "mean_excess_ret": float(np.mean(e_arr)),
        "median_excess_ret": float(np.median(e_arr)),
        "t_stat": t_stat,
        "p_value_ttest": p_t,
        "p_value_wilcoxon": p_w,
        "pct_beat_spy": float(np.mean(e_arr > 0.0)),
    }


def run_event_study(
    evaluated: list[EvaluatedCluster],
    bars: dict[str, BarSeries],
    spy_bars: BarSeries,
    window_start: date,
    window_end: date,
    sample_end_date: date,
    cfg: Form4StrategyConfig,
    horizons: tuple[int, ...] = (20, 45, 60, 90),
) -> dict[str, Any]:
    """Part A: Raw-Matched AND Survivorship-Penalized Event Study vs SPY.

    Signals are included in a window if ``window_start <= filing_date <= window_end``.
    Forward returns are measured strictly from ``FILING_DATE + 1 trading day`` Open
    using bars up to ``sample_end_date`` (right-censoring horizons beyond ``sample_end_date``).
    """
    spy_df = spy_bars.frame.loc[spy_bars.frame.index <= pd.Timestamp(sample_end_date)]
    spy_cal = spy_df.index.sort_values()
    end_ts = pd.Timestamp(sample_end_date)
    penalty = cfg.survivorship_penalty_return

    win_items = [
        ev for ev in evaluated
        if window_start <= ev.signal.filing_date <= window_end
    ]
    matched = [ev for ev in win_items if ev.status == "qualified_matched"]
    unmatched_elig = [
        ev for ev in win_items
        if ev.status == "unmatched_yfinance" and ev.form4_price_eligible
    ]

    raw_horizons: dict[int, tuple[list[float], list[float]]] = {h: ([], []) for h in horizons}
    pen_horizons: dict[int, tuple[list[float], list[float]]] = {h: ([], []) for h in horizons}

    # Right-tail multi-bagger & left-tail trap counters
    rt90_50_raw: list[int] = []
    rt90_100_raw: list[int] = []
    rt180_50_raw: list[int] = []
    rt180_100_raw: list[int] = []
    lt90_30_raw: list[int] = []
    lt90_50_raw: list[int] = []

    rt90_50_pen: list[int] = []
    rt90_100_pen: list[int] = []
    rt180_50_pen: list[int] = []
    rt180_100_pen: list[int] = []
    lt90_30_pen: list[int] = []
    lt90_50_pen: list[int] = []

    early_term_in_win = 0
    early_term_pen_in_win = 0

    for ev in matched:
        if ev.entry_date is None or ev.entry_raw_open is None:
            continue
        entry_ts = pd.Timestamp(ev.entry_date)
        if entry_ts > end_ts:
            continue
        df = bars[ev.signal.ticker].frame
        df = df.loc[(df.index >= entry_ts) & (df.index <= end_ts)]
        if df.empty:
            continue
        p0 = float(df["open"].iloc[0])
        if p0 <= 0:
            continue

        spy_pos = int(spy_cal.searchsorted(entry_ts, side="left"))
        if spy_pos >= len(spy_cal):
            continue
        spy_entry_open = float(spy_df["open"].iloc[spy_pos])

        if ev.early_terminated_before_90d:
            early_term_in_win += 1
            if not ev.verified_cash_buyout:
                early_term_pen_in_win += 1

        # Horizons
        for h in horizons:
            target_spy_pos = spy_pos + h - 1
            if target_spy_pos >= len(spy_cal):
                continue  # Right-censored by sample_end_date (not a delisting)
            target_spy_ts = spy_cal[target_spy_pos]
            spy_ret_h = float(spy_df["close"].iloc[target_spy_pos]) / spy_entry_open - 1.0

            if len(df) >= h:
                exit_ts = df.index[h - 1]
                s_ret = float(df["close"].iloc[h - 1]) / p0 - 1.0
                # Exact SPY return on same exit date if present
                s_spy_idx = int(spy_cal.searchsorted(exit_ts, side="right")) - 1
                s_spy_ret = (
                    float(spy_df["close"].iloc[s_spy_idx]) / spy_entry_open - 1.0
                    if s_spy_idx >= spy_pos
                    else spy_ret_h
                )
                raw_horizons[h][0].append(s_ret)
                raw_horizons[h][1].append(s_ret - s_spy_ret)
                pen_horizons[h][0].append(s_ret)
                pen_horizons[h][1].append(s_ret - s_spy_ret)
            elif ev.early_terminated_before_90d or (target_spy_ts - df.index[-1]).days >= 10:
                # Early-terminated before +H trading days
                last_ret = float(df["close"].iloc[-1]) / p0 - 1.0
                last_spy_idx = int(spy_cal.searchsorted(df.index[-1], side="right")) - 1
                last_spy_ret = (
                    float(spy_df["close"].iloc[last_spy_idx]) / spy_entry_open - 1.0
                    if last_spy_idx >= spy_pos
                    else spy_ret_h
                )
                raw_horizons[h][0].append(last_ret)
                raw_horizons[h][1].append(last_ret - last_spy_ret)

                if ev.verified_cash_buyout:
                    pen_horizons[h][0].append(last_ret)
                    pen_horizons[h][1].append(last_ret - last_spy_ret)
                else:
                    pen_horizons[h][0].append(penalty)
                    pen_horizons[h][1].append(penalty - spy_ret_h)

        # Tail metrics over 90d and 180d (only if SPY has >= window bars, or early-terminated)
        if spy_pos + 89 < len(spy_cal):
            w90 = df.iloc[:90]
            peak90 = float(w90["close"].max()) / p0 - 1.0
            trough90 = float(w90["close"].min()) / p0 - 1.0
            rt90_50_raw.append(1 if peak90 >= 0.50 else 0)
            rt90_100_raw.append(1 if peak90 >= 1.00 else 0)
            lt90_30_raw.append(1 if trough90 <= -0.30 else 0)
            lt90_50_raw.append(1 if trough90 <= -0.50 else 0)

            if ev.early_terminated_before_90d and not ev.verified_cash_buyout:
                rt90_50_pen.append(0)
                rt90_100_pen.append(0)
                lt90_30_pen.append(1)
                lt90_50_pen.append(1)
            else:
                rt90_50_pen.append(1 if peak90 >= 0.50 else 0)
                rt90_100_pen.append(1 if peak90 >= 1.00 else 0)
                lt90_30_pen.append(1 if trough90 <= -0.30 else 0)
                lt90_50_pen.append(1 if trough90 <= -0.50 else 0)

        if spy_pos + 179 < len(spy_cal):
            w180 = df.iloc[:180]
            peak180 = float(w180["close"].max()) / p0 - 1.0
            rt180_50_raw.append(1 if peak180 >= 0.50 else 0)
            rt180_100_raw.append(1 if peak180 >= 1.00 else 0)
            if ev.early_terminated_before_90d and not ev.verified_cash_buyout:
                rt180_50_pen.append(0)
                rt180_100_pen.append(0)
            else:
                rt180_50_pen.append(1 if peak180 >= 0.50 else 0)
                rt180_100_pen.append(1 if peak180 >= 1.00 else 0)

    # Add unmatched price-eligible events to Survivorship-Penalized pools
    unmatched_added_90d = 0
    for ev in unmatched_elig:
        if ev.entry_date is None:
            continue
        entry_ts = pd.Timestamp(ev.entry_date)
        if entry_ts > end_ts:
            continue
        spy_pos = int(spy_cal.searchsorted(entry_ts, side="left"))
        if spy_pos >= len(spy_cal):
            continue
        spy_entry_open = float(spy_df["open"].iloc[spy_pos])
        for h in horizons:
            target_spy_pos = spy_pos + h - 1
            if target_spy_pos >= len(spy_cal):
                continue
            spy_ret_h = float(spy_df["close"].iloc[target_spy_pos]) / spy_entry_open - 1.0
            pen_horizons[h][0].append(penalty)
            pen_horizons[h][1].append(penalty - spy_ret_h)
        if spy_pos + 89 < len(spy_cal):
            unmatched_added_90d += 1
            rt90_50_pen.append(0)
            rt90_100_pen.append(0)
            lt90_30_pen.append(1)
            lt90_50_pen.append(1)
        if spy_pos + 179 < len(spy_cal):
            rt180_50_pen.append(0)
            rt180_100_pen.append(0)

    def _rate(lst: list[int]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "window": {"start": str(window_start), "end": str(window_end)},
        "n_matched_signals": len(matched),
        "n_unmatched_price_eligible": len(unmatched_elig),
        "n_unmatched_added_at_90d": unmatched_added_90d,
        "n_early_terminated_in_window": early_term_in_win,
        "n_early_terminated_penalized_in_window": early_term_pen_in_win,
        "raw_matched": {
            "horizons": {
                f"+{h}d": _summarize_excess_array(*raw_horizons[h]) for h in horizons
            },
            "tails": {
                "n_90d_eligible": len(rt90_50_raw),
                "n_180d_eligible": len(rt180_50_raw),
                "right_tail_ge_50pct_90d": _rate(rt90_50_raw),
                "right_tail_ge_100pct_90d": _rate(rt90_100_raw),
                "right_tail_ge_50pct_180d": _rate(rt180_50_raw),
                "right_tail_ge_100pct_180d": _rate(rt180_100_raw),
                "left_tail_le_neg30pct_90d": _rate(lt90_30_raw),
                "left_tail_le_neg50pct_90d": _rate(lt90_50_raw),
            },
        },
        "survivorship_penalized": {
            "penalty_return": penalty,
            "horizons": {
                f"+{h}d": _summarize_excess_array(*pen_horizons[h]) for h in horizons
            },
            "tails": {
                "n_90d_eligible": len(rt90_50_pen),
                "n_180d_eligible": len(rt180_50_pen),
                "right_tail_ge_50pct_90d": _rate(rt90_50_pen),
                "right_tail_ge_100pct_90d": _rate(rt90_100_pen),
                "right_tail_ge_50pct_180d": _rate(rt180_50_pen),
                "right_tail_ge_100pct_180d": _rate(rt180_100_pen),
                "left_tail_le_neg30pct_90d": _rate(lt90_30_pen),
                "left_tail_le_neg50pct_90d": _rate(lt90_50_pen),
            },
        },
    }


@dataclass
class _OpenPos:
    ev: EvaluatedCluster
    entry_date: date
    entry_raw_open: float
    entry_fill: float
    hard_stop: float
    qty: int
    entry_comm: float
    entry_slip_usd: float
    cost_basis_usd: float
    last_close: float
    highest_close: float = -1e18
    peak_gain_pct: float = 0.0
    trailing_activated: bool = False
    bars_held: int = 0
    queued_exit_reason: ExitReason | None = None
    is_synthetic_unmatched: bool = False


def run_form4_portfolio_backtest(
    evaluated: list[EvaluatedCluster],
    bars: dict[str, BarSeries],
    spy_bars: BarSeries,
    cfg: Form4StrategyConfig,
    window_start: date,
    window_end: date,
    initial_capital_usd: float,
    initial_capital_eur: float,
    window_label: str = "FULL_5Y",
    survivorship_penalized: bool = False,
) -> Form4PortfolioResult:
    """Part B: Concentrated 4-slot IBKR Cash Account simulation (T+1, whole shares)."""
    spy_cal = spy_bars.frame.index.sort_values()
    cal = spy_cal[(spy_cal >= pd.Timestamp(window_start)) & (spy_cal <= pd.Timestamp(window_end))]
    if cal.empty:
        raise ValueError(f"Empty SPY calendar for window {window_start}..{window_end}")

    # Precompute causal ATR(14) for all matched tickers appearing in this window
    by_entry_date: dict[pd.Timestamp, list[EvaluatedCluster]] = {}
    needed_tickers: set[str] = set()
    for ev in evaluated:
        if not (window_start <= ev.signal.filing_date <= window_end):
            continue
        if ev.entry_date is None:
            continue
        e_ts = pd.Timestamp(ev.entry_date)
        if e_ts < cal[0] or e_ts > cal[-1]:
            continue
        if ev.status == "qualified_matched":
            by_entry_date.setdefault(e_ts, []).append(ev)
            needed_tickers.add(ev.signal.ticker)
        elif survivorship_penalized and ev.status == "unmatched_yfinance" and ev.form4_price_eligible:
            by_entry_date.setdefault(e_ts, []).append(ev)

    atr_map: dict[str, pd.Series] = {}
    frame_map: dict[str, pd.DataFrame] = {}
    for sym in needed_tickers:
        if sym in bars:
            f = bars[sym].frame
            frame_map[sym] = f
            atr_map[sym] = wilder_atr(f, cfg.atr_window)

    settled_cash = float(initial_capital_usd)
    unsettled: list[tuple[pd.Timestamp, float]] = []
    open_positions: list[_OpenPos] = []
    closed_trades: list[Form4TradeRecord] = []
    skipped: dict[str, int] = {
        "max_slots_full": 0,
        "already_held": 0,
        "insufficient_settled_cash": 0,
        "whole_share_too_expensive": 0,
    }
    total_orders = 0
    total_comm = 0.0
    total_slip = 0.0
    curve_rows: list[dict[str, Any]] = []

    slip = cfg.slippage_pct
    comm = cfg.commission_per_side_usd
    penalty_mult = 1.0 + cfg.survivorship_penalty_return  # 0.50 for -50%

    def _settle_date(idx_i: int) -> pd.Timestamp:
        if cfg.settlement_days <= 0:
            return cal[idx_i]
        tgt = idx_i + cfg.settlement_days
        if tgt < len(cal):
            return cal[tgt]
        # Beyond window end; settles after window
        return cal[-1] + pd.Timedelta(days=cfg.settlement_days)

    def _close_position(
        pos: _OpenPos,
        exit_ts: pd.Timestamp,
        raw_exit_px: float,
        reason: ExitReason,
        idx_i: int,
    ) -> None:
        nonlocal settled_cash, total_orders, total_comm, total_slip
        sell_fill = raw_exit_px * (1.0 - slip)
        sell_slip_usd = pos.qty * raw_exit_px * slip
        gross_proceeds = pos.qty * sell_fill
        net_proceeds = max(0.0, gross_proceeds - comm)

        total_orders += 1
        total_comm += comm
        total_slip += sell_slip_usd

        s_dt = _settle_date(idx_i)
        if cfg.settlement_days == 0:
            settled_cash += net_proceeds
        else:
            unsettled.append((s_dt, net_proceeds))

        gross_pnl = pos.qty * (raw_exit_px - pos.entry_raw_open)
        net_pnl = net_proceeds - pos.cost_basis_usd
        ret_pct = net_pnl / pos.cost_basis_usd if pos.cost_basis_usd > 0 else 0.0

        closed_trades.append(
            Form4TradeRecord(
                variant=cfg.variant,
                issuer_cik=pos.ev.signal.issuer_cik,
                ticker=pos.ev.signal.ticker,
                issuer_name=pos.ev.signal.issuer_name,
                filing_date=pos.ev.signal.filing_date,
                entry_date=pos.entry_date,
                exit_date=exit_ts.date(),
                distinct_insiders=pos.ev.signal.distinct_insiders,
                has_c_suite=pos.ev.signal.has_c_suite,
                combined_dollar_spend=pos.ev.signal.combined_dollar_spend,
                insiders_summary=pos.ev.signal.insiders_summary,
                qty=pos.qty,
                entry_raw_price=pos.entry_raw_open,
                entry_fill_price=pos.entry_fill,
                hard_stop_price=pos.hard_stop,
                exit_raw_price=raw_exit_px,
                exit_fill_price=sell_fill,
                bars_held=max(1, pos.bars_held),
                peak_gain_pct=pos.peak_gain_pct,
                trailing_activated=pos.trailing_activated,
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

    for i, ts in enumerate(cal):
        # 1. T+1 settlement BEFORE open
        if unsettled:
            still_pending: list[tuple[pd.Timestamp, float]] = []
            for s_dt, amt in unsettled:
                if s_dt <= ts:
                    settled_cash += amt
                else:
                    still_pending.append((s_dt, amt))
            unsettled = still_pending

        # 2. Open exits for existing held positions (queued exits + overnight gap-through stop)
        survivors_after_open: list[_OpenPos] = []
        for pos in open_positions:
            if pos.is_synthetic_unmatched:
                survivors_after_open.append(pos)
                continue
            df_s = frame_map[pos.ev.signal.ticker]
            if ts not in df_s.index:
                survivors_after_open.append(pos)
                continue
            row = df_s.loc[ts]
            open_px = float(row["open"])
            if open_px <= pos.hard_stop:
                _close_position(pos, ts, open_px, "HARD_STOP_GAP", i)
            elif pos.queued_exit_reason is not None:
                _close_position(pos, ts, open_px, pos.queued_exit_reason, i)
            else:
                survivors_after_open.append(pos)
        open_positions = survivors_after_open

        # 3. New entries at Open (strictly FILING_DATE + 1 trading day)
        cands = by_entry_date.get(ts)
        if cands:
            cands_sorted = sorted(
                cands,
                key=lambda ev: (
                    -ev.signal.combined_dollar_spend,
                    -ev.signal.distinct_insiders,
                    -int(ev.signal.has_c_suite),
                    ev.signal.ticker,
                ),
            )
            open_mtm_prior = sum(p.qty * p.last_close for p in open_positions)
            settled_equity = settled_cash + open_mtm_prior
            target_notional = cfg.target_slot_pct * settled_equity

            held_ciks = {p.ev.signal.issuer_cik for p in open_positions}
            held_tickers = {p.ev.signal.ticker for p in open_positions}

            for ev in cands_sorted:
                if len(open_positions) >= cfg.max_slots:
                    skipped["max_slots_full"] += 1
                    continue
                if ev.signal.issuer_cik in held_ciks or ev.signal.ticker in held_tickers:
                    skipped["already_held"] += 1
                    continue

                is_synth = ev.status == "unmatched_yfinance"
                if is_synth:
                    raw_open = float(ev.signal.weighted_avg_form4_price)
                else:
                    df_s = frame_map[ev.signal.ticker]
                    if ts not in df_s.index:
                        continue
                    raw_open = float(df_s.loc[ts, "open"])

                if raw_open <= 0:
                    continue

                buy_fill = raw_open * (1.0 + slip)
                budget = min(target_notional, settled_cash)
                if budget <= comm:
                    skipped["insufficient_settled_cash"] += 1
                    continue
                qty = int(math.floor((budget - comm) / buy_fill))
                if qty < 1:
                    if settled_cash < buy_fill + comm:
                        skipped["insufficient_settled_cash"] += 1
                    else:
                        skipped["whole_share_too_expensive"] += 1
                    continue

                cost_basis = qty * buy_fill + comm
                buy_slip_usd = qty * raw_open * slip
                settled_cash -= cost_basis
                total_orders += 1
                total_comm += comm
                total_slip += buy_slip_usd

                open_positions.append(
                    _OpenPos(
                        ev=ev,
                        entry_date=ts.date(),
                        entry_raw_open=raw_open,
                        entry_fill=buy_fill,
                        hard_stop=buy_fill * (1.0 - cfg.hard_stop_pct),
                        qty=qty,
                        entry_comm=comm,
                        entry_slip_usd=buy_slip_usd,
                        cost_basis_usd=cost_basis,
                        last_close=raw_open,
                        highest_close=-1e18,
                        is_synthetic_unmatched=is_synth,
                    )
                )
                held_ciks.add(ev.signal.issuer_cik)
                held_tickers.add(ev.signal.ticker)

        # 4. Intraday hard stop check + 5. Close MTM & exit queueing
        survivors_after_close: list[_OpenPos] = []
        for pos in open_positions:
            if pos.is_synthetic_unmatched:
                pos.bars_held += 1
                frac = min(1.0, pos.bars_held / float(cfg.penalized_unmatched_hold_days))
                pos.last_close = pos.entry_raw_open * (1.0 + cfg.survivorship_penalty_return * frac)
                if pos.bars_held >= cfg.penalized_unmatched_hold_days:
                    _close_position(
                        pos,
                        ts,
                        pos.entry_raw_open * penalty_mult,
                        "UNMATCHED_PENALIZED_50",
                        i,
                    )
                else:
                    survivors_after_close.append(pos)
                continue

            df_s = frame_map[pos.ev.signal.ticker]
            if ts not in df_s.index:
                survivors_after_close.append(pos)
                continue
            row = df_s.loc[ts]
            low_px = float(row["low"])
            close_px = float(row["close"])

            pos.bars_held += 1

            # Intraday -25% hard catastrophe stop (applies on entry bar after open and all held bars)
            if low_px <= pos.hard_stop:
                _close_position(pos, ts, pos.hard_stop, "HARD_STOP_INTRADAY", i)
                continue

            # Close MTM & causal trailing stop / 90d time exit
            pos.last_close = close_px
            if close_px > pos.highest_close:
                pos.highest_close = close_px
            gain_pct = close_px / pos.entry_fill - 1.0
            if gain_pct > pos.peak_gain_pct:
                pos.peak_gain_pct = gain_pct
            if gain_pct >= cfg.trail_activation_gain_pct:
                pos.trailing_activated = True

            # Check early delisting (stock's last bar strictly before window end)
            last_stock_ts = df_s.index[-1]
            if ts == last_stock_ts and (cal[-1] - ts).days >= 10:
                if survivorship_penalized and not pos.ev.verified_cash_buyout:
                    _close_position(
                        pos,
                        ts,
                        pos.entry_raw_open * penalty_mult,
                        "DELISTED_PENALIZED_50",
                        i,
                    )
                else:
                    _close_position(pos, ts, close_px, "DELISTED_LAST_BAR", i)
                continue

            # Queue next-open exit based on close_t (strictly causal)
            atr_val = float(atr_map[pos.ev.signal.ticker].loc[ts])
            if (
                pos.trailing_activated
                and not math.isnan(atr_val)
                and atr_val > 0.0
                and close_px < (pos.highest_close - cfg.trail_atr_mult * atr_val)
            ):
                pos.queued_exit_reason = "TRAILING_STOP_EXIT"
            elif pos.bars_held >= cfg.hold_trading_days:
                pos.queued_exit_reason = "TIME_EXIT_90D"

            survivors_after_close.append(pos)

        open_positions = survivors_after_close

        # Record daily equity curve
        unsettled_total = sum(amt for _, amt in unsettled)
        positions_val = sum(p.qty * p.last_close for p in open_positions)
        total_eq = settled_cash + unsettled_total + positions_val
        curve_rows.append({
            "date": ts,
            "equity": total_eq,
            "settled_cash": settled_cash,
            "unsettled_cash": unsettled_total,
            "positions_value": positions_val,
            "open_positions": len(open_positions),
            "exposure_pct": positions_val / total_eq if total_eq > 0 else 0.0,
        })

    # Record open positions at end of window (MTM, no phantom close commission)
    end_mtm_trades: list[Form4TradeRecord] = []
    end_ts = cal[-1]
    for pos in open_positions:
        gross_pnl = pos.qty * (pos.last_close - pos.entry_raw_open)
        mtm_val = pos.qty * pos.last_close
        net_pnl = mtm_val - pos.cost_basis_usd
        ret_pct = net_pnl / pos.cost_basis_usd if pos.cost_basis_usd > 0 else 0.0
        end_mtm_trades.append(
            Form4TradeRecord(
                variant=cfg.variant,
                issuer_cik=pos.ev.signal.issuer_cik,
                ticker=pos.ev.signal.ticker,
                issuer_name=pos.ev.signal.issuer_name,
                filing_date=pos.ev.signal.filing_date,
                entry_date=pos.entry_date,
                exit_date=end_ts.date(),
                distinct_insiders=pos.ev.signal.distinct_insiders,
                has_c_suite=pos.ev.signal.has_c_suite,
                combined_dollar_spend=pos.ev.signal.combined_dollar_spend,
                insiders_summary=pos.ev.signal.insiders_summary,
                qty=pos.qty,
                entry_raw_price=pos.entry_raw_open,
                entry_fill_price=pos.entry_fill,
                hard_stop_price=pos.hard_stop,
                exit_raw_price=pos.last_close,
                exit_fill_price=pos.last_close,
                bars_held=max(1, pos.bars_held),
                peak_gain_pct=pos.peak_gain_pct,
                trailing_activated=pos.trailing_activated,
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
    return Form4PortfolioResult(
        config=cfg,
        initial_capital_usd=initial_capital_usd,
        initial_capital_eur=initial_capital_eur,
        window_label=window_label,
        survivorship_penalized=survivorship_penalized,
        equity_curve=eq_df,
        trades=closed_trades,
        open_positions_at_end=end_mtm_trades,
        skipped_counts=skipped,
        total_orders=total_orders,
        total_commission_usd=total_comm,
        total_slippage_usd=total_slip,
    )


def compute_form4_portfolio_metrics(
    res: Form4PortfolioResult,
    mcfg: MetricsConfig,
    standalone_trials_n: int = 10,
    cumulative_trials_n: int = 45,
    cumulative_trials_conservative_n: int = 65,
) -> dict[str, Any]:
    """Compute CAGR, Sharpe, Sortino, DSR (N=10/45/65), Max DD, Calmar, Win Rate, PF."""
    eq = res.equity_curve["equity"]
    init = res.initial_capital_usd
    r = daily_returns(eq, init)

    days = (eq.index[-1] - eq.index[0]).days + 1
    years = max(days / 365.25, 1e-6)
    cagr_val = cagr(eq, init)
    mdd_val = max_drawdown(eq, init)
    calmar_val = (cagr_val / abs(mdd_val)) if (not math.isnan(cagr_val) and mdd_val < 0) else float("nan")

    all_trips = list(res.trades) + list(res.open_positions_at_end)
    closed = list(res.trades)
    pnl_arr = np.array([t.net_pnl_usd for t in closed], dtype=float) if closed else np.array([], dtype=float)
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
        "dsr_n45": deflated_sharpe(r, US_PERIODS_PER_YEAR, cumulative_trials_n, mcfg.dsr_trial_sharpe_std),
        "dsr_n65": deflated_sharpe(r, US_PERIODS_PER_YEAR, cumulative_trials_conservative_n, mcfg.dsr_trial_sharpe_std),
        "max_drawdown": mdd_val,
        "calmar": calmar_val,
        "avg_exposure_pct": float(res.equity_curve["exposure_pct"].mean()),
        "pct_days_invested": float((res.equity_curve["open_positions"] > 0).mean()),
        "total_orders": int(res.total_orders),
        "orders_per_year": float(res.total_orders / years),
        "closed_trades": len(closed),
        "open_mtm_trades": len(res.open_positions_at_end),
        "win_rate": float(wins.mean()) if len(pnl_arr) > 0 else float("nan"),
        "profit_factor": pf,
        "avg_return_pct": float(np.mean([t.return_pct for t in closed])) if closed else float("nan"),
        "median_return_pct": float(np.median([t.return_pct for t in closed])) if closed else float("nan"),
        "avg_bars_held": float(np.mean([t.bars_held for t in closed])) if closed else float("nan"),
        "total_commission_usd": float(res.total_commission_usd),
        "total_slippage_usd": float(res.total_slippage_usd),
        "total_friction_usd": float(total_friction),
        "friction_pct_of_initial": float(total_friction / init),
        "skipped_counts": dict(res.skipped_counts),
        "exit_breakdown": exit_breakdown,
    }


def extract_top_winners_and_traps(
    res: Form4PortfolioResult,
    n: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract Top ``n`` Winners and Worst ``n`` Traps for Stage 2 8-K/10-Q audit."""
    all_trips = list(res.trades) + list(res.open_positions_at_end)
    rows = [
        {
            "issuer_name": t.issuer_name,
            "issuer_cik": t.issuer_cik,
            "ticker": t.ticker,
            "filing_date": str(t.filing_date),
            "entry_date": str(t.entry_date),
            "exit_date": str(t.exit_date),
            "distinct_insiders": t.distinct_insiders,
            "has_c_suite": t.has_c_suite,
            "insiders_summary": t.insiders_summary,
            "combined_dollar_spend": round(t.combined_dollar_spend, 2),
            "qty": t.qty,
            "entry_fill_price": round(t.entry_fill_price, 4),
            "exit_fill_price": round(t.exit_fill_price, 4),
            "bars_held": t.bars_held,
            "net_pnl_usd": round(t.net_pnl_usd, 2),
            "return_pct": round(t.return_pct * 100.0, 2),
            "exit_reason": t.exit_reason,
        }
        for t in all_trips
    ]
    top_winners = sorted(rows, key=lambda x: (-x["return_pct"], -x["net_pnl_usd"]))[:n]
    worst_traps = sorted(rows, key=lambda x: (x["return_pct"], x["net_pnl_usd"]))[:n]
    return top_winners, worst_traps
