"""Strategy v9_form4 (Path 2, Stage 1): SEC Form 4 Cluster Buying Backtest & Audit.

Strict Anti-Bias Protocol (ALL rules locked and 162 unit tests passed BEFORE running
Blind OOS once; zero OOS tuning):
- SEC DERA Bulk Datasets: 2021q4 through 2026q3 (20 quarters, cached under data/sec_form4/)
- In-Sample (IS):  2021-11-22 to 2024-11-21 (3.00 years)
- Blind OOS:       2024-11-22 to 2026-11-20 (2.00 years; new Form 4 entries through 2026-09-30)
- Full 5Y:         2021-11-22 to 2026-11-20 (5.00 years)

Runs:
- Part A: Raw-Matched AND Survivorship-Penalized (-50%) Event Study vs SPY (+20/45/60/90d,
          p-values, right-tail multi-bagger rate, left-tail trap rate)
- Part B: Asymmetric 4-Slot IBKR Cash Account Portfolio Simulation (EUR 1,000 / $1,060 and
          EUR 5,000 / $5,300; T+1 settlement; whole shares; 0.25% slip + $0.38/side; EUR 0/mo)
          with Raw-Matched AND Survivorship-Penalized (-50%) Sensitivity vs SPY, CSPX.AS, v5b QQQ3.MI.

Outputs:
- outputs/v9_form4_results.json
- outputs/v9_form4_eur1000.png & Project store media/v9-form4-eur1000.png
- outputs/v9_form4_eur5000.png & Project store media/v9-form4-eur5000.png
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.metrics import (  # noqa: E402
    cagr,
    daily_returns,
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
)
from src.config import MetricsConfig  # noqa: E402
from src.data.cache import DiskCache  # noqa: E402
from src.data.sec_form4_loader import (  # noqa: E402
    DEFAULT_QUARTERS,
    SecForm4Loader,
    batch_load_yfinance_bars,
    load_sec_company_tickers_exchange,
)
from src.domain.models import BarSeries  # noqa: E402
from src.strategy.sec_form4_cluster import (  # noqa: E402
    US_PERIODS_PER_YEAR,
    EvaluatedCluster,
    Form4PortfolioResult,
    compute_form4_portfolio_metrics,
    detect_form4_clusters,
    evaluate_clusters_and_survivorship,
    extract_top_winners_and_traps,
    make_v9a_config,
    make_v9b_config,
    run_event_study,
    run_form4_portfolio_backtest,
)
from src.strategy.ucits_etf_regime import (  # noqa: E402
    RiskOffMode,
    SizingMode,
    UcitsRegimeConfig,
    run_ucits_regime_backtest,
)

IS_START = date(2021, 11, 22)
IS_END = date(2024, 11, 21)
OOS_START = date(2024, 11, 22)
OOS_END = date(2026, 11, 20)
BAR_FETCH_START = date(2020, 10, 18)
BAR_FETCH_END = date(2026, 11, 20)

EUR_USD_RATE = 1.06
CAPITALS: tuple[tuple[str, float, float], ...] = (
    ("eur1000", 1000.0, 1060.0),
    ("eur5000", 5000.0, 5300.0),
)

WINDOWS: dict[str, tuple[date, date]] = {
    "IS": (IS_START, IS_END),
    "OOS": (OOS_START, OOS_END),
    "FULL_5Y": (IS_START, OOS_END),
}

STORE_ROOT = Path(
    "/home/margarida/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-de418035-892d-40a4-a21a-bd834b742bee/files"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("v9_form4")


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (pd.Timestamp, date)):
        return str(obj.date() if isinstance(obj, pd.Timestamp) else obj)
    return obj


def _compute_spy_benchmark(
    spy_bars: BarSeries,
    window_start: date,
    window_end: date,
    initial_eur: float,
    mcfg: MetricsConfig,
) -> tuple[pd.Series, dict[str, Any]]:
    df = spy_bars.frame
    sub = df.loc[(df.index >= pd.Timestamp(window_start)) & (df.index <= pd.Timestamp(window_end))]
    # Buy at first open (or prior close) — compute close-to-close from first day's open / prev close
    prev_bars = df.loc[df.index < pd.Timestamp(window_start)]
    base_px = float(prev_bars["close"].iloc[-1]) if not prev_bars.empty else float(sub["open"].iloc[0])
    eq_eur = (sub["close"] / base_px) * initial_eur
    r = daily_returns(eq_eur, initial_eur)
    cagr_v = cagr(eq_eur, initial_eur)
    mdd_v = max_drawdown(eq_eur, initial_eur)
    calmar_v = cagr_v / abs(mdd_v) if mdd_v < 0 else float("nan")
    metrics = {
        "initial_eur": initial_eur,
        "final_eur": float(eq_eur.iloc[-1]),
        "total_return": float(eq_eur.iloc[-1] / initial_eur - 1.0),
        "cagr": cagr_v,
        "sharpe": sharpe(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "sortino": sortino(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "dsr_n10": deflated_sharpe(r, US_PERIODS_PER_YEAR, 10, mcfg.dsr_trial_sharpe_std),
        "dsr_n45": deflated_sharpe(r, US_PERIODS_PER_YEAR, 45, mcfg.dsr_trial_sharpe_std),
        "dsr_n65": deflated_sharpe(r, US_PERIODS_PER_YEAR, 65, mcfg.dsr_trial_sharpe_std),
        "max_drawdown": mdd_v,
        "calmar": calmar_v,
    }
    return eq_eur, metrics


def _load_v5b_and_cspx_curves(
    cache_root: Path,
    capital_eur: float,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Reconstruct exact saved v5b QQQ3.MI and CSPX.AS EUR curves from cached bars."""
    cache = DiskCache(cache_root, "yfinance")
    loaded: dict[str, BarSeries] = {}
    for sym in ("QQQ", "QQQ3.MI", "CSPX.AS"):
        key = f"{sym}_2020-10-18_2026-11-20"
        if cache.status("bars", key) == "ok":
            df = cache.read("bars", key)
            if df is not None and not df.empty:
                loaded[sym] = BarSeries(symbol=sym, frame=df)

    cspx_eq = None
    if "CSPX.AS" in loaded:
        df_c = loaded["CSPX.AS"].frame
        sub = df_c.loc[(df_c.index >= pd.Timestamp(IS_START)) & (df_c.index <= pd.Timestamp(OOS_END))]
        prev = df_c.loc[df_c.index < pd.Timestamp(IS_START)]
        base = float(prev["close"].iloc[-1]) if not prev.empty else float(sub["close"].iloc[0])
        cspx_eq = (sub["close"] / base) * capital_eur

    v5b_eq = None
    if "QQQ" in loaded and "QQQ3.MI" in loaded:
        cfg_v5b = UcitsRegimeConfig(
            name="v5b_QQQ3_cash",
            signal_symbol="QQQ",
            risk_on_symbol="QQQ3.MI",
            risk_off_mode=RiskOffMode.CASH,
            risk_off_symbol="XEON.DE",
            sma_window=200,
            hysteresis_pct=0.01,
            vol_window=20,
            vol_circuit_breaker=0.35,
            commission_eur=1.25,
            slippage_pct=0.0010,
            settlement_days=2,
            monthly_hosting_eur=0.0,
            sizing_mode=SizingMode.AS_TRADED,
        )
        res_v5b = run_ucits_regime_backtest(
            signal_bars=loaded["QQQ"],
            risk_on_bars=loaded["QQQ3.MI"],
            risk_off_bars=None,
            cfg=cfg_v5b,
            start_date=IS_START,
            end_date=OOS_END,
            initial_capital_eur=capital_eur,
        )
        v5b_eq = res_v5b.equity_curve["equity"]
    return v5b_eq, cspx_eq


def _window_survivorship_breakdown(
    evaluated: list[EvaluatedCluster],
    window_start: date,
    window_end: date,
) -> dict[str, Any]:
    sub = [ev for ev in evaluated if window_start <= ev.signal.filing_date <= window_end]
    counts: dict[str, int] = {}
    for ev in sub:
        counts[ev.status] = counts.get(ev.status, 0) + 1
    total = len(sub)
    rej_nc = counts.get("rejected_non_common", 0)
    common = total - rej_nc
    unmatched = counts.get("unmatched_yfinance", 0)
    unmatched_px_ok = sum(
        1 for ev in sub if ev.status == "unmatched_yfinance" and ev.form4_price_eligible
    )
    qual = counts.get("qualified_matched", 0)
    early_term = sum(
        1 for ev in sub if ev.status == "qualified_matched" and ev.early_terminated_before_90d
    )
    return {
        "total_clusters": total,
        "common_stock_clusters": common,
        "qualified_matched": qual,
        "unmatched_yfinance": unmatched,
        "unmatched_pct_of_common": (unmatched / common) if common > 0 else 0.0,
        "unmatched_price_eligible": unmatched_px_ok,
        "early_terminated_before_90d": early_term,
        "early_terminated_pct_of_qualified": (early_term / qual) if qual > 0 else 0.0,
        "status_counts": counts,
    }


def _plot_v9_png(
    *,
    capital_eur: float,
    v9a_raw: Form4PortfolioResult,
    v9a_pen: Form4PortfolioResult,
    v9b_raw: Form4PortfolioResult,
    v9b_pen: Form4PortfolioResult,
    spy_eq_eur: pd.Series,
    cspx_eq_eur: pd.Series | None,
    v5b_eq_eur: pd.Series | None,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Generate 3-panel EUR chart: Equity Curves, Drawdowns, Open Positions / Exposure."""
    init = float(capital_eur)
    scale = 1.0 / EUR_USD_RATE

    eq_9a_raw = v9a_raw.equity_curve["equity"] * scale
    eq_9a_pen = v9a_pen.equity_curve["equity"] * scale
    eq_9b_raw = v9b_raw.equity_curve["equity"] * scale
    eq_9b_pen = v9b_pen.equity_curve["equity"] * scale

    def _dd(s: pd.Series) -> pd.Series:
        return (s / s.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14.4, 10.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.25, 1.15]},
    )

    # Panel 1: Equity Curves
    ax1.plot(
        eq_9a_raw.index,
        eq_9a_raw.values,
        color="#1f77b4",
        lw=2.1,
        label=f"v9a Standard (Raw Matched)          | EUR {eq_9a_raw.iloc[-1]:,.0f} ({(eq_9a_raw.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        eq_9a_pen.index,
        eq_9a_pen.values,
        color="#1f77b4",
        lw=1.4,
        ls="--",
        alpha=0.80,
        label=f"v9a Standard (Surv-Penalized -50%)  | EUR {eq_9a_pen.iloc[-1]:,.0f} ({(eq_9a_pen.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        eq_9b_raw.index,
        eq_9b_raw.values,
        color="#d62728",
        lw=2.2,
        label=f"v9b High Conviction (Raw Matched)   | EUR {eq_9b_raw.iloc[-1]:,.0f} ({(eq_9b_raw.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        eq_9b_pen.index,
        eq_9b_pen.values,
        color="#d62728",
        lw=1.4,
        ls="--",
        alpha=0.80,
        label=f"v9b High Conviction (Surv-Pen -50%) | EUR {eq_9b_pen.iloc[-1]:,.0f} ({(eq_9b_pen.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        spy_eq_eur.index,
        spy_eq_eur.values,
        color="#555555",
        lw=1.5,
        ls=":",
        label=f"SPY B&H (USD eq)                    | EUR {spy_eq_eur.iloc[-1]:,.0f} ({(spy_eq_eur.iloc[-1]/init-1)*100:+.1f}%)",
    )
    if cspx_eq_eur is not None:
        ax1.plot(
            cspx_eq_eur.index,
            cspx_eq_eur.values,
            color="#2ca02c",
            lw=1.5,
            ls="-.",
            alpha=0.85,
            label=f"CSPX.AS B&H (EUR)                   | EUR {cspx_eq_eur.iloc[-1]:,.0f} ({(cspx_eq_eur.iloc[-1]/init-1)*100:+.1f}%)",
        )
    if v5b_eq_eur is not None:
        ax1.plot(
            v5b_eq_eur.index,
            v5b_eq_eur.values,
            color="#9467bd",
            lw=1.9,
            alpha=0.90,
            label=f"v5b QQQ3.MI (Saved Benchmark)       | EUR {v5b_eq_eur.iloc[-1]:,.0f} ({(v5b_eq_eur.iloc[-1]/init-1)*100:+.1f}%)",
        )

    ax1.axhline(init, color="#999999", lw=0.8, ls=":")
    ax1.axvline(oos_split, color="#222222", lw=1.3, ls="--", alpha=0.85)
    ax1.text(
        oos_split + pd.Timedelta(days=10),
        ax1.get_ylim()[1] * 0.94,
        "Blind OOS Start\n(2024-11-22)",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        va="top",
    )
    ax1.set_title(
        f"Strategy v9_form4 (SEC Form 4 C-Suite Cluster Buying) — EUR {int(capital_eur):,} "
        f"(${int(round(capital_eur * EUR_USD_RATE)):,}) IBKR Cash Account (T+1, Whole Shares, 2021-11-22 to 2026-11-20)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Equity (EUR equiv.)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    # Panel 2: Underwater Drawdown
    ax2.plot(eq_9a_raw.index, _dd(eq_9a_raw), color="#1f77b4", lw=1.4, label="v9a Raw DD")
    ax2.plot(eq_9b_raw.index, _dd(eq_9b_raw), color="#d62728", lw=1.5, label="v9b Raw DD")
    ax2.plot(spy_eq_eur.index, _dd(spy_eq_eur), color="#555555", lw=1.1, ls=":", label="SPY DD")
    if v5b_eq_eur is not None:
        ax2.plot(v5b_eq_eur.index, _dd(v5b_eq_eur), color="#9467bd", lw=1.2, alpha=0.8, label="v5b QQQ3.MI DD")
    ax2.axvline(oos_split, color="#222222", lw=1.1, ls="--", alpha=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=8.3, ncol=4)

    # Panel 3: Open Positions & Exposure
    ax3.fill_between(
        v9a_raw.equity_curve.index,
        0,
        v9a_raw.equity_curve["exposure_pct"] * 100.0,
        color="#1f77b4",
        alpha=0.28,
        label="v9a Raw Exposure (%)",
    )
    ax3.plot(
        v9b_raw.equity_curve.index,
        v9b_raw.equity_curve["exposure_pct"] * 100.0,
        color="#d62728",
        lw=1.3,
        alpha=0.9,
        label="v9b Raw Exposure (%)",
    )
    ax3.axvline(oos_split, color="#222222", lw=1.1, ls="--", alpha=0.8)
    ax3.set_ylim(0, 108)
    ax3.set_ylabel("Invested (%)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8.3, ncol=2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.monotonic()
    mcfg = MetricsConfig()
    cache_root = ROOT / "data" / "cache"
    sec_root = ROOT / "data" / "sec_form4"
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & parse SEC DERA quarterly Form 3/4/5 datasets (2021q4..2026q3)
    log.info("Step 1/6: Loading SEC DERA Form 3/4/5 datasets (%d quarters)...", len(DEFAULT_QUARTERS))
    loader = SecForm4Loader(data_root=sec_root)
    accessions_df, sec_manifest = loader.load_quarters(DEFAULT_QUARTERS)
    log.info(
        "SEC DERA loaded: %d/%d quarters, %d qualifying Form 4 open-market Officer/Director accessions "
        "(filing dates %s..%s, 10b5-1 excluded=%d, multi-owner collapsed=%d, outliers dropped=%d)",
        len(sec_manifest["downloaded_quarters"]),
        len(DEFAULT_QUARTERS),
        sec_manifest["total_qualifying_accessions"],
        sec_manifest["filing_date_min"],
        sec_manifest["filing_date_max"],
        sec_manifest["total_aff10b5one_excluded"],
        sec_manifest["total_multi_owner_accessions_collapsed"],
        sec_manifest["total_outlier_rows_excluded"],
    )

    # 2. Detect clusters for v9a_standard and v9b_high_conviction
    log.info("Step 2/6: Detecting strictly causal Form 4 open-market clusters...")
    cfg_v9a = make_v9a_config()
    cfg_v9b = make_v9b_config()
    clusters_v9a, cstats_v9a = detect_form4_clusters(accessions_df, cfg_v9a)
    clusters_v9b, cstats_v9b = detect_form4_clusters(accessions_df, cfg_v9b)

    # Restrict to clusters with filing_date in [IS_START, OOS_END] (warmup already applied cooldown causally!)
    clusters_v9a_5y = [c for c in clusters_v9a if IS_START <= c.filing_date <= OOS_END]
    clusters_v9b_5y = [c for c in clusters_v9b if IS_START <= c.filing_date <= OOS_END]
    log.info(
        "5Y Clusters (post-cooldown, post-SEC-name-filter): v9a=%d, v9b=%d",
        len(clusters_v9a_5y),
        len(clusters_v9b_5y),
    )

    # 3. Batch-load yfinance bars & metadata for all unique cluster tickers + SPY
    all_cluster_tickers = sorted(
        {c.ticker for c in clusters_v9a_5y} | {c.ticker for c in clusters_v9b_5y} | {"SPY"}
    )
    log.info("Step 3/6: Loading/caching yfinance bars for %d unique cluster tickers...", len(all_cluster_tickers))
    bars_map, fetch_statuses = batch_load_yfinance_bars(
        all_cluster_tickers,
        cache_root=cache_root,
        start=BAR_FETCH_START,
        end=BAR_FETCH_END,
    )
    if "SPY" not in bars_map:
        raise RuntimeError("SPY benchmark bars unavailable!")
    spy_bars = bars_map["SPY"]

    matched_tickers_for_meta = sorted(t for t in all_cluster_tickers if t in bars_map and t != "SPY")
    log.info(
        "yfinance bars loaded: %d/%d tickers have bars (%d cached_ok, %d fetched_ok, %d missing/delisted)",
        len(matched_tickers_for_meta),
        len(all_cluster_tickers) - 1,
        sum(1 for k, v in fetch_statuses.items() if k != "SPY" and v == "cached_ok"),
        sum(1 for k, v in fetch_statuses.items() if k != "SPY" and v == "fetched_ok"),
        sum(1 for k, v in fetch_statuses.items() if k != "SPY" and v.endswith("empty")),
    )
    ticker_meta = load_sec_company_tickers_exchange(data_root=sec_root)
    log.info("Loaded official SEC company_tickers_exchange metadata for %d tickers", len(ticker_meta))

    # 4. Evaluate causal liquidity filter & full survivorship audit
    log.info("Step 4/6: Evaluating causal tradability & survivorship audit...")
    eval_v9a, surv_v9a = evaluate_clusters_and_survivorship(
        clusters_v9a_5y, bars_map, ticker_meta, spy_bars, OOS_END, cfg_v9a
    )
    eval_v9b, surv_v9b = evaluate_clusters_and_survivorship(
        clusters_v9b_5y, bars_map, ticker_meta, spy_bars, OOS_END, cfg_v9b
    )

    # Log exact early-terminated matched tickers for complete transparency
    early_term_records: dict[str, list[dict[str, Any]]] = {}
    for vname, ev_list in (("v9a_standard", eval_v9a), ("v9b_high_conviction", eval_v9b)):
        et_rows = []
        for ev in ev_list:
            if ev.status == "qualified_matched" and ev.early_terminated_before_90d:
                df_s = bars_map[ev.signal.ticker].frame
                e_ts = pd.Timestamp(ev.entry_date)
                sub_f = df_s.loc[df_s.index >= e_ts]
                last_close = float(sub_f["close"].iloc[-1]) if not sub_f.empty else float("nan")
                raw_ret = (last_close / ev.entry_raw_open - 1.0) if (ev.entry_raw_open and ev.entry_raw_open > 0) else float("nan")
                et_rows.append({
                    "ticker": ev.signal.ticker,
                    "issuer_cik": ev.signal.issuer_cik,
                    "issuer_name": ev.signal.issuer_name,
                    "filing_date": str(ev.signal.filing_date),
                    "entry_date": str(ev.entry_date),
                    "last_bar_date": str(ev.last_bar_date),
                    "entry_open": round(ev.entry_raw_open or 0.0, 4),
                    "last_close": round(last_close, 4),
                    "terminal_return_pct": round(raw_ret * 100.0, 2) if not math.isnan(raw_ret) else None,
                    "bars_after_entry": int(len(sub_f)),
                })
        early_term_records[vname] = et_rows
        log.info("%s early-terminated before +90d: %d events", vname, len(et_rows))

    # 5. Part A: Event Study across IS, Blind OOS, FULL_5Y
    log.info("Step 5/6: Running Part A Event Study (Raw-Matched & Survivorship-Penalized)...")
    part_a: dict[str, dict[str, Any]] = {"v9a_standard": {}, "v9b_high_conviction": {}}
    window_surv: dict[str, dict[str, Any]] = {"v9a_standard": {}, "v9b_high_conviction": {}}
    for wname, (w_start, w_end) in WINDOWS.items():
        # Note: for IS event study, evaluate horizons strictly using bars <= IS_END or sample_end?
        # Using sample_end=OOS_END for Full 5Y & OOS, and sample_end=IS_END for IS avoids any IS->OOS price overlap!
        w_sample_end = IS_END if wname == "IS" else OOS_END
        part_a["v9a_standard"][wname] = run_event_study(
            eval_v9a, bars_map, spy_bars, w_start, w_end, w_sample_end, cfg_v9a
        )
        part_a["v9b_high_conviction"][wname] = run_event_study(
            eval_v9b, bars_map, spy_bars, w_start, w_end, w_sample_end, cfg_v9b
        )
        window_surv["v9a_standard"][wname] = _window_survivorship_breakdown(eval_v9a, w_start, w_end)
        window_surv["v9b_high_conviction"][wname] = _window_survivorship_breakdown(eval_v9b, w_start, w_end)

    # 6. Part B: Concentrated Portfolio Backtests + Benchmarks
    log.info("Step 6/6: Running Part B Portfolio Simulations (EUR 1k & EUR 5k, Raw & Penalized)...")
    benchmarks: dict[str, Any] = {"SPY": {}, "v5_saved": {}}
    spy_curves_5y: dict[str, pd.Series] = {}
    for wname, (w_start, w_end) in WINDOWS.items():
        eq_spy_5k, m_spy = _compute_spy_benchmark(spy_bars, w_start, w_end, 5000.0, mcfg)
        benchmarks["SPY"][wname] = m_spy
        if wname == "FULL_5Y":
            spy_curves_5y["eur5000"] = eq_spy_5k
            eq_spy_1k, _ = _compute_spy_benchmark(spy_bars, w_start, w_end, 1000.0, mcfg)
            spy_curves_5y["eur1000"] = eq_spy_1k

    v5_json_path = outputs_dir / "v5_etf_results.json"
    if v5_json_path.exists():
        v5_data = json.loads(v5_json_path.read_text())
        for wname in ("IS", "OOS", "FULL_5Y"):
            benchmarks["v5_saved"][wname] = {
                "CSPX.AS": v5_data["benchmarks_theoretical"][wname]["CSPX.AS"],
                "SPY_v5_eu_cal": v5_data["benchmarks_theoretical"][wname]["SPY"],
                "v5b_QQQ3_cash_eur1000": v5_data["strategies"]["v5b_QQQ3_cash"][wname]["eur1000"]["metrics"],
                "v5b_QQQ3_cash_eur5000": v5_data["strategies"]["v5b_QQQ3_cash"][wname]["eur5000"]["metrics"],
            }

    part_b: dict[str, Any] = {}
    full5y_results: dict[tuple[str, str, bool], Form4PortfolioResult] = {}

    for vname, cfg_v, ev_list in (
        ("v9a_standard", cfg_v9a, eval_v9a),
        ("v9b_high_conviction", cfg_v9b, eval_v9b),
    ):
        part_b[vname] = {}
        for mode_label, is_pen in (("raw_matched", False), ("survivorship_penalized", True)):
            part_b[vname][mode_label] = {}
            for wname, (w_start, w_end) in WINDOWS.items():
                part_b[vname][mode_label][wname] = {}
                for cap_key, cap_eur, cap_usd in CAPITALS:
                    res = run_form4_portfolio_backtest(
                        evaluated=ev_list,
                        bars=bars_map,
                        spy_bars=spy_bars,
                        cfg=cfg_v,
                        window_start=w_start,
                        window_end=w_end,
                        initial_capital_usd=cap_usd,
                        initial_capital_eur=cap_eur,
                        window_label=wname,
                        survivorship_penalized=is_pen,
                    )
                    m = compute_form4_portfolio_metrics(
                        res,
                        mcfg,
                        standalone_trials_n=10,
                        cumulative_trials_n=45,
                        cumulative_trials_conservative_n=65,
                    )
                    # Continuous OOS segment return inside FULL_5Y
                    if wname == "FULL_5Y":
                        full5y_results[(vname, cap_key, is_pen)] = res
                        eq = res.equity_curve["equity"]
                        pre_oos = eq.loc[eq.index <= pd.Timestamp(IS_END)]
                        if not pre_oos.empty:
                            m["continuous_oos_segment_return"] = float(eq.iloc[-1] / pre_oos.iloc[-1] - 1.0)
                        winners, traps = extract_top_winners_and_traps(res, n=15)
                        part_b[vname][mode_label][wname][cap_key] = {
                            "metrics": m,
                            "top_15_winners": winners,
                            "worst_15_traps": traps,
                            "total_trades_and_open": len(res.trades) + len(res.open_positions_at_end),
                        }
                    else:
                        part_b[vname][mode_label][wname][cap_key] = {
                            "metrics": m,
                            "total_trades_and_open": len(res.trades) + len(res.open_positions_at_end),
                        }

    # Generate EUR 1,000 and EUR 5,000 charts in BOTH outputs/ and Project store media/
    media_paths: dict[str, str] = {}
    oos_split_ts = pd.Timestamp(OOS_START)
    for cap_key, cap_eur, _ in CAPITALS:
        v5b_eq, cspx_eq = _load_v5b_and_cspx_curves(cache_root, cap_eur)
        local_png = outputs_dir / f"v9_form4_{cap_key}.png"
        store_png = STORE_ROOT / "media" / f"v9-form4-{cap_key}.png"
        _plot_v9_png(
            capital_eur=cap_eur,
            v9a_raw=full5y_results[("v9a_standard", cap_key, False)],
            v9a_pen=full5y_results[("v9a_standard", cap_key, True)],
            v9b_raw=full5y_results[("v9b_high_conviction", cap_key, False)],
            v9b_pen=full5y_results[("v9b_high_conviction", cap_key, True)],
            spy_eq_eur=spy_curves_5y[cap_key],
            cspx_eq_eur=cspx_eq,
            v5b_eq_eur=v5b_eq,
            oos_split=oos_split_ts,
            path=local_png,
        )
        store_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_png, store_png)
        media_paths[f"{cap_key}_local"] = str(local_png)
        media_paths[f"{cap_key}_store"] = str(store_png)

    # Save structured JSON
    results_payload = {
        "windows": {k: {"start": str(v[0]), "end": str(v[1])} for k, v in WINDOWS.items()},
        "sec_dera_manifest": sec_manifest,
        "cluster_detection_stats": {
            "v9a_standard": cstats_v9a,
            "v9b_high_conviction": cstats_v9b,
        },
        "survivorship_audit_5y": {
            "v9a_standard": surv_v9a,
            "v9b_high_conviction": surv_v9b,
        },
        "survivorship_by_window": window_surv,
        "early_terminated_matched_events": early_term_records,
        "part_a_event_study": part_a,
        "benchmarks": benchmarks,
        "part_b_portfolio": part_b,
        "media_paths": media_paths,
        "runtime_seconds": round(time.monotonic() - t0, 2),
    }

    out_json = outputs_dir / "v9_form4_results.json"
    out_json.write_text(json.dumps(_clean_json(results_payload), indent=2))
    log.info("Done in %.1fs! Results written to %s", time.monotonic() - t0, out_json)

    # Concise console summary
    for vname in ("v9a_standard", "v9b_high_conviction"):
        sa = results_payload["survivorship_audit_5y"][vname]
        es_raw = part_a[vname]["FULL_5Y"]["raw_matched"]["horizons"]["+90d"]
        es_pen = part_a[vname]["FULL_5Y"]["survivorship_penalized"]["horizons"]["+90d"]
        m_raw = part_b[vname]["raw_matched"]["FULL_5Y"]["eur5000"]["metrics"]
        m_pen = part_b[vname]["survivorship_penalized"]["FULL_5Y"]["eur5000"]["metrics"]
        log.info(
            "SUMMARY %s | Matched=%d Unmatched=%d (PxElig=%d) EarlyTerm=%d | "
            "PartA +90d ExRet Raw=%+.2f%% (p=%.4f) Pen=%+.2f%% | "
            "PartB 5Y EUR5k Raw=%+.1f%% (CAGR=%+.1f%%, SR=%+.2f, DD=%.1f%%) "
            "Pen=%+.1f%% (CAGR=%+.1f%%, SR=%+.2f)",
            vname,
            sa["qualified_matched"],
            sa["unmatched_yfinance"],
            sa["unmatched_price_eligible"],
            sa["early_terminated_before_90d"],
            (es_raw["mean_excess_ret"] or 0) * 100,
            es_raw["p_value_ttest"] or 1.0,
            (es_pen["mean_excess_ret"] or 0) * 100,
            m_raw["total_return"] * 100,
            m_raw["cagr"] * 100,
            m_raw["sharpe"] or 0,
            m_raw["max_drawdown"] * 100,
            m_pen["total_return"] * 100,
            m_pen["cagr"] * 100,
            m_pen["sharpe"] or 0,
        )


if __name__ == "__main__":
    main()
