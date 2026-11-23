"""Strategy v10_coffee_can (Layer 1): Quantitative Asymmetric Compounder Backtest.

Strict Anti-Bias Protocol (ALL rules locked and 168 unit tests passed BEFORE running
Blind OOS once; zero OOS tuning):
- SEC DERA Financial Statement Data Sets (XBRL): 2020q4 through 2026q3 (24 quarters,
  cached under data/sec_xbrl/).
- Rebalance Dates: First US trading day of April each year (2022-04 through 2026-04).
- IS Vintages:     2022-04, 2023-04, 2024-04 (evaluated over 2022-04-01..2025-03-31 & to end)
- Blind OOS:       2025-04, 2026-04 (evaluated standalone 2025-04-01..2026-11-20 & continuous)
- Full Window:     2022-04-01 to 2026-11-20 (4.64 years) — benchmarks computed over the
                   IDENTICAL 2022-04..2026-11-20 window alongside saved 5Y reference figures.

Outputs:
- outputs/v10_coffee_can_results.json
- outputs/v10_coffee_can_eur1000.png & Project store media/v10-coffee-can-eur1000.png
- outputs/v10_coffee_can_eur5000.png & Project store media/v10-coffee-can-eur5000.png
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
    batch_load_yfinance_bars,
    load_sec_company_tickers_exchange,
)
from src.data.sec_xbrl_loader import (  # noqa: E402
    DEFAULT_XBRL_QUARTERS,
    SecXbrlLoader,
)
from src.domain.models import BarSeries  # noqa: E402
from src.strategy.coffee_can_screener import (  # noqa: E402
    US_PERIODS_PER_YEAR,
    CoffeeCanConfig,
    CoffeeCanPortfolioResult,
    build_thesis_watch_events,
    compute_coffee_can_metrics,
    evaluate_vintage_candidates,
    first_trading_day_of_april,
    run_coffee_can_event_study,
    run_coffee_can_portfolio_backtest,
    screen_fundamentals_asof,
)
from src.strategy.ucits_etf_regime import (  # noqa: E402
    RiskOffMode,
    SizingMode,
    UcitsRegimeConfig,
    run_ucits_regime_backtest,
)

VINTAGE_YEARS: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
IS_VINTAGES: tuple[str, ...] = ("2022-04", "2023-04", "2024-04")
OOS_VINTAGES: tuple[str, ...] = ("2025-04", "2026-04")

SAMPLE_END = date(2026, 11, 20)
BAR_FETCH_START = date(2020, 10, 18)
BAR_FETCH_END = date(2026, 11, 20)
EUR_USD_RATE = 1.06

CAPITALS: tuple[tuple[str, float, float], ...] = (
    ("eur1000", 1000.0, 1060.0),
    ("eur5000", 5000.0, 5300.0),
)

STORE_ROOT = Path(
    "/home/margarida/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-de418035-892d-40a4-a21a-bd834b742bee/files"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("v10_coffee_can")


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


def _compute_bh_benchmark(
    bar_series: BarSeries,
    window_start: date,
    window_end: date,
    initial_eur: float,
    mcfg: MetricsConfig,
) -> tuple[pd.Series, dict[str, Any]]:
    df = bar_series.frame
    sub = df.loc[(df.index >= pd.Timestamp(window_start)) & (df.index <= pd.Timestamp(window_end))]
    prev = df.loc[df.index < pd.Timestamp(window_start)]
    base_px = float(prev["close"].iloc[-1]) if not prev.empty else float(sub["open"].iloc[0])
    eq = (sub["close"] / base_px) * initial_eur
    r = daily_returns(eq, initial_eur)
    cagr_v = cagr(eq, initial_eur)
    mdd_v = max_drawdown(eq, initial_eur)
    calmar_v = cagr_v / abs(mdd_v) if mdd_v < 0 else float("nan")
    return eq, {
        "initial_eur": initial_eur,
        "final_eur": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial_eur - 1.0),
        "cagr": cagr_v,
        "sharpe": sharpe(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "sortino": sortino(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "dsr_n10": deflated_sharpe(r, US_PERIODS_PER_YEAR, 10, mcfg.dsr_trial_sharpe_std),
        "dsr_n50": deflated_sharpe(r, US_PERIODS_PER_YEAR, 50, mcfg.dsr_trial_sharpe_std),
        "dsr_n70": deflated_sharpe(r, US_PERIODS_PER_YEAR, 70, mcfg.dsr_trial_sharpe_std),
        "max_drawdown": mdd_v,
        "calmar": calmar_v,
    }


def _compute_v5b_window(
    cache_root: Path,
    window_start: date,
    window_end: date,
    capital_eur: float,
    mcfg: MetricsConfig,
) -> tuple[pd.Series | None, dict[str, Any] | None]:
    cache = DiskCache(cache_root, "yfinance")
    loaded: dict[str, BarSeries] = {}
    for sym in ("QQQ", "QQQ3.MI"):
        key = f"{sym}_2020-10-18_2026-11-20"
        if cache.status("bars", key) == "ok":
            df = cache.read("bars", key)
            if df is not None and not df.empty:
                loaded[sym] = BarSeries(symbol=sym, frame=df)
    if "QQQ" not in loaded or "QQQ3.MI" not in loaded:
        return None, None

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
    res = run_ucits_regime_backtest(
        signal_bars=loaded["QQQ"],
        risk_on_bars=loaded["QQQ3.MI"],
        risk_off_bars=None,
        cfg=cfg_v5b,
        start_date=window_start,
        end_date=window_end,
        initial_capital_eur=capital_eur,
    )
    eq = res.equity_curve["equity"]
    r = daily_returns(eq, capital_eur)
    days = (eq.index[-1] - eq.index[0]).days + 1
    years = max(days / 365.25, 1e-6)
    cagr_v = cagr(eq, capital_eur)
    mdd_v = max_drawdown(eq, capital_eur)
    return eq, {
        "initial_eur": capital_eur,
        "final_eur": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / capital_eur - 1.0),
        "cagr": cagr_v,
        "sharpe": sharpe(r, US_PERIODS_PER_YEAR, mcfg.risk_free_rate),
        "dsr_n10": deflated_sharpe(r, US_PERIODS_PER_YEAR, 10, mcfg.dsr_trial_sharpe_std),
        "dsr_n50": deflated_sharpe(r, US_PERIODS_PER_YEAR, 50, mcfg.dsr_trial_sharpe_std),
        "dsr_n70": deflated_sharpe(r, US_PERIODS_PER_YEAR, 70, mcfg.dsr_trial_sharpe_std),
        "max_drawdown": mdd_v,
        "calmar": cagr_v / abs(mdd_v) if mdd_v < 0 else float("nan"),
        "orders_per_year": float(len(res.fills) / years),
    }


def _plot_v10_png(
    *,
    capital_eur: float,
    v10a_raw: CoffeeCanPortfolioResult,
    v10a_pen: CoffeeCanPortfolioResult,
    v10b_raw: CoffeeCanPortfolioResult,
    v10b_pen: CoffeeCanPortfolioResult,
    spy_eq: pd.Series,
    cspx_eq: pd.Series | None,
    v5b_eq: pd.Series | None,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    init = float(capital_eur)
    scale = 1.0 / EUR_USD_RATE

    eq_a_raw = v10a_raw.equity_curve["equity"] * scale
    eq_a_pen = v10a_pen.equity_curve["equity"] * scale
    eq_b_raw = v10b_raw.equity_curve["equity"] * scale
    eq_b_pen = v10b_pen.equity_curve["equity"] * scale

    def _dd(s: pd.Series) -> pd.Series:
        return (s / s.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14.4, 10.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.25, 1.15]},
    )

    ax1.plot(
        eq_a_raw.index,
        eq_a_raw.values,
        color="#1f77b4",
        lw=2.2,
        label=f"v10a Annual Top-5 (Raw Matched)      | EUR {eq_a_raw.iloc[-1]:,.0f} ({(eq_a_raw.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        eq_a_pen.index,
        eq_a_pen.values,
        color="#1f77b4",
        lw=1.35,
        ls="--",
        alpha=0.78,
        label=f"v10a Annual Top-5 (Surv-Pen -50%)    | EUR {eq_a_pen.iloc[-1]:,.0f} ({(eq_a_pen.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        eq_b_raw.index,
        eq_b_raw.values,
        color="#d62728",
        lw=2.2,
        label=f"v10b Coffee Can 3Y (Raw Matched)     | EUR {eq_b_raw.iloc[-1]:,.0f} ({(eq_b_raw.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        eq_b_pen.index,
        eq_b_pen.values,
        color="#d62728",
        lw=1.35,
        ls="--",
        alpha=0.78,
        label=f"v10b Coffee Can 3Y (Surv-Pen -50%)   | EUR {eq_b_pen.iloc[-1]:,.0f} ({(eq_b_pen.iloc[-1]/init-1)*100:+.1f}%)",
    )
    ax1.plot(
        spy_eq.index,
        spy_eq.values,
        color="#555555",
        lw=1.5,
        ls=":",
        label=f"SPY B&H (2022-04..2026-11)           | EUR {spy_eq.iloc[-1]:,.0f} ({(spy_eq.iloc[-1]/init-1)*100:+.1f}%)",
    )
    if cspx_eq is not None:
        ax1.plot(
            cspx_eq.index,
            cspx_eq.values,
            color="#2ca02c",
            lw=1.5,
            ls="-.",
            alpha=0.85,
            label=f"CSPX.AS B&H (2022-04..2026-11)       | EUR {cspx_eq.iloc[-1]:,.0f} ({(cspx_eq.iloc[-1]/init-1)*100:+.1f}%)",
        )
    if v5b_eq is not None:
        ax1.plot(
            v5b_eq.index,
            v5b_eq.values,
            color="#9467bd",
            lw=1.8,
            alpha=0.88,
            label=f"v5b QQQ3.MI (2022-04..2026-11)       | EUR {v5b_eq.iloc[-1]:,.0f} ({(v5b_eq.iloc[-1]/init-1)*100:+.1f}%)",
        )

    ax1.axhline(init, color="#999999", lw=0.8, ls=":")
    ax1.axvline(oos_split, color="#222222", lw=1.3, ls="--", alpha=0.85)
    ax1.text(
        oos_split + pd.Timedelta(days=10),
        ax1.get_ylim()[1] * 0.94,
        "Blind OOS Start\n(2025-04)",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        va="top",
    )
    ax1.set_title(
        f"Strategy v10_coffee_can (SEC XBRL Rule-of-40 Top-5 Compounders) — EUR {int(capital_eur):,} "
        f"(${int(round(capital_eur * EUR_USD_RATE)):,}) IBKR Cash Account (2022-04 to 2026-11-20)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Equity (EUR equiv.)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    ax2.plot(eq_a_raw.index, _dd(eq_a_raw), color="#1f77b4", lw=1.4, label="v10a Raw DD")
    ax2.plot(eq_b_raw.index, _dd(eq_b_raw), color="#d62728", lw=1.5, label="v10b Raw DD")
    ax2.plot(spy_eq.index, _dd(spy_eq), color="#555555", lw=1.1, ls=":", label="SPY DD")
    if v5b_eq is not None:
        ax2.plot(v5b_eq.index, _dd(v5b_eq), color="#9467bd", lw=1.2, alpha=0.8, label="v5b QQQ3.MI DD")
    ax2.axvline(oos_split, color="#222222", lw=1.1, ls="--", alpha=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=8.3, ncol=4)

    ax3.fill_between(
        v10a_raw.equity_curve.index,
        0,
        v10a_raw.equity_curve["exposure_pct"] * 100.0,
        color="#1f77b4",
        alpha=0.26,
        label="v10a Raw Exposure (%)",
    )
    ax3.plot(
        v10b_raw.equity_curve.index,
        v10b_raw.equity_curve["exposure_pct"] * 100.0,
        color="#d62728",
        lw=1.4,
        alpha=0.9,
        label="v10b Raw Exposure (%)",
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
    xbrl_root = ROOT / "data" / "sec_xbrl"
    form4_root = ROOT / "data" / "sec_form4"
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SEC XBRL Financial Statement Data Sets (2020q4..2026q3)
    log.info("Step 1/6: Loading SEC XBRL datasets (%d quarters)...", len(DEFAULT_XBRL_QUARTERS))
    loader = SecXbrlLoader(data_root=xbrl_root)
    annual_df, quarterly_df, xbrl_manifest = loader.load_quarters(DEFAULT_XBRL_QUARTERS)
    log.info(
        "SEC XBRL loaded: %d/%d quarters, %d 10-K rows (%s..%s), %d 10-Q rows, unresolved_filings=%d",
        len(xbrl_manifest["downloaded_quarters"]),
        len(DEFAULT_XBRL_QUARTERS),
        xbrl_manifest["total_10k_rows"],
        xbrl_manifest["filed_date_min_10k"],
        xbrl_manifest["filed_date_max_10k"],
        xbrl_manifest["total_10q_rows"],
        xbrl_manifest["unresolved_filings_count"],
    )

    # 2. Load SPY & CSPX.AS cached bars and resolve April rebalance dates
    log.info("Step 2/6: Resolving April rebalance dates & screening causal fundamentals...")
    base_bars, _ = batch_load_yfinance_bars(["SPY", "CSPX.AS"], cache_root=cache_root, start=BAR_FETCH_START, end=BAR_FETCH_END)
    if "SPY" not in base_bars:
        raise RuntimeError("SPY benchmark bars unavailable!")
    spy_bars = base_bars["SPY"]
    cspx_bars = base_bars.get("CSPX.AS")
    spy_cal = spy_bars.frame.index.sort_values()

    rebalance_dates: dict[str, date] = {}
    for yr in VINTAGE_YEARS:
        rebalance_dates[f"{yr}-04"] = first_trading_day_of_april(spy_cal, yr)
    log.info("Rebalance dates (first trading day of April): %s", {k: str(v) for k, v in rebalance_dates.items()})

    cfg_base = CoffeeCanConfig(rule="v10a_annual_top5")
    cands_by_vintage: dict[str, list[Any]] = {}
    fund_audit_by_vintage: dict[str, dict[str, int]] = {}
    all_cand_tickers: set[str] = {"SPY"}
    for vlabel, rdt in rebalance_dates.items():
        cands, f_aud = screen_fundamentals_asof(annual_df, rdt, cfg_base, vintage_label=vlabel)
        cands_by_vintage[vlabel] = cands
        fund_audit_by_vintage[vlabel] = f_aud
        for c in cands:
            if c.ticker:
                all_cand_tickers.add(c.ticker)
        log.info(
            "Vintage %s (%s): fresh_10k=%d -> fundamental_qualifiers=%d (unresolved_ticker=%d)",
            vlabel,
            rdt,
            f_aud["total_10k_asof_fresh"],
            f_aud["fundamental_qualifiers"],
            f_aud["fundamental_qualifiers_unresolved_ticker"],
        )

    # 3. Batch-load yfinance bars for all fundamental qualifiers (reusing existing cache!)
    log.info("Step 3/6: Loading/caching yfinance bars for %d unique fundamental qualifier tickers...", len(all_cand_tickers))
    bars_map, fetch_statuses = batch_load_yfinance_bars(
        sorted(all_cand_tickers),
        cache_root=cache_root,
        start=BAR_FETCH_START,
        end=BAR_FETCH_END,
    )
    ticker_meta = load_sec_company_tickers_exchange(data_root=form4_root)
    log.info(
        "yfinance bars loaded: %d/%d qualifier tickers have bars (%d missing/delisted, %d rate_limited)",
        sum(1 for t in all_cand_tickers if t != "SPY" and t in bars_map),
        len(all_cand_tickers) - 1,
        sum(1 for t, st in fetch_statuses.items() if t != "SPY" and st.endswith("empty")),
        sum(1 for t, st in fetch_statuses.items() if t != "SPY" and st == "rate_limited"),
    )

    # 4. Evaluate strictly causal tradability & survivorship audit per vintage
    log.info("Step 4/6: Evaluating causal tradability ($300M-$10B mcap, $5-$150 px, $2M ADV) & survivorship...")
    eval_by_vintage: dict[str, list[Any]] = {}
    surv_by_vintage: dict[str, dict[str, Any]] = {}
    for vlabel, cands in cands_by_vintage.items():
        ev_list, s_aud = evaluate_vintage_candidates(cands, bars_map, ticker_meta, spy_bars, cfg_base)
        eval_by_vintage[vlabel] = ev_list
        surv_by_vintage[vlabel] = s_aud
        log.info(
            "Vintage %s: fund_qual=%d -> qualified_matched=%d, unmatched_yf=%d (%.1f%% of common), rej_mcap=%d, rej_px=%d, rej_adv=%d",
            vlabel,
            s_aud["fundamental_qualifiers"],
            s_aud["qualified_matched"],
            s_aud["unmatched_yfinance"],
            s_aud["unmatched_pct_of_fundamental_qualifiers"] * 100.0,
            s_aud["rejected_market_cap"],
            s_aud["rejected_price"],
            s_aud["rejected_adv"],
        )

    # Aggregate IS vs OOS survivorship
    def _agg_surv(vkeys: tuple[str, ...]) -> dict[str, Any]:
        fq = sum(surv_by_vintage[v]["fundamental_qualifiers"] for v in vkeys)
        cfq = sum(surv_by_vintage[v]["common_fundamental_qualifiers"] for v in vkeys)
        qm = sum(surv_by_vintage[v]["qualified_matched"] for v in vkeys)
        unm = sum(surv_by_vintage[v]["unmatched_yfinance"] for v in vkeys)
        return {
            "fundamental_qualifiers": fq,
            "common_fundamental_qualifiers": cfq,
            "qualified_matched": qm,
            "unmatched_yfinance": unm,
            "unmatched_pct_of_common": unm / cfq if cfq > 0 else 0.0,
        }

    surv_summary = {
        "IS": _agg_surv(IS_VINTAGES),
        "OOS": _agg_surv(OOS_VINTAGES),
        "ALL": _agg_surv(tuple(rebalance_dates.keys())),
    }

    # 5. Part A — Qualifying Pool Event Study + Top-5 Tables
    log.info("Step 5/6: Running Part A Event Study (1Y/2Y/3Y vs SPY, Multi-Baggers, Blow-Ups)...")
    part_a = run_coffee_can_event_study(
        eval_by_vintage,
        bars_map,
        spy_bars,
        SAMPLE_END,
        cfg_base,
        is_vintages=IS_VINTAGES,
        oos_vintages=OOS_VINTAGES,
    )

    # 6. Part B — Portfolio Backtests (v10a & v10b) + Identical-Window & Full-5Y Benchmarks
    log.info("Step 6/6: Running Part B Portfolio Simulations (v10a & v10b, EUR 1k & EUR 5k, Raw & Penalized)...")
    thesis_events = build_thesis_watch_events(annual_df, quarterly_df)

    full_start = rebalance_dates["2022-04"]
    oos_start = rebalance_dates["2025-04"]
    is_end = date(2025, 3, 31)

    bt_windows: dict[str, tuple[date, date]] = {
        "IS": (full_start, is_end),
        "OOS": (oos_start, SAMPLE_END),
        "FULL_CC": (full_start, SAMPLE_END),
    }

    benchmarks: dict[str, Any] = {"identical_window": {}, "saved_full_5y": {}}
    spy_curves_full: dict[str, pd.Series] = {}
    cspx_curves_full: dict[str, pd.Series | None] = {}
    v5b_curves_full: dict[str, pd.Series | None] = {}

    for wname, (ws, we) in bt_windows.items():
        eq_spy5, m_spy5 = _compute_bh_benchmark(spy_bars, ws, we, 5000.0, mcfg)
        m_cspx5 = None
        eq_cspx5 = None
        if cspx_bars is not None:
            eq_cspx5, m_cspx5 = _compute_bh_benchmark(cspx_bars, ws, we, 5000.0, mcfg)
        eq_v5b5, m_v5b5 = _compute_v5b_window(cache_root, ws, we, 5000.0, mcfg)
        eq_v5b1, m_v5b1 = _compute_v5b_window(cache_root, ws, we, 1000.0, mcfg)
        benchmarks["identical_window"][wname] = {
            "start": str(ws),
            "end": str(we),
            "SPY": m_spy5,
            "CSPX.AS": m_cspx5,
            "v5b_QQQ3_cash_eur5000": m_v5b5,
            "v5b_QQQ3_cash_eur1000": m_v5b1,
        }
        if wname == "FULL_CC":
            spy_curves_full["eur5000"] = eq_spy5
            cspx_curves_full["eur5000"] = eq_cspx5
            v5b_curves_full["eur5000"] = eq_v5b5
            eq_spy1, _ = _compute_bh_benchmark(spy_bars, ws, we, 1000.0, mcfg)
            spy_curves_full["eur1000"] = eq_spy1
            cspx_curves_full["eur1000"] = (
                _compute_bh_benchmark(cspx_bars, ws, we, 1000.0, mcfg)[0] if cspx_bars is not None else None
            )
            v5b_curves_full["eur1000"] = eq_v5b1

    v5_json_path = outputs_dir / "v5_etf_results.json"
    if v5_json_path.exists():
        v5_d = json.loads(v5_json_path.read_text())
        benchmarks["saved_full_5y"] = {
            "CSPX.AS": v5_d["benchmarks_theoretical"]["FULL_5Y"]["CSPX.AS"],
            "v5b_QQQ3_cash_eur5000": v5_d["strategies"]["v5b_QQQ3_cash"]["FULL_5Y"]["eur5000"]["metrics"],
            "v5b_QQQ3_cash_eur1000": v5_d["strategies"]["v5b_QQQ3_cash"]["FULL_5Y"]["eur1000"]["metrics"],
        }
    v9_json_path = outputs_dir / "v9_form4_results.json"
    if v9_json_path.exists():
        v9_d = json.loads(v9_json_path.read_text())
        benchmarks["saved_full_5y"]["SPY"] = v9_d["benchmarks"]["SPY"]["FULL_5Y"]

    part_b: dict[str, Any] = {}
    full_results: dict[tuple[str, str, bool], CoffeeCanPortfolioResult] = {}

    for rule_name in ("v10a_annual_top5", "v10b_coffee_can_thesis_watch"):
        cfg_r = CoffeeCanConfig(rule=rule_name)  # type: ignore[arg-type]
        part_b[rule_name] = {}
        for mode_label, is_pen in (("raw_matched", False), ("survivorship_penalized", True)):
            part_b[rule_name][mode_label] = {}
            for wname, (ws, we) in bt_windows.items():
                part_b[rule_name][mode_label][wname] = {}
                for cap_key, cap_eur, cap_usd in CAPITALS:
                    res = run_coffee_can_portfolio_backtest(
                        evaluated_by_vintage=eval_by_vintage,
                        rebalance_dates=rebalance_dates,
                        bars=bars_map,
                        spy_bars=spy_bars,
                        thesis_events_by_cik=thesis_events,
                        cfg=cfg_r,
                        window_start=ws,
                        window_end=we,
                        initial_capital_usd=cap_usd,
                        initial_capital_eur=cap_eur,
                        window_label=wname,
                        survivorship_penalized=is_pen,
                    )
                    m = compute_coffee_can_metrics(
                        res,
                        mcfg,
                        standalone_trials_n=10,
                        cumulative_trials_n=50,
                        cumulative_trials_conservative_n=70,
                    )
                    if wname == "FULL_CC":
                        full_results[(rule_name, cap_key, is_pen)] = res
                        eq = res.equity_curve["equity"]
                        pre_oos = eq.loc[eq.index <= pd.Timestamp(is_end)]
                        if not pre_oos.empty:
                            m["continuous_oos_segment_return"] = float(eq.iloc[-1] / pre_oos.iloc[-1] - 1.0)
                        all_t = [t.model_dump(mode="json") for t in list(res.trades) + list(res.open_positions_at_end)]
                        part_b[rule_name][mode_label][wname][cap_key] = {
                            "metrics": m,
                            "trades_and_open": all_t,
                        }
                    else:
                        part_b[rule_name][mode_label][wname][cap_key] = {
                            "metrics": m,
                        }

    # Generate EUR 1,000 and EUR 5,000 charts in BOTH outputs/ and Project store media/
    media_paths: dict[str, str] = {}
    oos_split_ts = pd.Timestamp(oos_start)
    for cap_key, cap_eur, _ in CAPITALS:
        local_png = outputs_dir / f"v10_coffee_can_{cap_key}.png"
        store_png = STORE_ROOT / "media" / f"v10-coffee-can-{cap_key}.png"
        _plot_v10_png(
            capital_eur=cap_eur,
            v10a_raw=full_results[("v10a_annual_top5", cap_key, False)],
            v10a_pen=full_results[("v10a_annual_top5", cap_key, True)],
            v10b_raw=full_results[("v10b_coffee_can_thesis_watch", cap_key, False)],
            v10b_pen=full_results[("v10b_coffee_can_thesis_watch", cap_key, True)],
            spy_eq=spy_curves_full[cap_key],
            cspx_eq=cspx_curves_full[cap_key],
            v5b_eq=v5b_curves_full[cap_key],
            oos_split=oos_split_ts,
            path=local_png,
        )
        store_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_png, store_png)
        media_paths[f"{cap_key}_local"] = str(local_png)
        media_paths[f"{cap_key}_store"] = str(store_png)

    payload = {
        "rebalance_dates": {k: str(v) for k, v in rebalance_dates.items()},
        "windows": {k: {"start": str(v[0]), "end": str(v[1])} for k, v in bt_windows.items()},
        "xbrl_manifest": xbrl_manifest,
        "fundamental_screening_by_vintage": fund_audit_by_vintage,
        "survivorship_by_vintage": surv_by_vintage,
        "survivorship_summary": surv_summary,
        "part_a_event_study": part_a,
        "benchmarks": benchmarks,
        "part_b_portfolio": part_b,
        "media_paths": media_paths,
        "runtime_seconds": round(time.monotonic() - t0, 2),
    }

    out_json = outputs_dir / "v10_coffee_can_results.json"
    out_json.write_text(json.dumps(_clean_json(payload), indent=2))
    log.info("Done in %.1fs! Results written to %s", time.monotonic() - t0, out_json)

    for rule_name in ("v10a_annual_top5", "v10b_coffee_can_thesis_watch"):
        mr = part_b[rule_name]["raw_matched"]["FULL_CC"]["eur5000"]["metrics"]
        mp = part_b[rule_name]["survivorship_penalized"]["FULL_CC"]["eur5000"]["metrics"]
        log.info(
            "SUMMARY %s (EUR5k, 2022-04..2026-11) | Raw=%+.1f%% (CAGR=%+.1f%%, SR=%+.2f, DSR50=%.3f, DD=%.1f%%, ord/yr=%.1f) | "
            "Pen=%+.1f%% (CAGR=%+.1f%%, SR=%+.2f, DD=%.1f%%)",
            rule_name,
            mr["total_return"] * 100,
            mr["cagr"] * 100,
            mr["sharpe"] or 0.0,
            mr["dsr_n50"] or 0.0,
            mr["max_drawdown"] * 100,
            mr["orders_per_year"],
            mp["total_return"] * 100,
            mp["cagr"] * 100,
            mp["sharpe"] or 0.0,
            mp["max_drawdown"] * 100,
        )


if __name__ == "__main__":
    main()
