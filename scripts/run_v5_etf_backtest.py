"""Strategy v5 Backtest & Validation: Regime-Filtered Leveraged EU UCITS ETF/ETP Rotation.

Strict Out-of-Sample Protocol (ALL rules locked BEFORE touching OOS):
- Warmup:     2020-10-18 to 2021-11-21
- In-Sample:  2021-11-22 to 2024-11-21 (3 years)
- Blind OOS:  2024-11-22 to 2026-11-20 (2 years, run ONCE with locked rules)
- Full 5Y:    2021-11-22 to 2026-11-20

Executes native EUR simulations for IBKR Ireland EUR cash account:
- EUR 1,000 and EUR 5,000 starting capital (zero FX conversion fees)
- EUR 1.25/order commission + 0.10% slippage per side + EUR 0.00/mo hosting
- Strict whole-share as-traded sizing (plus labeled post-split affordability sensitivity)
- Strict T+2 European ETF settlement (zero Good Faith Violations)
- Strict US-close (22:00 CET) -> next-EU-open (eu_date > us_date) alignment

Outputs:
- Structured JSON: outputs/v5_etf_results.json
- Equity curve PNGs (IS/OOS split) in BOTH outputs/ and Project store media/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from src.config import MetricsConfig, load_settings  # noqa: E402
from src.data.yfinance_provider import YFinanceProvider  # noqa: E402
from src.domain.models import BarSeries  # noqa: E402
from src.strategy.ucits_etf_regime import (  # noqa: E402
    RiskOffMode,
    SizingMode,
    UcitsBacktestResult,
    UcitsRegimeConfig,
    UcitsRegimeState,
    compute_theoretical_bh_metrics,
    compute_ucits_metrics,
    run_buy_and_hold_etf,
    run_ucits_regime_backtest,
)

WARMUP_START = date(2020, 10, 18)
IS_START = date(2021, 11, 22)
IS_END = date(2024, 11, 21)
OOS_START = date(2024, 11, 22)
OOS_END = date(2026, 11, 20)

STORE_ROOT = Path(
    "/home/margarida/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-de418035-892d-40a4-a21a-bd834b742bee/files"
)

TICKERS = ["SPY", "QQQ", "CSPX.AS", "SXR8.DE", "XEON.DE", "CL2.PA", "LQQ.PA", "QQQ3.MI"]


def _regime_spans_from_curve(curve: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return contiguous spans of RISK_OFF vs VOL_BREAKER for chart shading."""
    if curve.empty:
        return []
    labels = []
    for ts, row in curve.iterrows():
        if bool(row["vol_breaker_active"]):
            labels.append((ts, "VOL_BREAKER"))
        elif str(row["target_state"]) == UcitsRegimeState.RISK_OFF.value:
            labels.append((ts, "RISK_OFF"))
        else:
            labels.append((ts, "RISK_ON"))

    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    start_ts, cur_lbl = labels[0]
    prev_ts = start_ts
    for ts, lbl in labels[1:]:
        if lbl != cur_lbl:
            spans.append((start_ts, prev_ts, cur_lbl))
            start_ts, cur_lbl = ts, lbl
        prev_ts = ts
    spans.append((start_ts, prev_ts, cur_lbl))
    return spans


def _pack_result(res: UcitsBacktestResult, mcfg: MetricsConfig) -> dict[str, Any]:
    m = compute_ucits_metrics(res, mcfg)
    fills_log = [
        {
            "date": str(f.date),
            "signal_us_date": str(f.signal_us_date),
            "symbol": f.symbol,
            "side": f.side,
            "raw_shares": f.raw_shares,
            "adj_shares": round(f.adj_shares, 4),
            "raw_fill_price": round(f.raw_fill_price, 4),
            "notional_eur": round(f.notional_eur, 2),
            "commission_eur": round(f.commission_eur, 2),
            "slippage_eur": round(f.slippage_eur, 2),
            "reason": f.reason,
        }
        for f in res.fills
    ]
    trips_log = [
        {
            "symbol": t.symbol,
            "role": t.role,
            "entry_date": str(t.entry_date),
            "exit_date": str(t.exit_date),
            "raw_shares_entry": t.raw_shares_entry,
            "bars_held": t.bars_held,
            "net_pnl_eur": round(t.net_pnl_eur, 2),
            "return_pct": round(t.return_pct * 100.0, 2),
            "exit_reason": t.exit_reason,
            "is_open_at_end": t.is_open_at_end,
        }
        for t in (res.round_trips + ([res.open_position_at_end] if res.open_position_at_end else []))
    ]
    return {
        "metrics": m,
        "fills": fills_log,
        "round_trips": trips_log,
    }


def _plot_v5_equity_png(
    *,
    capital_eur: float,
    v5a_cl2_as: UcitsBacktestResult,
    v5a_cl2_ps: UcitsBacktestResult,
    v5a_lqq_as: UcitsBacktestResult,
    v5a_lqq_ps: UcitsBacktestResult,
    v5b_qqq3_cash: UcitsBacktestResult,
    v5b_qqq3_xeon: UcitsBacktestResult,
    bh_cspx_theo: pd.Series,
    bh_cl2_theo: pd.Series,
    bh_qqq3_as: UcitsBacktestResult,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Generate publication-grade 2-panel EUR equity + drawdown chart with IS/OOS split."""
    init = float(capital_eur)

    # Align all series to common EU calendar (QQQ3.MI / v5b cash curve index)
    c_v5b_cash = v5b_qqq3_cash.equity_curve
    c_v5b_xeon = v5b_qqq3_xeon.equity_curve
    eq_v5b_cash = c_v5b_cash["equity"]
    eq_v5b_xeon = c_v5b_xeon["equity"]
    eq_cl2_as = v5a_cl2_as.equity_curve["equity"]
    eq_cl2_ps = v5a_cl2_ps.equity_curve["equity"]
    eq_lqq_as = v5a_lqq_as.equity_curve["equity"]
    eq_lqq_ps = v5a_lqq_ps.equity_curve["equity"]
    eq_bh_qqq3 = bh_qqq3_as.equity_curve["equity"]

    def _dd(s: pd.Series) -> pd.Series:
        return (s / s.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.8, 8.0), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.25]}
    )

    # Background regime shading from v5b (QQQ SMA200+-1% + sigma20 > 0.35)
    for a, b, lbl in _regime_spans_from_curve(c_v5b_cash):
        if lbl == "RISK_OFF":
            for ax in (ax1, ax2):
                ax.axvspan(a, b, color="#f0d9d9", alpha=0.35, lw=0)
        elif lbl == "VOL_BREAKER":
            for ax in (ax1, ax2):
                ax.axvspan(a, b, color="#e69f9f", alpha=0.45, lw=0)

    # For EUR 1,000, CL2.PA and LQQ.PA as-traded were > EUR 1,000 pre-split (uninvestable until Oct 2023 / Jul 2026),
    # so plot BOTH honest as-traded AND clearly labeled post-split (modern price) curves where they diverge!
    is_1k = capital_eur <= 1500.0

    ax1.plot(
        eq_v5b_cash.index,
        eq_v5b_cash.values,
        label=f"v5b QQQ3.MI (3x) Cash [as-traded]  | EUR {eq_v5b_cash.iloc[-1]:,.0f} ({(eq_v5b_cash.iloc[-1]/init-1)*100:+.1f}%)",
        color="#1b7837",
        lw=2.2,
    )
    ax1.plot(
        eq_v5b_xeon.index,
        eq_v5b_xeon.values,
        label=f"v5b QQQ3.MI (3x) XEON [as-traded]  | EUR {eq_v5b_xeon.iloc[-1]:,.0f} ({(eq_v5b_xeon.iloc[-1]/init-1)*100:+.1f}%)",
        color="#66bd63",
        lw=1.5,
        ls="--",
    )

    ax1.plot(
        eq_cl2_as.index,
        eq_cl2_as.values,
        label=f"v5a CL2.PA (2x) Cash [as-traded]   | EUR {eq_cl2_as.iloc[-1]:,.0f} ({(eq_cl2_as.iloc[-1]/init-1)*100:+.1f}%)",
        color="#1f4e79",
        lw=2.0,
    )
    if is_1k:
        ax1.plot(
            eq_cl2_ps.index,
            eq_cl2_ps.values,
            label=f"v5a CL2.PA (2x) Cash [post-split]  | EUR {eq_cl2_ps.iloc[-1]:,.0f} ({(eq_cl2_ps.iloc[-1]/init-1)*100:+.1f}%)",
            color="#5698c6",
            lw=1.4,
            ls="-.",
        )

    ax1.plot(
        eq_lqq_as.index,
        eq_lqq_as.values,
        label=f"v5a LQQ.PA (2x) Cash [as-traded]   | EUR {eq_lqq_as.iloc[-1]:,.0f} ({(eq_lqq_as.iloc[-1]/init-1)*100:+.1f}%)",
        color="#d95f02",
        lw=1.8,
    )
    if is_1k:
        ax1.plot(
            eq_lqq_ps.index,
            eq_lqq_ps.values,
            label=f"v5a LQQ.PA (2x) Cash [post-split]  | EUR {eq_lqq_ps.iloc[-1]:,.0f} ({(eq_lqq_ps.iloc[-1]/init-1)*100:+.1f}%)",
            color="#fdae6b",
            lw=1.3,
            ls="-.",
        )

    ax1.plot(
        bh_cspx_theo.index,
        bh_cspx_theo.values,
        label=f"CSPX.AS (1x S&P500 EUR) B&H        | EUR {bh_cspx_theo.iloc[-1]:,.0f} ({(bh_cspx_theo.iloc[-1]/init-1)*100:+.1f}%)",
        color="#333333",
        lw=1.6,
        ls="--",
    )
    ax1.plot(
        bh_cl2_theo.index,
        bh_cl2_theo.values,
        label=f"CL2.PA (2x) Buy & Hold (unfiltered)| EUR {bh_cl2_theo.iloc[-1]:,.0f} ({(bh_cl2_theo.iloc[-1]/init-1)*100:+.1f}%)",
        color="#7570b3",
        lw=1.3,
        ls=":",
    )
    ax1.plot(
        eq_bh_qqq3.index,
        eq_bh_qqq3.values,
        label=f"QQQ3.MI (3x) Buy & Hold (unfiltered)| EUR {eq_bh_qqq3.iloc[-1]:,.0f} ({(eq_bh_qqq3.iloc[-1]/init-1)*100:+.1f}%)",
        color="#b2182b",
        lw=1.3,
        ls=":",
    )
    ax1.axhline(init, color="black", lw=0.6, alpha=0.45)

    for ax in (ax1, ax2):
        ax.axvline(oos_split, color="#8b0000", lw=1.7, ls="--", alpha=0.92)

    ymin, ymax = ax1.get_ylim()
    y_label = ymin + 0.96 * (ymax - ymin)
    ax1.text(
        oos_split - pd.Timedelta(days=20),
        y_label,
        "In-Sample (2021-11 to 2024-11)",
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", alpha=0.90),
    )
    ax1.text(
        oos_split + pd.Timedelta(days=20),
        y_label,
        "Blind OOS (2024-11 to 2026-11)",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.22", fc="#fff5f5", ec="#d4a0a0", alpha=0.92),
    )

    ax1.set_ylabel("Portfolio Equity (EUR)")
    ax1.set_title(
        f"Strategy v5: Regime-Filtered Leveraged UCITS ETF/ETP Rotation vs Benchmarks — EUR {int(init):,} (IBKR Ireland, EUR 0/mo Hosting)",
        fontsize=11.0,
    )
    ax1.legend(loc="upper left", fontsize=7.9, framealpha=0.92)
    ax1.grid(alpha=0.3)

    # Bottom panel: Drawdowns
    ax2.plot(eq_v5b_cash.index, _dd(eq_v5b_cash).values, color="#1b7837", lw=1.6, label="v5b QQQ3.MI Cash DD%")
    ax2.plot(eq_cl2_as.index, _dd(eq_cl2_as).values, color="#1f4e79", lw=1.4, label="v5a CL2.PA Cash DD%")
    ax2.plot(eq_lqq_as.index, _dd(eq_lqq_as).values, color="#d95f02", lw=1.2, label="v5a LQQ.PA Cash DD%")
    ax2.plot(bh_cspx_theo.index, _dd(bh_cspx_theo).values, color="#333333", lw=1.2, ls="--", label="CSPX.AS B&H DD%")
    ax2.plot(eq_bh_qqq3.index, _dd(eq_bh_qqq3).values, color="#b2182b", lw=1.1, ls=":", label="QQQ3.MI 3x B&H DD%")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", ncol=5, fontsize=7.8, framealpha=0.90)
    ax2.grid(alpha=0.3)

    fig.text(
        0.99,
        0.005,
        "shading: pink=QQQ RISK_OFF (SMA200±1%), coral=sigma_20>35% vol breaker | crimson dashed: IS/OOS split (2024-11-22) | US-close->next-EU-open, T+2 settlement, EUR 1.25 + 0.10%/side",
        ha="right",
        fontsize=7.6,
        color="#555555",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Strategy v5 IS / Blind OOS / Full 5Y backtests.")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v5_etf_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    mcfg = settings.metrics
    provider = YFinanceProvider(settings.data.cache_path)

    bars: dict[str, BarSeries] = provider.get_daily_bars(TICKERS, WARMUP_START, OOS_END)
    missing = [s for s in TICKERS if s not in bars]
    if missing:
        raise RuntimeError(f"Missing real Yahoo Finance bars for: {missing}")
    logging.info("Loaded real EUR/USD bars from cache/yfinance: %s", {k: len(v) for k, v in bars.items()})

    # Verify ticker metadata (split counts & price ranges in 5Y window)
    ticker_audit: dict[str, Any] = {}
    for sym, bs in bars.items():
        df5 = bs.frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
        ratio = (df5["raw_close"] / df5["close"]).round(4)
        split_changes = ratio.ne(ratio.shift(1))
        split_dates = [str(ts.date()) for ts in df5.index[split_changes][1:]]
        ticker_audit[sym] = {
            "rows_total": len(bs),
            "rows_5y": len(df5),
            "first_date": str(bs.frame.index[0].date()),
            "last_date": str(bs.frame.index[-1].date()),
            "adj_close_min_5y": round(float(df5["close"].min()), 4),
            "adj_close_max_5y": round(float(df5["close"].max()), 4),
            "raw_close_min_5y": round(float(df5["raw_close"].min()), 4),
            "raw_close_max_5y": round(float(df5["raw_close"].max()), 4),
            "raw_close_start": round(float(df5["raw_close"].iloc[0]), 4),
            "raw_close_oos_start": round(float(df5.loc[pd.Timestamp(OOS_START) :]["raw_close"].iloc[0]), 4),
            "raw_close_end": round(float(df5["raw_close"].iloc[-1]), 4),
            "split_dates_in_5y": split_dates,
            "synthetic": False,
        }

    # LOCKED STRATEGY CONFIGURATIONS (zero OOS tuning)
    # Primary (AS_TRADED) + Post-Split Affordability Sensitivity (POST_SPLIT) for split ETFs
    strategy_specs: list[UcitsRegimeConfig] = [
        # v5a: CL2.PA (2x MSCI USA EUR) gated by QQQ SMA(200) ± 1%
        UcitsRegimeConfig(
            name="v5a_CL2_cash",
            signal_symbol="QQQ",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        UcitsRegimeConfig(
            name="v5a_CL2_xeon",
            signal_symbol="QQQ",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.XEON,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        # v5a: LQQ.PA (2x Nasdaq-100 EUR) gated by QQQ SMA(200) ± 1%
        UcitsRegimeConfig(
            name="v5a_LQQ_cash",
            signal_symbol="QQQ",
            risk_on_symbol="LQQ.PA",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        UcitsRegimeConfig(
            name="v5a_LQQ_xeon",
            signal_symbol="QQQ",
            risk_on_symbol="LQQ.PA",
            risk_off_mode=RiskOffMode.XEON,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        # Labeled Reference: CL2.PA gated by SPY SMA(200) ± 1%
        UcitsRegimeConfig(
            name="v5a_CL2_SPYsig_cash",
            signal_symbol="SPY",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        UcitsRegimeConfig(
            name="v5a_CL2_SPYsig_xeon",
            signal_symbol="SPY",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.XEON,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        # v5b: QQQ3.MI (3x Nasdaq-100 EUR) gated by QQQ SMA(200) ± 1% + sigma_20 > 0.35 circuit breaker
        UcitsRegimeConfig(
            name="v5b_QQQ3_cash",
            signal_symbol="QQQ",
            risk_on_symbol="QQQ3.MI",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=0.35,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        UcitsRegimeConfig(
            name="v5b_QQQ3_xeon",
            signal_symbol="QQQ",
            risk_on_symbol="QQQ3.MI",
            risk_off_mode=RiskOffMode.XEON,
            vol_circuit_breaker=0.35,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        # Also test v5b WITHOUT vol breaker (ablation reference on QQQ3.MI to isolate vol circuit breaker impact!)
        UcitsRegimeConfig(
            name="v5b_QQQ3_novol_cash",
            signal_symbol="QQQ",
            risk_on_symbol="QQQ3.MI",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.AS_TRADED,
        ),
        # Post-split modern-price affordability sensitivity (CL2.PA ~EUR 29, LQQ.PA ~EUR 7.40)
        UcitsRegimeConfig(
            name="v5a_CL2_cash_postsplit",
            signal_symbol="QQQ",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.POST_SPLIT,
        ),
        UcitsRegimeConfig(
            name="v5a_CL2_xeon_postsplit",
            signal_symbol="QQQ",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.XEON,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.POST_SPLIT,
        ),
        UcitsRegimeConfig(
            name="v5a_LQQ_cash_postsplit",
            signal_symbol="QQQ",
            risk_on_symbol="LQQ.PA",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.POST_SPLIT,
        ),
        UcitsRegimeConfig(
            name="v5a_LQQ_xeon_postsplit",
            signal_symbol="QQQ",
            risk_on_symbol="LQQ.PA",
            risk_off_mode=RiskOffMode.XEON,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.POST_SPLIT,
        ),
        UcitsRegimeConfig(
            name="v5a_CL2_SPYsig_cash_postsplit",
            signal_symbol="SPY",
            risk_on_symbol="CL2.PA",
            risk_off_mode=RiskOffMode.CASH,
            vol_circuit_breaker=None,
            sizing_mode=SizingMode.POST_SPLIT,
        ),
    ]

    windows = [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("FULL_5Y", IS_START, OOS_END),
    ]

    results: dict[str, Any] = {
        "windows": {w: {"start": str(s), "end": str(e)} for w, s, e in windows},
        "ticker_audit": ticker_audit,
        "locked_configs": {c.name: c.model_dump(mode="json") for c in strategy_specs},
        "benchmarks_theoretical": {},
        "benchmarks_whole_share": {},
        "strategies": {},
        "regime_transitions": {},
    }

    # Benchmarks (Theoretical fractional & Whole-share EUR)
    for w_name, w_start, w_end in windows:
        results["benchmarks_theoretical"][w_name] = {}
        for bsym in ["CSPX.AS", "SXR8.DE", "SPY", "QQQ", "XEON.DE", "CL2.PA", "LQQ.PA", "QQQ3.MI"]:
            results["benchmarks_theoretical"][w_name][bsym] = compute_theoretical_bh_metrics(
                bars[bsym], w_start, w_end, initial_capital=1000.0, metrics_cfg=mcfg
            )
        results["benchmarks_whole_share"][w_name] = {}
        for bsym in ["CSPX.AS", "CL2.PA", "LQQ.PA", "QQQ3.MI", "XEON.DE"]:
            results["benchmarks_whole_share"][w_name][bsym] = {}
            for cap in args.capitals_eur:
                cap_tag = f"eur{int(cap)}"
                bh_res = run_buy_and_hold_etf(
                    bars[bsym],
                    start_date=w_start,
                    end_date=w_end,
                    initial_capital_eur=cap,
                    sizing_mode=SizingMode.AS_TRADED,
                    benchmark_bars=bars["CSPX.AS"],
                )
                results["benchmarks_whole_share"][w_name][bsym][cap_tag] = _pack_result(bh_res, mcfg)
                if bsym in ("CL2.PA", "LQQ.PA"):
                    bh_ps = run_buy_and_hold_etf(
                        bars[bsym],
                        start_date=w_start,
                        end_date=w_end,
                        initial_capital_eur=cap,
                        sizing_mode=SizingMode.POST_SPLIT,
                        benchmark_bars=bars["CSPX.AS"],
                    )
                    results["benchmarks_whole_share"][w_name][bsym][f"{cap_tag}_postsplit"] = _pack_result(
                        bh_ps, mcfg
                    )

    # Run ALL locked strategies across IS, Blind OOS, Full 5Y
    full_5y_runs: dict[int, dict[str, UcitsBacktestResult]] = {}

    for cfg in strategy_specs:
        results["strategies"][cfg.name] = {}
        for w_name, w_start, w_end in windows:
            results["strategies"][cfg.name][w_name] = {}
            for cap in args.capitals_eur:
                cap_int = int(cap)
                cap_tag = f"eur{cap_int}"
                res = run_ucits_regime_backtest(
                    signal_bars=bars[cfg.signal_symbol],
                    risk_on_bars=bars[cfg.risk_on_symbol],
                    cfg=cfg,
                    start_date=w_start,
                    end_date=w_end,
                    initial_capital_eur=cap,
                    risk_off_bars=bars[cfg.risk_off_symbol] if cfg.risk_off_mode is RiskOffMode.XEON else None,
                    benchmark_bars=bars["CSPX.AS"],
                )
                packed = _pack_result(res, mcfg)
                results["strategies"][cfg.name][w_name][cap_tag] = packed
                if w_name == "FULL_5Y":
                    full_5y_runs.setdefault(cap_int, {})[cfg.name] = res

                m = packed["metrics"]
                logging.info(
                    "[%-7s] %-30s %s | eq=EUR%8.2f ret=%+6.1f%% cagr=%+6.1f%% sr=%+5.2f so=%+5.2f dsr=%.3f dd=%6.1f%% ord=%2d(%.1f/y) exp=%4.1f%% wr=%5.1f%% fric=EUR%5.2f skip0=%d",
                    w_name,
                    cfg.name,
                    cap_tag,
                    m["final_eur"],
                    m["total_return"] * 100,
                    m["cagr"] * 100,
                    m["sharpe"] if not math.isnan(m["sharpe"]) else 0.0,
                    m["sortino"] if not math.isnan(m["sortino"]) else 0.0,
                    m["deflated_sharpe"] if not math.isnan(m["deflated_sharpe"]) else 0.0,
                    m["max_drawdown"] * 100,
                    m["total_orders"],
                    m["orders_per_year"],
                    m["risk_on_exposure"] * 100,
                    (m["win_rate_risk_on"] * 100) if not math.isnan(m["win_rate_risk_on"]) else 0.0,
                    m["total_friction_eur"],
                    m["skipped_zero_shares"],
                )

    # Record exact US regime switch logs for Full 5Y (QQQ v5a, QQQ v5b, SPY v5a)
    for label, run_key in [
        ("QQQ_v5a", "v5a_CL2_cash"),
        ("QQQ_v5b", "v5b_QQQ3_cash"),
        ("SPY_v5a", "v5a_CL2_SPYsig_cash"),
    ]:
        sig_df = full_5y_runs[int(args.capitals_eur[0])][run_key].us_signals
        sub = sig_df.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
        flips = sub[sub["switched"]]
        init_row = sig_df.loc[: pd.Timestamp(IS_START) - pd.Timedelta(days=1)].iloc[-1]
        results["regime_transitions"][label] = {
            "init_prior_us_date": str(sig_df.loc[: pd.Timestamp(IS_START) - pd.Timedelta(days=1)].index[-1].date()),
            "init_state": str(init_row["state"]),
            "total_switches_5y": int(len(flips)),
            "switches_is": int(len(flips.loc[: pd.Timestamp(IS_END)])),
            "switches_oos": int(len(flips.loc[pd.Timestamp(OOS_START) :])),
            "events": [
                {
                    "us_date": str(ts.date()),
                    "window": "IS" if ts.date() <= IS_END else "OOS",
                    "new_state": str(r["state"]),
                    "close": round(float(r["close"]), 2),
                    "sma200": round(float(r["sma200"]), 2),
                    "sigma_20": round(float(r["sigma_20"]), 4),
                    "vol_breaker_active": bool(r["vol_breaker_active"]),
                }
                for ts, r in flips.iterrows()
            ],
        }

    # Generate equity curve PNGs (EUR 1,000 and EUR 5,000) in BOTH outputs/ and Project store media/
    media_paths: dict[str, str] = {}
    for cap in args.capitals_eur:
        cap_int = int(cap)
        runs = full_5y_runs[cap_int]

        # Theoretical continuous B&H series for CSPX.AS and CL2.PA on full 5Y
        cspx_5y = bars["CSPX.AS"].frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
        bh_cspx_theo = cspx_5y["close"] / float(cspx_5y["open"].iloc[0]) * cap
        cl2_5y = bars["CL2.PA"].frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
        bh_cl2_theo = cl2_5y["close"] / float(cl2_5y["open"].iloc[0]) * cap
        bh_qqq3_as = run_buy_and_hold_etf(
            bars["QQQ3.MI"],
            start_date=IS_START,
            end_date=OOS_END,
            initial_capital_eur=cap,
            sizing_mode=SizingMode.AS_TRADED,
            benchmark_bars=bars["CSPX.AS"],
        )

        local_png = ROOT / "outputs" / f"v5_etf_eur{cap_int}.png"
        store_png = STORE_ROOT / "media" / f"v5-etf-eur{cap_int}.png"
        for target_path in (local_png, store_png):
            _plot_v5_equity_png(
                capital_eur=cap,
                v5a_cl2_as=runs["v5a_CL2_cash"],
                v5a_cl2_ps=runs["v5a_CL2_cash_postsplit"],
                v5a_lqq_as=runs["v5a_LQQ_cash"],
                v5a_lqq_ps=runs["v5a_LQQ_cash_postsplit"],
                v5b_qqq3_cash=runs["v5b_QQQ3_cash"],
                v5b_qqq3_xeon=runs["v5b_QQQ3_xeon"],
                bh_cspx_theo=bh_cspx_theo,
                bh_cl2_theo=bh_cl2_theo,
                bh_qqq3_as=bh_qqq3_as,
                oos_split=pd.Timestamp(OOS_START),
                path=target_path,
            )
            logging.info("Wrote equity curve PNG -> %s", target_path)
        media_paths[f"eur{cap_int}_local"] = str(local_png)
        media_paths[f"eur{cap_int}_store"] = str(store_png)

    results["media_paths"] = media_paths

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logging.info("Strategy v5 backtest completed in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
