"""Strategy v3 (VCP + 52-Week High Momentum Breakout) IS vs Blind OOS Validation.

Strict Out-of-Sample Protocol:
- In-Sample (IS):  2021-11-22 to 2024-11-21 (first 3 years)
- Out-of-Sample:   2024-11-22 to 2026-11-22 (last 2 years, through latest cached bar)
- Full 5-Year:     2021-11-22 to 2026-11-22 (through latest cached bar)

All strategy parameters are locked in ``VcpConfig`` prior to OOS evaluation.
Runs strictly on the cached OHLCV bars (``--provider yfinance`` using ``data/cache``;
zero external fundamental API calls).

Outputs:
- Raw forward excess returns vs SPY (+5d, +20d, +45d, +60d from t+1 Open) with t-stats
  for IS, blind OOS, and Full 5-Year.
- Full portfolio simulation for EUR 1,000 ($1,060) and EUR 5,000 ($5,300) across IS,
  OOS, and Full 5-Year, with and without EUR 6.50/month Hetzner VPS fees.
- Full-period equity curve PNGs with IS/OOS split vertical marker.
- Structured JSON results and signal diagnostics CSV.

Usage:
    python scripts/run_v3_oos_backtest.py --provider yfinance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
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
from scipy import stats  # noqa: E402

from src.backtest.engine import BacktestResult  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.backtest.reporting import REGIME_COLORS, _regime_spans  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.factory import build_provider  # noqa: E402
from src.domain.models import TradeSignal  # noqa: E402
from src.strategy.vcp_momentum_screener import (  # noqa: E402
    PreparedVcpData,
    VcpConfig,
    prepare_vcp_data,
    run_backtest_v3,
    screen_all_sessions,
    slice_vcp_data,
)

IS_START = date(2021, 11, 22)
IS_END = date(2024, 11, 21)
OOS_START = date(2024, 11, 22)
OOS_END = date(2026, 11, 22)
HORIZONS: tuple[int, ...] = (5, 20, 45, 60)

STORE_ROOT = Path(
    "/home/margarida/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-de418035-892d-40a4-a21a-bd834b742bee/files"
)


def evaluate_raw_forward_returns(
    data: PreparedVcpData,
    signals_by_date: dict[pd.Timestamp, list[TradeSignal]],
    max_bar_date: date | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute raw and SPY-excess forward returns (+5d, +20d, +45d, +60d) from t+1 Open.

    When ``max_bar_date`` is provided (e.g. ``IS_END``), forward prices are strictly
    bounded to bars ``<= max_bar_date`` so IS diagnostics never touch OOS bars.
    """
    spy_full = data.benchmark.frame
    spy = spy_full.loc[: pd.Timestamp(max_bar_date)] if max_bar_date is not None else spy_full

    rows: list[dict[str, Any]] = []
    for ts in sorted(signals_by_date):
        for sig in signals_by_date[ts]:
            f_full = data.features[sig.symbol]
            f = f_full.loc[: pd.Timestamp(max_bar_date)] if max_bar_date is not None else f_full
            if ts not in f.index:
                continue
            i = int(f.index.get_loc(ts))
            if i + 1 >= len(f):
                continue

            entry_ts = f.index[i + 1]
            entry_open = float(f["open"].iloc[i + 1])
            gap = entry_open / sig.signal_close - 1.0
            j_spy = int(spy.index.get_loc(ts)) if ts in spy.index else -1
            spy_open_1 = (
                float(spy.loc[entry_ts, "open"])
                if entry_ts in spy.index
                else (float(spy["open"].iloc[j_spy + 1]) if 0 <= j_spy and j_spy + 1 < len(spy) else float("nan"))
            )

            row: dict[str, Any] = {
                "symbol": sig.symbol,
                "signal_date": ts.date(),
                "entry_date": entry_ts.date(),
                "year": ts.year,
                "regime": sig.regime.value,
                "signal_close": sig.signal_close,
                "entry_open": entry_open,
                "next_open_gap": gap,
                "within_gap_cap": gap <= 0.015,
                "rs_composite": sig.rank_score,
                "atr14": sig.atr14,
                "atr14_pct": sig.atr14 / entry_open,
                "volume_ratio": sig.volume_ratio,
                "close_location": sig.close_location,
            }

            for k in HORIZONS:
                if i + k < len(f):
                    exit_ts = f.index[i + k]
                    ret_k = float(f["close"].iloc[i + k]) / entry_open - 1.0
                    row[f"ret_{k}d"] = ret_k
                    if exit_ts in spy.index and not np.isnan(spy_open_1):
                        bench_k = float(spy.loc[exit_ts, "close"]) / spy_open_1 - 1.0
                        row[f"bench_{k}d"] = bench_k
                        row[f"excess_{k}d"] = ret_k - bench_k
                    elif 0 <= j_spy and j_spy + k < len(spy) and not np.isnan(spy_open_1):
                        bench_k = float(spy["close"].iloc[j_spy + k]) / spy_open_1 - 1.0
                        row[f"bench_{k}d"] = bench_k
                        row[f"excess_{k}d"] = ret_k - bench_k
                    else:
                        row[f"bench_{k}d"] = float("nan")
                        row[f"excess_{k}d"] = float("nan")
                else:
                    row[f"ret_{k}d"] = float("nan")
                    row[f"bench_{k}d"] = float("nan")
                    row[f"excess_{k}d"] = float("nan")

            rows.append(row)

    df = pd.DataFrame(rows)
    return df, summarize_forward_returns(df)


def summarize_forward_returns(df: pd.DataFrame) -> dict[str, Any]:
    """Compute horizon statistics including one-sample t-stats and two-sided p-values."""
    out: dict[str, Any] = {
        "total_signals": int(len(df)),
        "gap_above_1_5pct_count": int((~df["within_gap_cap"]).sum()) if len(df) else 0,
        "gap_above_1_5pct_rate": float((~df["within_gap_cap"]).mean()) if len(df) else float("nan"),
        "horizons": {},
    }
    for k in HORIZONS:
        ret_col = f"ret_{k}d"
        ex_col = f"excess_{k}d"
        r = df[ret_col].dropna() if ret_col in df.columns else pd.Series(dtype=float)
        ex = df[ex_col].dropna() if ex_col in df.columns else pd.Series(dtype=float)
        if len(r) >= 2 and r.std(ddof=1) > 0:
            r_t, r_p = stats.ttest_1samp(r, 0.0)
        else:
            r_t, r_p = float("nan"), float("nan")
        if len(ex) >= 2 and ex.std(ddof=1) > 0:
            ex_t, ex_p = stats.ttest_1samp(ex, 0.0)
        else:
            ex_t, ex_p = float("nan"), float("nan")
        out["horizons"][f"+{k}d"] = {
            "n": int(len(ex)),
            "raw_mean": float(r.mean()) if len(r) else float("nan"),
            "raw_median": float(r.median()) if len(r) else float("nan"),
            "raw_pos_rate": float((r > 0).mean()) if len(r) else float("nan"),
            "raw_tstat": float(r_t),
            "raw_pvalue": float(r_p),
            "excess_mean": float(ex.mean()) if len(ex) else float("nan"),
            "excess_median": float(ex.median()) if len(ex) else float("nan"),
            "excess_pos_rate": float((ex > 0).mean()) if len(ex) else float("nan"),
            "excess_tstat": float(ex_t),
            "excess_pvalue": float(ex_p),
        }
    return out


def _exit_breakdown(res: BacktestResult) -> dict[str, dict[str, float | int]]:
    by_reason: dict[str, list[float]] = {}
    for t in res.trades:
        by_reason.setdefault(t.exit_reason.value, []).append(t.r_multiple)
    return {
        k: {"count": len(v), "mean_r": float(sum(v) / len(v))}
        for k, v in sorted(by_reason.items())
    }


def _benchmark_total_return(curve: pd.DataFrame) -> float:
    b = curve["benchmark_close"]
    return float(b.iloc[-1] / b.iloc[0] - 1.0)


def _pack_run(res: BacktestResult, metrics: dict[str, float]) -> dict[str, Any]:
    m = dict(metrics)
    m["benchmark_total_return"] = _benchmark_total_return(res.equity_curve)
    return {
        "metrics": m,
        "rejections": res.rejections,
        "sizing_rejections": res.sizing_rejections,
        "exit_breakdown": _exit_breakdown(res),
        "skipped_reasons": dict(Counter(s.reason for s in res.skipped)),
        "fees_paid_usd": res.fees_paid,
    }


def plot_v3_equity_png(
    res_no_vps: BacktestResult,
    res_vps: BacktestResult,
    capital_eur: float,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Plot full 5-year equity curve + drawdown with explicit IS/OOS vertical split marker."""
    curve = res_no_vps.equity_curve
    eq = curve["equity"]
    eq_vps = res_vps.equity_curve["equity"]
    init = res_no_vps.initial_capital_usd
    bench = curve["benchmark_close"] / curve["benchmark_close"].iloc[0] * init
    dd = (eq / eq.cummax().clip(lower=init) - 1.0) * 100.0
    dd_vps = (eq_vps / eq_vps.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for a, b, reg in _regime_spans(curve):
        if REGIME_COLORS[reg]:
            for ax in (ax1, ax2):
                ax.axvspan(a, b, color=REGIME_COLORS[reg], alpha=0.32, lw=0)

    ax1.plot(eq.index, eq.values, label="v3 VCP Momentum (no VPS)", color="#1f4e79", lw=1.8)
    ax1.plot(
        eq_vps.index,
        eq_vps.values,
        label="v3 VCP Momentum (with EUR 6.50/mo VPS)",
        color="#d95f02",
        lw=1.3,
        ls="-.",
    )
    ax1.plot(bench.index, bench.values, label="SPY buy & hold (same capital)", color="#666666", lw=1.1, ls="--")
    ax1.axhline(init, color="black", lw=0.6, alpha=0.5)

    for ax in (ax1, ax2):
        ax.axvline(oos_split, color="#8b0000", lw=1.5, ls="--", alpha=0.9)

    ymin, ymax = ax1.get_ylim()
    y_label = ymin + 0.94 * (ymax - ymin)
    ax1.text(
        oos_split - pd.Timedelta(days=18),
        y_label,
        "In-Sample (2021-11 to 2024-11)",
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85),
    )
    ax1.text(
        oos_split + pd.Timedelta(days=18),
        y_label,
        "Blind OOS (2024-11 to 2026-11)",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#8b0000",
        bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec="#d4a0a0", alpha=0.9),
    )

    ax1.set_ylabel("Equity (USD)")
    ax1.set_title(
        f"Strategy v3 (VCP + 52w High Momentum Breakout) — EUR {int(capital_eur):,} (${init:,.0f}) | Full 5Y with IS/OOS Split",
        fontsize=11,
    )
    ax1.legend(loc="lower left", fontsize=8.8, framealpha=0.9)
    ax1.grid(alpha=0.3)

    ax2.fill_between(dd.index, dd.values, 0, color="#b03030", alpha=0.55, label="Drawdown % (no VPS)")
    ax2.plot(dd_vps.index, dd_vps.values, color="#d95f02", lw=0.9, ls="-.", label="Drawdown % (with VPS)")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", fontsize=8.0, framealpha=0.85)
    ax2.grid(alpha=0.3)

    fig.text(
        0.99,
        0.006,
        "shading: yellow=CAUTION, red=PANIC regime | dashed crimson line: IS/OOS split (2024-11-22)",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Strategy v3 IS vs blind OOS backtest & diagnostics.")
    ap.add_argument("--provider", choices=["yfinance", "sample", "auto", "fmp_massive"], default="yfinance")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v3_oos_results.json"))
    ap.add_argument("--diag-csv", default=str(ROOT / "outputs" / "v3_signal_diagnostics.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    settings = settings.model_copy(update={"data": settings.data.model_copy(update={"provider": args.provider})})
    provider, notes = build_provider(settings)
    for n in notes:
        logging.info("provider: %s", n)

    # LOCKED Strategy v3 configuration (zero OOS tuning)
    vcp_cfg = VcpConfig()
    full_data = prepare_vcp_data(settings, provider, vcp_cfg, max_symbols=args.max_symbols)
    logging.info("v3 OHLCV prepared in %.1fs: %s", time.time() - t0, json.dumps(full_data.data_stats))

    is_data = slice_vcp_data(full_data, IS_START, IS_END)
    oos_data = slice_vcp_data(full_data, OOS_START, OOS_END)

    windows: list[tuple[str, PreparedVcpData, date | None]] = [
        ("IS", is_data, IS_END),
        ("OOS", oos_data, None),
        ("FULL_5Y", full_data, None),
    ]

    results: dict[str, Any] = {
        "locked_config": vcp_cfg.model_dump(mode="json"),
        "provider_notes": notes,
        "provenance": full_data.provenance,
        "data_stats": {
            "FULL_5Y": full_data.data_stats,
            "IS": is_data.data_stats,
            "OOS": oos_data.data_stats,
        },
        "raw_signal_diagnostics": {},
        "portfolio": {},
    }

    diag_frames: list[pd.DataFrame] = []
    full_runs_for_plot: dict[int, dict[str, BacktestResult]] = {}

    for w_name, w_data, max_bar_d in windows:
        sigs_by_date, rej, elig_counts = screen_all_sessions(w_data, vcp_cfg)
        elig_vals = list(elig_counts.values())
        diag_df, diag_summary = evaluate_raw_forward_returns(w_data, sigs_by_date, max_bar_date=max_bar_d)
        diag_summary["eligible_universe_mean"] = float(np.mean(elig_vals)) if elig_vals else 0.0
        diag_summary["eligible_universe_min"] = int(np.min(elig_vals)) if elig_vals else 0
        diag_summary["eligible_universe_max"] = int(np.max(elig_vals)) if elig_vals else 0
        diag_summary["screen_rejections"] = rej
        results["raw_signal_diagnostics"][w_name] = diag_summary

        if not diag_df.empty:
            dcopy = diag_df.copy()
            dcopy.insert(0, "window", w_name)
            diag_frames.append(dcopy)

        logging.info(
            "=== RAW SIGNALS [%s] (%s..%s): N=%d, gap>1.5%%=%d (%.1f%%) ===",
            w_name,
            w_data.data_stats["first_session"],
            w_data.data_stats["last_session"],
            diag_summary["total_signals"],
            diag_summary["gap_above_1_5pct_count"],
            (diag_summary["gap_above_1_5pct_rate"] * 100) if not np.isnan(diag_summary["gap_above_1_5pct_rate"]) else 0.0,
        )
        for h in ("+5d", "+20d", "+45d", "+60d"):
            hs = diag_summary["horizons"][h]
            logging.info(
                "  %-4s n=%3d | raw mean=%+6.2f%% med=%+6.2f%% | excess vs SPY mean=%+6.2f%% med=%+6.2f%% t=%+5.2f p=%.4f",
                h,
                hs["n"],
                hs["raw_mean"] * 100,
                hs["raw_median"] * 100,
                hs["excess_mean"] * 100,
                hs["excess_median"] * 100,
                hs["excess_tstat"],
                hs["excess_pvalue"],
            )

        results["portfolio"][w_name] = {}
        for cap_eur in args.capitals_eur:
            cap_int = int(cap_eur)
            cap_usd = cap_eur * settings.backtest.eur_usd_rate
            for vps_on in (False, True):
                tag = f"eur{cap_int}_vps" if vps_on else f"eur{cap_int}"
                res = run_backtest_v3(
                    w_data,
                    settings,
                    vcp_cfg,
                    initial_capital_usd=cap_usd,
                    vps_enabled=vps_on,
                    precomputed_signals=(sigs_by_date, rej),
                )
                m = compute_metrics(res.equity_curve, cap_usd, res.trades, res.skipped, settings.metrics)
                results["portfolio"][w_name][tag] = _pack_run(res, m)
                if w_name == "FULL_5Y":
                    full_runs_for_plot.setdefault(cap_int, {})["vps" if vps_on else "no_vps"] = res
                logging.info(
                    "  [%s] %-12s eq=$%8.2f ret=%+6.1f%% cagr=%+6.1f%% spy=%+6.1f%% sr=%+5.2f sort=%+5.2f dsr=%.3f dd=%6.1f%% tr=%2d wr=%4.1f%% expR=%+5.2f pf=%4.2f gskip=%2d",
                    w_name,
                    tag,
                    m["final_equity"],
                    m["total_return"] * 100,
                    m["cagr"] * 100,
                    _benchmark_total_return(res.equity_curve) * 100,
                    m["sharpe"] if not np.isnan(m["sharpe"]) else 0.0,
                    m["sortino"] if not np.isnan(m["sortino"]) else 0.0,
                    m["deflated_sharpe"] if not np.isnan(m["deflated_sharpe"]) else 0.0,
                    m["max_drawdown"] * 100,
                    m["trades"],
                    (m["win_rate"] * 100) if m["trades"] else 0.0,
                    m["expectancy_r"] if m["trades"] else 0.0,
                    m["profit_factor"] if m["trades"] else 0.0,
                    m["skipped_gap"],
                )

    # Save equity curve PNGs (with IS/OOS vertical split) to both store media/ and local outputs/
    media_paths: dict[str, str] = {}
    for cap_int, runs in sorted(full_runs_for_plot.items()):
        store_png = STORE_ROOT / "media" / f"v3-equity-curve-eur{cap_int}.png"
        local_png = ROOT / "outputs" / f"v3-equity-curve-eur{cap_int}.png"
        plot_v3_equity_png(runs["no_vps"], runs["vps"], float(cap_int), pd.Timestamp(OOS_START), store_png)
        plot_v3_equity_png(runs["no_vps"], runs["vps"], float(cap_int), pd.Timestamp(OOS_START), local_png)
        media_paths[f"eur{cap_int}_store"] = str(store_png)
        media_paths[f"eur{cap_int}_local"] = str(local_png)
        logging.info("wrote equity curve PNG -> %s", store_png)

    results["media_paths"] = media_paths

    if diag_frames:
        diag_out = Path(args.diag_csv)
        diag_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(diag_frames, ignore_index=True).to_csv(diag_out, index=False)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logging.info("v3 IS/OOS validation finished in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
