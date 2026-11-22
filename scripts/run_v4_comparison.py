"""Strategy v4 Showdown: v4a (Panic Mean-Reversion) vs v4b (Regime-Gated Momentum Leaders).

Strict Out-of-Sample Protocol (zero tuning on OOS):
- In-Sample (IS):  2021-11-22 to 2024-11-21 (first 3 years)
- Blind OOS:       2024-11-22 to 2026-11-20 (last 2 years, run ONCE with locked configs)
- Full 5-Year:     2021-11-22 to 2026-11-20

Loads the cached yfinance OHLCV universe (1,232 symbols in ``data/cache``) ONCE,
evaluates raw signal edge before stops/fees, runs full portfolio simulations for
EUR 1,000 ($1,060) and EUR 5,000 ($5,300) with and without EUR 6.50/month VPS
fees (including v4a f=5%, v4a f=10%, and v4b), writes comparison equity curve
PNGs with IS/OOS split markers, and saves structured JSON results.

Usage:
    python scripts/run_v4_comparison.py --provider yfinance
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from datetime import date, timedelta
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

from src.backtest.metrics import compute_metrics  # noqa: E402
from src.backtest.reporting import REGIME_COLORS, _regime_spans  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.factory import build_provider  # noqa: E402
from src.domain.models import TradeSignal  # noqa: E402
from src.strategy.momentum_rotation_screener import (  # noqa: E402
    MomentumBacktestResult,
    MomentumCandidate,
    MomentumRotationConfig,
    PreparedMomentumData,
    build_momentum_data_from_bars,
    run_backtest_v4b,
    select_all_rebalance_leaders,
    slice_momentum_data,
)
from src.strategy.panic_reversion_screener import (  # noqa: E402
    PanicBacktestResult,
    PanicReversionConfig,
    PreparedPanicData,
    build_panic_data_from_bars,
    run_backtest_v4a,
    screen_panic_sessions,
    slice_panic_data,
)

IS_START = date(2021, 11, 22)
IS_END = date(2024, 11, 21)
OOS_START = date(2024, 11, 22)
OOS_END = date(2026, 11, 20)
V4A_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)

STORE_ROOT = Path(
    "/home/margarida/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-de418035-892d-40a4-a21a-bd834b742bee/files"
)


# ---------------------------------------------------------------------------
# Raw Signal Edge Diagnostics (before stops & fees, strictly zero look-ahead)
# ---------------------------------------------------------------------------


def _ttest_summary(raw: pd.Series, excess: pd.Series) -> dict[str, Any]:
    r = raw.dropna()
    ex = excess.dropna()
    if len(r) >= 2 and r.std(ddof=1) > 0:
        rt, rp = stats.ttest_1samp(r, 0.0)
    else:
        rt, rp = float("nan"), float("nan")
    if len(ex) >= 2 and ex.std(ddof=1) > 0:
        et, ep = stats.ttest_1samp(ex, 0.0)
    else:
        et, ep = float("nan"), float("nan")
    return {
        "n": int(len(ex)),
        "raw_mean": float(r.mean()) if len(r) else float("nan"),
        "raw_median": float(r.median()) if len(r) else float("nan"),
        "raw_pos_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "raw_tstat": float(rt),
        "raw_pvalue": float(rp),
        "excess_mean": float(ex.mean()) if len(ex) else float("nan"),
        "excess_median": float(ex.median()) if len(ex) else float("nan"),
        "excess_pos_rate": float((ex > 0).mean()) if len(ex) else float("nan"),
        "excess_tstat": float(et),
        "excess_pvalue": float(ep),
    }


def evaluate_v4a_raw_edge(
    data: PreparedPanicData,
    signals_by_date: dict[pd.Timestamp, list[TradeSignal]],
    max_hold_days: int = 10,
    max_bar_date: date | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate v4a raw signal edge before stops/fees from t+1 Open.

    Measures +1d, +3d, +5d, +10d (from t+1 Open to t+k Close) AND the exact
    signal exit rule ``to Close > SMA(5) (max 10d)`` exiting at next Open after
    trigger (strictly zero look-ahead). Reports both signal-level and
    day-clustered (daily equal-weight signal-day) t-stats/p-values.
    """
    spy_full = data.benchmark.frame
    spy = spy_full.loc[: pd.Timestamp(max_bar_date)] if max_bar_date is not None else spy_full
    removed_syms = {m.symbol for m in data.universe if m.removed_on is not None}

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
            j_spy = int(spy.index.get_loc(ts)) if ts in spy.index else -1
            spy_open_1 = (
                float(spy.loc[entry_ts, "open"])
                if entry_ts in spy.index
                else (float(spy["open"].iloc[j_spy + 1]) if 0 <= j_spy and j_spy + 1 < len(spy) else float("nan"))
            )

            row: dict[str, Any] = {
                "symbol": sig.symbol,
                "is_removed_member": sig.symbol in removed_syms,
                "signal_date": ts.date(),
                "entry_date": entry_ts.date(),
                "rsi2": -sig.rank_score,
                "ret_3d": sig.event_day_return,
                "atr14_pct": sig.atr14 / sig.signal_close,
            }

            # Fixed horizons +1d, +3d, +5d, +10d (from t+1 Open to t+k Close)
            for k in V4A_HORIZONS:
                if i + k < len(f):
                    exit_ts = f.index[i + k]
                    ret_k = float(f["close"].iloc[i + k]) / entry_open - 1.0
                    row[f"ret_{k}d"] = ret_k
                    if exit_ts in spy.index and not math.isnan(spy_open_1):
                        bench_k = float(spy.loc[exit_ts, "close"]) / spy_open_1 - 1.0
                        row[f"excess_{k}d"] = ret_k - bench_k
                    elif 0 <= j_spy and j_spy + k < len(spy) and not math.isnan(spy_open_1):
                        bench_k = float(spy["close"].iloc[j_spy + k]) / spy_open_1 - 1.0
                        row[f"excess_{k}d"] = ret_k - bench_k
                    else:
                        row[f"excess_{k}d"] = float("nan")
                else:
                    row[f"ret_{k}d"] = float("nan")
                    row[f"excess_{k}d"] = float("nan")

            # Signal exit: morning Open after Close > SMA5, or next Open after max_hold_days Close
            sma5_ret = float("nan")
            sma5_ex = float("nan")
            sma5_bars = float("nan")
            sma5_reason = "TRUNCATED"
            for hold_bar in range(1, max_hold_days + 1):
                j = i + hold_bar
                if j >= len(f):
                    break
                c_j = float(f["close"].iloc[j])
                s5_j = float(f["sma5"].iloc[j])
                trig_sma = (not math.isnan(s5_j)) and (c_j > s5_j)
                trig_time = hold_bar == max_hold_days
                if trig_sma or trig_time:
                    if j + 1 < len(f):
                        ex_ts = f.index[j + 1]
                        ex_open = float(f["open"].iloc[j + 1])
                        sma5_ret = ex_open / entry_open - 1.0
                        sma5_bars = float(hold_bar)
                        sma5_reason = "SMA5" if trig_sma else "MAX_10D"
                        if ex_ts in spy.index and not math.isnan(spy_open_1):
                            bench_s = float(spy.loc[ex_ts, "open"]) / spy_open_1 - 1.0
                            sma5_ex = sma5_ret - bench_s
                    break

            row["ret_sma5"] = sma5_ret
            row["excess_sma5"] = sma5_ex
            row["sma5_bars_held"] = sma5_bars
            row["sma5_exit_type"] = sma5_reason
            rows.append(row)

    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "total_signals": int(len(df)),
        "active_signal_days": int(df["signal_date"].nunique()) if len(df) else 0,
        "removed_member_signals": int(df["is_removed_member"].sum()) if len(df) else 0,
        "removed_member_signal_share": float(df["is_removed_member"].mean()) if len(df) else float("nan"),
        "avg_signals_per_active_day": float(len(df) / df["signal_date"].nunique()) if len(df) else 0.0,
        "horizons": {},
        "day_clustered_horizons": {},
    }
    if len(df):
        summary["sma5_avg_bars_held"] = float(df["sma5_bars_held"].dropna().mean())
        summary["sma5_hit_rate"] = float((df["sma5_exit_type"] == "SMA5").mean())

    horizon_specs = [(f"+{k}d", f"ret_{k}d", f"excess_{k}d") for k in V4A_HORIZONS] + [
        ("to_SMA5_max10d", "ret_sma5", "excess_sma5")
    ]
    for label, rcol, ecol in horizon_specs:
        if len(df):
            summary["horizons"][label] = _ttest_summary(df[rcol], df[ecol])
            daily = df.groupby("signal_date")[[rcol, ecol]].mean()
            summary["day_clustered_horizons"][label] = _ttest_summary(daily[rcol], daily[ecol])
        else:
            summary["horizons"][label] = _ttest_summary(pd.Series(dtype=float), pd.Series(dtype=float))
            summary["day_clustered_horizons"][label] = _ttest_summary(pd.Series(dtype=float), pd.Series(dtype=float))

    return df, summary


def evaluate_v4b_raw_edge(
    data: PreparedMomentumData,
    leaders_by_date: dict[pd.Timestamp, list[MomentumCandidate]],
    rebalance_every_days: int = 10,
    max_bar_date: date | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate v4b 10-day equal-weight top-3 basket excess vs SPY when risk-on.

    Measures each risk-on rebalance period from ``t+1`` Open to ``t+11`` Open
    (strictly bounded to ``<= max_bar_date`` for IS).
    """
    full_cal = data.benchmark.frame.index
    if max_bar_date is not None:
        full_cal = full_cal[full_cal <= pd.Timestamp(max_bar_date)]
    spy = data.benchmark.frame.loc[full_cal]
    removed_syms = {m.symbol for m in data.universe if m.removed_on is not None}

    rows: list[dict[str, Any]] = []
    n_risk_on_rebs = 0
    n_risk_off_rebs = 0
    total_picks = 0
    removed_picks = 0

    for ts in sorted(leaders_by_date):
        risk_on = data.spy_risk_on.get(ts, False)
        if not risk_on:
            n_risk_off_rebs += 1
            continue
        n_risk_on_rebs += 1
        cands = leaders_by_date[ts]
        if not cands or ts not in full_cal:
            continue

        pos = int(full_cal.get_loc(ts))
        if pos + 1 >= len(full_cal) or pos + 1 + rebalance_every_days >= len(full_cal):
            continue
        entry_ts = full_cal[pos + 1]
        exit_ts = full_cal[pos + 1 + rebalance_every_days]
        spy_ret = float(spy.loc[exit_ts, "open"]) / float(spy.loc[entry_ts, "open"]) - 1.0

        stock_rets: list[float] = []
        picked_syms: list[str] = []
        for c in cands:
            f = data.features[c.symbol]
            if entry_ts in f.index and exit_ts in f.index:
                r = float(f.loc[exit_ts, "open"]) / float(f.loc[entry_ts, "open"]) - 1.0
                stock_rets.append(r)
                picked_syms.append(c.symbol)
                total_picks += 1
                if c.symbol in removed_syms:
                    removed_picks += 1
        if not stock_rets:
            continue

        basket_ret = float(np.mean(stock_rets))
        rows.append(
            {
                "rebalance_date": ts.date(),
                "entry_date": entry_ts.date(),
                "exit_date": exit_ts.date(),
                "n_stocks": len(stock_rets),
                "symbols": ",".join(picked_syms),
                "basket_ret_10d": basket_ret,
                "spy_ret_10d": spy_ret,
                "excess_10d": basket_ret - spy_ret,
            }
        )

    df = pd.DataFrame(rows)
    stats_out = _ttest_summary(
        df["basket_ret_10d"] if len(df) else pd.Series(dtype=float),
        df["excess_10d"] if len(df) else pd.Series(dtype=float),
    )
    summary: dict[str, Any] = {
        "rebalance_days_total": n_risk_on_rebs + n_risk_off_rebs,
        "rebalance_days_risk_on": n_risk_on_rebs,
        "rebalance_days_risk_off": n_risk_off_rebs,
        "evaluated_10d_periods": int(len(df)),
        "spy_mean_10d": float(df["spy_ret_10d"].mean()) if len(df) else float("nan"),
        "removed_member_picks": removed_picks,
        "total_picks": total_picks,
        "removed_member_pick_share": float(removed_picks / total_picks) if total_picks else float("nan"),
        **stats_out,
    }
    return df, summary


# ---------------------------------------------------------------------------
# Portfolio Packing & Turnover / Friction Analytics
# ---------------------------------------------------------------------------


def _exit_breakdown(res: PanicBacktestResult | MomentumBacktestResult) -> dict[str, dict[str, float | int]]:
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


def _pack_v4_run(
    res: PanicBacktestResult | MomentumBacktestResult,
    metrics: dict[str, float],
) -> dict[str, Any]:
    curve = res.equity_curve
    days = (curve.index[-1] - curve.index[0]).days + 1
    years = max(days / 365.25, 1e-6)
    avg_eq = float(curve["equity"].mean())
    one_way_notional = 0.5 * (res.total_buy_notional + res.total_sell_notional)
    ann_turnover = one_way_notional / (avg_eq * years) if avg_eq > 0 else 0.0
    total_friction = res.total_commissions_usd + res.total_slippage_usd
    friction_pct_init = total_friction / res.initial_capital_usd if res.initial_capital_usd > 0 else 0.0
    friction_ann_pct_avg_eq = total_friction / (avg_eq * years) if avg_eq > 0 else 0.0

    m = dict(metrics)
    m["benchmark_total_return"] = _benchmark_total_return(curve)
    m["annualized_turnover"] = float(ann_turnover)
    m["total_buy_notional_usd"] = float(res.total_buy_notional)
    m["total_sell_notional_usd"] = float(res.total_sell_notional)
    m["total_commissions_usd"] = float(res.total_commissions_usd)
    m["total_slippage_usd"] = float(res.total_slippage_usd)
    m["total_friction_usd"] = float(total_friction)
    m["friction_drag_pct_initial"] = float(friction_pct_init)
    m["friction_drag_ann_pct_equity"] = float(friction_ann_pct_avg_eq)
    m["vps_fees_usd"] = float(res.fees_paid)

    payload: dict[str, Any] = {
        "metrics": m,
        "rejections": res.rejections,
        "sizing_rejections": res.sizing_rejections,
        "exit_breakdown": _exit_breakdown(res),
        "skipped_reasons": dict(Counter(s.reason for s in res.skipped)),
        "min_settled_cash_usd": float(curve["settled_cash"].min()),
    }
    if isinstance(res, MomentumBacktestResult):
        payload["rebalance_stats"] = {
            "total": res.rebalance_days_total,
            "risk_on": res.rebalance_days_risk_on,
            "risk_off": res.rebalance_days_risk_off,
            "kept_positions_total": res.kept_positions_total,
            "rotated_out_total": res.rotated_out_total,
        }
    return payload


# ---------------------------------------------------------------------------
# Equity Curve Comparison Plots (v4a vs v4b vs SPY, with IS/OOS split line)
# ---------------------------------------------------------------------------


def plot_v4_showdown_png(
    res_v4a_f5: PanicBacktestResult,
    res_v4a_f10: PanicBacktestResult,
    res_v4b: MomentumBacktestResult,
    capital_eur: float,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Plot full 5Y side-by-side equity curves and drawdowns for v4a (f=5%, f=10%), v4b, and SPY."""
    c_a5 = res_v4a_f5.equity_curve
    c_a10 = res_v4a_f10.equity_curve
    c_b = res_v4b.equity_curve
    init = res_v4a_f5.initial_capital_usd

    eq_a5 = c_a5["equity"]
    eq_a10 = c_a10["equity"]
    eq_b = c_b["equity"]
    bench = c_a5["benchmark_close"] / c_a5["benchmark_close"].iloc[0] * init

    dd_a5 = (eq_a5 / eq_a5.cummax().clip(lower=init) - 1.0) * 100.0
    dd_a10 = (eq_a10 / eq_a10.cummax().clip(lower=init) - 1.0) * 100.0
    dd_b = (eq_b / eq_b.cummax().clip(lower=init) - 1.0) * 100.0
    dd_spy = (bench / bench.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.2, 7.6), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.15]}
    )

    for a, b, reg in _regime_spans(c_a5):
        if REGIME_COLORS[reg]:
            for ax in (ax1, ax2):
                ax.axvspan(a, b, color=REGIME_COLORS[reg], alpha=0.25, lw=0)

    ax1.plot(
        eq_a5.index,
        eq_a5.values,
        label=f"v4a Panic Mean-Rev (f=5%)  |  ${eq_a5.iloc[-1]:,.0f} ({(eq_a5.iloc[-1]/init - 1)*100:+.1f}%)",
        color="#1f4e79",
        lw=1.8,
    )
    ax1.plot(
        eq_a10.index,
        eq_a10.values,
        label=f"v4a Panic Mean-Rev (f=10%) |  ${eq_a10.iloc[-1]:,.0f} ({(eq_a10.iloc[-1]/init - 1)*100:+.1f}%)",
        color="#4e9ad6",
        lw=1.3,
        ls="-.",
    )
    ax1.plot(
        eq_b.index,
        eq_b.values,
        label=f"v4b Bi-Weekly Momentum Top-3 |  ${eq_b.iloc[-1]:,.0f} ({(eq_b.iloc[-1]/init - 1)*100:+.1f}%)",
        color="#d95f02",
        lw=1.9,
    )
    ax1.plot(
        bench.index,
        bench.values,
        label=f"SPY Buy & Hold               |  ${bench.iloc[-1]:,.0f} ({(bench.iloc[-1]/init - 1)*100:+.1f}%)",
        color="#555555",
        lw=1.3,
        ls="--",
    )
    ax1.axhline(init, color="black", lw=0.6, alpha=0.45)

    for ax in (ax1, ax2):
        ax.axvline(oos_split, color="#8b0000", lw=1.6, ls="--", alpha=0.9)

    ymin, ymax = ax1.get_ylim()
    y_label = ymin + 0.95 * (ymax - ymin)
    ax1.text(
        oos_split - pd.Timedelta(days=18),
        y_label,
        "In-Sample (2021-11 to 2024-11)",
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", alpha=0.88),
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
        bbox=dict(boxstyle="round,pad=0.22", fc="#fff5f5", ec="#d4a0a0", alpha=0.92),
    )

    ax1.set_ylabel("Equity (USD)")
    ax1.set_title(
        f"v4 Showdown: v4a (Panic Mean-Reversion) vs v4b (Momentum Leaders) vs SPY — EUR {int(capital_eur):,} (${init:,.0f}, No VPS)",
        fontsize=11.2,
    )
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax1.grid(alpha=0.3)

    ax2.plot(dd_a5.index, dd_a5.values, color="#1f4e79", lw=1.3, label="v4a (f=5%) DD%")
    ax2.plot(dd_a10.index, dd_a10.values, color="#4e9ad6", lw=1.0, ls="-.", label="v4a (f=10%) DD%")
    ax2.plot(dd_b.index, dd_b.values, color="#d95f02", lw=1.4, label="v4b DD%")
    ax2.plot(dd_spy.index, dd_spy.values, color="#666666", lw=1.0, ls="--", label="SPY DD%")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", ncol=4, fontsize=8.0, framealpha=0.88)
    ax2.grid(alpha=0.3)

    fig.text(
        0.99,
        0.006,
        "shading: yellow=CAUTION, red=PANIC (SPY regime) | crimson dashed line: IS/OOS split (2024-11-22) | whole shares, T+1 settled cash",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Strategy v4a vs v4b IS/OOS/Full 5Y showdown.")
    ap.add_argument("--provider", choices=["yfinance", "sample", "auto", "fmp_massive"], default="yfinance")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v4_comparison_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    settings = settings.model_copy(update={"data": settings.data.model_copy(update={"provider": args.provider})})
    provider, notes = build_provider(settings)
    for n in notes:
        logging.info("provider: %s", n)

    # Load cached OHLCV bars ONCE using exact settings.backtest dates to hit disk cache cleanly
    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    universe = [m for m in provider.get_universe(bt.end_date) if not m.is_etf]
    if args.max_symbols:
        universe = universe[: args.max_symbols]
    symbols = [m.symbol for m in universe]
    logging.info("Loading cached OHLCV for %d universe candidates (%s..%s)...", len(symbols), fetch_start, bt.end_date)
    benchmark = provider.get_benchmark_bars(fetch_start, bt.end_date, bt.benchmark_symbol)
    bars = provider.get_daily_bars(symbols, fetch_start, bt.end_date)
    logging.info("Loaded %d symbols with bars in %.1fs", len(bars), time.time() - t0)

    # LOCKED Strategy Configurations (zero OOS tuning)
    cfg_v4a_f5 = PanicReversionConfig(risk_per_trade=0.05, rsi_method="wilder")
    cfg_v4a_f10 = PanicReversionConfig(risk_per_trade=0.10, rsi_method="wilder")
    cfg_v4a_simple_diag = PanicReversionConfig(risk_per_trade=0.05, rsi_method="simple")
    cfg_v4b = MomentumRotationConfig()

    # Build strictly causal features ONCE for v4a (wilder) and v4b
    t_feat = time.time()
    v4a_full = build_panic_data_from_bars(
        bars=bars,
        benchmark=benchmark,
        universe=universe,
        settings=settings,
        cfg=cfg_v4a_f5,
        provenance=provider.describe(),
        is_synthetic=provider.is_synthetic,
    )
    v4b_full = build_momentum_data_from_bars(
        bars=bars,
        benchmark=benchmark,
        universe=universe,
        settings=settings,
        cfg=cfg_v4b,
        provenance=provider.describe(),
        is_synthetic=provider.is_synthetic,
    )
    logging.info("Computed v4a & v4b causal features in %.1fs", time.time() - t_feat)

    # Document Wilder vs Simple RSI(2) IS-only signal count (never peeking at OOS)
    v4a_simple_is = slice_panic_data(
        build_panic_data_from_bars(
            bars=bars,
            benchmark=benchmark,
            universe=universe,
            settings=settings,
            cfg=cfg_v4a_simple_diag,
            provenance=provider.describe(),
            is_synthetic=provider.is_synthetic,
        ),
        IS_START,
        IS_END,
    )
    simple_is_sigs, _, _ = screen_panic_sessions(v4a_simple_is, cfg_v4a_simple_diag)
    simple_is_count = sum(len(v) for v in simple_is_sigs.values())
    simple_is_days = len(simple_is_sigs)

    v4a_windows = [
        ("IS", slice_panic_data(v4a_full, IS_START, IS_END), IS_END),
        ("OOS", slice_panic_data(v4a_full, OOS_START, OOS_END), None),
        ("FULL_5Y", v4a_full, None),
    ]
    v4b_windows = [
        ("IS", slice_momentum_data(v4b_full, IS_START, IS_END), IS_END),
        ("OOS", slice_momentum_data(v4b_full, OOS_START, OOS_END), None),
        ("FULL_5Y", v4b_full, None),
    ]

    results: dict[str, Any] = {
        "locked_configs": {
            "v4a_f5": cfg_v4a_f5.model_dump(mode="json"),
            "v4a_f10": cfg_v4a_f10.model_dump(mode="json"),
            "v4b": cfg_v4b.model_dump(mode="json"),
        },
        "is_rsi2_formula_check": {
            "simple_rsi2_is_signals": simple_is_count,
            "simple_rsi2_is_active_days": simple_is_days,
        },
        "data_stats": {
            "FULL_5Y": v4a_full.data_stats,
            "IS": v4a_windows[0][1].data_stats,
            "OOS": v4a_windows[1][1].data_stats,
        },
        "provenance": provider.describe(),
        "raw_signal_edge": {"v4a": {}, "v4b": {}},
        "portfolio": {"v4a_f5": {}, "v4a_f10": {}, "v4b": {}},
    }

    full_runs_for_plot: dict[int, dict[str, Any]] = {}

    # ---- Run v4a (f=5% and f=10%) across IS, Blind OOS, Full 5Y ----
    for w_name, w_data, max_bar_d in v4a_windows:
        sigs_by_date, rej, elig_counts = screen_panic_sessions(w_data, cfg_v4a_f5)
        elig_vals = list(elig_counts.values())
        _, raw_sum = evaluate_v4a_raw_edge(
            w_data, sigs_by_date, max_hold_days=cfg_v4a_f5.max_hold_days, max_bar_date=max_bar_d
        )
        raw_sum["eligible_universe_mean"] = float(np.mean(elig_vals)) if elig_vals else 0.0
        raw_sum["eligible_universe_min"] = int(np.min(elig_vals)) if elig_vals else 0
        raw_sum["eligible_universe_max"] = int(np.max(elig_vals)) if elig_vals else 0
        raw_sum["screen_rejections"] = rej
        results["raw_signal_edge"]["v4a"][w_name] = raw_sum

        if w_name == "IS":
            results["is_rsi2_formula_check"]["wilder_rsi2_is_signals"] = raw_sum["total_signals"]
            results["is_rsi2_formula_check"]["wilder_rsi2_is_active_days"] = raw_sum["active_signal_days"]

        logging.info(
            "=== v4a RAW EDGE [%s] N=%d signals across %d days (elig univ mean=%.0f) ===",
            w_name,
            raw_sum["total_signals"],
            raw_sum["active_signal_days"],
            raw_sum["eligible_universe_mean"],
        )
        for h in ("+1d", "+3d", "+5d", "+10d", "to_SMA5_max10d"):
            hs = raw_sum["horizons"][h]
            dc = raw_sum["day_clustered_horizons"][h]
            logging.info(
                "  %-14s n=%4d | raw=%+6.2f%% | ex=%+6.2f%% t=%+5.2f p=%.4f | day-clustered(n=%3d) ex=%+6.2f%% t=%+5.2f p=%.4f",
                h,
                hs["n"],
                hs["raw_mean"] * 100,
                hs["excess_mean"] * 100,
                hs["excess_tstat"],
                hs["excess_pvalue"],
                dc["n"],
                dc["excess_mean"] * 100,
                dc["excess_tstat"],
                dc["excess_pvalue"],
            )

        results["portfolio"]["v4a_f5"][w_name] = {}
        results["portfolio"]["v4a_f10"][w_name] = {}
        for variant_name, v_cfg in (("v4a_f5", cfg_v4a_f5), ("v4a_f10", cfg_v4a_f10)):
            for cap_eur in args.capitals_eur:
                cap_int = int(cap_eur)
                cap_usd = cap_eur * settings.backtest.eur_usd_rate
                for vps_on in (False, True):
                    tag = f"eur{cap_int}_vps" if vps_on else f"eur{cap_int}"
                    res = run_backtest_v4a(
                        w_data,
                        settings,
                        v_cfg,
                        initial_capital_usd=cap_usd,
                        vps_enabled=vps_on,
                        precomputed_signals=(sigs_by_date, rej),
                    )
                    m = compute_metrics(res.equity_curve, cap_usd, res.trades, res.skipped, settings.metrics)
                    packed = _pack_v4_run(res, m)
                    results["portfolio"][variant_name][w_name][tag] = packed
                    if w_name == "FULL_5Y" and not vps_on:
                        full_runs_for_plot.setdefault(cap_int, {})[variant_name] = res
                    pm = packed["metrics"]
                    logging.info(
                        "  [%s] %-7s %-12s eq=$%8.2f ret=%+6.1f%% cagr=%+6.1f%% spy=%+6.1f%% sr=%+5.2f dsr=%.3f dd=%6.1f%% tr=%3d wr=%4.1f%% expR=%+5.2f pf=%4.2f exp=%4.1f%% turn=%4.1fx fric=$%5.0f",
                        w_name,
                        variant_name,
                        tag,
                        pm["final_equity"],
                        pm["total_return"] * 100,
                        pm["cagr"] * 100,
                        pm["benchmark_total_return"] * 100,
                        pm["sharpe"] if not math.isnan(pm["sharpe"]) else 0.0,
                        pm["deflated_sharpe"] if not math.isnan(pm["deflated_sharpe"]) else 0.0,
                        pm["max_drawdown"] * 100,
                        pm["trades"],
                        (pm["win_rate"] * 100) if pm["trades"] else 0.0,
                        pm["expectancy_r"] if pm["trades"] else 0.0,
                        pm["profit_factor"] if pm["trades"] else 0.0,
                        pm["exposure"] * 100,
                        pm["annualized_turnover"],
                        pm["total_friction_usd"],
                    )

    # ---- Run v4b across IS, Blind OOS, Full 5Y ----
    for w_name, w_data, max_bar_d in v4b_windows:
        leaders_by_date, rej, liq_counts = select_all_rebalance_leaders(w_data, cfg_v4b)
        liq_vals = list(liq_counts.values())
        _, raw_sum = evaluate_v4b_raw_edge(
            w_data, leaders_by_date, rebalance_every_days=cfg_v4b.rebalance_every_days, max_bar_date=max_bar_d
        )
        raw_sum["liquid_50m_universe_mean"] = float(np.mean(liq_vals)) if liq_vals else 0.0
        raw_sum["liquid_50m_universe_min"] = int(np.min(liq_vals)) if liq_vals else 0
        raw_sum["liquid_50m_universe_max"] = int(np.max(liq_vals)) if liq_vals else 0
        raw_sum["screen_rejections"] = rej
        results["raw_signal_edge"]["v4b"][w_name] = raw_sum

        logging.info(
            "=== v4b RAW EDGE [%s] reb=%d (on=%d, off=%d, eval=%d) | basket=%+6.2f%% spy=%+6.2f%% | ex=%+6.2f%% med=%+6.2f%% hit=%.1f%% t=%+5.2f p=%.4f ===",
            w_name,
            raw_sum["rebalance_days_total"],
            raw_sum["rebalance_days_risk_on"],
            raw_sum["rebalance_days_risk_off"],
            raw_sum["evaluated_10d_periods"],
            raw_sum["raw_mean"] * 100,
            raw_sum["spy_mean_10d"] * 100,
            raw_sum["excess_mean"] * 100,
            raw_sum["excess_median"] * 100,
            raw_sum["excess_pos_rate"] * 100,
            raw_sum["excess_tstat"],
            raw_sum["excess_pvalue"],
        )

        results["portfolio"]["v4b"][w_name] = {}
        for cap_eur in args.capitals_eur:
            cap_int = int(cap_eur)
            cap_usd = cap_eur * settings.backtest.eur_usd_rate
            for vps_on in (False, True):
                tag = f"eur{cap_int}_vps" if vps_on else f"eur{cap_int}"
                res = run_backtest_v4b(
                    w_data,
                    settings,
                    cfg_v4b,
                    initial_capital_usd=cap_usd,
                    vps_enabled=vps_on,
                    precomputed_leaders=(leaders_by_date, rej),
                )
                m = compute_metrics(res.equity_curve, cap_usd, res.trades, res.skipped, settings.metrics)
                packed = _pack_v4_run(res, m)
                results["portfolio"]["v4b"][w_name][tag] = packed
                if w_name == "FULL_5Y" and not vps_on:
                    full_runs_for_plot.setdefault(cap_int, {})["v4b"] = res
                pm = packed["metrics"]
                logging.info(
                    "  [%s] v4b     %-12s eq=$%8.2f ret=%+6.1f%% cagr=%+6.1f%% spy=%+6.1f%% sr=%+5.2f dsr=%.3f dd=%6.1f%% tr=%3d wr=%4.1f%% expR=%+5.2f pf=%4.2f exp=%4.1f%% turn=%4.1fx fric=$%5.0f",
                    w_name,
                    tag,
                    pm["final_equity"],
                    pm["total_return"] * 100,
                    pm["cagr"] * 100,
                    pm["benchmark_total_return"] * 100,
                    pm["sharpe"] if not math.isnan(pm["sharpe"]) else 0.0,
                    pm["deflated_sharpe"] if not math.isnan(pm["deflated_sharpe"]) else 0.0,
                    pm["max_drawdown"] * 100,
                    pm["trades"],
                    (pm["win_rate"] * 100) if pm["trades"] else 0.0,
                    pm["expectancy_r"] if pm["trades"] else 0.0,
                    pm["profit_factor"] if pm["trades"] else 0.0,
                    pm["exposure"] * 100,
                    pm["annualized_turnover"],
                    pm["total_friction_usd"],
                )

    # Save side-by-side comparison equity curve PNGs (with IS/OOS split line)
    media_paths: dict[str, str] = {}
    for cap_int, runs in sorted(full_runs_for_plot.items()):
        store_png = STORE_ROOT / "media" / f"v4-showdown-eur{cap_int}.png"
        local_png = ROOT / "outputs" / f"v4-showdown-eur{cap_int}.png"
        plot_v4_showdown_png(
            runs["v4a_f5"],
            runs["v4a_f10"],
            runs["v4b"],
            float(cap_int),
            pd.Timestamp(OOS_START),
            store_png,
        )
        plot_v4_showdown_png(
            runs["v4a_f5"],
            runs["v4a_f10"],
            runs["v4b"],
            float(cap_int),
            pd.Timestamp(OOS_START),
            local_png,
        )
        media_paths[f"eur{cap_int}_store"] = str(store_png)
        media_paths[f"eur{cap_int}_local"] = str(local_png)
        logging.info("wrote showdown PNG -> %s", store_png)

    results["media_paths"] = media_paths

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logging.info("v4 comparison finished in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
