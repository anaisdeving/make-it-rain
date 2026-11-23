"""Signal-level diagnostic study for all raw signals passing ``pead_screener.py``.

Evaluates every signal emitted by ``pead_screener.screen`` (ignoring the +0.3% entry
cap and ignoring stop-losses), strictly from t+1 Open with strictly past-only ATR14
and benchmark SMAs (zero look-ahead).

Prints:
1. Mean and median forward returns at +5d, +10d, +20d, +45d, +60d trading days from
   t+1 Open, overall, by year (2021-2026), and by MarketRegime (NORMAL, CAUTION, PANIC).
2. Performance split by t+1 Open gap vs day t Close:
   <= +0.3%, (+0.3%, +2.5%], and > +2.5% (plus (+0.3%, +2.0%] vs > +2.0%).
3. 45-trading-day MAE and MFE in ATR14 multiples (intraday Low/High from t+1 Open),
   stop-distance binding breakdown (ATR vs EventDayLow vs 15% cap for 1.5x and 2.5x),
   and how often winning trades (positive at +45d, or reaching >= +1.5R / >= +2.0R)
   dipped past 1.5 * ATR14 and 2.5 * ATR14 (and actual stops) before running.

Usage:
    python scripts/diagnose_signal.py --provider yfinance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.engine import PreparedData, prepare_data, snapshot  # noqa: E402
from src.config import RiskConfig, Settings, load_settings  # noqa: E402
from src.data.factory import build_provider  # noqa: E402
from src.domain.models import BarSeries, RegimeState  # noqa: E402
from src.risk.risk_manager import stop_distance  # noqa: E402
from src.strategy.pead_screener import screen  # noqa: E402
from src.strategy.pead_v2_screener import compute_benchmark_trend  # noqa: E402

HORIZONS: tuple[int, ...] = (5, 10, 20, 45, 60)
MAE_WINDOW: int = 45
YEARS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)
GAP_BUCKETS: tuple[str, ...] = ("<= +0.3%", "(+0.3%, +2.5%]", "> +2.5%")


def classify_gap_bucket(gap: float) -> str:
    if gap <= 0.003:
        return "<= +0.3%"
    if gap <= 0.025:
        return "(+0.3%, +2.5%]"
    return "> +2.5%"


def _stop_binding_component(entry: float, atr14: float, event_day_low: float, mult: float, max_stop_pct: float) -> str:
    atr_d = mult * atr14
    low_d = entry - event_day_low
    cap_d = max_stop_pct * entry
    raw = max(atr_d, low_d)
    if raw >= cap_d - 1e-12:
        return "max_stop_cap_15pct"
    if atr_d >= low_d:
        return f"atr_{mult:.1f}x"
    return "event_day_low"


def _first_idx(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def collect_signal_diagnostics(
    data: PreparedData,
    settings: Settings,
    iwm: BarSeries | None = None,
) -> pd.DataFrame:
    """Collect per-signal forward returns and 45d intraday MAE/MFE (zero look-ahead)."""
    spy = data.benchmark.frame
    iwm_trends = compute_benchmark_trend(iwm, settings.regime.sma_window) if iwm is not None else {}
    rcfg_v1 = settings.risk.model_copy(update={"atr_stop_multiple": 1.5})
    rcfg_v2 = settings.risk.model_copy(update={"atr_stop_multiple": 2.5})

    rows: list[dict[str, Any]] = []
    for ts in data.calendar:
        events = data.events_by_session.get(ts, [])
        if not events:
            continue
        cands = [
            (e, snapshot(e.symbol, ts, data.features[e.symbol].loc[ts]))
            for e in events
            if e.symbol in data.features and ts in data.features[e.symbol].index
        ]
        if not cands:
            continue
        regime = data.regimes[ts]
        sigs, _ = screen(cands, regime, settings.screener)
        iwm_snap = iwm_trends.get(ts)

        for sig in sigs:
            f = data.features[sig.symbol]
            i = int(f.index.get_loc(ts))
            if i + 1 >= len(f):
                continue

            entry_ts = f.index[i + 1]
            entry_open = float(f["open"].iloc[i + 1])
            gap = entry_open / sig.signal_close - 1.0
            atr14 = float(sig.atr14)

            d_v1 = stop_distance(entry_open, atr14, sig.event_day_low, rcfg_v1)
            d_v2 = stop_distance(entry_open, atr14, sig.event_day_low, rcfg_v2)

            j_spy = int(spy.index.get_loc(ts))
            spy_open_1 = (
                float(spy.loc[entry_ts, "open"])
                if entry_ts in spy.index
                else (float(spy["open"].iloc[j_spy + 1]) if j_spy + 1 < len(spy) else float("nan"))
            )

            row: dict[str, Any] = {
                "symbol": sig.symbol,
                "signal_date": ts.date(),
                "year": ts.year,
                "regime": sig.regime.value,
                "spy_above_sma200": bool(regime.spy_close > regime.sma200),
                "iwm_above_sma200": bool(iwm_snap.is_bullish) if iwm_snap is not None else None,
                "timing": sig.event.timing.value,
                "event_day_return": sig.event_day_return,
                "signal_close": sig.signal_close,
                "event_day_low": sig.event_day_low,
                "entry_open": entry_open,
                "next_open_gap": gap,
                "gap_bucket": classify_gap_bucket(gap),
                "within_cap_03": gap <= 0.003,
                "within_cap_20": gap <= 0.020,
                "atr14": atr14,
                "atr14_pct": atr14 / entry_open,
                "event_low_dist_pct": (entry_open - sig.event_day_low) / entry_open,
                "d_v1": d_v1,
                "d_v1_pct": d_v1 / entry_open,
                "d_v1_atr": d_v1 / atr14 if atr14 > 0 else float("nan"),
                "d_v1_binding": _stop_binding_component(
                    entry_open, atr14, sig.event_day_low, 1.5, rcfg_v1.max_stop_pct
                ),
                "d_v2": d_v2,
                "d_v2_pct": d_v2 / entry_open,
                "d_v2_atr": d_v2 / atr14 if atr14 > 0 else float("nan"),
                "d_v2_binding": _stop_binding_component(
                    entry_open, atr14, sig.event_day_low, 2.5, rcfg_v2.max_stop_pct
                ),
            }

            for k in HORIZONS:
                if i + k < len(f):
                    exit_ts = f.index[i + k]
                    ret_k = float(f["close"].iloc[i + k]) / entry_open - 1.0
                    row[f"ret_{k}d"] = ret_k
                    if exit_ts in spy.index and not np.isnan(spy_open_1):
                        bench_k = float(spy.loc[exit_ts, "close"]) / spy_open_1 - 1.0
                        row[f"excess_{k}d"] = ret_k - bench_k
                    elif j_spy + k < len(spy) and not np.isnan(spy_open_1):
                        bench_k = float(spy["close"].iloc[j_spy + k]) / spy_open_1 - 1.0
                        row[f"excess_{k}d"] = ret_k - bench_k
                    else:
                        row[f"excess_{k}d"] = float("nan")
                else:
                    row[f"ret_{k}d"] = float("nan")
                    row[f"excess_{k}d"] = float("nan")

            # 45-trading-day intraday MAE / MFE from t+1 Open (bars i+1 .. i+45 inclusive)
            if i + MAE_WINDOW < len(f) and atr14 > 0:
                win = f.iloc[i + 1 : i + 1 + MAE_WINDOW]
                lows = win["low"].to_numpy(dtype=float)
                highs = win["high"].to_numpy(dtype=float)

                min_low = float(lows.min())
                max_high = float(highs.max())
                mae_atr = (entry_open - min_low) / atr14
                mfe_atr = (max_high - entry_open) / atr14

                dip_1_5_atr = _first_idx(lows <= entry_open - 1.5 * atr14)
                dip_2_5_atr = _first_idx(lows <= entry_open - 2.5 * atr14)
                dip_v1_stop = _first_idx(lows <= entry_open - d_v1)
                dip_v2_stop = _first_idx(lows <= entry_open - d_v2)

                hit_1_5r_v1 = _first_idx(highs >= entry_open + 1.5 * d_v1)
                hit_2_0r_v1 = _first_idx(highs >= entry_open + 2.0 * d_v1)
                hit_1_5r_atr = _first_idx(highs >= entry_open + 1.5 * (1.5 * atr14))
                hit_2_0r_atr = _first_idx(highs >= entry_open + 2.0 * (1.5 * atr14))

                row.update(
                    {
                        "has_45d_window": True,
                        "mae_atr": mae_atr,
                        "mfe_atr": mfe_atr,
                        "mae_pct": (entry_open - min_low) / entry_open,
                        "mfe_pct": (max_high - entry_open) / entry_open,
                        "dipped_1_5_atr": dip_1_5_atr is not None,
                        "dipped_2_5_atr": dip_2_5_atr is not None,
                        "dipped_v1_stop": dip_v1_stop is not None,
                        "dipped_v2_stop": dip_v2_stop is not None,
                        "reached_1_5r_v1": hit_1_5r_v1 is not None,
                        "reached_2_0r_v1": hit_2_0r_v1 is not None,
                        "reached_1_5r_atr": hit_1_5r_atr is not None,
                        "reached_2_0r_atr": hit_2_0r_atr is not None,
                        "dip_1_5_atr_before_1_5r_v1": (
                            dip_1_5_atr is not None and hit_1_5r_v1 is not None and dip_1_5_atr <= hit_1_5r_v1
                        ),
                        "dip_2_5_atr_before_1_5r_v1": (
                            dip_2_5_atr is not None and hit_1_5r_v1 is not None and dip_2_5_atr <= hit_1_5r_v1
                        ),
                        "dip_v1_stop_before_1_5r_v1": (
                            dip_v1_stop is not None and hit_1_5r_v1 is not None and dip_v1_stop <= hit_1_5r_v1
                        ),
                        "dip_v2_stop_before_1_5r_v1": (
                            dip_v2_stop is not None and hit_1_5r_v1 is not None and dip_v2_stop <= hit_1_5r_v1
                        ),
                        "dip_1_5_atr_before_2_0r_v1": (
                            dip_1_5_atr is not None and hit_2_0r_v1 is not None and dip_1_5_atr <= hit_2_0r_v1
                        ),
                        "dip_2_5_atr_before_2_0r_v1": (
                            dip_2_5_atr is not None and hit_2_0r_v1 is not None and dip_2_5_atr <= hit_2_0r_v1
                        ),
                        "dip_v1_stop_before_2_0r_v1": (
                            dip_v1_stop is not None and hit_2_0r_v1 is not None and dip_v1_stop <= hit_2_0r_v1
                        ),
                        "dip_v2_stop_before_2_0r_v1": (
                            dip_v2_stop is not None and hit_2_0r_v1 is not None and dip_v2_stop <= hit_2_0r_v1
                        ),
                        "dip_1_5_atr_before_1_5r_atr": (
                            dip_1_5_atr is not None and hit_1_5r_atr is not None and dip_1_5_atr <= hit_1_5r_atr
                        ),
                        "dip_2_5_atr_before_1_5r_atr": (
                            dip_2_5_atr is not None and hit_1_5r_atr is not None and dip_2_5_atr <= hit_1_5r_atr
                        ),
                        "dip_1_5_atr_before_2_0r_atr": (
                            dip_1_5_atr is not None and hit_2_0r_atr is not None and dip_1_5_atr <= hit_2_0r_atr
                        ),
                        "dip_2_5_atr_before_2_0r_atr": (
                            dip_2_5_atr is not None and hit_2_0r_atr is not None and dip_2_5_atr <= hit_2_0r_atr
                        ),
                    }
                )
            else:
                row["has_45d_window"] = False

            rows.append(row)

    return pd.DataFrame(rows)


def _horizon_stats(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(df))}
    for k in HORIZONS:
        col = f"ret_{k}d"
        ex_col = f"excess_{k}d"
        s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        ex = df[ex_col].dropna() if ex_col in df.columns else pd.Series(dtype=float)
        out[f"n_{k}d"] = int(len(s))
        out[f"mean_{k}d"] = float(s.mean()) if len(s) else float("nan")
        out[f"median_{k}d"] = float(s.median()) if len(s) else float("nan")
        out[f"pos_rate_{k}d"] = float((s > 0).mean()) if len(s) else float("nan")
        out[f"excess_mean_{k}d"] = float(ex.mean()) if len(ex) else float("nan")
        out[f"excess_median_{k}d"] = float(ex.median()) if len(ex) else float("nan")
    return out


def summarize_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Build structured summary tables for printing and reporting."""
    summary: dict[str, Any] = {"total_signals": int(len(df))}
    if df.empty:
        return summary

    summary["overall"] = _horizon_stats(df)

    by_year: dict[str, Any] = {}
    for y in YEARS:
        sub = df[df["year"] == y]
        by_year[str(y)] = _horizon_stats(sub)
    summary["by_year"] = by_year

    by_regime: dict[str, Any] = {}
    for r in (RegimeState.NORMAL.value, RegimeState.CAUTION.value, RegimeState.PANIC.value):
        sub = df[df["regime"] == r]
        by_regime[r] = _horizon_stats(sub)
    summary["by_regime"] = by_regime

    by_gap: dict[str, Any] = {}
    for b in GAP_BUCKETS:
        sub = df[df["gap_bucket"] == b]
        by_gap[b] = _horizon_stats(sub)
    summary["by_gap_bucket"] = by_gap

    summary["by_v2_gap_cap"] = {
        "<= +0.3%": _horizon_stats(df[df["next_open_gap"] <= 0.003]),
        "(+0.3%, +2.0%]": _horizon_stats(df[(df["next_open_gap"] > 0.003) & (df["next_open_gap"] <= 0.020)]),
        "<= +2.0%": _horizon_stats(df[df["next_open_gap"] <= 0.020]),
        "> +2.0%": _horizon_stats(df[df["next_open_gap"] > 0.020]),
    }

    if "iwm_above_sma200" in df.columns and df["iwm_above_sma200"].notna().any():
        both = df[(df["spy_above_sma200"] == True) & (df["iwm_above_sma200"] == True)]  # noqa: E712
        spy_only = df[(df["spy_above_sma200"] == True) & (df["iwm_above_sma200"] == False)]  # noqa: E712
        spy_below = df[df["spy_above_sma200"] == False]  # noqa: E712
        summary["by_trend_gate"] = {
            "SPY>200 & IWM>200": _horizon_stats(both),
            "SPY>200 & IWM<=200": _horizon_stats(spy_only),
            "SPY<=200": _horizon_stats(spy_below),
        }

    # Stop binding breakdown across all signals
    summary["stop_binding"] = {
        "v1_1_5x_atr": {
            "counts": df["d_v1_binding"].value_counts().to_dict(),
            "shares": (df["d_v1_binding"].value_counts(normalize=True)).to_dict(),
            "median_stop_pct": float(df["d_v1_pct"].median()),
            "mean_stop_pct": float(df["d_v1_pct"].mean()),
            "median_stop_atr": float(df["d_v1_atr"].median()),
            "mean_stop_atr": float(df["d_v1_atr"].mean()),
        },
        "v2_2_5x_atr": {
            "counts": df["d_v2_binding"].value_counts().to_dict(),
            "shares": (df["d_v2_binding"].value_counts(normalize=True)).to_dict(),
            "median_stop_pct": float(df["d_v2_pct"].median()),
            "mean_stop_pct": float(df["d_v2_pct"].mean()),
            "median_stop_atr": float(df["d_v2_atr"].median()),
            "mean_stop_atr": float(df["d_v2_atr"].mean()),
        },
        "median_atr14_pct": float(df["atr14_pct"].median()),
        "median_event_low_dist_pct": float(df["event_low_dist_pct"].median()),
    }

    # 45d MAE / MFE analysis
    w = df[df["has_45d_window"] == True].copy()  # noqa: E712
    mae_summary: dict[str, Any] = {"n_45d": int(len(w))}
    if len(w):
        for col in ("mae_atr", "mfe_atr", "mae_pct", "mfe_pct"):
            s = w[col].dropna()
            mae_summary[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
            }
        mae_summary["all_signals_dip_rates"] = {
            "dipped_1_5_atr": float(w["dipped_1_5_atr"].mean()),
            "dipped_2_5_atr": float(w["dipped_2_5_atr"].mean()),
            "dipped_v1_stop": float(w["dipped_v1_stop"].mean()),
            "dipped_v2_stop": float(w["dipped_v2_stop"].mean()),
        }

        pos45 = w[w["ret_45d"] > 0]
        mae_summary["winners_pos_45d"] = {
            "n": int(len(pos45)),
            "share_of_45d": float(len(pos45) / len(w)),
            "dipped_1_5_atr": float(pos45["dipped_1_5_atr"].mean()) if len(pos45) else float("nan"),
            "dipped_2_5_atr": float(pos45["dipped_2_5_atr"].mean()) if len(pos45) else float("nan"),
            "dipped_v1_stop": float(pos45["dipped_v1_stop"].mean()) if len(pos45) else float("nan"),
            "dipped_v2_stop": float(pos45["dipped_v2_stop"].mean()) if len(pos45) else float("nan"),
            "median_mae_atr": float(pos45["mae_atr"].median()) if len(pos45) else float("nan"),
            "median_mfe_atr": float(pos45["mfe_atr"].median()) if len(pos45) else float("nan"),
        }

        for r_label, flag, b15, b25, bv1, bv2 in [
            (
                "reached_1_5r_v1",
                "reached_1_5r_v1",
                "dip_1_5_atr_before_1_5r_v1",
                "dip_2_5_atr_before_1_5r_v1",
                "dip_v1_stop_before_1_5r_v1",
                "dip_v2_stop_before_1_5r_v1",
            ),
            (
                "reached_2_0r_v1",
                "reached_2_0r_v1",
                "dip_1_5_atr_before_2_0r_v1",
                "dip_2_5_atr_before_2_0r_v1",
                "dip_v1_stop_before_2_0r_v1",
                "dip_v2_stop_before_2_0r_v1",
            ),
        ]:
            sub = w[w[flag] == True]  # noqa: E712
            mae_summary[r_label] = {
                "n": int(len(sub)),
                "share_of_45d": float(len(sub) / len(w)),
                "dipped_1_5_atr_before": float(sub[b15].mean()) if len(sub) else float("nan"),
                "dipped_2_5_atr_before": float(sub[b25].mean()) if len(sub) else float("nan"),
                "dipped_v1_stop_before": float(sub[bv1].mean()) if len(sub) else float("nan"),
                "dipped_v2_stop_before": float(sub[bv2].mean()) if len(sub) else float("nan"),
                "dipped_1_5_atr_ever": float(sub["dipped_1_5_atr"].mean()) if len(sub) else float("nan"),
                "dipped_2_5_atr_ever": float(sub["dipped_2_5_atr"].mean()) if len(sub) else float("nan"),
            }

        for r_label, flag, b15, b25 in [
            ("reached_1_5r_pure_atr", "reached_1_5r_atr", "dip_1_5_atr_before_1_5r_atr", "dip_2_5_atr_before_1_5r_atr"),
            ("reached_2_0r_pure_atr", "reached_2_0r_atr", "dip_1_5_atr_before_2_0r_atr", "dip_2_5_atr_before_2_0r_atr"),
        ]:
            sub = w[w[flag] == True]  # noqa: E712
            mae_summary[r_label] = {
                "n": int(len(sub)),
                "share_of_45d": float(len(sub) / len(w)),
                "dipped_1_5_atr_before": float(sub[b15].mean()) if len(sub) else float("nan"),
                "dipped_2_5_atr_before": float(sub[b25].mean()) if len(sub) else float("nan"),
            }

    summary["mae_mfe_45d"] = mae_summary
    return summary


def _fmt_pct(v: float) -> str:
    return "   n/a" if np.isnan(v) else f"{v:+6.2%}"


def _print_horizon_table(title: str, groups: dict[str, dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    hdr = f"{'Group':<20} {'N':>5} | " + " | ".join(
        f"+{k}d mean / med (ex)" for k in HORIZONS
    )
    print(hdr)
    print("-" * len(hdr))
    for name, st in groups.items():
        cells = []
        for k in HORIZONS:
            m = _fmt_pct(st[f"mean_{k}d"])
            med = _fmt_pct(st[f"median_{k}d"])
            ex = _fmt_pct(st[f"excess_mean_{k}d"])
            cells.append(f"{m}/{med} ({ex})")
        print(f"{name:<20} {st['n']:>5} | " + " | ".join(cells))


def print_diagnostics(summary: dict[str, Any]) -> None:
    print(f"Total raw signals passing pead_screener.py (with t+1 bar): {summary['total_signals']}")
    if not summary["total_signals"]:
        return

    _print_horizon_table("Overall Forward Returns from t+1 Open (raw mean/median, excess vs SPY mean)", {"ALL": summary["overall"]})
    _print_horizon_table("Forward Returns by Year", summary["by_year"])
    _print_horizon_table("Forward Returns by MarketRegime (PANIC=0 because pead_screener rejects PANIC)", summary["by_regime"])
    _print_horizon_table("Forward Returns by t+1 Open Gap vs Day t Close", summary["by_gap_bucket"])
    _print_horizon_table("Forward Returns by v2 Gap Cap (+2.0% split)", summary["by_v2_gap_cap"])
    if "by_trend_gate" in summary:
        _print_horizon_table("Forward Returns by Dual Benchmark Trend Gate (SPY & IWM vs SMA200)", summary["by_trend_gate"])

    sb = summary["stop_binding"]
    print("\n=== Initial Stop Binding Breakdown (min(max(mult*ATR14, Entry-EventDayLow), 0.15*Entry)) ===")
    print(f"Median ATR14 / Entry: {sb['median_atr14_pct']:.2%} | Median (Entry - EventDayLow) / Entry: {sb['median_event_low_dist_pct']:.2%}")
    for label, key in [("v1 (1.5 * ATR14)", "v1_1_5x_atr"), ("v2 (2.5 * ATR14)", "v2_2_5x_atr")]:
        b = sb[key]
        parts = ", ".join(f"{k}: {b['counts'][k]} ({b['shares'][k]:.1%})" for k in sorted(b["counts"]))
        print(f"  {label}: median stop={b['median_stop_pct']:.2%} ({b['median_stop_atr']:.2f} ATR) | {parts}")

    mm = summary["mae_mfe_45d"]
    print(f"\n=== 45-Trading-Day MAE / MFE in ATR14 Multiples (N={mm['n_45d']} with full 45d window) ===")
    if mm["n_45d"]:
        for label, key in [("MAE (ATR14)", "mae_atr"), ("MFE (ATR14)", "mfe_atr")]:
            d = mm[key]
            print(f"  {label}: mean={d['mean']:.2f}  median={d['median']:.2f}  p25={d['p25']:.2f}  p75={d['p75']:.2f}  p90={d['p90']:.2f}")
        ar = mm["all_signals_dip_rates"]
        print(
            f"  All signals dipping past: 1.5*ATR14={ar['dipped_1_5_atr']:.1%}, "
            f"2.5*ATR14={ar['dipped_2_5_atr']:.1%}, v1 stop={ar['dipped_v1_stop']:.1%}, v2 stop={ar['dipped_v2_stop']:.1%}"
        )
        p45 = mm["winners_pos_45d"]
        print(
            f"  Winners (+45d > 0, N={p45['n']}, {p45['share_of_45d']:.1%}): "
            f"dipped past 1.5*ATR14={p45['dipped_1_5_atr']:.1%}, "
            f"2.5*ATR14={p45['dipped_2_5_atr']:.1%}, "
            f"v1 stop={p45['dipped_v1_stop']:.1%}, v2 stop={p45['dipped_v2_stop']:.1%}"
        )
        for label, key in [
            ("Reached >= +1.5R (R = v1 stop d)", "reached_1_5r_v1"),
            ("Reached >= +2.0R (R = v1 stop d)", "reached_2_0r_v1"),
        ]:
            r = mm[key]
            print(
                f"  {label} (N={r['n']}, {r['share_of_45d']:.1%}): "
                f"dipped BEFORE target past 1.5*ATR14={r['dipped_1_5_atr_before']:.1%}, "
                f"2.5*ATR14={r['dipped_2_5_atr_before']:.1%}, "
                f"v1 stop={r['dipped_v1_stop_before']:.1%}, v2 stop={r['dipped_v2_stop_before']:.1%}"
            )
        for label, key in [
            ("Reached >= +1.5R (R = 1.5*ATR14)", "reached_1_5r_pure_atr"),
            ("Reached >= +2.0R (R = 1.5*ATR14)", "reached_2_0r_pure_atr"),
        ]:
            r = mm[key]
            print(
                f"  {label} (N={r['n']}, {r['share_of_45d']:.1%}): "
                f"dipped BEFORE target past 1.5*ATR14={r['dipped_1_5_atr_before']:.1%}, "
                f"2.5*ATR14={r['dipped_2_5_atr_before']:.1%}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose raw PEAD screener signals (zero look-ahead).")
    ap.add_argument("--provider", choices=["yfinance", "sample", "auto", "fmp_massive"], default="yfinance")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--output-csv", default=str(ROOT / "outputs" / "diagnose_signal.csv"))
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "diagnose_signal_summary.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    settings = load_settings()
    settings = settings.model_copy(update={"data": settings.data.model_copy(update={"provider": args.provider})})
    provider, _ = build_provider(settings)
    data = prepare_data(settings, provider, max_symbols=args.max_symbols)

    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    iwm = provider.get_benchmark_bars(fetch_start, bt.end_date, "IWM")

    df = collect_signal_diagnostics(data, settings, iwm=iwm)
    summary = summarize_diagnostics(df)
    print_diagnostics(summary)

    if args.output_csv:
        p = Path(args.output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
