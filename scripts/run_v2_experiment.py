"""Run Phase 0 Baseline v1 vs Variant v2, v2b, and single-lever ablations.

Uses ``--provider yfinance`` by default so FMP Basic is never called.
Fetches/caches ``IWM`` via the same provider for the exact 5-year + warmup window.
Runs:
  - Raw signal diagnostics (via ``scripts.diagnose_signal``)
  - Baseline v1 (untouched ``src.backtest.engine.run_backtest``)
  - Single-lever ablations:
      * ablation_trend:   IWM > SMA200 & SPY > SMA200 only
      * ablation_gap:     +2.0% t+1 Open gap cap only
      * ablation_stop:    2.5 * ATR14 initial stop only (f = 5% via config DI)
  - Variant v2:           all 3 levers combined (IWM+SPY trend, +2.0% gap cap, 2.5*ATR14)
  - Variant v2_cap25:     v2 with max_stop_pct=0.25 (diagnostic of 15% cap interaction)
  - Variant v2b:          v2 + 2-to-5 day consolidation breakout above event-day high

Usage:
    python scripts/run_v2_experiment.py --provider yfinance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_signal import (  # noqa: E402
    collect_signal_diagnostics,
    print_diagnostics,
    summarize_diagnostics,
)
from src.backtest.engine import BacktestResult, prepare_data, run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.factory import build_provider  # noqa: E402
from src.strategy.pead_v2_screener import V2Config, compute_benchmark_trend, run_backtest_v2  # noqa: E402


def _exit_breakdown(res: BacktestResult) -> dict[str, dict[str, float | int]]:
    by_reason: dict[str, list[float]] = {}
    for t in res.trades:
        by_reason.setdefault(t.exit_reason.value, []).append(t.r_multiple)
    return {
        k: {"count": len(v), "mean_r": float(sum(v) / len(v))}
        for k, v in sorted(by_reason.items())
    }


def _pack_run(res: BacktestResult, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "metrics": metrics,
        "rejections": res.rejections,
        "sizing_rejections": res.sizing_rejections,
        "exit_breakdown": _exit_breakdown(res),
        "skipped_reasons": dict(Counter(s.reason for s in res.skipped)),
        "fees_paid": res.fees_paid,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Baseline v1 vs Variant v2 experiment.")
    ap.add_argument("--provider", choices=["yfinance", "sample", "auto", "fmp_massive"], default="yfinance")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v1_vs_v2_results.json"))
    ap.add_argument("--diag-csv", default=str(ROOT / "outputs" / "diagnose_signal.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    settings = settings.model_copy(update={"data": settings.data.model_copy(update={"provider": args.provider})})
    provider, notes = build_provider(settings)
    for n in notes:
        logging.info("provider: %s", n)

    data = prepare_data(settings, provider, max_symbols=args.max_symbols)
    bt = settings.backtest
    fetch_start = bt.start_date - timedelta(days=bt.warmup_calendar_days)
    iwm = provider.get_benchmark_bars(fetch_start, bt.end_date, "IWM")
    logging.info("data + IWM prepared in %.1fs: %s", time.time() - t0, json.dumps(data.data_stats))

    # Benchmark trend session counts inside backtest calendar
    iwm_trends = compute_benchmark_trend(iwm, settings.regime.sma_window)
    cal = data.calendar
    trend_days = {
        "total_sessions": len(cal),
        "spy_above_sma200": sum(1 for ts in cal if data.regimes[ts].spy_close > data.regimes[ts].sma200),
        "iwm_above_sma200": sum(1 for ts in cal if (t := iwm_trends.get(ts)) is not None and t.is_bullish),
        "both_above_sma200": sum(
            1
            for ts in cal
            if data.regimes[ts].spy_close > data.regimes[ts].sma200
            and (t := iwm_trends.get(ts)) is not None
            and t.is_bullish
        ),
        "spy_only_above_sma200": sum(
            1
            for ts in cal
            if data.regimes[ts].spy_close > data.regimes[ts].sma200
            and not ((t := iwm_trends.get(ts)) is not None and t.is_bullish)
        ),
    }

    # Task 1: signal diagnostics
    diag_df = collect_signal_diagnostics(data, settings, iwm=iwm)
    diag_summary = summarize_diagnostics(diag_df)
    print_diagnostics(diag_summary)
    Path(args.diag_csv).parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(args.diag_csv, index=False)

    # Task 2/3: variant suite
    variants: list[tuple[str, V2Config | None]] = [
        ("baseline_v1", None),
        (
            "ablation_trend",
            V2Config(
                require_spy_above_sma200=True,
                require_iwm_above_sma200=True,
                entry_limit_cap_pct=settings.execution.entry_limit_cap_pct,
                atr_stop_multiple=settings.risk.atr_stop_multiple,
            ),
        ),
        (
            "ablation_gap_2pct",
            V2Config(
                require_spy_above_sma200=False,
                require_iwm_above_sma200=False,
                entry_limit_cap_pct=0.02,
                atr_stop_multiple=settings.risk.atr_stop_multiple,
            ),
        ),
        (
            "ablation_stop_2_5atr",
            V2Config(
                require_spy_above_sma200=False,
                require_iwm_above_sma200=False,
                entry_limit_cap_pct=settings.execution.entry_limit_cap_pct,
                atr_stop_multiple=2.5,
            ),
        ),
        (
            "variant_v2",
            V2Config(
                require_spy_above_sma200=True,
                require_iwm_above_sma200=True,
                entry_limit_cap_pct=0.02,
                atr_stop_multiple=2.5,
            ),
        ),
        (
            "variant_v2_cap25",
            V2Config(
                require_spy_above_sma200=True,
                require_iwm_above_sma200=True,
                entry_limit_cap_pct=0.02,
                atr_stop_multiple=2.5,
                max_stop_pct=0.25,
            ),
        ),
        (
            "variant_v2b_breakout",
            V2Config(
                require_spy_above_sma200=True,
                require_iwm_above_sma200=True,
                entry_limit_cap_pct=0.02,
                atr_stop_multiple=2.5,
                consolidation_breakout=True,
            ),
        ),
    ]

    results: dict[str, Any] = {
        "provider_notes": notes,
        "provenance": data.provenance,
        "data_stats": data.data_stats,
        "trend_days": trend_days,
        "diagnostics": diag_summary,
        "variants": {},
        "vps_comparison": {},
    }

    for name, v2_cfg in variants:
        results["variants"][name] = {}
        for cap_eur in args.capitals_eur:
            cap_usd = cap_eur * bt.eur_usd_rate
            tag = f"eur{int(cap_eur)}"
            if v2_cfg is None:
                res = run_backtest(data, settings, cap_usd, vps_enabled=False)
            else:
                res = run_backtest_v2(data, settings, v2_cfg, iwm=iwm, initial_capital_usd=cap_usd, vps_enabled=False)
            m = compute_metrics(res.equity_curve, cap_usd, res.trades, res.skipped, settings.metrics)
            results["variants"][name][tag] = _pack_run(res, m)
            logging.info(
                "%-22s %-7s eq=$%7.2f ret=%+6.1f%% cagr=%+6.1f%% sr=%+5.2f dd=%6.1f%% trades=%3d wr=%4.1f%% expR=%+5.2f pf=%4.2f",
                name,
                tag,
                m["final_equity"],
                m["total_return"] * 100,
                m["cagr"] * 100,
                m["sharpe"],
                m["max_drawdown"] * 100,
                m["trades"],
                m["win_rate"] * 100 if m["trades"] else 0.0,
                m["expectancy_r"] if m["trades"] else 0.0,
                m["profit_factor"] if m["trades"] else 0.0,
            )

    # Also record VPS-on comparison for baseline_v1 and variant_v2
    for name, v2_cfg in [("baseline_v1", None), ("variant_v2", variants[4][1])]:
        results["vps_comparison"][name] = {}
        for cap_eur in args.capitals_eur:
            cap_usd = cap_eur * bt.eur_usd_rate
            tag = f"eur{int(cap_eur)}-vps"
            if v2_cfg is None:
                res = run_backtest(data, settings, cap_usd, vps_enabled=True)
            else:
                res = run_backtest_v2(data, settings, v2_cfg, iwm=iwm, initial_capital_usd=cap_usd, vps_enabled=True)
            m = compute_metrics(res.equity_curve, cap_usd, res.trades, res.skipped, settings.metrics)
            results["vps_comparison"][name][tag] = _pack_run(res, m)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logging.info("experiment complete in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
