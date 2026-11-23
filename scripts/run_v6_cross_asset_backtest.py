"""Strategy v6 Backtest & Validation: Cross-Asset Regime-Filtered Momentum Rotation.

Strict Out-of-Sample Protocol (ALL rules locked BEFORE running Blind OOS once):
- Warmup:     2020-10-18 to 2021-11-21
- In-Sample:  2021-11-22 to 2024-11-21 (3.00 years)
- Blind OOS:  2024-11-22 to 2026-11-20 (2.00 years — evaluated ONCE, zero tuning)
- Full 5Y:    2021-11-22 to 2026-11-20 (5.00 years)

Executes native EUR simulations on an IBKR Ireland EUR cash account:
- EUR 1,000 and EUR 5,000 starting capital (zero FX fees, EUR 0.00/mo hosting)
- EUR 1.25/order commission + 0.10%/side slippage, whole as-traded shares only
- Strict T+2 settlement (zero Good Faith Violations)
- Zero look-ahead bias (day t EU close -> day t+1 EU open)

Outputs:
- Structured JSON: outputs/v6_cross_asset_results.json
- Charts (IS/OOS split + drawdowns + allocation ribbon) in BOTH outputs/ and Project store media/
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
from src.data.cross_asset_eu_loader import (  # noqa: E402
    DEFAULT_CROSS_ASSET_BUCKETS,
    EQUITIES_2X_BUCKET,
    CrossAssetEULoader,
)
from src.data.yfinance_provider import YFinanceProvider  # noqa: E402
from src.strategy.cross_asset_momentum import (  # noqa: E402
    CrossAssetBacktestResult,
    CrossAssetConfig,
    compute_asset_attribution,
    compute_cross_asset_metrics,
    run_cross_asset_backtest,
)
from src.strategy.ucits_etf_regime import (  # noqa: E402
    RiskOffMode,
    SizingMode,
    UcitsRegimeConfig,
    compute_theoretical_bh_metrics,
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


def _pack_v6_result(
    res: CrossAssetBacktestResult,
    mcfg: MetricsConfig,
    symbol_to_bucket: dict[str, str],
) -> dict[str, Any]:
    m = compute_cross_asset_metrics(res, mcfg)
    attr = compute_asset_attribution(res)

    attr_out = {}
    for sym, a in attr.items():
        d = a.model_dump(mode="json")
        d["bucket"] = symbol_to_bucket.get(sym, "cash")
        attr_out[sym] = d

    fills_log = [
        {
            "date": str(f.date),
            "signal_date": str(f.signal_date),
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
    all_trips = list(res.round_trips) + list(res.open_positions_at_end)
    trips_log = [
        {
            "symbol": t.symbol,
            "bucket": symbol_to_bucket.get(t.symbol, t.symbol),
            "entry_date": str(t.entry_date),
            "exit_date": str(t.exit_date),
            "raw_shares_entry": t.raw_shares_entry,
            "bars_held": t.bars_held,
            "net_pnl_eur": round(t.net_pnl_eur, 2),
            "return_pct": round(t.return_pct * 100.0, 2),
            "exit_reason": t.exit_reason,
            "is_open_at_end": t.is_open_at_end,
        }
        for t in all_trips
    ]
    return {
        "metrics": m,
        "attribution": attr_out,
        "fills": fills_log,
        "round_trips": trips_log,
    }


def _plot_v6_png(
    *,
    capital_eur: float,
    v6a: CrossAssetBacktestResult,
    v6b: CrossAssetBacktestResult,
    v6a_2x: CrossAssetBacktestResult,
    v6b_2x: CrossAssetBacktestResult,
    v5b_eq: pd.Series,
    cspx_eq: pd.Series,
    bucket_to_symbol: dict[str, str],
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Generate 3-panel EUR chart: Equity Curves, Drawdowns, and v6b Cross-Asset Allocation."""
    init = float(capital_eur)
    eq_v6a = v6a.equity_curve["equity"]
    eq_v6b = v6b.equity_curve["equity"]
    eq_v6a_2x = v6a_2x.equity_curve["equity"]
    eq_v6b_2x = v6b_2x.equity_curve["equity"]

    def _dd(s: pd.Series) -> pd.Series:
        return (s / s.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14.0, 10.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.1]},
    )

    # Panel 1: Equity Curves
    ax1.plot(
        eq_v6b.index,
        eq_v6b.values,
        label=f"v6b Top-2 (3x Eq + BTC/ETH/Ag/Au) | EUR {eq_v6b.iloc[-1]:,.0f} ({(eq_v6b.iloc[-1]/init-1)*100:+.1f}%)",
        color="#1b7837",
        lw=2.3,
    )
    ax1.plot(
        eq_v6a.index,
        eq_v6a.values,
        label=f"v6a Top-1 (3x Eq + BTC/ETH/Ag/Au) | EUR {eq_v6a.iloc[-1]:,.0f} ({(eq_v6a.iloc[-1]/init-1)*100:+.1f}%)",
        color="#005a9c",
        lw=2.1,
    )
    ax1.plot(
        eq_v6b_2x.index,
        eq_v6b_2x.values,
        label=f"v6b Top-2 (2x CL2.PA sensitivity) | EUR {eq_v6b_2x.iloc[-1]:,.0f} ({(eq_v6b_2x.iloc[-1]/init-1)*100:+.1f}%)",
        color="#66bd63",
        lw=1.4,
        ls="--",
    )
    ax1.plot(
        eq_v6a_2x.index,
        eq_v6a_2x.values,
        label=f"v6a Top-1 (2x CL2.PA sensitivity) | EUR {eq_v6a_2x.iloc[-1]:,.0f} ({(eq_v6a_2x.iloc[-1]/init-1)*100:+.1f}%)",
        color="#6baed6",
        lw=1.3,
        ls="-.",
    )
    ax1.plot(
        v5b_eq.index,
        v5b_eq.values,
        label=f"v5b QQQ3.MI (3x) Baseline         | EUR {v5b_eq.iloc[-1]:,.0f} ({(v5b_eq.iloc[-1]/init-1)*100:+.1f}%)",
        color="#d95f02",
        lw=1.7,
        ls="--",
    )
    ax1.plot(
        cspx_eq.index,
        cspx_eq.values,
        label=f"CSPX.AS (1x S&P500 EUR) B&H       | EUR {cspx_eq.iloc[-1]:,.0f} ({(cspx_eq.iloc[-1]/init-1)*100:+.1f}%)",
        color="#333333",
        lw=1.5,
        ls=":",
    )
    ax1.axhline(init, color="black", lw=0.6, alpha=0.45)

    for ax in (ax1, ax2, ax3):
        ax.axvline(oos_split, color="#8b0000", lw=1.7, ls="--", alpha=0.92)

    ymin, ymax = ax1.get_ylim()
    y_label = ymin + 0.96 * (ymax - ymin)
    ax1.text(
        oos_split - pd.Timedelta(days=18),
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

    ax1.set_ylabel("Portfolio Equity (EUR)")
    ax1.set_title(
        f"Strategy v6: Cross-Asset Momentum Rotation (EUR PRIIPs ETPs/ETCs) vs v5b & CSPX.AS — EUR {int(init):,} (IBKR Ireland)",
        fontsize=11.0,
    )
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax1.grid(alpha=0.3)

    # Panel 2: Drawdowns
    ax2.plot(eq_v6b.index, _dd(eq_v6b).values, color="#1b7837", lw=1.6, label="v6b Top-2 DD%")
    ax2.plot(eq_v6a.index, _dd(eq_v6a).values, color="#005a9c", lw=1.4, label="v6a Top-1 DD%")
    ax2.plot(v5b_eq.index, _dd(v5b_eq).values, color="#d95f02", lw=1.3, ls="--", label="v5b QQQ3.MI DD%")
    ax2.plot(cspx_eq.index, _dd(cspx_eq).values, color="#333333", lw=1.2, ls=":", label="CSPX.AS B&H DD%")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", ncol=4, fontsize=8.0, framealpha=0.90)
    ax2.grid(alpha=0.3)

    # Panel 3: v6b_top2 Dynamic Portfolio Weight Stackplot
    curve_b = v6b.equity_curve
    order = [
        ("equities_3x", f"3x Eq ({bucket_to_symbol['equities_3x']})", "#2b8cbe"),
        ("btc", f"BTC ({bucket_to_symbol['btc']})", "#f7931a"),
        ("eth", f"ETH ({bucket_to_symbol['eth']})", "#627eea"),
        ("silver", f"Silver ({bucket_to_symbol['silver']})", "#969696"),
        ("gold", f"Gold ({bucket_to_symbol['gold']})", "#e6ab02"),
    ]
    series_list = []
    labels = []
    colors = []
    for bkt, lbl, col in order:
        sym = bucket_to_symbol[bkt]
        series_list.append(curve_b[f"w_{sym}"].to_numpy(dtype=float) * 100.0)
        labels.append(lbl)
        colors.append(col)
    series_list.append(curve_b["w_CASH"].to_numpy(dtype=float) * 100.0)
    labels.append("EUR Cash")
    colors.append("#e8ecee")

    ax3.stackplot(curve_b.index, *series_list, labels=labels, colors=colors, alpha=0.88)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("v6b Weight %")
    ax3.legend(loc="upper left", ncol=6, fontsize=7.6, framealpha=0.92)
    ax3.grid(alpha=0.2)

    fig.text(
        0.99,
        0.005,
        "Weekly Fri EU close -> Mon EU open | SMA100+SMA50+VolCap(45%/85%) | 15% hysteresis | -15%/-20% daily catastrophe stop | T+2 settlement, EUR 1.25 + 0.10%/side, EUR 0/mo hosting",
        ha="right",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Strategy v6 Cross-Asset Momentum IS / Blind OOS / Full 5Y backtests.")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v6_cross_asset_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    mcfg = settings.metrics
    cache_root = settings.data.cache_path

    loader = CrossAssetEULoader(cache_root)
    uni_3x = loader.resolve_universe(
        DEFAULT_CROSS_ASSET_BUCKETS,
        warmup_start=WARMUP_START,
        eval_start=IS_START,
        eval_end=OOS_END,
        oos_start=OOS_START,
    )

    buckets_2x = (EQUITIES_2X_BUCKET,) + DEFAULT_CROSS_ASSET_BUCKETS[1:]
    uni_2x = loader.resolve_universe(
        buckets_2x,
        warmup_start=WARMUP_START,
        eval_start=IS_START,
        eval_end=OOS_END,
        oos_start=OOS_START,
    )

    # Load CSPX.AS and QQQ from existing cache (for benchmark curve & v5b read-only chart overlay)
    yf_prov = YFinanceProvider(cache_root)
    ref_bars = yf_prov.get_daily_bars(["CSPX.AS", "QQQ", "QQQ3.MI"], WARMUP_START, OOS_END)
    cspx_bars = ref_bars["CSPX.AS"]

    # Load saved v5 baseline numbers directly from outputs/v5_etf_results.json
    v5_json_path = ROOT / "outputs" / "v5_etf_results.json"
    v5_saved = json.loads(v5_json_path.read_text()) if v5_json_path.exists() else {}

    # LOCKED STRATEGY CONFIGURATIONS (zero OOS tuning)
    run_specs: list[tuple[str, CrossAssetConfig, Any]] = [
        (
            "v6a_top1",
            CrossAssetConfig(
                name="v6a_top1",
                top_n=1,
                vol_caps=uni_3x.vol_caps,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            uni_3x,
        ),
        (
            "v6b_top2",
            CrossAssetConfig(
                name="v6b_top2",
                top_n=2,
                vol_caps=uni_3x.vol_caps,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            uni_3x,
        ),
        # Optional 2x equities sensitivity (CL2.PA instead of QQQ3.MI)
        (
            "v6a_top1_cl2",
            CrossAssetConfig(
                name="v6a_top1_cl2",
                top_n=1,
                vol_caps=uni_2x.vol_caps,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            uni_2x,
        ),
        (
            "v6b_top2_cl2",
            CrossAssetConfig(
                name="v6b_top2_cl2",
                top_n=2,
                vol_caps=uni_2x.vol_caps,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            uni_2x,
        ),
        # Labeled modern post-split affordability sensitivity for CL2.PA (pre-split CL2.PA was > EUR 3,000/sh)
        (
            "v6a_top1_cl2_postsplit",
            CrossAssetConfig(
                name="v6a_top1_cl2_postsplit",
                top_n=1,
                vol_caps=uni_2x.vol_caps,
                sizing_mode=SizingMode.POST_SPLIT,
            ),
            uni_2x,
        ),
        (
            "v6b_top2_cl2_postsplit",
            CrossAssetConfig(
                name="v6b_top2_cl2_postsplit",
                top_n=2,
                vol_caps=uni_2x.vol_caps,
                sizing_mode=SizingMode.POST_SPLIT,
            ),
            uni_2x,
        ),
    ]

    windows = [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("FULL_5Y", IS_START, OOS_END),
    ]

    # Combine audits across 3x universe + CL2.PA sensitivity
    all_audits = {k: v.model_dump(mode="json") for k, v in uni_3x.audits.items()}
    all_audits["equities_2x"] = uni_2x.audits["equities_2x"].model_dump(mode="json")

    results: dict[str, Any] = {
        "windows": {w: {"start": str(s), "end": str(e)} for w, s, e in windows},
        "resolved_universe_3x": uni_3x.bucket_to_symbol,
        "resolved_universe_2x": uni_2x.bucket_to_symbol,
        "ticker_audits": all_audits,
        "locked_configs": {name: cfg.model_dump(mode="json") for name, cfg, _ in run_specs},
        "asset_buy_and_hold": {},
        "v5_baselines": {},
        "strategies": {},
    }

    # Individual underlying Buy & Hold references (theoretical fractional EUR open-to-close)
    bh_universe = {**uni_3x.bars, **uni_2x.bars, "CSPX.AS": cspx_bars}
    for w_name, w_start, w_end in windows:
        results["asset_buy_and_hold"][w_name] = {}
        for sym, bs in sorted(bh_universe.items()):
            results["asset_buy_and_hold"][w_name][sym] = compute_theoretical_bh_metrics(
                bs, w_start, w_end, initial_capital=1000.0, metrics_cfg=mcfg
            )
        if v5_saved:
            results["v5_baselines"][w_name] = {
                "CSPX.AS_theo": v5_saved["benchmarks_theoretical"][w_name]["CSPX.AS"],
                "v5b_QQQ3_cash_eur1000": v5_saved["strategies"]["v5b_QQQ3_cash"][w_name]["eur1000"]["metrics"],
                "v5b_QQQ3_cash_eur5000": v5_saved["strategies"]["v5b_QQQ3_cash"][w_name]["eur5000"]["metrics"],
                "v5a_CL2_cash_eur1000": v5_saved["strategies"]["v5a_CL2_cash"][w_name]["eur1000"]["metrics"],
                "v5a_CL2_cash_eur5000": v5_saved["strategies"]["v5a_CL2_cash"][w_name]["eur5000"]["metrics"],
            }

    full_5y_runs: dict[int, dict[str, CrossAssetBacktestResult]] = {}

    for spec_name, cfg, uni in run_specs:
        results["strategies"][spec_name] = {}
        for w_name, w_start, w_end in windows:
            results["strategies"][spec_name][w_name] = {}
            for cap in args.capitals_eur:
                cap_int = int(cap)
                cap_tag = f"eur{cap_int}"
                res = run_cross_asset_backtest(
                    uni.bars,
                    cfg,
                    start_date=w_start,
                    end_date=w_end,
                    initial_capital_eur=cap,
                    benchmark_bars=cspx_bars,
                )
                packed = _pack_v6_result(res, mcfg, uni.symbol_to_bucket)
                results["strategies"][spec_name][w_name][cap_tag] = packed
                if w_name == "FULL_5Y":
                    full_5y_runs.setdefault(cap_int, {})[spec_name] = res

                m = packed["metrics"]
                logging.info(
                    "[%-7s] %-24s %s | eq=EUR%8.2f ret=%+6.1f%% cagr=%+6.1f%% sr=%+5.2f dsr10=%.3f dsr30=%.3f dd=%6.1f%% ord=%3d(%.1f/y) stops=%2d fric=EUR%6.2f skip0=%d",
                    w_name,
                    spec_name,
                    cap_tag,
                    m["final_eur"],
                    m["total_return"] * 100,
                    m["cagr"] * 100,
                    m["sharpe"] if not math.isnan(m["sharpe"]) else 0.0,
                    m["deflated_sharpe"] if not math.isnan(m["deflated_sharpe"]) else 0.0,
                    m["deflated_sharpe_cum30"] if not math.isnan(m["deflated_sharpe_cum30"]) else 0.0,
                    m["max_drawdown"] * 100,
                    m["total_orders"],
                    m["orders_per_year"],
                    m["catastrophe_stops_triggered"],
                    m["total_friction_eur"],
                    m["skipped_zero_shares"],
                )

    # Generate Full-5Y PNG charts (EUR 1,000 and EUR 5,000) in BOTH outputs/ and Project store media/
    media_paths: dict[str, str] = {}
    v5b_cfg = UcitsRegimeConfig(
        name="v5b_QQQ3_cash",
        signal_symbol="QQQ",
        risk_on_symbol="QQQ3.MI",
        risk_off_mode=RiskOffMode.CASH,
        vol_circuit_breaker=0.35,
        sizing_mode=SizingMode.AS_TRADED,
    )

    for cap in args.capitals_eur:
        cap_int = int(cap)
        runs = full_5y_runs[cap_int]

        # Compute v5b QQQ3.MI curve read-only for chart overlay
        v5b_res = run_ucits_regime_backtest(
            signal_bars=ref_bars["QQQ"],
            risk_on_bars=ref_bars["QQQ3.MI"],
            cfg=v5b_cfg,
            start_date=IS_START,
            end_date=OOS_END,
            initial_capital_eur=cap,
            benchmark_bars=cspx_bars,
        )
        cspx_5y = cspx_bars.frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
        cspx_eq = cspx_5y["close"] / float(cspx_5y["open"].iloc[0]) * cap

        local_png = ROOT / "outputs" / f"v6_cross_asset_eur{cap_int}.png"
        store_png = STORE_ROOT / "media" / f"v6-cross-asset-eur{cap_int}.png"
        for target_path in (local_png, store_png):
            _plot_v6_png(
                capital_eur=cap,
                v6a=runs["v6a_top1"],
                v6b=runs["v6b_top2"],
                v6a_2x=runs["v6a_top1_cl2_postsplit"] if cap_int <= 1500 else runs["v6a_top1_cl2"],
                v6b_2x=runs["v6b_top2_cl2_postsplit"] if cap_int <= 1500 else runs["v6b_top2_cl2"],
                v5b_eq=v5b_res.equity_curve["equity"],
                cspx_eq=cspx_eq,
                bucket_to_symbol=uni_3x.bucket_to_symbol,
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
    logging.info("Strategy v6 backtest completed in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
