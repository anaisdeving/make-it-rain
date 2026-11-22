"""Strategy v7 Backtest & Validation: VIX Term-Structure Tail-Capture & Leveraged Equity Rotation.

Strict Out-of-Sample Protocol (ALL rules locked and unit-tested BEFORE running Blind OOS once):
- Warmup:     2020-10-18 to 2021-11-21
- In-Sample:  2021-11-22 to 2024-11-21 (3.00 years)
- Blind OOS:  2024-11-22 to 2026-11-20 (2.00 years — evaluated ONCE, zero OOS tuning)
- Full 5Y:    2021-11-22 to 2026-11-20 (5.00 years)

Executes native EUR simulations on an IBKR Ireland EUR cash account:
- EUR 1,000 and EUR 5,000 starting capital (zero FX fees, EUR 0.00/mo hosting)
- EUR 1.25/order commission + 0.10%/side slippage, whole as-traded shares only
- Strict T+2 settlement (zero Good Faith Violations)
- Strict US-close (22:00 CET) -> next-EU-open (us_date < eu_date) alignment

Outputs:
- Structured JSON: outputs/v7_vix_results.json
- Charts (IS/OOS split + drawdowns + VIX/VIX3M term structure & regime shading)
  in BOTH outputs/ and Project store media/
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
import pandas as pd  # noqa: E402

from src.config import MetricsConfig, load_settings  # noqa: E402
from src.data.vix_term_loader import VixTermLoader  # noqa: E402
from src.strategy.ucits_etf_regime import (  # noqa: E402
    RiskOffMode,
    SizingMode,
    UcitsRegimeConfig,
    compute_theoretical_bh_metrics,
    run_ucits_regime_backtest,
)
from src.strategy.vix_term_regime import (  # noqa: E402
    VixRegimeState,
    VixTermBacktestResult,
    VixTermRegimeConfig,
    compute_sleeve_attribution,
    compute_vix_regime_metrics,
    run_vix_term_backtest,
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


def _regime_spans(curve: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return contiguous spans of RISK_ON / NEUTRAL / TAIL_HEDGE for chart shading."""
    if curve.empty:
        return []
    labels = [(ts, str(row["target_state"])) for ts, row in curve.iterrows()]
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


def _pack_v7_result(res: VixTermBacktestResult, mcfg: MetricsConfig) -> dict[str, Any]:
    m = compute_vix_regime_metrics(res, mcfg)
    attr = compute_sleeve_attribution(res)
    attr_out = {k: v.model_dump(mode="json") for k, v in attr.items()}

    fills_log = [
        {
            "date": str(f.date),
            "signal_us_date": str(f.signal_us_date),
            "symbol": f.symbol,
            "role": f.role,
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
    all_trips = list(res.round_trips) + ([res.open_position_at_end] if res.open_position_at_end else [])
    trips_log = [
        {
            "symbol": t.symbol,
            "role": t.role,
            "signal_entry_us_date": str(t.signal_entry_us_date),
            "entry_date": str(t.entry_date),
            "exit_date": str(t.exit_date),
            "raw_shares_entry": t.raw_shares_entry,
            "bars_held": t.bars_held,
            "gross_pnl_eur": round(t.gross_pnl_eur, 2),
            "commissions_eur": round(t.commissions_eur, 2),
            "slippage_eur": round(t.slippage_eur, 2),
            "net_pnl_eur": round(t.net_pnl_eur, 2),
            "return_pct": round(t.return_pct * 100.0, 2),
            "exit_reason": t.exit_reason,
            "is_open_at_end": t.is_open_at_end,
        }
        for t in all_trips
    ]
    episodes_log = [
        {
            "episode_id": ep.episode_id,
            "prior_state": ep.prior_state,
            "signal_start_us_date": str(ep.signal_start_us_date),
            "signal_end_us_date": str(ep.signal_end_us_date) if ep.signal_end_us_date else None,
            "first_actionable_eu_date": str(ep.first_actionable_eu_date),
            "entry_date": str(ep.entry_date) if ep.entry_date else None,
            "exit_date": str(ep.exit_date) if ep.exit_date else None,
            "t2_delay_sessions": ep.t2_delay_sessions,
            "missed_due_to_t2": ep.missed_due_to_t2,
            "entry_vix_ratio": round(ep.entry_vix_ratio, 4),
            "peak_vix_ratio": round(ep.peak_vix_ratio, 4),
            "exit_vix_ratio": round(ep.exit_vix_ratio, 4) if ep.exit_vix_ratio is not None else None,
            "lvo_open_on_signal_day": round(ep.lvo_open_on_signal_day, 4)
            if ep.lvo_open_on_signal_day is not None
            else None,
            "lvo_open_on_fill_day": round(ep.lvo_open_on_fill_day, 4)
            if ep.lvo_open_on_fill_day is not None
            else None,
            "lvo_move_during_t2_pct": round(ep.lvo_move_during_t2_pct * 100.0, 2)
            if ep.lvo_move_during_t2_pct is not None
            else None,
            "raw_shares_entry": ep.raw_shares_entry,
            "bars_held": ep.bars_held,
            "net_pnl_eur": round(ep.net_pnl_eur, 2),
            "return_pct": round(ep.return_pct * 100.0, 2) if ep.return_pct is not None else None,
            "exit_reason": ep.exit_reason,
            "is_open_at_end": ep.is_open_at_end,
        }
        for ep in res.tail_hedge_episodes
    ]
    return {
        "metrics": m,
        "attribution": attr_out,
        "fills": fills_log,
        "round_trips": trips_log,
        "tail_hedge_episodes": episodes_log,
    }


def _plot_v7_png(
    *,
    capital_eur: float,
    tail_sym: str,
    v7a_as: VixTermBacktestResult,
    v7a_ps: VixTermBacktestResult,
    v7b: VixTermBacktestResult,
    v7c: VixTermBacktestResult,
    v7b_cash: VixTermBacktestResult,
    v5b_eq: pd.Series,
    cspx_eq: pd.Series,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Generate 3-panel EUR chart: Equity Curves, Drawdowns, and VIX/VIX3M Term Structure."""
    init = float(capital_eur)
    is_1k = capital_eur <= 1500.0

    c_v7b = v7b.equity_curve
    eq_v7b = c_v7b["equity"]
    eq_v7c = v7c.equity_curve["equity"]
    eq_v7b_cash = v7b_cash.equity_curve["equity"]
    eq_v7a_as = v7a_as.equity_curve["equity"]
    eq_v7a_ps = v7a_ps.equity_curve["equity"]

    def _dd(s: pd.Series) -> pd.Series:
        return (s / s.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14.2, 10.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25, 1.15]},
    )

    # Background regime shading across all 3 panels (red=TAIL_HEDGE, beige=NEUTRAL)
    for a, b, lbl in _regime_spans(c_v7b):
        if lbl == VixRegimeState.TAIL_HEDGE.value:
            for ax in (ax1, ax2, ax3):
                ax.axvspan(a, b, color="#e69f9f", alpha=0.42, lw=0)
        elif lbl == VixRegimeState.NEUTRAL.value:
            for ax in (ax1, ax2, ax3):
                ax.axvspan(a, b, color="#ece7d5", alpha=0.35, lw=0)

    # Panel 1: Equity Curves
    ax1.plot(
        eq_v7b.index,
        eq_v7b.values,
        label=f"v7b 3x (QQQ3.MI + 100% {tail_sym})     | EUR {eq_v7b.iloc[-1]:,.0f} ({(eq_v7b.iloc[-1]/init-1)*100:+.1f}%)",
        color="#1b7837",
        lw=2.2,
    )
    ax1.plot(
        eq_v7c.index,
        eq_v7c.values,
        label=f"v7c 3x Half-Hedge (50% {tail_sym}/Cash) | EUR {eq_v7c.iloc[-1]:,.0f} ({(eq_v7c.iloc[-1]/init-1)*100:+.1f}%)",
        color="#005a9c",
        lw=2.0,
    )
    ax1.plot(
        eq_v7b_cash.index,
        eq_v7b_cash.values,
        label=f"v7b Cash-Tail Ablation (0% {tail_sym})  | EUR {eq_v7b_cash.iloc[-1]:,.0f} ({(eq_v7b_cash.iloc[-1]/init-1)*100:+.1f}%)",
        color="#66bd63",
        lw=1.5,
        ls="--",
    )
    ax1.plot(
        eq_v7a_as.index,
        eq_v7a_as.values,
        label=f"v7a 2x (CL2.PA + 100% {tail_sym}) [as-tr]| EUR {eq_v7a_as.iloc[-1]:,.0f} ({(eq_v7a_as.iloc[-1]/init-1)*100:+.1f}%)",
        color="#7570b3",
        lw=1.9,
    )
    if is_1k:
        ax1.plot(
            eq_v7a_ps.index,
            eq_v7a_ps.values,
            label=f"v7a 2x (CL2.PA + 100% {tail_sym}) [post] | EUR {eq_v7a_ps.iloc[-1]:,.0f} ({(eq_v7a_ps.iloc[-1]/init-1)*100:+.1f}%)",
            color="#9e9ac8",
            lw=1.4,
            ls="-.",
        )
    ax1.plot(
        v5b_eq.index,
        v5b_eq.values,
        label=f"v5b QQQ3.MI (3x) Baseline          | EUR {v5b_eq.iloc[-1]:,.0f} ({(v5b_eq.iloc[-1]/init-1)*100:+.1f}%)",
        color="#d95f02",
        lw=1.7,
        ls="--",
    )
    ax1.plot(
        cspx_eq.index,
        cspx_eq.values,
        label=f"CSPX.AS (1x S&P500 EUR) B&H        | EUR {cspx_eq.iloc[-1]:,.0f} ({(cspx_eq.iloc[-1]/init-1)*100:+.1f}%)",
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
        f"Strategy v7: VIX Term-Structure Tail-Capture & Leveraged Equity Rotation ({tail_sym}) — EUR {int(init):,} (IBKR Ireland)",
        fontsize=11.0,
    )
    ax1.legend(loc="upper left", fontsize=7.9, framealpha=0.92)
    ax1.grid(alpha=0.3)

    # Panel 2: Drawdowns
    ax2.plot(eq_v7b.index, _dd(eq_v7b).values, color="#1b7837", lw=1.5, label="v7b 3x DD%")
    ax2.plot(eq_v7c.index, _dd(eq_v7c).values, color="#005a9c", lw=1.4, label="v7c Half-Hedge DD%")
    ax2.plot(eq_v7a_as.index, _dd(eq_v7a_as).values, color="#7570b3", lw=1.3, label="v7a 2x DD%")
    ax2.plot(v5b_eq.index, _dd(v5b_eq).values, color="#d95f02", lw=1.2, ls="--", label="v5b QQQ3.MI DD%")
    ax2.plot(cspx_eq.index, _dd(cspx_eq).values, color="#333333", lw=1.1, ls=":", label="CSPX.AS B&H DD%")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", ncol=5, fontsize=7.8, framealpha=0.90)
    ax2.grid(alpha=0.3)

    # Panel 3: VIX / VIX3M Ratio + Hysteresis Thresholds
    ax3.plot(c_v7b.index, c_v7b["vix_ratio"].values, color="#222222", lw=1.25, label="VIX / VIX3M Ratio (prior US close)")
    ax3.axhline(1.03, color="#b2182b", lw=1.1, ls="--", label="Tail Enter (>1.03)")
    ax3.axhline(1.00, color="#d95f02", lw=0.9, ls=":", label="Risk-On Exit (>1.00)")
    ax3.axhline(0.97, color="#7570b3", lw=0.9, ls="-.", label="Tail Exit (<0.97)")
    ax3.axhline(0.95, color="#1b7837", lw=1.0, ls="--", label="Risk-On Enter (<0.95)")
    ax3.set_ylabel("VIX / VIX3M")
    ax3.legend(loc="upper left", ncol=5, fontsize=7.6, framealpha=0.92)
    ax3.grid(alpha=0.3)

    fig.text(
        0.99,
        0.005,
        f"shading: coral=TAIL_HEDGE ({tail_sym}), beige=NEUTRAL (EUR Cash), white=RISK_ON | crimson dashed: IS/OOS (2024-11-22) | us_date<eu_date, T+2 settlement, EUR 1.25 + 0.10%/side",
        ha="right",
        fontsize=7.4,
        color="#555555",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Strategy v7 VIX Term-Structure IS / Blind OOS / Full 5Y backtests.")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v7_vix_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    mcfg = settings.metrics
    cache_root = settings.data.cache_path

    loader = VixTermLoader(cache_root)
    ds = loader.load_dataset(
        warmup_start=WARMUP_START,
        eval_start=IS_START,
        eval_end=OOS_END,
        oos_start=OOS_START,
    )
    tail_sym = ds.tail_hedge_symbol

    # Also cross-check VOOL.DE, VIXL.MI, VIXL.DE integrity for transparent audit reporting
    extra_integrity_audit: dict[str, Any] = {}
    for sym in ("VOOL.DE", "VIXL.MI", "VIXL.DE"):
        s_chk, meta_chk, rej = loader.validate_eu_candidate(
            sym, warmup_start=WARMUP_START, eval_start=IS_START, eval_end=OOS_END
        )
        entry: dict[str, Any] = {
            "currency": meta_chk.currency,
            "exchange": meta_chk.exchange,
            "name": meta_chk.name,
            "passed": rej is None,
            "rejection_reason": rej,
        }
        if s_chk is not None:
            r_main = ds.tail_hedge.frame["close"].pct_change()
            r_chk = s_chk.frame["close"].pct_change()
            common = r_main.index.intersection(r_chk.index)
            entry["corr_with_resolved"] = round(float(r_main.loc[common].corr(r_chk.loc[common])), 5)
            entry["rows_5y"] = int(len(s_chk.frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]))
        extra_integrity_audit[sym] = entry

    # Load saved v5 baseline numbers directly from outputs/v5_etf_results.json
    v5_json_path = ROOT / "outputs" / "v5_etf_results.json"
    v5_saved = json.loads(v5_json_path.read_text()) if v5_json_path.exists() else {}

    # LOCKED STRATEGY CONFIGURATIONS (zero OOS tuning)
    run_specs: list[tuple[str, VixTermRegimeConfig, Any, Any]] = [
        (
            "v7a_2x",
            VixTermRegimeConfig(
                name="v7a_2x",
                risk_on_symbol="CL2.PA",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=1.0,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            ds.equity_2x,
            ds.tail_hedge,
        ),
        (
            "v7a_2x_postsplit",
            VixTermRegimeConfig(
                name="v7a_2x_postsplit",
                risk_on_symbol="CL2.PA",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=1.0,
                sizing_mode=SizingMode.POST_SPLIT,
            ),
            ds.equity_2x,
            ds.tail_hedge,
        ),
        (
            "v7b_3x",
            VixTermRegimeConfig(
                name="v7b_3x",
                risk_on_symbol="QQQ3.MI",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=1.0,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            ds.equity_3x,
            ds.tail_hedge,
        ),
        (
            "v7c_3x_half_hedge",
            VixTermRegimeConfig(
                name="v7c_3x_half_hedge",
                risk_on_symbol="QQQ3.MI",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=0.50,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            ds.equity_3x,
            ds.tail_hedge,
        ),
        # Diagnostic ablations (0% tail hedge -> EUR Cash during TAIL_HEDGE) to isolate LVO alpha
        (
            "v7b_cash_tail",
            VixTermRegimeConfig(
                name="v7b_cash_tail",
                risk_on_symbol="QQQ3.MI",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=0.0,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            ds.equity_3x,
            None,
        ),
        (
            "v7a_cash_tail",
            VixTermRegimeConfig(
                name="v7a_cash_tail",
                risk_on_symbol="CL2.PA",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=0.0,
                sizing_mode=SizingMode.AS_TRADED,
            ),
            ds.equity_2x,
            None,
        ),
        (
            "v7a_cash_tail_postsplit",
            VixTermRegimeConfig(
                name="v7a_cash_tail_postsplit",
                risk_on_symbol="CL2.PA",
                tail_hedge_symbol=tail_sym,
                tail_hedge_weight=0.0,
                sizing_mode=SizingMode.POST_SPLIT,
            ),
            ds.equity_2x,
            None,
        ),
    ]

    windows = [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("FULL_5Y", IS_START, OOS_END),
    ]

    results: dict[str, Any] = {
        "windows": {w: {"start": str(s), "end": str(e)} for w, s, e in windows},
        "resolved_tail_hedge_symbol": tail_sym,
        "ticker_audits": {k: v.model_dump(mode="json") for k, v in ds.audits.items()},
        "extra_candidate_integrity_audit": extra_integrity_audit,
        "locked_configs": {name: cfg.model_dump(mode="json") for name, cfg, _, _ in run_specs},
        "asset_buy_and_hold": {},
        "v5_baselines": {},
        "strategies": {},
        "us_regime_summary": {},
    }

    bh_map = {
        "CSPX.AS": ds.benchmark,
        "CL2.PA": ds.equity_2x,
        "QQQ3.MI": ds.equity_3x,
        tail_sym: ds.tail_hedge,
    }
    for w_name, w_start, w_end in windows:
        results["asset_buy_and_hold"][w_name] = {}
        for sym, bs in bh_map.items():
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

    full_5y_runs: dict[int, dict[str, VixTermBacktestResult]] = {}

    for spec_name, cfg, ro_bars, th_bars in run_specs:
        results["strategies"][spec_name] = {}
        for w_name, w_start, w_end in windows:
            results["strategies"][spec_name][w_name] = {}
            for cap in args.capitals_eur:
                cap_int = int(cap)
                cap_tag = f"eur{cap_int}"
                res = run_vix_term_backtest(
                    qqq_bars=ds.qqq,
                    vix_bars=ds.vix,
                    vix3m_bars=ds.vix3m,
                    risk_on_bars=ro_bars,
                    cfg=cfg,
                    start_date=w_start,
                    end_date=w_end,
                    initial_capital_eur=cap,
                    tail_hedge_bars=th_bars,
                    benchmark_bars=ds.benchmark,
                )
                packed = _pack_v7_result(res, mcfg)
                results["strategies"][spec_name][w_name][cap_tag] = packed
                if w_name == "FULL_5Y":
                    full_5y_runs.setdefault(cap_int, {})[spec_name] = res

                m = packed["metrics"]
                ro_pnl = packed["attribution"]["RISK_ON"]["net_pnl_eur"]
                th_pnl = packed["attribution"]["TAIL_HEDGE"]["net_pnl_eur"]
                logging.info(
                    "[%-7s] %-24s %s | eq=EUR%8.2f ret=%+6.1f%% cagr=%+6.1f%% sr=%+5.2f dsr10=%.3f dsr35=%.3f dd=%6.1f%% ord=%2d(%.1f/y) t2w=%2d(th=%2d) | RO=%+8.2f TH=%+8.2f",
                    w_name,
                    spec_name,
                    cap_tag,
                    m["final_eur"],
                    m["total_return"] * 100,
                    m["cagr"] * 100,
                    m["sharpe"] if not math.isnan(m["sharpe"]) else 0.0,
                    m["deflated_sharpe"] if not math.isnan(m["deflated_sharpe"]) else 0.0,
                    m["deflated_sharpe_cum35"] if not math.isnan(m["deflated_sharpe_cum35"]) else 0.0,
                    m["max_drawdown"] * 100,
                    m["total_orders"],
                    m["orders_per_year"],
                    m["settlement_wait_days"],
                    m["tail_hedge_t2_wait_days"],
                    ro_pnl,
                    th_pnl,
                )

    # Record US regime distribution & transitions across IS, OOS, Full 5Y
    sig_df = full_5y_runs[int(args.capitals_eur[0])]["v7b_3x"].us_signals
    for w_name, w_start, w_end in windows:
        sub = sig_df.loc[pd.Timestamp(w_start) : pd.Timestamp(w_end)]
        flips = sub[sub["switched"]]
        counts = sub["state"].value_counts().to_dict()
        results["us_regime_summary"][w_name] = {
            "us_sessions": int(len(sub)),
            "state_days": {k: int(v) for k, v in counts.items()},
            "state_pct": {k: round(float(v) / len(sub) * 100.0, 2) for k, v in counts.items()},
            "vix_ratio_min": round(float(sub["vix_ratio"].min()), 4),
            "vix_ratio_median": round(float(sub["vix_ratio"].median()), 4),
            "vix_ratio_max": round(float(sub["vix_ratio"].max()), 4),
            "days_vix_ratio_gt_103": int((sub["vix_ratio"] > 1.03).sum()),
            "total_switches": int(len(flips)),
            "events": [
                {
                    "us_date": str(ts.date()),
                    "new_state": str(r["state"]),
                    "qqq_close": round(float(r["qqq_close"]), 2),
                    "qqq_sma200": round(float(r["qqq_sma200"]), 2),
                    "vix_close": round(float(r["vix_close"]), 2),
                    "vix3m_close": round(float(r["vix3m_close"]), 2),
                    "vix_ratio": round(float(r["vix_ratio"]), 4),
                }
                for ts, r in flips.iterrows()
            ],
        }

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

        v5b_res = run_ucits_regime_backtest(
            signal_bars=ds.qqq,
            risk_on_bars=ds.equity_3x,
            cfg=v5b_cfg,
            start_date=IS_START,
            end_date=OOS_END,
            initial_capital_eur=cap,
            benchmark_bars=ds.benchmark,
        )
        cspx_5y = ds.benchmark.frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
        cspx_eq = cspx_5y["close"] / float(cspx_5y["open"].iloc[0]) * cap

        local_png = ROOT / "outputs" / f"v7_vix_eur{cap_int}.png"
        store_png = STORE_ROOT / "media" / f"v7-vix-eur{cap_int}.png"
        for target_path in (local_png, store_png):
            _plot_v7_png(
                capital_eur=cap,
                tail_sym=tail_sym,
                v7a_as=runs["v7a_2x"],
                v7a_ps=runs["v7a_2x_postsplit"],
                v7b=runs["v7b_3x"],
                v7c=runs["v7c_3x_half_hedge"],
                v7b_cash=runs["v7b_cash_tail"],
                v5b_eq=v5b_res.equity_curve["equity"],
                cspx_eq=cspx_eq,
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
    logging.info("Strategy v7 backtest completed in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
