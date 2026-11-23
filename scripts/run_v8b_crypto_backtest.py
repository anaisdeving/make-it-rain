"""Strategy v8b-clean Backtest & Validation: Bias-Corrected EUR Spot Crypto Trend Breakout.

Strict Out-of-Sample Protocol (ALL rules locked and 151 unit tests passed BEFORE running Blind OOS once):
- Warmup:     2021-06-01 to 2021-11-21 (>= 120 daily crypto bars before IS start)
- In-Sample:  2021-11-22 to 2024-11-21 (3.00 years)
- Blind OOS:  2024-11-22 to 2026-11-20 (2.00 years — evaluated ONCE, zero OOS tuning)
- Full 5Y:    2021-11-22 to 2026-11-20 (5.00 years)

Executes native EUR spot simulations (Bitvavo Tier 1 primary: 0.25% taker + 0.10% slippage,
fractional EUR sizing min EUR 10, T+0 instant settlement, EUR 0/mo hosting; plus Kraken Pro
Tier 1 fee sensitivity: 0.80% taker + 0.10% slippage) at EUR 1,000 and EUR 5,000.

Benchmarks:
- BTC-EUR Buy & Hold (365-day annualization)
- 50/50 BTC-EUR + ETH-EUR Buy & Hold (365-day annualization)
- Saved v5b QQQ3.MI (outputs/v5_etf_results.json, 252-day EU exchange calendar)
- Saved CSPX.AS B&H (outputs/v5_etf_results.json, 252-day EU exchange calendar)

Outputs:
- Structured JSON: outputs/v8b_crypto_results.json
- Charts (EUR 1,000 and EUR 5,000) in BOTH outputs/ and Project store media/
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
from src.data.cache import DiskCache  # noqa: E402
from src.data.crypto_eur_loader import (  # noqa: E402
    UNIVERSE_MAJORS,
    UNIVERSE_NOV2021_TOP10,
    CryptoEurLoader,
)
from src.domain.models import BarSeries  # noqa: E402
from src.strategy.crypto_eur_breakout import (  # noqa: E402
    CRYPTO_PERIODS_PER_YEAR,
    CryptoBacktestResult,
    CryptoBreakoutConfig,
    compute_coin_attribution,
    compute_crypto_bh_benchmark,
    compute_crypto_metrics,
    run_crypto_breakout_backtest,
)
from src.strategy.ucits_etf_regime import (  # noqa: E402
    RiskOffMode,
    SizingMode,
    UcitsRegimeConfig,
    run_ucits_regime_backtest,
)

WARMUP_START = date(2021, 6, 1)
IS_START = date(2021, 11, 22)
IS_END = date(2024, 11, 21)
OOS_START = date(2024, 11, 22)
OOS_END = date(2026, 11, 20)

STORE_ROOT = Path(
    "/home/margarida/.local/state/cursor/agent-stores/cursor_agent_stores/"
    "bc-de418035-892d-40a4-a21a-bd834b742bee/files"
)


def _btc_bear_spans(curve: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous spans where BTC-EUR <= SMA(100) for chart shading."""
    if curve.empty:
        return []
    bear = ~curve["btc_bull"].astype(bool)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_span = False
    start_ts: pd.Timestamp | None = None
    prev_ts: pd.Timestamp | None = None
    for ts, is_bear in bear.items():
        if is_bear and not in_span:
            start_ts = ts
            in_span = True
        elif not is_bear and in_span and start_ts is not None and prev_ts is not None:
            spans.append((start_ts, prev_ts))
            in_span = False
            start_ts = None
        prev_ts = ts
    if in_span and start_ts is not None and prev_ts is not None:
        spans.append((start_ts, prev_ts))
    return spans


def _pack_v8b_result(res: CryptoBacktestResult, mcfg: MetricsConfig) -> dict[str, Any]:
    m = compute_crypto_metrics(
        res,
        mcfg,
        periods_per_year=CRYPTO_PERIODS_PER_YEAR,
        cumulative_trials_n=40,
        cumulative_trials_conservative_n=60,
    )
    attr = compute_coin_attribution(res)
    attr_out = {sym: a.model_dump(mode="json") for sym, a in attr.items()}

    all_trips = list(res.round_trips) + list(res.open_positions_at_end)
    trips_log = [
        {
            "symbol": t.symbol,
            "signal_entry_date": str(t.signal_entry_date),
            "entry_date": str(t.entry_date),
            "exit_date": str(t.exit_date),
            "qty": t.qty,
            "entry_raw_price": t.entry_raw_price,
            "entry_fill_price": t.entry_fill_price,
            "entry_atr14": t.entry_atr14,
            "hard_stop_price": t.hard_stop_price,
            "exit_raw_price": t.exit_raw_price,
            "exit_fill_price": t.exit_fill_price,
            "bars_held": t.bars_held,
            "cost_basis_eur": round(t.cost_basis_eur, 2),
            "gross_pnl_eur": round(t.gross_pnl_eur, 2),
            "fees_eur": round(t.fees_eur, 2),
            "slippage_eur": round(t.slippage_eur, 2),
            "net_pnl_eur": round(t.net_pnl_eur, 2),
            "return_pct": round(t.return_pct * 100.0, 2),
            "exit_reason": t.exit_reason,
            "exit_category": t.exit_category,
            "is_open_at_end": t.is_open_at_end,
        }
        for t in all_trips
    ]
    return {
        "metrics": m,
        "coin_attribution": attr_out,
        "round_trips": trips_log,
        "num_fills": len(res.fills),
    }


def _load_cached_etf_bar(cache: DiskCache, symbol: str) -> BarSeries | None:
    key = f"{symbol}_2020-10-18_2026-11-20"
    if cache.status("bars", key) == "ok":
        df = cache.read("bars", key)
        if df is not None and not df.empty:
            return BarSeries(symbol=symbol, frame=df)
    return None


def _plot_v8b_png(
    *,
    capital_eur: float,
    majors: CryptoBacktestResult,
    top10_s2: CryptoBacktestResult,
    top10_s3: CryptoBacktestResult,
    btc_bh_eq: pd.Series,
    btc_eth_bh_eq: pd.Series,
    v5b_eq: pd.Series | None,
    cspx_eq: pd.Series | None,
    oos_split: pd.Timestamp,
    path: Path,
) -> None:
    """Generate 3-panel EUR chart: Equity Curves, Drawdowns, BTC Macro Regime & Exposure."""
    init = float(capital_eur)

    c_maj = majors.equity_curve
    eq_maj = c_maj["equity"]
    eq_s2 = top10_s2.equity_curve["equity"]
    eq_s3 = top10_s3.equity_curve["equity"]

    def _dd(s: pd.Series) -> pd.Series:
        return (s / s.cummax().clip(lower=init) - 1.0) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(14.4, 10.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25, 1.2]},
    )

    # Background shading where BTC-EUR <= SMA(100) (macro risk-off)
    for a, b in _btc_bear_spans(c_maj):
        for ax in (ax1, ax2, ax3):
            ax.axvspan(a, b, color="#ece7d5", alpha=0.45, lw=0)

    # Panel 1: Equity Curves
    ax1.plot(
        eq_maj.index,
        eq_maj.values,
        label=f"v8b_majors (BTC+ETH, 2 slots)       | EUR {eq_maj.iloc[-1]:,.0f} ({(eq_maj.iloc[-1]/init-1)*100:+.1f}%)",
        color="#1b7837",
        lw=2.2,
    )
    ax1.plot(
        eq_s2.index,
        eq_s2.values,
        label=f"v8b_top10_slots2 (Nov21 Top10, 2sl) | EUR {eq_s2.iloc[-1]:,.0f} ({(eq_s2.iloc[-1]/init-1)*100:+.1f}%)",
        color="#005a9c",
        lw=2.1,
    )
    ax1.plot(
        eq_s3.index,
        eq_s3.values,
        label=f"v8b_top10_slots3 (Nov21 Top10, 3sl) | EUR {eq_s3.iloc[-1]:,.0f} ({(eq_s3.iloc[-1]/init-1)*100:+.1f}%)",
        color="#7570b3",
        lw=1.9,
    )
    ax1.plot(
        btc_bh_eq.index,
        btc_bh_eq.values,
        label=f"BTC-EUR Buy & Hold (0-fee)          | EUR {btc_bh_eq.iloc[-1]:,.0f} ({(btc_bh_eq.iloc[-1]/init-1)*100:+.1f}%)",
        color="#e6ab02",
        lw=1.6,
        ls="--",
    )
    ax1.plot(
        btc_eth_bh_eq.index,
        btc_eth_bh_eq.values,
        label=f"50/50 BTC+ETH EUR B&H (0-fee)       | EUR {btc_eth_bh_eq.iloc[-1]:,.0f} ({(btc_eth_bh_eq.iloc[-1]/init-1)*100:+.1f}%)",
        color="#a6761d",
        lw=1.4,
        ls="-.",
    )
    if v5b_eq is not None:
        ax1.plot(
            v5b_eq.index,
            v5b_eq.values,
            label=f"v5b QQQ3.MI (3x ETF Baseline)       | EUR {v5b_eq.iloc[-1]:,.0f} ({(v5b_eq.iloc[-1]/init-1)*100:+.1f}%)",
            color="#d95f02",
            lw=1.7,
            ls="--",
        )
    if cspx_eq is not None:
        ax1.plot(
            cspx_eq.index,
            cspx_eq.values,
            label=f"CSPX.AS (1x S&P500 EUR B&H)         | EUR {cspx_eq.iloc[-1]:,.0f} ({(cspx_eq.iloc[-1]/init-1)*100:+.1f}%)",
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
        f"Strategy v8b-clean: EUR Spot Crypto Trend Breakout (Bitvavo Tier 1: 0.25% + 0.10%/side) — EUR {int(init):,}",
        fontsize=11.0,
    )
    ax1.legend(loc="upper left", fontsize=7.8, framealpha=0.92)
    ax1.grid(alpha=0.3)

    # Panel 2: Drawdowns
    ax2.plot(eq_maj.index, _dd(eq_maj).values, color="#1b7837", lw=1.5, label="v8b_majors DD%")
    ax2.plot(eq_s2.index, _dd(eq_s2).values, color="#005a9c", lw=1.4, label="v8b_top10_slots2 DD%")
    ax2.plot(eq_s3.index, _dd(eq_s3).values, color="#7570b3", lw=1.3, label="v8b_top10_slots3 DD%")
    ax2.plot(btc_bh_eq.index, _dd(btc_bh_eq).values, color="#e6ab02", lw=1.2, ls="--", label="BTC-EUR B&H DD%")
    if v5b_eq is not None:
        ax2.plot(v5b_eq.index, _dd(v5b_eq).values, color="#d95f02", lw=1.1, ls=":", label="v5b QQQ3.MI DD%")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", ncol=5, fontsize=7.8, framealpha=0.90)
    ax2.grid(alpha=0.3)

    # Panel 3: BTC-EUR vs SMA(100) Macro Regime + Deployed Capital %
    ax3.plot(c_maj.index, c_maj["btc_close"].values / 1000.0, color="#222222", lw=1.2, label="BTC-EUR (k EUR)")
    ax3.plot(c_maj.index, c_maj["btc_sma100"].values / 1000.0, color="#b2182b", lw=1.1, ls="--", label="BTC-EUR SMA(100)")
    ax3.set_ylabel("BTC-EUR (k EUR)")
    ax3_r = ax3.twinx()
    ax3_r.fill_between(
        top10_s2.equity_curve.index,
        0.0,
        top10_s2.equity_curve["deployed_weight"].values * 100.0,
        color="#005a9c",
        alpha=0.18,
        label="v8b_top10_slots2 Deployed %",
    )
    ax3_r.set_ylabel("Deployed %")
    ax3_r.set_ylim(0.0, 115.0)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=3, fontsize=7.6, framealpha=0.92)
    ax3.grid(alpha=0.3)

    fig.text(
        0.99,
        0.005,
        "beige shading: BTC-EUR <= SMA(100) macro gate | crimson dashed: IS/Blind OOS split (2024-11-22) | Bitvavo 0.25% taker + 0.10% slip, T+0, EUR 10 min, 365-day crypto calendar",
        ha="right",
        fontsize=7.3,
        color="#555555",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Strategy v8b-clean EUR Spot Crypto IS / Blind OOS / Full 5Y backtests.")
    ap.add_argument("--capitals-eur", type=float, nargs="+", default=[1000.0, 5000.0])
    ap.add_argument("--output-json", default=str(ROOT / "outputs" / "v8b_crypto_results.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    t0 = time.time()
    settings = load_settings()
    mcfg = settings.metrics
    cache_root = settings.data.cache_path

    loader = CryptoEurLoader(cache_root)
    ds = loader.load_dataset(
        symbols=UNIVERSE_NOV2021_TOP10,
        warmup_start=WARMUP_START,
        eval_start=IS_START,
        is_end=IS_END,
        oos_start=OOS_START,
        eval_end=OOS_END,
        min_warmup_bars=120,
        min_eval_bars=1750,
        inspect_lunc=True,
    )

    usable_top10 = tuple(sym for sym in UNIVERSE_NOV2021_TOP10 if sym in ds.bars)
    usable_majors = tuple(sym for sym in UNIVERSE_MAJORS if sym in ds.bars)
    dropped_coins = [sym for sym in UNIVERSE_NOV2021_TOP10 if sym not in ds.bars]
    if dropped_coins:
        logging.warning("Explicitly logged unusable coins (excluded from simulation): %s", dropped_coins)

    # LOCKED STRATEGY CONFIGURATIONS (zero OOS tuning)
    primary_configs: list[CryptoBreakoutConfig] = [
        CryptoBreakoutConfig(
            name="v8b_majors",
            universe=usable_majors,
            max_slots=2,
            taker_fee_pct=0.0025,
            slippage_pct=0.0010,
            fee_tier_label="Bitvavo EUR Tier 1 (0.25% taker + 0.10% slip)",
        ),
        CryptoBreakoutConfig(
            name="v8b_top10_slots2",
            universe=usable_top10,
            max_slots=2,
            taker_fee_pct=0.0025,
            slippage_pct=0.0010,
            fee_tier_label="Bitvavo EUR Tier 1 (0.25% taker + 0.10% slip)",
        ),
        CryptoBreakoutConfig(
            name="v8b_top10_slots3",
            universe=usable_top10,
            max_slots=3,
            taker_fee_pct=0.0025,
            slippage_pct=0.0010,
            fee_tier_label="Bitvavo EUR Tier 1 (0.25% taker + 0.10% slip)",
        ),
    ]

    # Kraken Pro Tier 1 post-July 2026 fee sensitivity (0.80% taker + 0.10% slippage)
    kraken_configs: list[CryptoBreakoutConfig] = [
        cfg.model_copy(
            update={
                "name": f"{cfg.name}_kraken",
                "taker_fee_pct": 0.0080,
                "slippage_pct": 0.0010,
                "fee_tier_label": "Kraken Pro Tier 1 post-July 2026 (0.80% taker + 0.10% slip)",
            }
        )
        for cfg in primary_configs
    ]

    windows = [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("FULL_5Y", IS_START, OOS_END),
    ]

    # Load saved v5 baselines (QQQ3.MI & CSPX.AS) verbatim from outputs/v5_etf_results.json
    v5_json_path = ROOT / "outputs" / "v5_etf_results.json"
    v5_saved = json.loads(v5_json_path.read_text()) if v5_json_path.exists() else {}

    results: dict[str, Any] = {
        "windows": {w: {"start": str(s), "end": str(e)} for w, s, e in windows},
        "warmup_start": str(WARMUP_START),
        "crypto_periods_per_year": CRYPTO_PERIODS_PER_YEAR,
        "eurusd_audit": ds.eurusd_audit,
        "coin_audits": {sym: audit.model_dump(mode="json") for sym, audit in ds.audits.items()},
        "usable_majors": list(usable_majors),
        "usable_top10": list(usable_top10),
        "unusable_coins": dropped_coins,
        "lunc_inspections": [x.model_dump(mode="json") for x in ds.lunc_inspections],
        "lunc_selected_source": ds.lunc_selected_source,
        "lunc_audit": ds.lunc_audit.model_dump(mode="json") if ds.lunc_audit else None,
        "locked_configs": {cfg.name: cfg.model_dump(mode="json") for cfg in primary_configs},
        "benchmarks_crypto": {},
        "coin_buy_and_hold": {},
        "v5_baselines": {},
        "strategies_bitvavo": {},
        "strategies_kraken_sensitivity": {},
        "lunc_stress_test": {},
    }

    # 1. Crypto benchmarks (BTC-EUR B&H, 50/50 BTC+ETH B&H) + per-coin B&H across IS / OOS / FULL_5Y
    bh_curves_5y: dict[int, dict[str, pd.Series]] = {}
    for w_name, w_start, w_end in windows:
        results["benchmarks_crypto"][w_name] = {}
        results["coin_buy_and_hold"][w_name] = {}

        for cap in args.capitals_eur:
            cap_int = int(cap)
            cap_tag = f"eur{cap_int}"
            btc_eq, btc_m = compute_crypto_bh_benchmark(
                ds.bars, {"BTC-EUR": 1.0}, w_start, w_end, initial_capital_eur=cap, metrics_cfg=mcfg
            )
            mix_eq, mix_m = compute_crypto_bh_benchmark(
                ds.bars,
                {"BTC-EUR": 0.5, "ETH-EUR": 0.5},
                w_start,
                w_end,
                initial_capital_eur=cap,
                metrics_cfg=mcfg,
            )
            results["benchmarks_crypto"][w_name][f"BTC_EUR_BH_{cap_tag}"] = btc_m
            results["benchmarks_crypto"][w_name][f"BTC_ETH_5050_BH_{cap_tag}"] = mix_m
            if w_name == "FULL_5Y":
                bh_curves_5y[cap_int] = {"BTC-EUR": btc_eq, "BTC_ETH_5050": mix_eq}

        for sym in usable_top10:
            _, cm = compute_crypto_bh_benchmark(
                ds.bars, {sym: 1.0}, w_start, w_end, initial_capital_eur=1000.0, metrics_cfg=mcfg
            )
            results["coin_buy_and_hold"][w_name][sym] = cm

        if v5_saved:
            v5_block: dict[str, Any] = {
                "CSPX.AS_theo": v5_saved["benchmarks_theoretical"][w_name]["CSPX.AS"],
                "v5b_QQQ3_cash_eur1000": v5_saved["strategies"]["v5b_QQQ3_cash"][w_name]["eur1000"]["metrics"],
                "v5b_QQQ3_cash_eur5000": v5_saved["strategies"]["v5b_QQQ3_cash"][w_name]["eur5000"]["metrics"],
            }
            for k, blk in v5_block.items():
                cg = float(blk["cagr"])
                mdd = float(blk["max_drawdown"])
                blk_copy = dict(blk)
                blk_copy["calmar"] = (cg / abs(mdd)) if mdd < -1e-9 else float("nan")
                v5_block[k] = blk_copy
            results["v5_baselines"][w_name] = v5_block

    # 2. Primary Bitvavo EUR Tier 1 runs (all 3 variants x IS / Blind OOS / Full 5Y x EUR 1,000 / EUR 5,000)
    full_5y_runs: dict[int, dict[str, CryptoBacktestResult]] = {}
    for cfg in primary_configs:
        results["strategies_bitvavo"][cfg.name] = {}
        for w_name, w_start, w_end in windows:
            results["strategies_bitvavo"][cfg.name][w_name] = {}
            for cap in args.capitals_eur:
                cap_int = int(cap)
                cap_tag = f"eur{cap_int}"
                res = run_crypto_breakout_backtest(
                    ds.bars,
                    cfg,
                    start_date=w_start,
                    end_date=w_end,
                    initial_capital_eur=cap,
                )
                packed = _pack_v8b_result(res, mcfg)
                results["strategies_bitvavo"][cfg.name][w_name][cap_tag] = packed
                if w_name == "FULL_5Y":
                    full_5y_runs.setdefault(cap_int, {})[cfg.name] = res

                m = packed["metrics"]
                logging.info(
                    "[%-7s] %-18s %s | eq=EUR%9.2f ret=%+6.1f%% cagr=%+6.1f%% sr=%+5.2f dsr(10/40/60)=%.3f/%.3f/%.3f dd=%6.1f%% cal=%.2f wr=%4.1f%% pf=%4.2f ord=%3d(%.1f/y) fric=EUR%.1f hs=%d sig=%d",
                    w_name,
                    cfg.name,
                    cap_tag,
                    m["final_eur"],
                    m["total_return"] * 100,
                    m["cagr"] * 100,
                    m["sharpe"] if m["sharpe"] is not None and not math.isnan(m["sharpe"]) else 0.0,
                    m["deflated_sharpe_n10"] or 0.0,
                    m["deflated_sharpe_cum40"] or 0.0,
                    m["deflated_sharpe_cum60"] or 0.0,
                    m["max_drawdown"] * 100,
                    m["calmar"] if m["calmar"] is not None and not math.isnan(m["calmar"]) else 0.0,
                    (m["win_rate"] * 100) if m["win_rate"] is not None and not math.isnan(m["win_rate"]) else 0.0,
                    m["profit_factor"] if (m["profit_factor"] is not None and math.isfinite(m["profit_factor"])) else 0.0,
                    m["total_orders"],
                    m["orders_per_year"],
                    m["total_friction_eur"],
                    m["hard_stop_exits"],
                    m["signal_exits"],
                )

    # 3. Kraken Pro Tier 1 Fee Sensitivity (0.80% taker + 0.10% slip) for Blind OOS and Full 5Y
    for kcfg in kraken_configs:
        results["strategies_kraken_sensitivity"][kcfg.name] = {}
        for w_name, w_start, w_end in (("OOS", OOS_START, OOS_END), ("FULL_5Y", IS_START, OOS_END)):
            results["strategies_kraken_sensitivity"][kcfg.name][w_name] = {}
            for cap in args.capitals_eur:
                cap_int = int(cap)
                cap_tag = f"eur{cap_int}"
                res = run_crypto_breakout_backtest(
                    ds.bars,
                    kcfg,
                    start_date=w_start,
                    end_date=w_end,
                    initial_capital_eur=cap,
                )
                packed = _pack_v8b_result(res, mcfg)
                results["strategies_kraken_sensitivity"][kcfg.name][w_name][cap_tag] = packed
                m = packed["metrics"]
                logging.info(
                    "[KRAKEN %-7s] %-24s %s | eq=EUR%9.2f ret=%+6.1f%% cagr=%+6.1f%% sr=%+5.2f dsr40=%.3f dd=%6.1f%% fric=EUR%.1f",
                    w_name,
                    kcfg.name,
                    cap_tag,
                    m["final_eur"],
                    m["total_return"] * 100,
                    m["cagr"] * 100,
                    m["sharpe"] if m["sharpe"] is not None and not math.isnan(m["sharpe"]) else 0.0,
                    m["deflated_sharpe_cum40"] or 0.0,
                    m["max_drawdown"] * 100,
                    m["total_friction_eur"],
                )

    # 4. Terra Classic (LUNC) Stress Test (if usable pre-May-2022 data exists on Yahoo Finance)
    if ds.lunc_bars is not None:
        bars_with_lunc = dict(ds.bars)
        bars_with_lunc["LUNC-EUR"] = ds.lunc_bars
        top10_plus_lunc = usable_top10 + ("LUNC-EUR",)
        lunc_cfgs = [
            CryptoBreakoutConfig(
                name="v8b_top10_slots2_plus_lunc",
                universe=top10_plus_lunc,
                max_slots=2,
                taker_fee_pct=0.0025,
                slippage_pct=0.0010,
            ),
            CryptoBreakoutConfig(
                name="v8b_top10_slots3_plus_lunc",
                universe=top10_plus_lunc,
                max_slots=3,
                taker_fee_pct=0.0025,
                slippage_pct=0.0010,
            ),
        ]
        results["lunc_stress_test"]["status"] = "executed"
        results["lunc_stress_test"]["source"] = ds.lunc_selected_source
        results["lunc_stress_test"]["variants"] = {}
        for lcfg in lunc_cfgs:
            results["lunc_stress_test"]["variants"][lcfg.name] = {}
            for w_name, w_start, w_end in windows:
                results["lunc_stress_test"]["variants"][lcfg.name][w_name] = {}
                for cap in args.capitals_eur:
                    cap_int = int(cap)
                    cap_tag = f"eur{cap_int}"
                    res = run_crypto_breakout_backtest(
                        bars_with_lunc,
                        lcfg,
                        start_date=w_start,
                        end_date=w_end,
                        initial_capital_eur=cap,
                    )
                    packed = _pack_v8b_result(res, mcfg)
                    results["lunc_stress_test"]["variants"][lcfg.name][w_name][cap_tag] = packed
                    m = packed["metrics"]
                    lunc_attr = packed["coin_attribution"].get("LUNC-EUR", {})
                    logging.info(
                        "[LUNC %-7s] %-26s %s | eq=EUR%9.2f ret=%+6.1f%% dd=%6.1f%% | LUNC trades=%s hs=%s pnl=EUR%+.2f",
                        w_name,
                        lcfg.name,
                        cap_tag,
                        m["final_eur"],
                        m["total_return"] * 100,
                        m["max_drawdown"] * 100,
                        lunc_attr.get("evaluated_trades", 0),
                        lunc_attr.get("hard_stop_exits", 0),
                        lunc_attr.get("net_pnl_eur", 0.0),
                    )
    else:
        results["lunc_stress_test"]["status"] = "unavailable_or_corrupted"
        results["lunc_stress_test"]["reason"] = (
            "Yahoo Finance lacks usable pre-May-2022 Terra Classic history across "
            "LUNC-EUR, LUNC-USD, and LUNA1-USD; see lunc_inspections for exact details."
        )
        logging.warning("LUNC stress test skipped cleanly: %s", results["lunc_stress_test"]["reason"])

    # 5. Generate Full-5Y PNG charts (EUR 1,000 and EUR 5,000) in BOTH outputs/ and Project store media/
    yf_cache = DiskCache(cache_root, "yfinance")
    qqq_bars = _load_cached_etf_bar(yf_cache, "QQQ")
    qqq3_bars = _load_cached_etf_bar(yf_cache, "QQQ3.MI")
    cspx_bars = _load_cached_etf_bar(yf_cache, "CSPX.AS")
    v5b_cfg = UcitsRegimeConfig(
        name="v5b_QQQ3_cash",
        signal_symbol="QQQ",
        risk_on_symbol="QQQ3.MI",
        risk_off_mode=RiskOffMode.CASH,
        vol_circuit_breaker=0.35,
        sizing_mode=SizingMode.AS_TRADED,
    )

    media_paths: dict[str, str] = {}
    for cap in args.capitals_eur:
        cap_int = int(cap)
        runs = full_5y_runs[cap_int]

        v5b_eq: pd.Series | None = None
        cspx_eq: pd.Series | None = None
        if qqq_bars is not None and qqq3_bars is not None and cspx_bars is not None:
            v5b_res = run_ucits_regime_backtest(
                signal_bars=qqq_bars,
                risk_on_bars=qqq3_bars,
                cfg=v5b_cfg,
                start_date=IS_START,
                end_date=OOS_END,
                initial_capital_eur=cap,
                benchmark_bars=cspx_bars,
            )
            v5b_eq = v5b_res.equity_curve["equity"]
            cspx_5y = cspx_bars.frame.loc[pd.Timestamp(IS_START) : pd.Timestamp(OOS_END)]
            cspx_eq = cspx_5y["close"] / float(cspx_5y["open"].iloc[0]) * cap

        local_png = ROOT / "outputs" / f"v8b_crypto_eur{cap_int}.png"
        store_png = STORE_ROOT / "media" / f"v8b-crypto-eur{cap_int}.png"
        for target_path in (local_png, store_png):
            _plot_v8b_png(
                capital_eur=cap,
                majors=runs["v8b_majors"],
                top10_s2=runs["v8b_top10_slots2"],
                top10_s3=runs["v8b_top10_slots3"],
                btc_bh_eq=bh_curves_5y[cap_int]["BTC-EUR"],
                btc_eth_bh_eq=bh_curves_5y[cap_int]["BTC_ETH_5050"],
                v5b_eq=v5b_eq,
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
    logging.info("Strategy v8b-clean backtest completed in %.1fs -> %s", time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
