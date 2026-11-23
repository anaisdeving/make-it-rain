"""Run the Phase 0 backtest.

    python scripts/run_backtest.py                       # EUR 1,000, auto provider chain
    python scripts/run_backtest.py --capital-eur 1000 5000 --vps both
    python scripts/run_backtest.py --provider sample     # offline, SYNTHETIC data
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import prepare_data, run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.backtest.reporting import export_all  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.factory import build_provider  # noqa: E402


def _fmt(m: dict) -> str:
    keys = ["final_equity", "total_return", "cagr", "sharpe", "sortino", "deflated_sharpe", "max_drawdown",
            "win_rate", "avg_win_r", "avg_loss_r", "expectancy_r", "profit_factor", "trades", "skipped_gap"]
    return "  ".join(f"{k}={m[k]:.3f}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", choices=["auto", "fmp_massive", "yfinance", "sample"], default=None)
    ap.add_argument("--capital-eur", type=float, nargs="+", default=None)
    ap.add_argument("--vps", choices=["on", "off", "both"], default=None)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--max-symbols", type=int, default=None, help="debug: cap universe size")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    settings = load_settings(args.config)
    if args.provider:
        settings = settings.model_copy(update={"data": settings.data.model_copy(update={"provider": args.provider})})

    t0 = time.time()
    provider, notes = build_provider(settings)
    for n in notes:
        logging.info("provider chain: %s", n)
    data = prepare_data(settings, provider, args.max_symbols)
    logging.info("data prepared in %.0fs: %s", time.time() - t0, json.dumps(data.data_stats))

    capitals = args.capital_eur or [settings.backtest.initial_capital_eur]
    vps_modes = {"on": [True], "off": [False], "both": [False, True], None: [settings.execution.vps_enabled]}[args.vps]
    out_dir = Path(args.output_dir)
    summary = {"provider_notes": notes, "provenance": data.provenance, "data_stats": data.data_stats, "runs": {}}
    synth = " [SYNTHETIC DATA - NOT REAL]" if data.is_synthetic else ""
    for cap in capitals:
        for vps in vps_modes:
            usd = cap * settings.backtest.eur_usd_rate
            res = run_backtest(data, settings, usd, vps)
            m = compute_metrics(res.equity_curve, usd, res.trades, res.skipped, settings.metrics)
            tag = f"eur{int(cap)}" + ("-vps" if vps else "")
            title = (f"PEAD + regime momentum | EUR {cap:,.0f} (${usd:,.0f}) | "
                     f"{settings.backtest.start_date} to {settings.backtest.end_date}"
                     f"{' | VPS cost on' if vps else ''} | data: {provider.name}{synth}")
            meta = {"capital_eur": cap, "capital_usd": usd, "vps": vps, "provider": provider.name,
                    "synthetic": data.is_synthetic}
            paths = export_all(res, m, meta, out_dir, tag, title)
            summary["runs"][tag] = {"metrics": m, "rejections": res.rejections,
                                    "sizing_rejections": res.sizing_rejections,
                                    "files": {k: str(v) for k, v in paths.items()}}
            logging.info("%s: %s", tag, _fmt(m))
            if not vps and cap == capitals[0]:
                shutil.copy(paths["trades"], out_dir / "trade_log.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logging.info("done in %.0fs -> %s", time.time() - t0, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
