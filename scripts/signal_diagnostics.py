"""Event study of every signal the screener emits (independent of capital and slots).

For each signal at close t: next-open fill vs the +0.3% limit cap, and forward returns
from the t+1 open to the close of t+k, raw and in excess of SPY over the same window.
Uses the same cached data and screener as the backtest; reads no data beyond each window.

    python scripts/signal_diagnostics.py --output outputs/signal-diagnostics.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtest.engine import prepare_data, snapshot  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.data.factory import build_provider  # noqa: E402
from src.strategy.pead_screener import screen  # noqa: E402

HORIZONS = (5, 20, 45)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "signal-diagnostics.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = load_settings()
    if args.provider:
        s = s.model_copy(update={"data": s.data.model_copy(update={"provider": args.provider})})
    provider, _ = build_provider(s)
    data = prepare_data(s, provider)
    spy = data.benchmark.frame
    cap = s.execution.entry_limit_cap_pct
    rows = []
    for ts, events in data.events_by_session.items():
        cands = [(e, snapshot(e.symbol, ts, data.features[e.symbol].loc[ts])) for e in events]
        sigs, _ = screen(cands, data.regimes[ts], s.screener)
        for sig in sigs:
            f = data.features[sig.symbol]
            i = f.index.get_loc(ts)
            if i + 1 >= len(f):
                continue
            o = float(f["open"].iloc[i + 1])
            j = spy.index.get_loc(ts)
            row = {"symbol": sig.symbol, "signal_date": ts.date(), "regime": sig.regime.value,
                   "timing": sig.event.timing.value, "event_day_return": sig.event_day_return,
                   "next_open_gap": o / sig.signal_close - 1, "gap_skipped": o > sig.signal_close * (1 + cap)}
            for k in HORIZONS:
                if i + k < len(f) and j + k < len(spy):
                    r = float(f["close"].iloc[i + k]) / o - 1
                    b = float(spy["close"].iloc[j + k]) / float(spy["open"].iloc[j + 1]) - 1
                    row[f"ret_{k}d"], row[f"excess_{k}d"] = r, r - b
            rows.append(row)
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    cols = [c for c in df.columns if c.startswith(("ret_", "excess_"))]
    print(f"signals: {len(df)}  gap-skipped share: {df.gap_skipped.mean():.1%}")
    print(df.groupby("gap_skipped")[cols].agg(["mean", "median"]).round(4).T.to_string())
    print(df.groupby("regime")[cols].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
