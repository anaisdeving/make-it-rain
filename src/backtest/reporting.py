"""Exports: PNG + HTML equity curves, trade log CSV, metrics JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.backtest.engine import BacktestResult  # noqa: E402

REGIME_COLORS = {"NORMAL": None, "CAUTION": "#f5d38a", "PANIC": "#f2a0a0"}


def trades_frame(result: BacktestResult) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.model_dump(mode="json") for t in result.trades])
    return df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def _regime_spans(curve: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    spans, start, cur = [], curve.index[0], curve["regime"].iloc[0]
    for ts, reg in curve["regime"].items():
        if reg != cur:
            spans.append((start, ts, cur))
            start, cur = ts, reg
    spans.append((start, curve.index[-1], cur))
    return spans


def plot_png(result: BacktestResult, title: str, path: Path) -> None:
    curve = result.equity_curve
    eq = curve["equity"]
    bench = curve["benchmark_close"] / curve["benchmark_close"].iloc[0] * result.initial_capital_usd
    dd = eq / eq.cummax().clip(lower=result.initial_capital_usd) - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for a, b, reg in _regime_spans(curve):
        if REGIME_COLORS[reg]:
            for ax in (ax1, ax2):
                ax.axvspan(a, b, color=REGIME_COLORS[reg], alpha=0.35, lw=0)
    ax1.plot(eq.index, eq.values, label="Strategy equity (USD)", color="#1f4e79", lw=1.6)
    ax1.plot(bench.index, bench.values, label="SPY buy & hold (same capital)", color="#888888", lw=1.0, ls="--")
    ax1.axhline(result.initial_capital_usd, color="black", lw=0.6, alpha=0.5)
    ax1.set_ylabel("USD")
    ax1.set_title(title, fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#b03030", alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.text(0.99, 0.005, "shading: yellow=CAUTION, red=PANIC regime", ha="right", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_html(result: BacktestResult, title: str, path: Path) -> None:
    curve = result.equity_curve
    eq = curve["equity"]
    bench = curve["benchmark_close"] / curve["benchmark_close"].iloc[0] * result.initial_capital_usd
    dd = (eq / eq.cummax().clip(lower=result.initial_capital_usd) - 1) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy equity (USD)", line=dict(color="#1f4e79")), 1, 1)
    fig.add_trace(go.Scatter(x=bench.index, y=bench, name="SPY buy & hold", line=dict(color="#888", dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown %", fill="tozeroy", line=dict(color="#b03030")), 2, 1)
    tdf = trades_frame(result)
    if not tdf.empty:
        ent = pd.to_datetime(tdf["entry_date"])
        fig.add_trace(go.Scatter(
            x=ent, y=eq.reindex(ent, method="ffill"), mode="markers", name="Entries",
            marker=dict(symbol="triangle-up", size=8, color="#2a9d8f"),
            text=[f"{s} R={r:.2f} ({x})" for s, r, x in zip(tdf["symbol"], tdf["r_multiple"], tdf["exit_reason"])],
        ), 1, 1)
    for a, b, reg in _regime_spans(curve):
        if REGIME_COLORS[reg]:
            fig.add_vrect(x0=a, x1=b, fillcolor=REGIME_COLORS[reg], opacity=0.3, line_width=0)
    fig.update_layout(title=title, template="plotly_white", height=650, hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")


def export_all(result: BacktestResult, metrics: dict, meta: dict, out_dir: Path, tag: str, title: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": out_dir / f"equity-curve-{tag}.png",
        "html": out_dir / f"equity-curve-{tag}.html",
        "trades": out_dir / f"trade-log-{tag}.csv",
        "skipped": out_dir / f"skipped-entries-{tag}.csv",
        "equity": out_dir / f"equity-{tag}.csv",
        "metrics": out_dir / f"metrics-{tag}.json",
    }
    plot_png(result, title, paths["png"])
    plot_html(result, title, paths["html"])
    trades_frame(result).to_csv(paths["trades"], index=False)
    pd.DataFrame([s.model_dump(mode="json") for s in result.skipped]).to_csv(paths["skipped"], index=False)
    result.equity_curve.to_csv(paths["equity"])
    payload = {"metrics": metrics, "meta": meta, "rejections": result.rejections,
               "sizing_rejections": result.sizing_rejections, "fees_paid_usd": result.fees_paid}
    paths["metrics"].write_text(json.dumps(payload, indent=2, default=str))
    return paths
