"""Performance and trade statistics. Pure functions of the equity curve / trade list."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MetricsConfig
from src.domain.models import ClosedTrade, SkippedEntry

EULER_GAMMA = 0.5772156649015329


def daily_returns(equity: pd.Series, initial: float) -> pd.Series:
    e = pd.concat([pd.Series([initial]), equity.reset_index(drop=True)], ignore_index=True)
    return e.pct_change().dropna()


def cagr(equity: pd.Series, initial: float) -> float:
    days = (equity.index[-1] - equity.index[0]).days + 1
    if days <= 0 or initial <= 0 or equity.iloc[-1] <= 0:
        return float("nan") if equity.iloc[-1] > 0 else -1.0
    return (equity.iloc[-1] / initial) ** (365.25 / days) - 1.0


def sharpe(r: pd.Series, periods: int, rf: float = 0.0) -> float:
    ex = r - rf / periods
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def sortino(r: pd.Series, periods: int, rf: float = 0.0) -> float:
    ex = r - rf / periods
    dd = math.sqrt(float((np.minimum(ex, 0.0) ** 2).mean()))
    return float(ex.mean() / dd * math.sqrt(periods)) if dd > 0 else float("nan")


def max_drawdown(equity: pd.Series, initial: float) -> float:
    e = pd.concat([pd.Series([initial]), equity.reset_index(drop=True)], ignore_index=True)
    return float((e / e.cummax() - 1.0).min())


def expected_max_sharpe(n_trials: int, trial_sr_std: float) -> float:
    """E[max SR] across ``n_trials`` independent zero-skill trials (Bailey & Lopez de Prado)."""
    if n_trials <= 1 or trial_sr_std <= 0:
        return 0.0
    z = stats.norm.ppf
    return trial_sr_std * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(r: pd.Series, periods: int, n_trials: int, trial_sr_std_annual: float) -> float:
    """Probability that the true Sharpe exceeds the best of ``n_trials`` lucky trials."""
    t = len(r)
    sd = r.std(ddof=1)
    if t < 3 or not sd > 0:
        return float("nan")
    sr = r.mean() / sd
    sr0 = expected_max_sharpe(n_trials, trial_sr_std_annual / math.sqrt(periods))
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)))


def trade_stats(trades: list[ClosedTrade]) -> dict[str, float]:
    if not trades:
        return {"trades": 0, "win_rate": float("nan"), "avg_win_r": float("nan"), "avg_loss_r": float("nan"),
                "expectancy_r": float("nan"), "profit_factor": float("nan"), "avg_bars_held": float("nan")}
    r = np.array([t.r_multiple for t in trades])
    pnl = np.array([t.net_pnl for t in trades])
    wins, losses = pnl > 0, pnl <= 0
    gross_win, gross_loss = pnl[wins].sum(), -pnl[losses].sum()
    return {
        "trades": len(trades),
        "win_rate": float(wins.mean()),
        "avg_win_r": float(r[wins].mean()) if wins.any() else float("nan"),
        "avg_loss_r": float(r[losses].mean()) if losses.any() else float("nan"),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
    }


def compute_metrics(
    equity_curve: pd.DataFrame,
    initial: float,
    trades: list[ClosedTrade],
    skipped: list[SkippedEntry],
    cfg: MetricsConfig,
) -> dict[str, float]:
    eq = equity_curve["equity"]
    r = daily_returns(eq, initial)
    bench = equity_curve["benchmark_close"]
    bench_eq = bench / bench.iloc[0] * initial
    out: dict[str, float] = {
        "initial_equity": initial,
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial - 1),
        "cagr": cagr(eq, initial),
        "sharpe": sharpe(r, cfg.periods_per_year, cfg.risk_free_rate),
        "sortino": sortino(r, cfg.periods_per_year, cfg.risk_free_rate),
        "deflated_sharpe": deflated_sharpe(r, cfg.periods_per_year, cfg.dsr_n_trials, cfg.dsr_trial_sharpe_std),
        "max_drawdown": max_drawdown(eq, initial),
        "exposure": float((equity_curve["open_positions"] > 0).mean()),
        "benchmark_cagr": cagr(bench_eq, initial),
        "skipped_gap": sum(1 for s in skipped if s.reason == "gap_above_limit"),
        "skipped_other": sum(1 for s in skipped if s.reason != "gap_above_limit"),
    }
    out.update(trade_stats(trades))
    return out
