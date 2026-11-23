"""Strategy v6: Cross-Asset Regime-Filtered Momentum Rotation (EU PRIIPs ETPs/ETCs).

Designed for an IBKR Ireland EUR cash account (zero FX conversion fees, EUR 0/mo
cron/local hosting, whole shares only, strict T+2 European ETF/ETP settlement).

Locked Strategy Rules (zero OOS tuning):
- **Weekly Rebalance (last EU trading day of each ISO week -> next EU trading day Open):**
  Detected strictly causally at session ``e`` Open via
  ``prev_eu_ts.isocalendar()[:2] != e.isocalendar()[:2]``, where ``prev_eu_ts`` is the
  immediately preceding session on the aligned EU calendar. Handles Friday closes,
  Good Friday (Thursday close -> Tuesday open), and all cross-exchange holidays with
  exact truncation invariance and zero look-ahead bias.
- **Eligibility Filter (evaluated at day ``t`` EU Close on split-adjusted series):**
  1. ``Close > SMA(100)``
  2. ``Close > SMA(50)``
  3. 20-day annualised log-return realised volatility ``sigma_20 <= vol_cap``
     (``0.45`` for equities & metals, ``0.85`` for crypto ETPs), with ``sigma_20 > 0``.
- **Composite Risk-Adjusted Momentum Score:**
  ``score = (0.5 * Return_20d + 0.5 * Return_60d) / sigma_20``.
- **Hysteresis Buffer (15%):**
  Keep an incumbent eligible asset unless a challenger satisfies
  ``challenger_score > held_score + abs(held_score) * 0.15`` (exact ``1.15 * held_score``
  for positive scores, ``> 0`` for zero, and ``> 0.85 * held_score`` for negative scores so
  non-positive scores never invert the comparison) OR the incumbent fails eligibility.
- **Intra-Week Catastrophe Stop (evaluated DAILY at EU Close ``t`` -> ``t+1`` EU Open):**
  Exit any held position to EUR Cash at next trading day ``t+1`` Open if it closes
  ``>= 15%`` below split-adjusted entry fill price OR ``>= 20%`` below its peak
  split-adjusted close since entry. The freed slot stays in EUR Cash until the next
  weekly rebalance to prevent intra-week whipsaw.
- **Variants:**
  - ``v6a_top1``: 100% allocated to the #1 eligible asset (100% EUR Cash if 0 qualify).
  - ``v6b_top2``: 50% / 50% equal-slot target across Top 2 eligible assets (50% Cash if
    only 1 qualifies; 100% Cash if 0 qualify). Uses no-churn incumbent retention:
    when an incumbent remains in the target set, its shares are left untouched (no
    weight-drift trimming/rebalancing), avoiding unnecessary EUR 2.50 commissions,
    0.20% slippage, and T+2 settlement delays.
- **IBKR Europe Execution & T+2 Settlement:**
  EUR 1.25/order commission, 0.10%/side slippage, whole as-traded shares only,
  strict T+2 settlement (sale proceeds settle on ``day_idx + 2`` before buying incoming
  assets; already-settled EUR Cash deploys immediately at ``t+1`` Open; zero GFVs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.backtest.metrics import (
    cagr,
    daily_returns,
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
)
from src.config import MetricsConfig
from src.domain.models import BarSeries
from src.strategy.ucits_etf_regime import SizingMode


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CrossAssetConfig(_Frozen):
    """Frozen configuration for Strategy v6 (``v6a_top1`` / ``v6b_top2``)."""

    name: str = "v6a_top1"
    top_n: int = Field(1, ge=1, le=5)
    vol_caps: dict[str, float] = Field(default_factory=dict)
    default_vol_cap: float = Field(0.45, gt=0.0)
    sma_fast: int = Field(50, ge=2)
    sma_slow: int = Field(100, ge=2)
    ret_short: int = Field(20, ge=1)
    ret_long: int = Field(60, ge=1)
    vol_window: int = Field(20, ge=2)
    hysteresis_pct: float = Field(0.15, ge=0.0, lt=1.0)
    catastrophe_entry_dd: float = Field(0.15, gt=0.0, lt=1.0)
    catastrophe_peak_dd: float = Field(0.20, gt=0.0, lt=1.0)
    commission_eur: float = Field(1.25, ge=0.0)
    slippage_pct: float = Field(0.001, ge=0.0, lt=0.1)
    settlement_days: int = Field(2, ge=0, le=5)
    monthly_hosting_eur: float = Field(0.0, ge=0.0)
    sizing_mode: SizingMode = SizingMode.AS_TRADED

    @model_validator(mode="after")
    def _validate_windows(self) -> "CrossAssetConfig":
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be strictly smaller than sma_slow")
        if self.ret_short >= self.ret_long:
            raise ValueError("ret_short must be strictly smaller than ret_long")
        for sym, cap in self.vol_caps.items():
            if cap <= 0:
                raise ValueError(f"vol_cap for {sym} must be positive")
        return self

    def vol_cap_for(self, symbol: str) -> float:
        return float(self.vol_caps.get(symbol, self.default_vol_cap))


class CrossAssetOrderFill(_Frozen):
    """One executed EU Open fill (buy or sell) in EUR."""

    date: date
    signal_date: date
    symbol: str
    side: Literal["BUY", "SELL"]
    raw_shares: int = Field(gt=0)
    adj_shares: float = Field(gt=0)
    split_factor_after: float = Field(gt=0)
    raw_fill_price: float = Field(gt=0)
    adj_fill_price: float = Field(gt=0)
    notional_eur: float = Field(gt=0)
    commission_eur: float = Field(ge=0)
    slippage_eur: float = Field(ge=0)
    reason: str


class CrossAssetRoundTrip(_Frozen):
    """Completed (or end-of-window open MTM) position lifecycle in EUR."""

    symbol: str
    signal_entry_date: date
    entry_date: date
    exit_date: date
    raw_shares_entry: int = Field(gt=0)
    adj_shares: float = Field(gt=0)
    entry_price_adj: float = Field(gt=0)
    exit_price_adj: float = Field(gt=0)
    peak_close_adj: float = Field(gt=0)
    bars_held: int = Field(ge=0)
    gross_pnl_eur: float
    commissions_eur: float
    slippage_eur: float
    net_pnl_eur: float
    return_pct: float
    exit_reason: str
    is_open_at_end: bool = False


class AssetAttribution(_Frozen):
    """Per-asset performance, holding-time, trade, and catastrophe-stop attribution."""

    symbol: str
    days_held: int
    pct_days_held: float
    avg_weight_pct: float
    net_pnl_eur: float
    return_contribution_pct: float
    closed_trades: int
    open_at_end: int
    evaluated_trades: int
    win_rate: float | None
    catastrophe_stops: int
    avg_bars_held: float | None
    commissions_eur: float
    slippage_eur: float


@dataclass
class _OpenPosition:
    symbol: str
    signal_entry_date: date
    entry_date: date
    raw_shares_entry: int
    adj_shares: float
    entry_price_adj: float
    peak_close_adj: float
    entry_notional_eur: float
    entry_commission_eur: float
    entry_slippage_eur: float
    bars_held: int = 0
    pending_stop_exit: bool = False
    pending_stop_reason: str = ""
    pending_stop_signal_date: date | None = None


@dataclass(frozen=True)
class CrossAssetBacktestResult:
    """Full output of a Strategy v6 EUR cash-account simulation."""

    config: CrossAssetConfig
    symbols: tuple[str, ...]
    initial_capital_eur: float
    start_date: date
    end_date: date
    equity_curve: pd.DataFrame
    weekly_signals: pd.DataFrame
    fills: list[CrossAssetOrderFill]
    round_trips: list[CrossAssetRoundTrip]
    open_positions_at_end: list[CrossAssetRoundTrip]
    total_commissions_eur: float
    total_slippage_eur: float
    total_buy_notional_eur: float
    total_sell_notional_eur: float
    hosting_fees_eur: float
    skipped_zero_shares: int
    settlement_wait_days: int
    min_settled_cash_eur: float
    catastrophe_stops_triggered: int
    rebalance_events: int


# ---------------------------------------------------------------------------
# Pure Signal Math, Hysteresis, & Target Selection
# ---------------------------------------------------------------------------


def compute_asset_indicators(
    bars: BarSeries,
    cfg: CrossAssetConfig,
    *,
    vol_cap: float | None = None,
) -> pd.DataFrame:
    """Compute strictly causal daily indicators on one asset's native EU calendar.

    Uses split-adjusted ``close`` throughout so stock splits never distort SMAs,
    momentum returns, or realised volatility.
    """
    close = bars.frame["close"].astype(float)
    sma_fast = close.rolling(cfg.sma_fast, min_periods=cfg.sma_fast).mean()
    sma_slow = close.rolling(cfg.sma_slow, min_periods=cfg.sma_slow).mean()
    ret_short = close / close.shift(cfg.ret_short) - 1.0
    ret_long = close / close.shift(cfg.ret_long) - 1.0
    sigma_20 = (
        np.log(close)
        .diff()
        .rolling(cfg.vol_window, min_periods=cfg.vol_window)
        .std()
        * math.sqrt(252)
    )

    cap = cfg.vol_cap_for(bars.symbol) if vol_cap is None else float(vol_cap)
    raw_mom = 0.5 * ret_short + 0.5 * ret_long
    valid_sigma = sigma_20.notna() & np.isfinite(sigma_20) & (sigma_20 > 0.0)
    score = pd.Series(np.nan, index=close.index, dtype=float)
    score.loc[valid_sigma] = raw_mom.loc[valid_sigma] / sigma_20.loc[valid_sigma]

    eligible = (
        close.notna()
        & sma_slow.notna()
        & sma_fast.notna()
        & valid_sigma
        & score.notna()
        & np.isfinite(score)
        & (close > sma_slow)
        & (close > sma_fast)
        & (sigma_20 <= cap)
    )

    return pd.DataFrame(
        {
            "close": close,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "ret_short": ret_short,
            "ret_long": ret_long,
            "sigma_20": sigma_20,
            "score": score,
            "eligible": eligible.astype(bool),
            "vol_cap": cap,
        },
        index=close.index,
    )


def challenger_beats_held(
    challenger_score: float,
    held_score: float,
    hysteresis_pct: float = 0.15,
) -> bool:
    """Return True iff ``challenger_score`` clears the hysteresis hurdle over ``held_score``.

    Unified non-inverting threshold across all signs of ``held_score``:
    ``threshold = held_score + abs(held_score) * hysteresis_pct``.
    - ``held_score > 0``: requires ``challenger_score > (1 + hysteresis_pct) * held_score``.
    - ``held_score == 0``: requires ``challenger_score > 0``.
    - ``held_score < 0``: requires ``challenger_score > (1 - hysteresis_pct) * held_score``
      (strictly higher than ``held_score`` by ``hysteresis_pct * |held_score|``, never
      inverting to allow a worse score to win).
    """
    if not (math.isfinite(challenger_score) and math.isfinite(held_score)):
        return False
    threshold = held_score + abs(held_score) * hysteresis_pct
    return challenger_score > threshold


def select_target_assets(
    eligible_scores: dict[str, float],
    held_symbols: set[str],
    *,
    top_n: int,
    hysteresis_pct: float = 0.15,
    blocked_symbols: set[str] | None = None,
) -> list[str]:
    """Select up to ``top_n`` target assets with hysteresis retention for incumbents.

    Deterministic tie-breaking: descending score, then ascending symbol name.
    ``blocked_symbols`` (e.g. an asset that triggered an intra-week catastrophe stop on
    the same Friday close) are excluded from selection.
    """
    blocked = blocked_symbols or set()
    clean_eligible = {
        sym: float(sc)
        for sym, sc in eligible_scores.items()
        if sym not in blocked and math.isfinite(sc)
    }
    if not clean_eligible:
        return []

    def _rank_key(sym: str) -> tuple[float, str]:
        return (-clean_eligible[sym], sym)

    # 1. Retain eligible incumbents (up to top_n, strongest first if somehow > top_n)
    incumbents = sorted([s for s in held_symbols if s in clean_eligible], key=_rank_key)[:top_n]
    challengers = sorted([s for s in clean_eligible if s not in incumbents], key=_rank_key)

    selected_incumbents = list(incumbents)
    selected_new: list[str] = []

    # 2. Fill any vacant slots directly with highest-ranked challengers (no hysteresis)
    while len(selected_incumbents) + len(selected_new) < top_n and challengers:
        selected_new.append(challengers.pop(0))

    # 3. Remaining challengers must beat the weakest remaining incumbent by hysteresis_pct
    while challengers and selected_incumbents:
        best_chal = challengers[0]
        weakest_inc = max(selected_incumbents, key=lambda s: (-clean_eligible[s], s))
        if challenger_beats_held(clean_eligible[best_chal], clean_eligible[weakest_inc], hysteresis_pct):
            selected_incumbents.remove(weakest_inc)
            selected_new.append(challengers.pop(0))
        else:
            break

    final = selected_incumbents + selected_new
    return sorted(final, key=_rank_key)


# ---------------------------------------------------------------------------
# Calendar, Split, & Benchmark Helpers
# ---------------------------------------------------------------------------


def _split_factor_after(row: pd.Series, sizing_mode: SizingMode) -> float:
    if sizing_mode is SizingMode.POST_SPLIT:
        return 1.0
    c = float(row["close"])
    rc = float(row["raw_close"])
    if c <= 0 or math.isnan(rc) or rc <= 0:
        return 1.0
    return rc / c


def _build_union_calendar(
    bars_by_symbol: dict[str, BarSeries],
    end_date: date,
) -> pd.DatetimeIndex:
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol must be non-empty")
    idx: pd.DatetimeIndex | None = None
    end_ts = pd.Timestamp(end_date)
    for bs in bars_by_symbol.values():
        sub = bs.frame.index[bs.frame.index <= end_ts]
        idx = sub if idx is None else idx.union(sub)
    assert idx is not None
    idx = idx.sort_values().unique()
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _is_weekly_rebalance_execution_day(prev_ts: pd.Timestamp | None, cur_ts: pd.Timestamp) -> bool:
    """True iff ``prev_ts`` was the last EU trading day of its ISO week and ``cur_ts`` is the next."""
    if prev_ts is None:
        return False
    prev_iso = prev_ts.isocalendar()
    cur_iso = cur_ts.isocalendar()
    return (prev_iso[0], prev_iso[1]) != (cur_iso[0], cur_iso[1])


def _months_between_crossed(prev_date: date | None, cur_date: date) -> int:
    if prev_date is None:
        return 0
    return (cur_date.year - prev_date.year) * 12 + (cur_date.month - prev_date.month)


def _causal_benchmark_series(
    benchmark_bars: BarSeries | None,
    eu_calendar: pd.DatetimeIndex,
) -> pd.Series:
    if benchmark_bars is None:
        return pd.Series(np.nan, index=eu_calendar, dtype=float)
    end_ts = eu_calendar[-1]
    bench_cut = benchmark_bars.frame.loc[:end_ts, "close"]
    union_idx = bench_cut.index.union(eu_calendar).sort_values()
    return bench_cut.reindex(union_idx).ffill().reindex(eu_calendar)


# ---------------------------------------------------------------------------
# T+2 EUR Cash-Account Simulator
# ---------------------------------------------------------------------------


def run_cross_asset_backtest(
    bars_by_symbol: dict[str, BarSeries],
    cfg: CrossAssetConfig,
    *,
    start_date: date,
    end_date: date,
    initial_capital_eur: float = 1000.0,
    benchmark_bars: BarSeries | None = None,
) -> CrossAssetBacktestResult:
    """Simulate Strategy v6 on an IBKR Ireland EUR cash account with strict T+2 settlement.

    Zero look-ahead guarantees:
    1. Every asset frame is truncated strictly to ``<= end_date`` before computing any
       indicators.
    2. Weekly rebalance on session ``e`` Open uses indicators strictly from ``prev_ts < e``
       (the last EU trading day of the prior ISO week).
    3. Daily catastrophe stops evaluated at day ``t`` Close execute strictly at day ``t+1``
       Open (or the next session where that asset's exchange is open).
    4. Cross-exchange holidays: signals forward-fill causally (``<= t``), while fills ONLY
       execute on sessions where the traded asset actually has a real bar on its home
       exchange.
    5. Strict T+2 settlement: sale proceeds on session ``day_idx`` settle at
       ``day_idx + settlement_days`` and are never spent early (zero Good Faith Violations).
    """
    if initial_capital_eur <= 0:
        raise ValueError("initial_capital_eur must be positive")
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol must be non-empty")

    symbols = tuple(sorted(bars_by_symbol.keys()))
    end_ts = pd.Timestamp(end_date)
    start_ts = pd.Timestamp(start_date)

    # 1. Truncate every asset strictly to <= end_date and compute native-calendar indicators
    asset_frames: dict[str, pd.DataFrame] = {}
    native_indicators: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df_cut = bars_by_symbol[sym].frame.loc[:end_ts].copy()
        asset_frames[sym] = df_cut
        native_indicators[sym] = compute_asset_indicators(BarSeries(symbol=sym, frame=df_cut), cfg)

    # 2. Build aligned EU union calendar (warmup..end_date) and evaluation slice
    full_calendar = _build_union_calendar({s: BarSeries(symbol=s, frame=asset_frames[s]) for s in symbols}, end_date)
    eval_mask = (full_calendar >= start_ts) & (full_calendar <= end_ts)
    eval_calendar = full_calendar[eval_mask]
    if len(eval_calendar) == 0:
        raise ValueError(f"No EU trading sessions in [{start_date}, {end_date}]")

    # 3. Causally align indicators to full_calendar (forward-fill across single-exchange holidays)
    aligned_scores: dict[str, pd.Series] = {}
    aligned_eligible: dict[str, pd.Series] = {}
    aligned_sigma: dict[str, pd.Series] = {}
    for sym in symbols:
        ind = native_indicators[sym]
        aligned_scores[sym] = ind["score"].reindex(full_calendar).ffill()
        aligned_eligible[sym] = ind["eligible"].reindex(full_calendar).ffill().fillna(False).astype(bool)
        aligned_sigma[sym] = ind["sigma_20"].reindex(full_calendar).ffill()

    bench_series = _causal_benchmark_series(benchmark_bars, eval_calendar)

    # Causal last-known close per asset initialized strictly < start_date
    last_close: dict[str, float] = {}
    for sym in symbols:
        prior = asset_frames[sym].loc[asset_frames[sym].index < eval_calendar[0], "close"]
        if len(prior):
            last_close[sym] = float(prior.iloc[-1])

    # Portfolio state
    settled_cash = float(initial_capital_eur)
    unsettled_buckets: dict[int, float] = {}
    positions: dict[str, _OpenPosition] = {}
    active_targets: list[str] = []
    target_signal_date: date = start_date
    stopped_on_signal_date: dict[str, pd.Timestamp] = {}

    fills: list[CrossAssetOrderFill] = []
    round_trips: list[CrossAssetRoundTrip] = []
    weekly_signal_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    total_commissions = 0.0
    total_slippage = 0.0
    total_buy_notional = 0.0
    total_sell_notional = 0.0
    hosting_fees = 0.0
    skipped_zero_shares = 0
    settlement_wait_days = 0
    catastrophe_stops_triggered = 0
    rebalance_events = 0
    min_settled_cash = settled_cash

    # Map each eval_ts to its preceding session on full_calendar (strictly < eval_ts)
    full_pos_map = {ts: idx for idx, ts in enumerate(full_calendar)}
    prev_eval_date: date | None = None

    for day_idx, eu_ts in enumerate(eval_calendar):
        cur_date = eu_ts.date()
        fpos = full_pos_map[eu_ts]
        prev_ts: pd.Timestamp | None = full_calendar[fpos - 1] if fpos > 0 else None

        # Step 1: Settle T+2 proceeds ready at or before day_idx
        ready_keys = [k for k in unsettled_buckets if k <= day_idx]
        for k in ready_keys:
            settled_cash += unsettled_buckets.pop(k)

        # Step 2: Deduct monthly hosting fee on month boundaries (EUR 0.00 for v6)
        if cfg.monthly_hosting_eur > 0.0:
            months = _months_between_crossed(prev_eval_date, cur_date)
            if months > 0:
                fee = months * cfg.monthly_hosting_eur
                settled_cash = max(0.0, settled_cash - fee)
                hosting_fees += fee

        # Step 3: Weekly rebalance decision (if prev_ts was the last EU trading day of prior ISO week)
        is_rebalance_day = _is_weekly_rebalance_execution_day(prev_ts, eu_ts)
        if is_rebalance_day and prev_ts is not None:
            rebalance_events += 1
            elig_scores: dict[str, float] = {}
            all_scores_snap: dict[str, float] = {}
            for sym in symbols:
                sc = float(aligned_scores[sym].loc[prev_ts])
                all_scores_snap[sym] = sc
                if bool(aligned_eligible[sym].loc[prev_ts]) and math.isfinite(sc):
                    elig_scores[sym] = sc

            # Exclude any symbol that hit a catastrophe stop on this exact prev_ts close
            blocked_today = {
                sym for sym, stop_ts in stopped_on_signal_date.items() if stop_ts == prev_ts
            }
            # Incumbents exclude any position already scheduled for catastrophe stop exit
            held_for_hyst = {
                sym for sym, pos in positions.items() if not pos.pending_stop_exit
            }

            active_targets = select_target_assets(
                elig_scores,
                held_for_hyst,
                top_n=cfg.top_n,
                hysteresis_pct=cfg.hysteresis_pct,
                blocked_symbols=blocked_today,
            )
            target_signal_date = prev_ts.date()

            sig_row: dict[str, object] = {
                "exec_date": cur_date,
                "signal_date": prev_ts.date(),
                "n_eligible": len(elig_scores),
                "targets": ",".join(active_targets) if active_targets else "EUR_CASH",
            }
            for sym in symbols:
                sig_row[f"score_{sym}"] = all_scores_snap[sym]
                sig_row[f"elig_{sym}"] = sym in elig_scores
            weekly_signal_rows.append(sig_row)

        # Step 4: Open Step A — Execute SELLS (catastrophe stops OR holdings no longer in active_targets)
        target_set = set(active_targets)
        to_sell = [
            sym
            for sym, pos in sorted(positions.items())
            if pos.pending_stop_exit or sym not in target_set
        ]
        for sym in to_sell:
            pos = positions[sym]
            df_held = asset_frames[sym]
            if eu_ts not in df_held.index:
                continue  # Home exchange closed today; retry next open session

            bar = df_held.loc[eu_ts]
            open_adj = float(bar["open"])
            split_f = _split_factor_after(bar, cfg.sizing_mode)
            fill_adj = open_adj * (1.0 - cfg.slippage_pct)
            fill_raw = fill_adj * split_f
            gross_proceeds = pos.adj_shares * fill_adj
            slip_eur = pos.adj_shares * open_adj * cfg.slippage_pct
            comm = cfg.commission_eur
            net_proceeds = gross_proceeds - comm

            if cfg.settlement_days == 0:
                settled_cash += net_proceeds
            else:
                settle_idx = day_idx + cfg.settlement_days
                unsettled_buckets[settle_idx] = unsettled_buckets.get(settle_idx, 0.0) + net_proceeds

            raw_shares_exit = max(1, int(round(pos.adj_shares / split_f)))
            if pos.pending_stop_exit:
                exit_reason = pos.pending_stop_reason
                sig_dt = pos.pending_stop_signal_date or (prev_ts.date() if prev_ts is not None else cur_date)
                catastrophe_stops_triggered += 1
            else:
                sig_dt = target_signal_date
                if prev_ts is not None and not bool(aligned_eligible[sym].loc[prev_ts]):
                    exit_reason = "REBAL_FILTER_EXIT"
                elif len(active_targets) > 0:
                    exit_reason = "REBAL_ROTATE_OUT"
                else:
                    exit_reason = "REBAL_TO_CASH"

            fills.append(
                CrossAssetOrderFill(
                    date=cur_date,
                    signal_date=sig_dt,
                    symbol=sym,
                    side="SELL",
                    raw_shares=raw_shares_exit,
                    adj_shares=pos.adj_shares,
                    split_factor_after=split_f,
                    raw_fill_price=fill_raw,
                    adj_fill_price=fill_adj,
                    notional_eur=gross_proceeds,
                    commission_eur=comm,
                    slippage_eur=slip_eur,
                    reason=exit_reason,
                )
            )

            entry_open_adj = pos.entry_price_adj / (1.0 + cfg.slippage_pct)
            gross_pnl = pos.adj_shares * (open_adj - entry_open_adj)
            tot_comm = pos.entry_commission_eur + comm
            tot_slip = pos.entry_slippage_eur + slip_eur
            cost_basis = pos.entry_notional_eur + pos.entry_commission_eur
            net_pnl = net_proceeds - cost_basis
            ret_pct = net_pnl / cost_basis if cost_basis > 0 else 0.0

            round_trips.append(
                CrossAssetRoundTrip(
                    symbol=sym,
                    signal_entry_date=pos.signal_entry_date,
                    entry_date=pos.entry_date,
                    exit_date=cur_date,
                    raw_shares_entry=pos.raw_shares_entry,
                    adj_shares=pos.adj_shares,
                    entry_price_adj=pos.entry_price_adj,
                    exit_price_adj=fill_adj,
                    peak_close_adj=pos.peak_close_adj,
                    bars_held=pos.bars_held,
                    gross_pnl_eur=gross_pnl,
                    commissions_eur=tot_comm,
                    slippage_eur=tot_slip,
                    net_pnl_eur=net_pnl,
                    return_pct=ret_pct,
                    exit_reason=exit_reason,
                    is_open_at_end=False,
                )
            )

            total_commissions += comm
            total_slippage += slip_eur
            total_sell_notional += gross_proceeds
            del positions[sym]

        # Step 5: Open Step B — Execute BUYS for active_targets not currently held (SETTLED cash only!)
        pending_buys = [sym for sym in active_targets if sym not in positions]
        if pending_buys:
            # Estimate open-of-day portfolio equity causally using open (if trading) or last_close
            unsettled_now = float(sum(unsettled_buckets.values()))
            mv_open_est = 0.0
            for hsym, hpos in positions.items():
                if eu_ts in asset_frames[hsym].index:
                    px = float(asset_frames[hsym].loc[eu_ts, "open"])
                else:
                    px = last_close.get(hsym, hpos.entry_price_adj)
                mv_open_est += hpos.adj_shares * px
            equity_open_est = settled_cash + unsettled_now + mv_open_est
            slot_budget_target = equity_open_est / float(cfg.top_n)

            waited_today = False
            for sym in pending_buys:
                # Strict T+2 guard:
                # - top_n == 1: wait if ANY sale proceeds are settling so we deploy full capital once.
                # - top_n >= 2: deploy immediately ONLY if already-settled cash covers the target slot
                #   budget (e.g. cash was already settled before today); if proceeds needed for this
                #   slot are still settling, wait cleanly for T+2 settlement.
                if unsettled_now > 1e-6:
                    if cfg.top_n == 1 or settled_cash + 1e-6 < slot_budget_target:
                        waited_today = True
                        continue

                df_tgt = asset_frames[sym]
                if eu_ts not in df_tgt.index:
                    continue  # Home exchange closed today; retry next open session

                bar = df_tgt.loc[eu_ts]
                open_adj = float(bar["open"])
                split_f = _split_factor_after(bar, cfg.sizing_mode)
                fill_adj = open_adj * (1.0 + cfg.slippage_pct)
                fill_raw = fill_adj * split_f
                comm = cfg.commission_eur

                budget = settled_cash if cfg.top_n == 1 else min(settled_cash, slot_budget_target)
                available_for_shares = budget - comm
                raw_shares = (
                    int(math.floor(available_for_shares / fill_raw))
                    if available_for_shares > 0.0 and fill_raw > 0.0
                    else 0
                )

                if raw_shares >= 1:
                    adj_shares = raw_shares * split_f
                    notional = adj_shares * fill_adj
                    slip_eur = adj_shares * open_adj * cfg.slippage_pct
                    total_cost = notional + comm
                    if settled_cash - total_cost < -1e-6:
                        raise AssertionError(
                            f"Good Faith Violation on {cur_date}: settled={settled_cash}, cost={total_cost}"
                        )
                    settled_cash = max(0.0, settled_cash - total_cost)
                    min_settled_cash = min(min_settled_cash, settled_cash)

                    fills.append(
                        CrossAssetOrderFill(
                            date=cur_date,
                            signal_date=target_signal_date,
                            symbol=sym,
                            side="BUY",
                            raw_shares=raw_shares,
                            adj_shares=adj_shares,
                            split_factor_after=split_f,
                            raw_fill_price=fill_raw,
                            adj_fill_price=fill_adj,
                            notional_eur=notional,
                            commission_eur=comm,
                            slippage_eur=slip_eur,
                            reason="REBAL_BUY",
                        )
                    )
                    positions[sym] = _OpenPosition(
                        symbol=sym,
                        signal_entry_date=target_signal_date,
                        entry_date=cur_date,
                        raw_shares_entry=raw_shares,
                        adj_shares=adj_shares,
                        entry_price_adj=fill_adj,
                        peak_close_adj=fill_adj,
                        entry_notional_eur=notional,
                        entry_commission_eur=comm,
                        entry_slippage_eur=slip_eur,
                        bars_held=0,
                    )
                    total_commissions += comm
                    total_slippage += slip_eur
                    total_buy_notional += notional
                else:
                    skipped_zero_shares += 1

            if waited_today:
                settlement_wait_days += 1

        # Step 6: Close Mark-to-Market & Daily Intra-Week Catastrophe Stop Check (for t+1 Open)
        for sym in symbols:
            if eu_ts in asset_frames[sym].index:
                last_close[sym] = float(asset_frames[sym].loc[eu_ts, "close"])

        total_mv = 0.0
        mv_by_sym: dict[str, float] = {sym: 0.0 for sym in symbols}
        stop_triggered_today = False

        for sym, pos in positions.items():
            pos.bars_held += 1
            c_adj = last_close[sym]
            mv = pos.adj_shares * c_adj
            mv_by_sym[sym] = mv
            total_mv += mv

            # Evaluate catastrophe stop ONLY on sessions where the asset actually traded
            if eu_ts in asset_frames[sym].index and not pos.pending_stop_exit:
                if c_adj > pos.peak_close_adj:
                    pos.peak_close_adj = c_adj
                dd_entry = c_adj / pos.entry_price_adj - 1.0
                dd_peak = c_adj / pos.peak_close_adj - 1.0
                hit_entry = dd_entry <= -cfg.catastrophe_entry_dd + 1e-12
                hit_peak = dd_peak <= -cfg.catastrophe_peak_dd + 1e-12
                if hit_entry or hit_peak:
                    pos.pending_stop_exit = True
                    pos.pending_stop_signal_date = cur_date
                    if hit_entry and hit_peak:
                        pos.pending_stop_reason = "CATASTROPHE_STOP_ENTRY_AND_PEAK"
                    elif hit_entry:
                        pos.pending_stop_reason = "CATASTROPHE_STOP_ENTRY"
                    else:
                        pos.pending_stop_reason = "CATASTROPHE_STOP_PEAK"
                    # Stay in EUR Cash for this slot until the next weekly rebalance
                    active_targets = [t for t in active_targets if t != sym]
                    stopped_on_signal_date[sym] = eu_ts
                    stop_triggered_today = True

        unsettled_cash = float(sum(unsettled_buckets.values()))
        equity = settled_cash + unsettled_cash + total_mv
        min_settled_cash = min(min_settled_cash, settled_cash)

        held_sorted = sorted(positions.keys())
        row: dict[str, object] = {
            "date": eu_ts,
            "equity": equity,
            "settled_cash": settled_cash,
            "unsettled_cash": unsettled_cash,
            "market_value": total_mv,
            "cash_total": settled_cash + unsettled_cash,
            "open_positions": len(positions),
            "holdings": ",".join(held_sorted) if held_sorted else "EUR_CASH",
            "targets": ",".join(active_targets) if active_targets else "EUR_CASH",
            "is_rebalance_day": is_rebalance_day,
            "stop_triggered_at_close": stop_triggered_today,
            "benchmark_close": float(bench_series.loc[eu_ts]),
        }
        for sym in symbols:
            row[f"mv_{sym}"] = mv_by_sym[sym]
            row[f"w_{sym}"] = mv_by_sym[sym] / equity if equity > 0 else 0.0
        row["w_CASH"] = (settled_cash + unsettled_cash) / equity if equity > 0 else 1.0
        curve_rows.append(row)
        prev_eval_date = cur_date

    # Snapshot any open positions at end_date (mark-to-market, NOT executed exit fills)
    open_at_end: list[CrossAssetRoundTrip] = []
    for sym, pos in sorted(positions.items()):
        end_close = last_close[sym]
        mv_end = pos.adj_shares * end_close
        cost_basis = pos.entry_notional_eur + pos.entry_commission_eur
        net_pnl_mtm = mv_end - cost_basis
        entry_open_adj = pos.entry_price_adj / (1.0 + cfg.slippage_pct)
        gross_pnl_mtm = pos.adj_shares * (end_close - entry_open_adj)
        open_at_end.append(
            CrossAssetRoundTrip(
                symbol=sym,
                signal_entry_date=pos.signal_entry_date,
                entry_date=pos.entry_date,
                exit_date=eval_calendar[-1].date(),
                raw_shares_entry=pos.raw_shares_entry,
                adj_shares=pos.adj_shares,
                entry_price_adj=pos.entry_price_adj,
                exit_price_adj=end_close,
                peak_close_adj=pos.peak_close_adj,
                bars_held=pos.bars_held,
                gross_pnl_eur=gross_pnl_mtm,
                commissions_eur=pos.entry_commission_eur,
                slippage_eur=pos.entry_slippage_eur,
                net_pnl_eur=net_pnl_mtm,
                return_pct=net_pnl_mtm / cost_basis if cost_basis > 0 else 0.0,
                exit_reason="OPEN_MTM",
                is_open_at_end=True,
            )
        )

    curve = pd.DataFrame(curve_rows).set_index("date")
    weekly_df = pd.DataFrame(weekly_signal_rows)

    return CrossAssetBacktestResult(
        config=cfg,
        symbols=symbols,
        initial_capital_eur=float(initial_capital_eur),
        start_date=start_date,
        end_date=end_date,
        equity_curve=curve,
        weekly_signals=weekly_df,
        fills=fills,
        round_trips=round_trips,
        open_positions_at_end=open_at_end,
        total_commissions_eur=float(total_commissions),
        total_slippage_eur=float(total_slippage),
        total_buy_notional_eur=float(total_buy_notional),
        total_sell_notional_eur=float(total_sell_notional),
        hosting_fees_eur=float(hosting_fees),
        skipped_zero_shares=skipped_zero_shares,
        settlement_wait_days=settlement_wait_days,
        min_settled_cash_eur=float(min_settled_cash),
        catastrophe_stops_triggered=catastrophe_stops_triggered,
        rebalance_events=rebalance_events,
    )


# ---------------------------------------------------------------------------
# Metrics & Full Per-Asset Attribution
# ---------------------------------------------------------------------------


def compute_asset_attribution(res: CrossAssetBacktestResult) -> dict[str, AssetAttribution]:
    """Compute additive return contribution, % days held, trades, win rate, and stops per asset + Cash."""
    curve = res.equity_curve
    n_days = len(curve)
    init = res.initial_capital_eur
    all_trips = list(res.round_trips) + list(res.open_positions_at_end)

    out: dict[str, AssetAttribution] = {}
    for sym in res.symbols:
        mv_col = f"mv_{sym}"
        w_col = f"w_{sym}"
        days_held = int((curve[mv_col] > 0.0).sum()) if mv_col in curve.columns else 0
        avg_w = float(curve[w_col].mean()) if w_col in curve.columns and n_days > 0 else 0.0

        closed_s = [t for t in res.round_trips if t.symbol == sym]
        open_s = [t for t in res.open_positions_at_end if t.symbol == sym]
        eval_s = [t for t in all_trips if t.symbol == sym]

        net_pnl = float(sum(t.net_pnl_eur for t in eval_s))
        win_rate = float(np.mean([t.net_pnl_eur > 0.0 for t in eval_s])) if eval_s else None
        avg_bars = float(np.mean([t.bars_held for t in eval_s])) if eval_s else None
        cat_stops = sum(1 for t in closed_s if t.exit_reason.startswith("CATASTROPHE_STOP"))
        comm = float(sum(t.commissions_eur for t in eval_s))
        slip = float(sum(t.slippage_eur for t in eval_s))

        out[sym] = AssetAttribution(
            symbol=sym,
            days_held=days_held,
            pct_days_held=float(days_held / n_days) if n_days > 0 else 0.0,
            avg_weight_pct=avg_w,
            net_pnl_eur=net_pnl,
            return_contribution_pct=net_pnl / init if init > 0 else 0.0,
            closed_trades=len(closed_s),
            open_at_end=len(open_s),
            evaluated_trades=len(eval_s),
            win_rate=win_rate,
            catastrophe_stops=cat_stops,
            avg_bars_held=avg_bars,
            commissions_eur=comm,
            slippage_eur=slip,
        )

    # EUR Cash attribution row
    cash_only_days = int((curve["open_positions"] == 0).sum()) if n_days > 0 else 0
    avg_cash_w = float(curve["w_CASH"].mean()) if "w_CASH" in curve.columns and n_days > 0 else 1.0
    out["EUR_CASH"] = AssetAttribution(
        symbol="EUR_CASH",
        days_held=cash_only_days,
        pct_days_held=float(cash_only_days / n_days) if n_days > 0 else 0.0,
        avg_weight_pct=avg_cash_w,
        net_pnl_eur=-res.hosting_fees_eur,
        return_contribution_pct=-res.hosting_fees_eur / init if init > 0 else 0.0,
        closed_trades=0,
        open_at_end=0,
        evaluated_trades=0,
        win_rate=None,
        catastrophe_stops=0,
        avg_bars_held=None,
        commissions_eur=0.0,
        slippage_eur=0.0,
    )
    return out


def compute_cross_asset_metrics(
    res: CrossAssetBacktestResult,
    metrics_cfg: MetricsConfig | None = None,
    *,
    cumulative_trials_n: int = 30,
    cumulative_trials_conservative_n: int = 50,
    include_open_in_win_rate: bool = True,
) -> dict[str, float | int]:
    """Compute standardized EUR performance, risk, standalone & cumulative DSR, turnover, and drag."""
    mcfg = metrics_cfg or MetricsConfig()
    curve = res.equity_curve
    eq = curve["equity"]
    init = res.initial_capital_eur
    r = daily_returns(eq, init)

    days = (curve.index[-1] - curve.index[0]).days + 1
    years = max(days / 365.25, 1e-6)

    n_fills = len(res.fills)
    eval_trips = list(res.round_trips)
    if include_open_in_win_rate:
        eval_trips.extend(res.open_positions_at_end)

    win_rate = float(np.mean([t.net_pnl_eur > 0.0 for t in eval_trips])) if eval_trips else float("nan")
    total_friction = res.total_commissions_eur + res.total_slippage_eur + res.hosting_fees_eur

    # Cash-drag diagnostics on invested days:
    # - invested_days: days with >= 1 open position
    # - full_target_days: days where open_positions == top_n (or == n_targets > 0)
    invested_mask = curve["open_positions"] > 0
    avg_deployed_overall = float((curve["market_value"] / curve["equity"].replace(0.0, np.nan)).fillna(0.0).mean())
    avg_deployed_when_invested = (
        float(
            (curve.loc[invested_mask, "market_value"] / curve.loc[invested_mask, "equity"].replace(0.0, np.nan))
            .fillna(0.0)
            .mean()
        )
        if bool(invested_mask.any())
        else 0.0
    )
    full_slot_mask = curve["open_positions"] >= res.config.top_n
    avg_cash_drag_full_slots = (
        float(curve.loc[full_slot_mask, "w_CASH"].mean())
        if bool(full_slot_mask.any())
        else float("nan")
    )

    return {
        "initial_eur": init,
        "final_eur": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / init - 1.0),
        "cagr": cagr(eq, init),
        "sharpe": sharpe(r, mcfg.periods_per_year, mcfg.risk_free_rate),
        "sortino": sortino(r, mcfg.periods_per_year, mcfg.risk_free_rate),
        "deflated_sharpe": deflated_sharpe(
            r, mcfg.periods_per_year, mcfg.dsr_n_trials, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum30": deflated_sharpe(
            r, mcfg.periods_per_year, cumulative_trials_n, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum50": deflated_sharpe(
            r, mcfg.periods_per_year, cumulative_trials_conservative_n, mcfg.dsr_trial_sharpe_std
        ),
        "max_drawdown": max_drawdown(eq, init),
        "total_orders": n_fills,
        "orders_per_year": float(n_fills / years),
        "closed_round_trips": len(res.round_trips),
        "open_at_end": len(res.open_positions_at_end),
        "evaluated_trips": len(eval_trips),
        "win_rate": win_rate,
        "any_invested_exposure": float(invested_mask.mean()),
        "cash_only_exposure": float((~invested_mask).mean()),
        "avg_capital_deployed_pct": avg_deployed_overall,
        "avg_deployed_when_invested_pct": avg_deployed_when_invested,
        "avg_cash_drag_when_full_slots_pct": avg_cash_drag_full_slots,
        "commissions_eur": res.total_commissions_eur,
        "slippage_eur": res.total_slippage_eur,
        "hosting_eur": res.hosting_fees_eur,
        "total_friction_eur": total_friction,
        "friction_pct_initial": total_friction / init,
        "catastrophe_stops_triggered": res.catastrophe_stops_triggered,
        "rebalance_events": res.rebalance_events,
        "skipped_zero_shares": res.skipped_zero_shares,
        "settlement_wait_days": res.settlement_wait_days,
        "min_settled_cash_eur": res.min_settled_cash_eur,
    }
