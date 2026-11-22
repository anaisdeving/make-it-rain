"""Strategy v7: VIX Term-Structure Tail-Capture & Leveraged Equity Rotation.

Designed for an IBKR Ireland EUR cash account (EUR 1.25/order commission, 0.10%/side
slippage, whole as-traded shares only, strict T+2 settlement, EUR 0.00/mo hosting).

Core Guarantees (zero look-ahead, zero Good Faith Violations):
- Strictly causal US-close signal computation on the common US trading calendar
  (``QQQ ∩ ^VIX ∩ ^VIX3M``), truncated to ``<= end_date`` before computing any indicator.
- Strict cross-timezone alignment: US close at 22:00 CET on ``us_date`` is actionable
  strictly on the next EU trading session ``eu_date > us_date`` Open (09:00 CET) via
  ``searchsorted(..., side="left") - 1`` (never same-day EU open).
- Home-exchange execution guard: fills execute ONLY on sessions where the traded EU
  instrument actually has a real bar on its home exchange.
- Strict T+2 settlement: sale proceeds on EU session ``day_idx`` enter unsettled bucket
  ``day_idx + settlement_days`` and are never spent early (``settled_cash >= -1e-6``
  asserted every day). Direct asset-to-asset rotations (``RISK_ON <-> TAIL_HEDGE``)
  wait T+2 for proceeds to settle before buying the incoming leg; already-settled
  EUR Cash (from ``NEUTRAL``) deploys immediately at ``t+1`` Open.

Locked Three-State Hysteresis State Machine (US close ``t``):
- ``vix_ratio = VIX_close / VIX3M_close``
- ``qqq_sma200 = SMA(QQQ_close, 200)``
- **``TAIL_HEDGE`` (Long Volatility UCITS ETF):**
  - Enter when ``vix_ratio > 1.03`` (backwardation).
  - Stay in ``TAIL_HEDGE`` while ``vix_ratio >= 0.97``.
  - Exit as soon as ``vix_ratio < 0.97``: transition directly to ``RISK_ON`` if
    ``QQQ > 1.01 * qqq_sma200`` AND ``vix_ratio < 0.95``, else to ``NEUTRAL``.
- **``RISK_ON`` (Long Leveraged Equity):**
  - When NOT in ``TAIL_HEDGE``, enter if ``QQQ > 1.01 * qqq_sma200`` AND
    ``vix_ratio < 0.95``.
  - Exit ``RISK_ON`` to ``TAIL_HEDGE`` if ``vix_ratio > 1.03``, or to ``NEUTRAL``
    if ``QQQ < 0.99 * qqq_sma200`` OR ``vix_ratio > 1.00``.
- **``NEUTRAL`` (100% Settled EUR Cash):** All other states.

Variants:
- ``v7a_2x``: ``RISK_ON`` = 100% ``CL2.PA``, ``TAIL_HEDGE`` = 100% VIX ETF, ``NEUTRAL`` = EUR Cash.
- ``v7b_3x``: ``RISK_ON`` = 100% ``QQQ3.MI``, ``TAIL_HEDGE`` = 100% VIX ETF, ``NEUTRAL`` = EUR Cash.
- ``v7c_3x_half_hedge``: ``RISK_ON`` = 100% ``QQQ3.MI``, ``TAIL_HEDGE`` = 50% VIX ETF / 50% EUR Cash, ``NEUTRAL`` = EUR Cash.
- ``v7b_cash_tail`` (diagnostic ablation): ``RISK_ON`` = 100% ``QQQ3.MI``, ``TAIL_HEDGE`` = 0% VIX ETF / 100% EUR Cash (isolates VIX ETF alpha vs cash during backwardation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
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


class VixRegimeState(str, Enum):
    RISK_ON = "RISK_ON"
    TAIL_HEDGE = "TAIL_HEDGE"
    NEUTRAL = "NEUTRAL"


SleeveRole = Literal["RISK_ON", "TAIL_HEDGE"]


class VixTermRegimeConfig(_Frozen):
    """Frozen configuration for Strategy v7."""

    name: str = "v7b_3x"
    risk_on_symbol: str = "QQQ3.MI"
    tail_hedge_symbol: str = "LVO.MI"
    tail_hedge_weight: float = Field(1.0, ge=0.0, le=1.0)
    sma_window: int = Field(200, ge=2)
    sma_enter_mult: float = Field(1.01, gt=1.0, lt=1.25)
    sma_exit_mult: float = Field(0.99, gt=0.75, lt=1.0)
    vix_ratio_tail_enter: float = Field(1.03, gt=0.5)
    vix_ratio_tail_exit: float = Field(0.97, gt=0.5)
    vix_ratio_risk_on_enter: float = Field(0.95, gt=0.5)
    vix_ratio_risk_on_exit: float = Field(1.00, gt=0.5)
    commission_eur: float = Field(1.25, ge=0.0)
    slippage_pct: float = Field(0.001, ge=0.0, lt=0.1)
    settlement_days: int = Field(2, ge=0, le=5)
    monthly_hosting_eur: float = Field(0.0, ge=0.0)
    sizing_mode: SizingMode = SizingMode.AS_TRADED

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "VixTermRegimeConfig":
        if not self.risk_on_symbol.strip():
            raise ValueError("risk_on_symbol must be non-empty")
        if self.tail_hedge_weight > 0.0 and not self.tail_hedge_symbol.strip():
            raise ValueError("tail_hedge_symbol must be non-empty when tail_hedge_weight > 0")
        if self.vix_ratio_tail_exit >= self.vix_ratio_tail_enter:
            raise ValueError("vix_ratio_tail_exit must be strictly below vix_ratio_tail_enter")
        if self.vix_ratio_risk_on_enter >= self.vix_ratio_risk_on_exit:
            raise ValueError("vix_ratio_risk_on_enter must be strictly below vix_ratio_risk_on_exit")
        if self.vix_ratio_risk_on_exit >= self.vix_ratio_tail_enter:
            raise ValueError("vix_ratio_risk_on_exit must be strictly below vix_ratio_tail_enter")
        return self


class VixOrderFill(_Frozen):
    """One executed EU Open fill (buy or sell) in EUR."""

    date: date
    signal_us_date: date
    symbol: str
    role: SleeveRole
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


class VixRoundTrip(_Frozen):
    """Completed (or end-of-window open MTM) sleeve position lifecycle in EUR."""

    symbol: str
    role: SleeveRole
    signal_entry_us_date: date
    entry_date: date
    exit_date: date
    raw_shares_entry: int = Field(gt=0)
    adj_shares: float = Field(gt=0)
    entry_price_adj: float = Field(gt=0)
    exit_price_adj: float = Field(gt=0)
    bars_held: int = Field(ge=0)
    gross_pnl_eur: float
    commissions_eur: float
    slippage_eur: float
    net_pnl_eur: float
    return_pct: float
    exit_reason: str
    is_open_at_end: bool = False


class TailHedgeEpisode(_Frozen):
    """Episode-level audit for every US TAIL_HEDGE regime episode in the evaluation window."""

    episode_id: int
    prior_state: str
    signal_start_us_date: date
    signal_end_us_date: date | None
    first_actionable_eu_date: date
    entry_date: date | None
    exit_date: date | None
    t2_delay_sessions: int
    missed_due_to_t2: bool
    entry_vix_ratio: float
    peak_vix_ratio: float
    exit_vix_ratio: float | None
    lvo_open_on_signal_day: float | None
    lvo_open_on_fill_day: float | None
    lvo_move_during_t2_pct: float | None
    raw_shares_entry: int
    bars_held: int
    net_pnl_eur: float
    return_pct: float | None
    exit_reason: str
    is_open_at_end: bool = False


class SleeveAttribution(_Frozen):
    """Performance and holding-time attribution for one strategy sleeve."""

    sleeve: Literal["RISK_ON", "TAIL_HEDGE", "EUR_CASH"]
    symbol: str
    days_held: int
    pct_days_held: float
    avg_weight_pct: float
    gross_pnl_eur: float
    commissions_eur: float
    slippage_eur: float
    net_pnl_eur: float
    return_contribution_pct: float
    closed_trades: int
    open_at_end: int
    evaluated_trades: int
    win_rate: float | None
    avg_bars_held: float | None


@dataclass
class _OpenPosition:
    symbol: str
    role: SleeveRole
    signal_entry_us_date: date
    entry_date: date
    raw_shares_entry: int
    adj_shares: float
    entry_price_adj: float
    entry_notional_eur: float
    entry_commission_eur: float
    entry_slippage_eur: float
    bars_held: int = 0
    episode_id: int | None = None


@dataclass
class _ActiveTailEpisode:
    episode_id: int
    prior_state: str
    signal_start_us_date: date
    first_actionable_eu_date: date
    first_actionable_day_idx: int
    entry_vix_ratio: float
    peak_vix_ratio: float
    lvo_open_on_signal_day: float | None
    t2_delay_sessions: int = 0
    entry_date: date | None = None
    lvo_open_on_fill_day: float | None = None
    raw_shares_entry: int = 0
    linked_trip: VixRoundTrip | None = None


@dataclass(frozen=True)
class VixTermBacktestResult:
    """Full output of a Strategy v7 EUR cash-account simulation."""

    config: VixTermRegimeConfig
    initial_capital_eur: float
    start_date: date
    end_date: date
    equity_curve: pd.DataFrame
    us_signals: pd.DataFrame
    fills: list[VixOrderFill]
    round_trips: list[VixRoundTrip]
    open_position_at_end: VixRoundTrip | None
    tail_hedge_episodes: list[TailHedgeEpisode]
    total_commissions_eur: float
    total_slippage_eur: float
    total_buy_notional_eur: float
    total_sell_notional_eur: float
    hosting_fees_eur: float
    skipped_zero_shares: int
    settlement_wait_days: int
    tail_hedge_t2_wait_days: int
    risk_on_t2_wait_days: int
    min_settled_cash_eur: float
    regime_switches_us: int


# ---------------------------------------------------------------------------
# Pure Signal Generator (US Close) & Strict Cross-Timezone Alignment
# ---------------------------------------------------------------------------


def compute_vix_term_signals(
    qqq_bars: BarSeries,
    vix_bars: BarSeries,
    vix3m_bars: BarSeries,
    cfg: VixTermRegimeConfig,
) -> pd.DataFrame:
    """Compute strictly causal 3-state VIX term-structure + QQQ SMA(200) regime signals.

    Evaluated strictly on the common US trading calendar ``QQQ ∩ ^VIX ∩ ^VIX3M``
    so holiday stray bars (e.g. CBOE-only prints when US equity markets are closed)
    never introduce stale or unmatched inputs.
    """
    common_idx = (
        qqq_bars.frame.index.intersection(vix_bars.frame.index)
        .intersection(vix3m_bars.frame.index)
        .sort_values()
        .unique()
    )
    if len(common_idx) == 0:
        raise ValueError("No overlapping US trading dates across QQQ, ^VIX, and ^VIX3M")

    qqq_close = qqq_bars.frame.loc[common_idx, "close"].astype(float)
    vix_close = vix_bars.frame.loc[common_idx, "close"].astype(float)
    vix3m_close = vix3m_bars.frame.loc[common_idx, "close"].astype(float)

    qqq_sma200 = qqq_close.rolling(cfg.sma_window, min_periods=cfg.sma_window).mean()
    vix_ratio = vix_close / vix3m_close.replace(0.0, np.nan)

    q_vals = qqq_close.to_numpy(dtype=float)
    sma_vals = qqq_sma200.to_numpy(dtype=float)
    v_vals = vix_close.to_numpy(dtype=float)
    v3_vals = vix3m_close.to_numpy(dtype=float)
    r_vals = vix_ratio.to_numpy(dtype=float)

    n = len(common_idx)
    states: list[str] = []
    switched: list[bool] = []
    prev_state = VixRegimeState.NEUTRAL

    for i in range(n):
        q = q_vals[i]
        m = sma_vals[i]
        r = r_vals[i]

        if not (math.isfinite(q) and math.isfinite(m) and math.isfinite(r)):
            cur = VixRegimeState.NEUTRAL
        elif prev_state is VixRegimeState.TAIL_HEDGE:
            if r >= cfg.vix_ratio_tail_exit:
                cur = VixRegimeState.TAIL_HEDGE
            elif q > cfg.sma_enter_mult * m and r < cfg.vix_ratio_risk_on_enter:
                cur = VixRegimeState.RISK_ON
            else:
                cur = VixRegimeState.NEUTRAL
        elif prev_state is VixRegimeState.RISK_ON:
            if r > cfg.vix_ratio_tail_enter:
                cur = VixRegimeState.TAIL_HEDGE
            elif q < cfg.sma_exit_mult * m or r > cfg.vix_ratio_risk_on_exit:
                cur = VixRegimeState.NEUTRAL
            else:
                cur = VixRegimeState.RISK_ON
        else:  # NEUTRAL
            if r > cfg.vix_ratio_tail_enter:
                cur = VixRegimeState.TAIL_HEDGE
            elif q > cfg.sma_enter_mult * m and r < cfg.vix_ratio_risk_on_enter:
                cur = VixRegimeState.RISK_ON
            else:
                cur = VixRegimeState.NEUTRAL

        states.append(cur.value)
        switched.append(i > 0 and cur is not prev_state)
        prev_state = cur

    return pd.DataFrame(
        {
            "qqq_close": q_vals,
            "qqq_sma200": sma_vals,
            "vix_close": v_vals,
            "vix3m_close": v3_vals,
            "vix_ratio": r_vals,
            "state": states,
            "switched": switched,
        },
        index=common_idx,
    )


def align_vix_signals_to_eu_calendar(
    us_signals: pd.DataFrame,
    eu_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each EU session ``e`` to the latest US close strictly BEFORE ``e`` (``us_date < e``)."""
    if not isinstance(eu_calendar, pd.DatetimeIndex):
        raise ValueError("eu_calendar must be a DatetimeIndex")
    if eu_calendar.tz is not None:
        raise ValueError("eu_calendar must be tz-naive")
    if not eu_calendar.is_monotonic_increasing or not eu_calendar.is_unique:
        raise ValueError("eu_calendar must be ascending and unique")

    us_idx = us_signals.index
    positions = us_idx.searchsorted(eu_calendar, side="left") - 1

    sig_dates: list[pd.Timestamp] = []
    target_states: list[str] = []
    qqq_closes: list[float] = []
    qqq_smas: list[float] = []
    vix_closes: list[float] = []
    vix3m_closes: list[float] = []
    vix_ratios: list[float] = []

    state_arr = us_signals["state"].to_numpy()
    q_arr = us_signals["qqq_close"].to_numpy(dtype=float)
    sma_arr = us_signals["qqq_sma200"].to_numpy(dtype=float)
    v_arr = us_signals["vix_close"].to_numpy(dtype=float)
    v3_arr = us_signals["vix3m_close"].to_numpy(dtype=float)
    r_arr = us_signals["vix_ratio"].to_numpy(dtype=float)

    for eu_ts, pos in zip(eu_calendar, positions):
        if pos < 0:
            sig_dates.append(pd.NaT)
            target_states.append(VixRegimeState.NEUTRAL.value)
            qqq_closes.append(float("nan"))
            qqq_smas.append(float("nan"))
            vix_closes.append(float("nan"))
            vix3m_closes.append(float("nan"))
            vix_ratios.append(float("nan"))
        else:
            us_ts = us_idx[pos]
            if not (us_ts < eu_ts):
                raise AssertionError(f"look-ahead violation: us_ts={us_ts} >= eu_ts={eu_ts}")
            sig_dates.append(us_ts)
            target_states.append(str(state_arr[pos]))
            qqq_closes.append(float(q_arr[pos]))
            qqq_smas.append(float(sma_arr[pos]))
            vix_closes.append(float(v_arr[pos]))
            vix3m_closes.append(float(v3_arr[pos]))
            vix_ratios.append(float(r_arr[pos]))

    return pd.DataFrame(
        {
            "signal_us_date": sig_dates,
            "target_state": target_states,
            "qqq_close": qqq_closes,
            "qqq_sma200": qqq_smas,
            "vix_close": vix_closes,
            "vix3m_close": vix3m_closes,
            "vix_ratio": vix_ratios,
        },
        index=eu_calendar,
    )


# ---------------------------------------------------------------------------
# Split, Calendar, & Benchmark Helpers
# ---------------------------------------------------------------------------


def _split_factor_after(row: pd.Series, sizing_mode: SizingMode) -> float:
    if sizing_mode is SizingMode.POST_SPLIT:
        return 1.0
    c = float(row["close"])
    rc = float(row["raw_close"])
    if c <= 0 or math.isnan(rc) or rc <= 0:
        return 1.0
    return rc / c


def _build_eu_calendar(
    risk_on_bars: BarSeries,
    tail_hedge_bars: BarSeries | None,
    start_date: date,
    end_date: date,
) -> pd.DatetimeIndex:
    idx = risk_on_bars.frame.index
    if tail_hedge_bars is not None:
        idx = idx.union(tail_hedge_bars.frame.index)
    idx = idx.sort_values().unique()
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    mask = (idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))
    cal = idx[mask]
    if len(cal) == 0:
        raise ValueError(f"No EU trading sessions in [{start_date}, {end_date}]")
    return cal


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


def _months_between_crossed(prev_date: date | None, cur_date: date) -> int:
    if prev_date is None:
        return 0
    return (cur_date.year - prev_date.year) * 12 + (cur_date.month - prev_date.month)


# ---------------------------------------------------------------------------
# Strict T+2 EUR Cash-Account Simulator
# ---------------------------------------------------------------------------


def run_vix_term_backtest(
    qqq_bars: BarSeries,
    vix_bars: BarSeries,
    vix3m_bars: BarSeries,
    risk_on_bars: BarSeries,
    cfg: VixTermRegimeConfig,
    *,
    start_date: date,
    end_date: date,
    initial_capital_eur: float = 1000.0,
    tail_hedge_bars: BarSeries | None = None,
    benchmark_bars: BarSeries | None = None,
) -> VixTermBacktestResult:
    """Simulate Strategy v7 on an IBKR Ireland EUR cash account with strict T+2 settlement."""
    if initial_capital_eur <= 0:
        raise ValueError("initial_capital_eur must be positive")
    if cfg.tail_hedge_weight > 0.0 and tail_hedge_bars is None:
        raise ValueError("tail_hedge_bars is required when tail_hedge_weight > 0")

    end_ts = pd.Timestamp(end_date)

    # 1. Truncate all input series strictly to <= end_date before any indicator math
    qqq_cut = BarSeries(symbol=qqq_bars.symbol, frame=qqq_bars.frame.loc[:end_ts])
    vix_cut = BarSeries(symbol=vix_bars.symbol, frame=vix_bars.frame.loc[:end_ts])
    vix3m_cut = BarSeries(symbol=vix3m_bars.symbol, frame=vix3m_bars.frame.loc[:end_ts])
    us_signals = compute_vix_term_signals(qqq_cut, vix_cut, vix3m_cut, cfg)

    ro_cut = BarSeries(symbol=risk_on_bars.symbol, frame=risk_on_bars.frame.loc[:end_ts])
    th_cut: BarSeries | None = None
    if cfg.tail_hedge_weight > 0.0 and tail_hedge_bars is not None:
        th_cut = BarSeries(symbol=tail_hedge_bars.symbol, frame=tail_hedge_bars.frame.loc[:end_ts])

    eu_calendar = _build_eu_calendar(ro_cut, th_cut, start_date, end_date)
    aligned = align_vix_signals_to_eu_calendar(us_signals, eu_calendar)
    bench_series = _causal_benchmark_series(benchmark_bars, eu_calendar)

    asset_frames: dict[str, pd.DataFrame] = {cfg.risk_on_symbol: ro_cut.frame}
    if th_cut is not None:
        asset_frames[cfg.tail_hedge_symbol] = th_cut.frame

    # Causal last-known close per asset initialized strictly < start_date
    last_close: dict[str, float] = {}
    for sym, df in asset_frames.items():
        prior = df.loc[df.index < eu_calendar[0], "close"]
        if len(prior):
            last_close[sym] = float(prior.iloc[-1])

    settled_cash = float(initial_capital_eur)
    unsettled_buckets: dict[int, float] = {}
    pos: _OpenPosition | None = None

    fills: list[VixOrderFill] = []
    round_trips: list[VixRoundTrip] = []
    tail_episodes: list[TailHedgeEpisode] = []
    active_tail_ep: _ActiveTailEpisode | None = None
    next_ep_id = 1

    total_commissions = 0.0
    total_slippage = 0.0
    total_buy_notional = 0.0
    total_sell_notional = 0.0
    hosting_fees = 0.0
    skipped_zero_shares = 0
    settlement_wait_days = 0
    tail_hedge_t2_wait_days = 0
    risk_on_t2_wait_days = 0
    min_settled_cash = settled_cash

    rows: list[dict[str, object]] = []
    prev_date: date | None = None
    prev_target_state: str | None = None

    def _finalize_tail_episode(
        ep: _ActiveTailEpisode,
        *,
        signal_end_us_date: date | None,
        exit_vix_ratio: float | None,
        exit_trip: VixRoundTrip | None,
        is_open_at_end: bool,
    ) -> TailHedgeEpisode:
        trip = exit_trip or ep.linked_trip
        lvo_move: float | None = None
        if (
            ep.lvo_open_on_signal_day is not None
            and ep.lvo_open_on_fill_day is not None
            and ep.lvo_open_on_signal_day > 0.0
        ):
            lvo_move = (ep.lvo_open_on_fill_day / ep.lvo_open_on_signal_day) - 1.0

        missed = ep.entry_date is None and cfg.tail_hedge_weight > 0.0
        if trip is not None:
            exit_dt = trip.exit_date
            bars_held = trip.bars_held
            net_pnl = trip.net_pnl_eur
            ret_pct: float | None = trip.return_pct
            exit_reason = trip.exit_reason
        else:
            exit_dt = None
            bars_held = 0
            net_pnl = 0.0
            ret_pct = None
            if cfg.tail_hedge_weight == 0.0:
                exit_reason = "CASH_TAIL_ABLATION"
            elif ep.t2_delay_sessions > 0:
                exit_reason = "MISSED_T2_SETTLEMENT"
            else:
                exit_reason = "UNFILLED"

        return TailHedgeEpisode(
            episode_id=ep.episode_id,
            prior_state=ep.prior_state,
            signal_start_us_date=ep.signal_start_us_date,
            signal_end_us_date=signal_end_us_date,
            first_actionable_eu_date=ep.first_actionable_eu_date,
            entry_date=ep.entry_date,
            exit_date=exit_dt,
            t2_delay_sessions=ep.t2_delay_sessions,
            missed_due_to_t2=missed and ep.t2_delay_sessions > 0,
            entry_vix_ratio=ep.entry_vix_ratio,
            peak_vix_ratio=ep.peak_vix_ratio,
            exit_vix_ratio=exit_vix_ratio,
            lvo_open_on_signal_day=ep.lvo_open_on_signal_day,
            lvo_open_on_fill_day=ep.lvo_open_on_fill_day,
            lvo_move_during_t2_pct=lvo_move,
            raw_shares_entry=ep.raw_shares_entry,
            bars_held=bars_held,
            net_pnl_eur=net_pnl,
            return_pct=ret_pct,
            exit_reason=exit_reason,
            is_open_at_end=is_open_at_end,
        )

    for day_idx, eu_ts in enumerate(eu_calendar):
        cur_date = eu_ts.date()

        # 1. Settle T+2 proceeds ready at or before day_idx
        ready_keys = [k for k in unsettled_buckets if k <= day_idx]
        for k in ready_keys:
            settled_cash += unsettled_buckets.pop(k)

        # 2. Monthly hosting fee on month boundaries (EUR 0.00 for Strategy v7)
        if cfg.monthly_hosting_eur > 0.0:
            months = _months_between_crossed(prev_date, cur_date)
            if months > 0:
                fee = months * cfg.monthly_hosting_eur
                settled_cash = max(0.0, settled_cash - fee)
                hosting_fees += fee

        # 3. Target state strictly from prior US close (us_date < eu_ts)
        al_row = aligned.loc[eu_ts]
        sig_us_ts = al_row["signal_us_date"]
        sig_us_date = sig_us_ts.date() if pd.notna(sig_us_ts) else cur_date
        target_state = str(al_row["target_state"])
        vix_ratio_now = float(al_row["vix_ratio"])
        vix_close_now = float(al_row["vix_close"])
        vix3m_close_now = float(al_row["vix3m_close"])

        # Track TAIL_HEDGE episode lifecycle
        if target_state == VixRegimeState.TAIL_HEDGE.value:
            if active_tail_ep is None:
                lvo_open_sig: float | None = None
                if th_cut is not None and eu_ts in th_cut.frame.index:
                    lvo_open_sig = float(th_cut.frame.loc[eu_ts, "open"])
                active_tail_ep = _ActiveTailEpisode(
                    episode_id=next_ep_id,
                    prior_state=prev_target_state or "WINDOW_START",
                    signal_start_us_date=sig_us_date,
                    first_actionable_eu_date=cur_date,
                    first_actionable_day_idx=day_idx,
                    entry_vix_ratio=vix_ratio_now,
                    peak_vix_ratio=vix_ratio_now,
                    lvo_open_on_signal_day=lvo_open_sig,
                )
                next_ep_id += 1
            else:
                if math.isfinite(vix_ratio_now) and vix_ratio_now > active_tail_ep.peak_vix_ratio:
                    active_tail_ep.peak_vix_ratio = vix_ratio_now
                if active_tail_ep.lvo_open_on_signal_day is None and th_cut is not None and eu_ts in th_cut.frame.index:
                    active_tail_ep.lvo_open_on_signal_day = float(th_cut.frame.loc[eu_ts, "open"])

        if target_state == VixRegimeState.RISK_ON.value:
            desired_symbol: str | None = cfg.risk_on_symbol
            desired_role: SleeveRole | None = "RISK_ON"
            desired_weight = 1.0
        elif target_state == VixRegimeState.TAIL_HEDGE.value and cfg.tail_hedge_weight > 0.0:
            desired_symbol = cfg.tail_hedge_symbol
            desired_role = "TAIL_HEDGE"
            desired_weight = cfg.tail_hedge_weight
        else:
            desired_symbol = None
            desired_role = None
            desired_weight = 0.0

        # 4. Open Step A: Liquidate non-target holding if its home exchange is open today
        closed_tail_trip_today: VixRoundTrip | None = None
        if pos is not None and pos.symbol != desired_symbol:
            df_held = asset_frames[pos.symbol]
            if eu_ts in df_held.index:
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
                if pos.role == "RISK_ON":
                    exit_reason = (
                        "ROTATE_TO_TAIL_HEDGE"
                        if target_state == VixRegimeState.TAIL_HEDGE.value
                        else "REGIME_TO_NEUTRAL"
                    )
                else:
                    exit_reason = (
                        "TAIL_EXIT_TO_RISK_ON"
                        if target_state == VixRegimeState.RISK_ON.value
                        else "TAIL_EXIT_TO_NEUTRAL"
                    )

                fills.append(
                    VixOrderFill(
                        date=cur_date,
                        signal_us_date=sig_us_date,
                        symbol=pos.symbol,
                        role=pos.role,
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

                trip = VixRoundTrip(
                    symbol=pos.symbol,
                    role=pos.role,
                    signal_entry_us_date=pos.signal_entry_us_date,
                    entry_date=pos.entry_date,
                    exit_date=cur_date,
                    raw_shares_entry=pos.raw_shares_entry,
                    adj_shares=pos.adj_shares,
                    entry_price_adj=pos.entry_price_adj,
                    exit_price_adj=fill_adj,
                    bars_held=pos.bars_held,
                    gross_pnl_eur=gross_pnl,
                    commissions_eur=tot_comm,
                    slippage_eur=tot_slip,
                    net_pnl_eur=net_pnl,
                    return_pct=ret_pct,
                    exit_reason=exit_reason,
                    is_open_at_end=False,
                )
                round_trips.append(trip)
                if pos.role == "TAIL_HEDGE":
                    closed_tail_trip_today = trip

                total_commissions += comm
                total_slippage += slip_eur
                total_sell_notional += gross_proceeds
                pos = None

        # Finalize active TAIL_HEDGE episode once target state leaves TAIL_HEDGE AND any held TAIL_HEDGE position is closed
        if active_tail_ep is not None and target_state != VixRegimeState.TAIL_HEDGE.value:
            if pos is None or pos.role != "TAIL_HEDGE":
                tail_episodes.append(
                    _finalize_tail_episode(
                        active_tail_ep,
                        signal_end_us_date=sig_us_date,
                        exit_vix_ratio=vix_ratio_now if math.isfinite(vix_ratio_now) else None,
                        exit_trip=closed_tail_trip_today,
                        is_open_at_end=False,
                    )
                )
                active_tail_ep = None

        # 5. Open Step B: Buy target holding using SETTLED cash only (zero Good Faith Violations).
        # Wait if sale proceeds are currently settling (T+2) so we deploy the full target budget
        # in one clean order without spending unsettled proceeds or fragmenting capital.
        if pos is None and desired_symbol is not None and desired_role is not None:
            unsettled_now = float(sum(unsettled_buckets.values()))
            if unsettled_now > 1e-6:
                settlement_wait_days += 1
                if desired_role == "TAIL_HEDGE":
                    tail_hedge_t2_wait_days += 1
                    if active_tail_ep is not None and active_tail_ep.entry_date is None:
                        active_tail_ep.t2_delay_sessions += 1
                else:
                    risk_on_t2_wait_days += 1
            else:
                df_tgt = asset_frames[desired_symbol]
                if eu_ts in df_tgt.index:
                    bar = df_tgt.loc[eu_ts]
                    open_adj = float(bar["open"])
                    split_f = _split_factor_after(bar, cfg.sizing_mode)
                    fill_adj = open_adj * (1.0 + cfg.slippage_pct)
                    fill_raw = fill_adj * split_f
                    comm = cfg.commission_eur

                    budget = settled_cash * desired_weight
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

                        entry_reason = (
                            "REGIME_RISK_ON"
                            if desired_role == "RISK_ON"
                            else (
                                "REGIME_TAIL_HEDGE_50PCT"
                                if desired_weight < 1.0 - 1e-9
                                else "REGIME_TAIL_HEDGE"
                            )
                        )
                        fills.append(
                            VixOrderFill(
                                date=cur_date,
                                signal_us_date=sig_us_date,
                                symbol=desired_symbol,
                                role=desired_role,
                                side="BUY",
                                raw_shares=raw_shares,
                                adj_shares=adj_shares,
                                split_factor_after=split_f,
                                raw_fill_price=fill_raw,
                                adj_fill_price=fill_adj,
                                notional_eur=notional,
                                commission_eur=comm,
                                slippage_eur=slip_eur,
                                reason=entry_reason,
                            )
                        )
                        pos = _OpenPosition(
                            symbol=desired_symbol,
                            role=desired_role,
                            signal_entry_us_date=sig_us_date,
                            entry_date=cur_date,
                            raw_shares_entry=raw_shares,
                            adj_shares=adj_shares,
                            entry_price_adj=fill_adj,
                            entry_notional_eur=notional,
                            entry_commission_eur=comm,
                            entry_slippage_eur=slip_eur,
                            bars_held=0,
                            episode_id=active_tail_ep.episode_id
                            if (desired_role == "TAIL_HEDGE" and active_tail_ep is not None)
                            else None,
                        )
                        if desired_role == "TAIL_HEDGE" and active_tail_ep is not None:
                            active_tail_ep.entry_date = cur_date
                            active_tail_ep.lvo_open_on_fill_day = open_adj
                            active_tail_ep.raw_shares_entry = raw_shares

                        total_commissions += comm
                        total_slippage += slip_eur
                        total_buy_notional += notional
                    else:
                        skipped_zero_shares += 1

        # 6. Close Mark-to-Market (strictly causal)
        for sym, df in asset_frames.items():
            if eu_ts in df.index:
                last_close[sym] = float(df.loc[eu_ts, "close"])

        if pos is not None:
            pos.bars_held += 1
            mv = pos.adj_shares * last_close[pos.symbol]
            holding_symbol = pos.symbol
            holding_role = pos.role
            open_positions = 1
        else:
            mv = 0.0
            holding_symbol = "EUR_CASH"
            holding_role = "EUR_CASH"
            open_positions = 0

        unsettled_cash = float(sum(unsettled_buckets.values()))
        equity = settled_cash + unsettled_cash + mv
        if settled_cash < -1e-6:
            raise AssertionError(f"Good Faith Violation on {cur_date}: settled_cash={settled_cash}")
        min_settled_cash = min(min_settled_cash, settled_cash)

        w_risk_on = (mv / equity) if (pos is not None and pos.role == "RISK_ON" and equity > 0) else 0.0
        w_tail = (mv / equity) if (pos is not None and pos.role == "TAIL_HEDGE" and equity > 0) else 0.0
        w_cash = ((settled_cash + unsettled_cash) / equity) if equity > 0 else 1.0

        rows.append(
            {
                "date": eu_ts,
                "equity": equity,
                "settled_cash": settled_cash,
                "unsettled_cash": unsettled_cash,
                "market_value": mv,
                "holding_symbol": holding_symbol,
                "holding_role": holding_role,
                "target_state": target_state,
                "signal_us_date": sig_us_ts,
                "vix_close": vix_close_now,
                "vix3m_close": vix3m_close_now,
                "vix_ratio": vix_ratio_now,
                "open_positions": open_positions,
                "w_RISK_ON": w_risk_on,
                "w_TAIL_HEDGE": w_tail,
                "w_CASH": w_cash,
                "benchmark_close": float(bench_series.loc[eu_ts]),
            }
        )
        prev_date = cur_date
        prev_target_state = target_state

    # Snapshot any open position at end_date (mark-to-market, NOT an executed exit fill)
    open_at_end: VixRoundTrip | None = None
    if pos is not None:
        end_close = last_close[pos.symbol]
        mv_end = pos.adj_shares * end_close
        cost_basis = pos.entry_notional_eur + pos.entry_commission_eur
        net_pnl_mtm = mv_end - cost_basis
        entry_open_adj = pos.entry_price_adj / (1.0 + cfg.slippage_pct)
        gross_pnl_mtm = pos.adj_shares * (end_close - entry_open_adj)
        open_at_end = VixRoundTrip(
            symbol=pos.symbol,
            role=pos.role,
            signal_entry_us_date=pos.signal_entry_us_date,
            entry_date=pos.entry_date,
            exit_date=eu_calendar[-1].date(),
            raw_shares_entry=pos.raw_shares_entry,
            adj_shares=pos.adj_shares,
            entry_price_adj=pos.entry_price_adj,
            exit_price_adj=end_close,
            bars_held=pos.bars_held,
            gross_pnl_eur=gross_pnl_mtm,
            commissions_eur=pos.entry_commission_eur,
            slippage_eur=pos.entry_slippage_eur,
            net_pnl_eur=net_pnl_mtm,
            return_pct=net_pnl_mtm / cost_basis if cost_basis > 0 else 0.0,
            exit_reason="OPEN_MTM",
            is_open_at_end=True,
        )

    # Finalize any active TAIL_HEDGE episode still open at end_date
    if active_tail_ep is not None:
        last_vix_ratio = float(aligned["vix_ratio"].iloc[-1])
        tail_episodes.append(
            _finalize_tail_episode(
                active_tail_ep,
                signal_end_us_date=None,
                exit_vix_ratio=last_vix_ratio if math.isfinite(last_vix_ratio) else None,
                exit_trip=open_at_end if (open_at_end is not None and open_at_end.role == "TAIL_HEDGE") else None,
                is_open_at_end=True,
            )
        )

    curve = pd.DataFrame(rows).set_index("date")
    us_win = us_signals.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
    regime_switches_us = int(us_win["switched"].sum()) if len(us_win) else 0

    return VixTermBacktestResult(
        config=cfg,
        initial_capital_eur=float(initial_capital_eur),
        start_date=start_date,
        end_date=end_date,
        equity_curve=curve,
        us_signals=us_signals,
        fills=fills,
        round_trips=round_trips,
        open_position_at_end=open_at_end,
        tail_hedge_episodes=tail_episodes,
        total_commissions_eur=float(total_commissions),
        total_slippage_eur=float(total_slippage),
        total_buy_notional_eur=float(total_buy_notional),
        total_sell_notional_eur=float(total_sell_notional),
        hosting_fees_eur=float(hosting_fees),
        skipped_zero_shares=skipped_zero_shares,
        settlement_wait_days=settlement_wait_days,
        tail_hedge_t2_wait_days=tail_hedge_t2_wait_days,
        risk_on_t2_wait_days=risk_on_t2_wait_days,
        min_settled_cash_eur=float(min_settled_cash),
        regime_switches_us=regime_switches_us,
    )


# ---------------------------------------------------------------------------
# Metrics & Exact Sleeve P&L Attribution
# ---------------------------------------------------------------------------


def compute_sleeve_attribution(res: VixTermBacktestResult) -> dict[str, SleeveAttribution]:
    """Compute exact additive P&L, holding-time, and win-rate attribution per sleeve."""
    curve = res.equity_curve
    n_days = len(curve)
    init = res.initial_capital_eur
    all_trips = list(res.round_trips)
    if res.open_position_at_end is not None:
        all_trips.append(res.open_position_at_end)

    out: dict[str, SleeveAttribution] = {}
    for sleeve, sym, w_col in (
        ("RISK_ON", res.config.risk_on_symbol, "w_RISK_ON"),
        ("TAIL_HEDGE", res.config.tail_hedge_symbol if res.config.tail_hedge_weight > 0 else "NONE", "w_TAIL_HEDGE"),
    ):
        days_held = int((curve["holding_role"] == sleeve).sum()) if n_days > 0 else 0
        avg_w = float(curve[w_col].mean()) if n_days > 0 else 0.0
        closed_s = [t for t in res.round_trips if t.role == sleeve]
        open_s = [res.open_position_at_end] if (res.open_position_at_end and res.open_position_at_end.role == sleeve) else []
        eval_s = [t for t in all_trips if t.role == sleeve]

        gross_pnl = float(sum(t.gross_pnl_eur for t in eval_s))
        comm = float(sum(t.commissions_eur for t in eval_s))
        slip = float(sum(t.slippage_eur for t in eval_s))
        net_pnl = float(sum(t.net_pnl_eur for t in eval_s))
        win_rate = float(np.mean([t.net_pnl_eur > 0.0 for t in eval_s])) if eval_s else None
        avg_bars = float(np.mean([t.bars_held for t in eval_s])) if eval_s else None

        out[sleeve] = SleeveAttribution(
            sleeve=sleeve,  # type: ignore[arg-type]
            symbol=sym,
            days_held=days_held,
            pct_days_held=float(days_held / n_days) if n_days > 0 else 0.0,
            avg_weight_pct=avg_w,
            gross_pnl_eur=gross_pnl,
            commissions_eur=comm,
            slippage_eur=slip,
            net_pnl_eur=net_pnl,
            return_contribution_pct=net_pnl / init if init > 0 else 0.0,
            closed_trades=len(closed_s),
            open_at_end=len(open_s),
            evaluated_trades=len(eval_s),
            win_rate=win_rate,
            avg_bars_held=avg_bars,
        )

    cash_only_days = int((curve["holding_role"] == "EUR_CASH").sum()) if n_days > 0 else 0
    avg_cash_w = float(curve["w_CASH"].mean()) if n_days > 0 else 1.0
    out["EUR_CASH"] = SleeveAttribution(
        sleeve="EUR_CASH",
        symbol="EUR_CASH",
        days_held=cash_only_days,
        pct_days_held=float(cash_only_days / n_days) if n_days > 0 else 0.0,
        avg_weight_pct=avg_cash_w,
        gross_pnl_eur=0.0,
        commissions_eur=0.0,
        slippage_eur=0.0,
        net_pnl_eur=-res.hosting_fees_eur,
        return_contribution_pct=-res.hosting_fees_eur / init if init > 0 else 0.0,
        closed_trades=0,
        open_at_end=0,
        evaluated_trades=0,
        win_rate=None,
        avg_bars_held=None,
    )
    return out


def compute_vix_regime_metrics(
    res: VixTermBacktestResult,
    metrics_cfg: MetricsConfig | None = None,
    *,
    cumulative_trials_n: int = 35,
    cumulative_trials_conservative_n: int = 50,
    include_open_in_win_rate: bool = True,
) -> dict[str, float | int]:
    """Compute standardized EUR performance, risk, standalone & cumulative DSR, and T+2 metrics."""
    mcfg = metrics_cfg or MetricsConfig()
    curve = res.equity_curve
    eq = curve["equity"]
    init = res.initial_capital_eur
    r = daily_returns(eq, init)

    days = (curve.index[-1] - curve.index[0]).days + 1
    years = max(days / 365.25, 1e-6)

    n_fills = len(res.fills)
    n_risk_on_fills = sum(1 for f in res.fills if f.role == "RISK_ON")
    n_tail_fills = sum(1 for f in res.fills if f.role == "TAIL_HEDGE")

    eval_trips = list(res.round_trips)
    if include_open_in_win_rate and res.open_position_at_end is not None:
        eval_trips.append(res.open_position_at_end)

    eval_risk_on = [t for t in eval_trips if t.role == "RISK_ON"]
    eval_tail = [t for t in eval_trips if t.role == "TAIL_HEDGE"]

    win_rate_all = float(np.mean([t.net_pnl_eur > 0.0 for t in eval_trips])) if eval_trips else float("nan")
    win_rate_risk_on = float(np.mean([t.net_pnl_eur > 0.0 for t in eval_risk_on])) if eval_risk_on else float("nan")
    win_rate_tail = float(np.mean([t.net_pnl_eur > 0.0 for t in eval_tail])) if eval_tail else float("nan")

    total_friction = res.total_commissions_eur + res.total_slippage_eur + res.hosting_fees_eur
    avg_deployed = float((curve["market_value"] / curve["equity"].replace(0.0, np.nan)).fillna(0.0).mean())

    n_tail_episodes = len(res.tail_hedge_episodes)
    n_tail_delayed = sum(1 for ep in res.tail_hedge_episodes if ep.t2_delay_sessions > 0)
    n_tail_missed_t2 = sum(1 for ep in res.tail_hedge_episodes if ep.missed_due_to_t2)

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
        "deflated_sharpe_cum35": deflated_sharpe(
            r, mcfg.periods_per_year, cumulative_trials_n, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum50": deflated_sharpe(
            r, mcfg.periods_per_year, cumulative_trials_conservative_n, mcfg.dsr_trial_sharpe_std
        ),
        "max_drawdown": max_drawdown(eq, init),
        "total_orders": n_fills,
        "orders_per_year": float(n_fills / years),
        "risk_on_orders": n_risk_on_fills,
        "tail_hedge_orders": n_tail_fills,
        "closed_round_trips": len(res.round_trips),
        "evaluated_trips": len(eval_trips),
        "win_rate": win_rate_all,
        "win_rate_risk_on": win_rate_risk_on,
        "win_rate_tail_hedge": win_rate_tail,
        "risk_on_exposure_days_pct": float((curve["holding_role"] == "RISK_ON").mean()),
        "tail_hedge_exposure_days_pct": float((curve["holding_role"] == "TAIL_HEDGE").mean()),
        "cash_only_days_pct": float((curve["holding_role"] == "EUR_CASH").mean()),
        "avg_capital_deployed_pct": avg_deployed,
        "target_risk_on_days_pct": float((curve["target_state"] == VixRegimeState.RISK_ON.value).mean()),
        "target_tail_hedge_days_pct": float((curve["target_state"] == VixRegimeState.TAIL_HEDGE.value).mean()),
        "target_neutral_days_pct": float((curve["target_state"] == VixRegimeState.NEUTRAL.value).mean()),
        "commissions_eur": res.total_commissions_eur,
        "slippage_eur": res.total_slippage_eur,
        "hosting_eur": res.hosting_fees_eur,
        "total_friction_eur": total_friction,
        "friction_pct_initial": total_friction / init,
        "regime_switches_us": res.regime_switches_us,
        "skipped_zero_shares": res.skipped_zero_shares,
        "settlement_wait_days": res.settlement_wait_days,
        "tail_hedge_t2_wait_days": res.tail_hedge_t2_wait_days,
        "risk_on_t2_wait_days": res.risk_on_t2_wait_days,
        "tail_hedge_episodes": n_tail_episodes,
        "tail_hedge_episodes_t2_delayed": n_tail_delayed,
        "tail_hedge_episodes_missed_t2": n_tail_missed_t2,
        "min_settled_cash_eur": res.min_settled_cash_eur,
    }
