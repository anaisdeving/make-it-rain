"""Strategy v5: Low-turnover Regime-Filtered Leveraged EU UCITS ETF/ETP Rotation.

Designed for an IBKR Ireland EUR cash account (zero FX conversion fees, EUR 0/mo
cron/local hosting, whole shares only, strict T+2 European ETF settlement).

Core guarantees:
- Strictly zero look-ahead bias, including cross-timezone alignment:
  US ``QQQ`` / ``SPY`` close at 22:00 CET (after European exchanges close at 17:30 CET),
  so a US close on date ``d`` is strictly actionable on the next EU trading day
  ``eu_date > d`` Open (never same-day EU Open). US-only and EU-only holidays are
  handled causally via ``searchsorted(..., side="left") - 1``.
- Split-consistent as-traded whole-share sizing:
  ``raw_open[t] = open_adj[t] * (raw_close[t] / close_adj[t])`` reconstructs the
  historical as-traded EUR share price for whole-share sizing while maintaining exact
  split-adjusted return continuity across stock splits (e.g. CL2.PA 300:1 on 2023-10-10,
  LQQ.PA 200:1 on 2026-07-09).
- Strict T+2 European ETF settlement (zero Good Faith Violations: sale proceeds on EU
  session ``i`` settle before Open on EU session ``i + 2`` and are never spent early).

Locked signal definitions (zero OOS tuning):
- ``v5a`` (2x Leveraged Trend Rotation):
  US daily Close vs ``SMA(200)`` with ``±1.0%`` hysteresis band:
  enter ``RISK_ON`` when ``Close > 1.01 * SMA(200)``, exit to ``RISK_OFF`` when
  ``Close < 0.99 * SMA(200)``, hold prior state inside ``[0.99, 1.01] * SMA(200)``.
- ``v5b`` (3x Nasdaq-100 Asymmetric Trend Rotation):
  Same ``SMA(200) ± 1.0%`` hysteresis band PLUS emergency volatility circuit breaker:
  immediately force ``RISK_OFF`` when 20-day annualised realised volatility ``sigma_20``
  (matching ``src/strategy/regime_filter.py``) exceeds ``0.35``. Re-enter ``RISK_ON``
  only when ``sigma_20 <= 0.35`` AND ``Close > 1.01 * SMA(200)``.
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


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UcitsRegimeState(str, Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"


class RiskOffMode(str, Enum):
    CASH = "cash"
    XEON = "xeon"


class SizingMode(str, Enum):
    AS_TRADED = "as_traded"
    POST_SPLIT = "post_split"


class UcitsRegimeConfig(_Frozen):
    """Frozen configuration for Strategy v5 (v5a / v5b)."""

    name: str = "v5a"
    signal_symbol: str = "QQQ"
    risk_on_symbol: str = "CL2.PA"
    risk_off_mode: RiskOffMode = RiskOffMode.CASH
    risk_off_symbol: str = "XEON.DE"
    sma_window: int = Field(200, ge=2)
    hysteresis_pct: float = Field(0.01, ge=0.0, lt=0.25)
    vol_window: int = Field(20, ge=2)
    vol_circuit_breaker: float | None = Field(None, gt=0.0)
    commission_eur: float = Field(1.25, ge=0.0)
    slippage_pct: float = Field(0.001, ge=0.0, lt=0.1)
    settlement_days: int = Field(2, ge=0, le=5)
    monthly_hosting_eur: float = Field(0.0, ge=0.0)
    sizing_mode: SizingMode = SizingMode.AS_TRADED

    @model_validator(mode="after")
    def _validate_symbols(self) -> "UcitsRegimeConfig":
        if not self.signal_symbol.strip():
            raise ValueError("signal_symbol must be non-empty")
        if not self.risk_on_symbol.strip():
            raise ValueError("risk_on_symbol must be non-empty")
        if self.risk_off_mode is RiskOffMode.XEON and not self.risk_off_symbol.strip():
            raise ValueError("risk_off_symbol must be non-empty when risk_off_mode='xeon'")
        return self


class UcitsOrderFill(_Frozen):
    """One executed EU Open fill (buy or sell) in EUR."""

    date: date
    signal_us_date: date
    symbol: str
    role: Literal["RISK_ON", "RISK_OFF"]
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


class UcitsRoundTrip(_Frozen):
    """Completed (or end-of-window open MTM) position lifecycle in EUR."""

    symbol: str
    role: Literal["RISK_ON", "RISK_OFF"]
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


@dataclass
class _OpenPosition:
    symbol: str
    role: Literal["RISK_ON", "RISK_OFF"]
    signal_entry_us_date: date
    entry_date: date
    raw_shares_entry: int
    adj_shares: float
    entry_price_adj: float
    entry_notional_eur: float
    entry_commission_eur: float
    entry_slippage_eur: float
    bars_held: int = 0


@dataclass(frozen=True)
class UcitsBacktestResult:
    """Full output of a Strategy v5 EUR cash-account simulation."""

    config: UcitsRegimeConfig
    initial_capital_eur: float
    start_date: date
    end_date: date
    equity_curve: pd.DataFrame
    us_signals: pd.DataFrame
    fills: list[UcitsOrderFill]
    round_trips: list[UcitsRoundTrip]
    open_position_at_end: UcitsRoundTrip | None
    total_commissions_eur: float
    total_slippage_eur: float
    total_buy_notional_eur: float
    total_sell_notional_eur: float
    hosting_fees_eur: float
    skipped_zero_shares: int
    settlement_wait_days: int
    min_settled_cash_eur: float
    regime_switches_us: int


# ---------------------------------------------------------------------------
# Pure Signal Generator (US Close) & Cross-Timezone Alignment (US -> Next EU Open)
# ---------------------------------------------------------------------------


def compute_us_regime_signals(
    signal_bars: BarSeries,
    cfg: UcitsRegimeConfig,
) -> pd.DataFrame:
    """Compute strictly causal US-close regime signals (SMA200 ± hysteresis + optional vol breaker).

    ``sigma_20`` matches ``src/strategy/regime_filter.py`` exactly:
    ``np.log(close).diff().rolling(vol_window, min_periods=vol_window).std() * sqrt(252)``.

    State transition rules on US bar ``t``:
    - If ``sma200`` is NaN (or ``vol_circuit_breaker`` is enabled and ``sigma_20`` is NaN):
      ``RISK_OFF``.
    - Else if ``vol_circuit_breaker`` is enabled and ``sigma_20 > vol_circuit_breaker``:
      immediately force ``RISK_OFF`` (``vol_breaker_active = True``).
    - Else:
      - From ``RISK_ON``: exit to ``RISK_OFF`` iff ``close < (1 - hysteresis_pct) * sma200``.
      - From ``RISK_OFF``: enter ``RISK_ON`` iff ``close > (1 + hysteresis_pct) * sma200``.
      - Inside the hysteresis band, hold the previous state.
    """
    close = signal_bars.frame["close"]
    sma = close.rolling(cfg.sma_window, min_periods=cfg.sma_window).mean()
    sigma = (
        np.log(close)
        .diff()
        .rolling(cfg.vol_window, min_periods=cfg.vol_window)
        .std()
        * math.sqrt(252)
    )

    enter_mult = 1.0 + cfg.hysteresis_pct
    exit_mult = 1.0 - cfg.hysteresis_pct
    vol_cap = cfg.vol_circuit_breaker

    n = len(close)
    states: list[str] = []
    vol_flags: list[bool] = []
    switched: list[bool] = []
    prev_on = False

    close_vals = close.to_numpy(dtype=float)
    sma_vals = sma.to_numpy(dtype=float)
    sig_vals = sigma.to_numpy(dtype=float)

    for i in range(n):
        c = close_vals[i]
        m = sma_vals[i]
        v = sig_vals[i]
        vol_active = False

        if math.isnan(m) or (vol_cap is not None and math.isnan(v)):
            cur_on = False
        elif vol_cap is not None and v > vol_cap:
            cur_on = False
            vol_active = True
        elif prev_on:
            cur_on = not (c < exit_mult * m)
        else:
            cur_on = c > enter_mult * m

        states.append(UcitsRegimeState.RISK_ON.value if cur_on else UcitsRegimeState.RISK_OFF.value)
        vol_flags.append(vol_active)
        switched.append(i > 0 and cur_on != prev_on)
        prev_on = cur_on

    return pd.DataFrame(
        {
            "close": close_vals,
            "sma200": sma_vals,
            "sigma_20": sig_vals,
            "state": states,
            "vol_breaker_active": vol_flags,
            "switched": switched,
        },
        index=close.index,
    )


def align_us_signals_to_eu_calendar(
    us_signals: pd.DataFrame,
    eu_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each EU session ``e`` to the latest US close strictly BEFORE ``e`` (``us_date < e``).

    Because US exchanges close at ~22:00 CET (after European exchanges close at 17:30 CET),
    a US close on date ``d`` is NEVER actionable on EU date ``d``; it is strictly first
    actionable on the next EU trading session ``e > d`` Open.

    Uses ``searchsorted(eu_calendar, side="left") - 1`` which guarantees strict
    inequality ``us_index[pos] < eu_ts`` across all weekends, US-only holidays, and
    EU-only holidays.
    """
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
    vol_flags: list[bool] = []
    us_closes: list[float] = []
    us_sma200s: list[float] = []
    us_sigmas: list[float] = []

    state_arr = us_signals["state"].to_numpy()
    vol_arr = us_signals["vol_breaker_active"].to_numpy(dtype=bool)
    close_arr = us_signals["close"].to_numpy(dtype=float)
    sma_arr = us_signals["sma200"].to_numpy(dtype=float)
    sig_arr = us_signals["sigma_20"].to_numpy(dtype=float)

    for eu_ts, pos in zip(eu_calendar, positions):
        if pos < 0:
            sig_dates.append(pd.NaT)
            target_states.append(UcitsRegimeState.RISK_OFF.value)
            vol_flags.append(False)
            us_closes.append(float("nan"))
            us_sma200s.append(float("nan"))
            us_sigmas.append(float("nan"))
        else:
            us_ts = us_idx[pos]
            if not (us_ts < eu_ts):
                raise AssertionError(f"look-ahead violation: us_ts={us_ts} >= eu_ts={eu_ts}")
            sig_dates.append(us_ts)
            target_states.append(str(state_arr[pos]))
            vol_flags.append(bool(vol_arr[pos]))
            us_closes.append(float(close_arr[pos]))
            us_sma200s.append(float(sma_arr[pos]))
            us_sigmas.append(float(sig_arr[pos]))

    return pd.DataFrame(
        {
            "signal_us_date": sig_dates,
            "target_state": target_states,
            "vol_breaker_active": vol_flags,
            "us_close": us_closes,
            "us_sma200": us_sma200s,
            "us_sigma_20": us_sigmas,
        },
        index=eu_calendar,
    )


# ---------------------------------------------------------------------------
# Split Factor & Calendar Helpers
# ---------------------------------------------------------------------------


def _split_factor_after(row: pd.Series, sizing_mode: SizingMode) -> float:
    """Return multiplier converting split-adjusted price to as-traded price on this bar."""
    if sizing_mode is SizingMode.POST_SPLIT:
        return 1.0
    c = float(row["close"])
    rc = float(row["raw_close"])
    if c <= 0 or math.isnan(rc) or rc <= 0:
        return 1.0
    return rc / c


def _build_eu_calendar(
    risk_on_bars: BarSeries,
    risk_off_bars: BarSeries | None,
    start_date: date,
    end_date: date,
) -> pd.DatetimeIndex:
    idx = risk_on_bars.frame.index
    if risk_off_bars is not None:
        idx = idx.union(risk_off_bars.frame.index)
    idx = idx.sort_values().unique()
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
    union_idx = benchmark_bars.frame.index.union(eu_calendar).sort_values()
    return benchmark_bars.frame["close"].reindex(union_idx).ffill().reindex(eu_calendar)


def _months_between_crossed(prev_date: date | None, cur_date: date) -> int:
    if prev_date is None:
        return 0
    return (cur_date.year - prev_date.year) * 12 + (cur_date.month - prev_date.month)


# ---------------------------------------------------------------------------
# T+2 EUR Cash-Account Simulator
# ---------------------------------------------------------------------------


def run_ucits_regime_backtest(
    signal_bars: BarSeries,
    risk_on_bars: BarSeries,
    cfg: UcitsRegimeConfig,
    *,
    start_date: date,
    end_date: date,
    initial_capital_eur: float = 1000.0,
    risk_off_bars: BarSeries | None = None,
    benchmark_bars: BarSeries | None = None,
) -> UcitsBacktestResult:
    """Simulate Strategy v5 on an IBKR Ireland EUR cash account with strict T+2 settlement.

    Execution sequence on each EU trading session ``i`` (date ``eu_ts``):
    1. Move unsettled sale proceeds scheduled for ``day_idx <= i`` into ``settled_cash``.
    2. Deduct monthly hosting fee on month boundaries (``0.0`` for Strategy v5).
    3. Read pre-aligned ``target_state`` from the latest US Close strictly before ``eu_ts``.
    4. If holding a non-target ETF and its exchange is open today, sell all shares at
       ``Open * (1 - slippage_pct)`` minus ``commission_eur``; net proceeds enter T+2
       unsettled bucket ``i + settlement_days``.
    5. If flat and target requires an ETF (``RISK_ON`` ETF, or ``XEON.DE`` when
       ``risk_off_mode == 'xeon'``) and its exchange is open today, buy the maximum
       whole number of as-traded shares affordable from ``settled_cash`` only (never
       spending unsettled cash — strictly zero Good Faith Violations).
    6. Mark-to-market at session Close using causal last-known Close for held asset.
    """
    if initial_capital_eur <= 0:
        raise ValueError("initial_capital_eur must be positive")
    if cfg.risk_off_mode is RiskOffMode.XEON and risk_off_bars is None:
        raise ValueError("risk_off_bars is required when risk_off_mode='xeon'")

    # Truncate US signals strictly to <= end_date so future data never touches anything
    us_frame_cut = signal_bars.frame.loc[: pd.Timestamp(end_date)]
    us_signals = compute_us_regime_signals(BarSeries(symbol=signal_bars.symbol, frame=us_frame_cut), cfg)

    active_risk_off_bars = risk_off_bars if cfg.risk_off_mode is RiskOffMode.XEON else None
    eu_calendar = _build_eu_calendar(risk_on_bars, active_risk_off_bars, start_date, end_date)
    aligned = align_us_signals_to_eu_calendar(us_signals, eu_calendar)
    bench_series = _causal_benchmark_series(benchmark_bars, eu_calendar)

    asset_frames: dict[str, pd.DataFrame] = {cfg.risk_on_symbol: risk_on_bars.frame}
    if active_risk_off_bars is not None:
        asset_frames[cfg.risk_off_symbol] = active_risk_off_bars.frame

    # Causal last-known close per asset (initialized from prior warmup bars strictly < start_date)
    last_close: dict[str, float] = {}
    for sym, df in asset_frames.items():
        prior = df.loc[df.index < eu_calendar[0], "close"]
        if len(prior):
            last_close[sym] = float(prior.iloc[-1])

    settled_cash = float(initial_capital_eur)
    unsettled_buckets: dict[int, float] = {}
    pos: _OpenPosition | None = None

    fills: list[UcitsOrderFill] = []
    round_trips: list[UcitsRoundTrip] = []
    total_commissions = 0.0
    total_slippage = 0.0
    total_buy_notional = 0.0
    total_sell_notional = 0.0
    hosting_fees = 0.0
    skipped_zero_shares = 0
    settlement_wait_days = 0
    min_settled_cash = settled_cash

    rows: list[dict[str, object]] = []
    prev_date: date | None = None

    for day_idx, eu_ts in enumerate(eu_calendar):
        cur_date = eu_ts.date()

        # 1. Settle T+2 proceeds ready by day_idx
        ready_keys = [k for k in unsettled_buckets if k <= day_idx]
        for k in ready_keys:
            settled_cash += unsettled_buckets.pop(k)

        # 2. Monthly hosting fee (EUR 0.00 for Strategy v5)
        if cfg.monthly_hosting_eur > 0.0:
            months = _months_between_crossed(prev_date, cur_date)
            if months > 0:
                fee = months * cfg.monthly_hosting_eur
                settled_cash = max(0.0, settled_cash - fee)
                hosting_fees += fee

        # 3. Target state strictly from prior US close (< eu_ts)
        al_row = aligned.loc[eu_ts]
        sig_us_ts = al_row["signal_us_date"]
        sig_us_date = sig_us_ts.date() if pd.notna(sig_us_ts) else cur_date
        target_state = str(al_row["target_state"])
        vol_active = bool(al_row["vol_breaker_active"])

        if target_state == UcitsRegimeState.RISK_ON.value:
            desired_symbol: str | None = cfg.risk_on_symbol
            desired_role: Literal["RISK_ON", "RISK_OFF"] | None = "RISK_ON"
        elif cfg.risk_off_mode is RiskOffMode.XEON:
            desired_symbol = cfg.risk_off_symbol
            desired_role = "RISK_OFF"
        else:
            desired_symbol = None
            desired_role = None

        # 4. Open Step A: Liquidate non-target holding if its exchange is open today
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
                exit_reason = (
                    "VOL_BREAKER_EXIT"
                    if (pos.role == "RISK_ON" and vol_active)
                    else ("REGIME_RISK_OFF" if pos.role == "RISK_ON" else "ROTATE_TO_RISK_ON")
                )

                fills.append(
                    UcitsOrderFill(
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

                gross_pnl = pos.adj_shares * (open_adj - (pos.entry_price_adj / (1.0 + cfg.slippage_pct)))
                tot_comm = pos.entry_commission_eur + comm
                tot_slip = pos.entry_slippage_eur + slip_eur
                net_pnl = (gross_proceeds - comm) - (pos.entry_notional_eur + pos.entry_commission_eur)
                ret_pct = net_pnl / (pos.entry_notional_eur + pos.entry_commission_eur)

                round_trips.append(
                    UcitsRoundTrip(
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
                )

                total_commissions += comm
                total_slippage += slip_eur
                total_sell_notional += gross_proceeds
                pos = None

        # 5. Open Step B: Buy target holding using SETTLED cash only (zero GFV).
        # Wait if sale proceeds are currently settling (T+2) so we deploy full capital
        # in one clean order rather than buying a stub with leftover cash and stranding proceeds.
        if pos is None and desired_symbol is not None and desired_role is not None:
            unsettled_now = sum(unsettled_buckets.values())
            if unsettled_now > 1e-6:
                settlement_wait_days += 1
            else:
                df_tgt = asset_frames[desired_symbol]
                if eu_ts in df_tgt.index:
                    bar = df_tgt.loc[eu_ts]
                    open_adj = float(bar["open"])
                    split_f = _split_factor_after(bar, cfg.sizing_mode)
                    fill_adj = open_adj * (1.0 + cfg.slippage_pct)
                    fill_raw = fill_adj * split_f
                    comm = cfg.commission_eur

                    available_for_shares = settled_cash - comm
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

                        entry_reason = "REGIME_RISK_ON" if desired_role == "RISK_ON" else "PARK_XEON"
                        fills.append(
                            UcitsOrderFill(
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
                        )
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
            risk_on_exposure = 1.0 if pos.role == "RISK_ON" else 0.0
        else:
            mv = 0.0
            holding_symbol = "EUR_CASH"
            holding_role = "CASH"
            open_positions = 0
            risk_on_exposure = 0.0

        unsettled_cash = float(sum(unsettled_buckets.values()))
        equity = settled_cash + unsettled_cash + mv
        min_settled_cash = min(min_settled_cash, settled_cash)

        if eu_ts in risk_on_bars.frame.index:
            ro_bar = risk_on_bars.frame.loc[eu_ts]
            as_traded_ro_px = float(ro_bar["close"]) * _split_factor_after(ro_bar, cfg.sizing_mode)
        else:
            as_traded_ro_px = float("nan")

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
                "open_positions": open_positions,
                "risk_on_exposure": risk_on_exposure,
                "vol_breaker_active": vol_active,
                "benchmark_close": float(bench_series.loc[eu_ts]),
                "as_traded_risk_on_price": as_traded_ro_px,
            }
        )
        prev_date = cur_date

    # Snapshot any open position at end_date (mark-to-market, NOT an executed exit fill)
    open_at_end: UcitsRoundTrip | None = None
    if pos is not None:
        end_close = last_close[pos.symbol]
        mv_end = pos.adj_shares * end_close
        cost_basis = pos.entry_notional_eur + pos.entry_commission_eur
        net_pnl_mtm = mv_end - cost_basis
        gross_pnl_mtm = pos.adj_shares * (end_close - (pos.entry_price_adj / (1.0 + cfg.slippage_pct)))
        open_at_end = UcitsRoundTrip(
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

    curve = pd.DataFrame(rows).set_index("date")
    us_win = us_signals.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
    regime_switches_us = int(us_win["switched"].sum()) if len(us_win) else 0

    return UcitsBacktestResult(
        config=cfg,
        initial_capital_eur=float(initial_capital_eur),
        start_date=start_date,
        end_date=end_date,
        equity_curve=curve,
        us_signals=us_signals,
        fills=fills,
        round_trips=round_trips,
        open_position_at_end=open_at_end,
        total_commissions_eur=float(total_commissions),
        total_slippage_eur=float(total_slippage),
        total_buy_notional_eur=float(total_buy_notional),
        total_sell_notional_eur=float(total_sell_notional),
        hosting_fees_eur=float(hosting_fees),
        skipped_zero_shares=skipped_zero_shares,
        settlement_wait_days=settlement_wait_days,
        min_settled_cash_eur=float(min_settled_cash),
        regime_switches_us=regime_switches_us,
    )


def run_buy_and_hold_etf(
    etf_bars: BarSeries,
    *,
    start_date: date,
    end_date: date,
    initial_capital_eur: float = 1000.0,
    commission_eur: float = 1.25,
    slippage_pct: float = 0.001,
    sizing_mode: SizingMode = SizingMode.AS_TRADED,
    benchmark_bars: BarSeries | None = None,
) -> UcitsBacktestResult:
    """Whole-share EUR Buy & Hold baseline with identical IBKR commission/slippage rules.

    Buys as many whole as-traded shares as affordable on the first EU session where
    ``settled_cash >= raw_open * (1 + slippage_pct) + commission_eur`` and holds to end.
    """
    # Construct a strictly always-RISK_ON causal signal guaranteed active before start_date
    t0 = pd.Timestamp(start_date)
    pre_idx = pd.DatetimeIndex(
        [t0 - pd.Timedelta(days=3), t0 - pd.Timedelta(days=2), t0 - pd.Timedelta(days=1)]
    )
    sig_idx = pre_idx.union(etf_bars.frame.index).sort_values().unique()
    ramp = np.linspace(100.0, 1000.0, len(sig_idx))
    sig_df = pd.DataFrame(
        {
            "open": ramp,
            "high": ramp,
            "low": ramp,
            "close": ramp,
            "volume": np.full(len(sig_idx), 1_000_000.0),
            "raw_close": ramp,
        },
        index=sig_idx,
    )
    cfg = UcitsRegimeConfig(
        name=f"B&H_{etf_bars.symbol}",
        signal_symbol=etf_bars.symbol,
        risk_on_symbol=etf_bars.symbol,
        risk_off_mode=RiskOffMode.CASH,
        sma_window=2,
        hysteresis_pct=0.0,
        vol_window=2,
        vol_circuit_breaker=None,
        commission_eur=commission_eur,
        slippage_pct=slippage_pct,
        settlement_days=2,
        sizing_mode=sizing_mode,
    )
    return run_ucits_regime_backtest(
        signal_bars=BarSeries(symbol=etf_bars.symbol, frame=sig_df),
        risk_on_bars=etf_bars,
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
        initial_capital_eur=initial_capital_eur,
        benchmark_bars=benchmark_bars,
    )


# ---------------------------------------------------------------------------
# Metrics Summary (Reusing src/backtest/metrics.py)
# ---------------------------------------------------------------------------


def compute_ucits_metrics(
    res: UcitsBacktestResult,
    metrics_cfg: MetricsConfig | None = None,
    *,
    include_open_in_win_rate: bool = True,
) -> dict[str, float | int]:
    """Compute standardized EUR performance, risk, DSR, turnover, and friction metrics."""
    mcfg = metrics_cfg or MetricsConfig()
    curve = res.equity_curve
    eq = curve["equity"]
    init = res.initial_capital_eur
    r = daily_returns(eq, init)

    days = (curve.index[-1] - curve.index[0]).days + 1
    years = max(days / 365.25, 1e-6)

    n_fills = len(res.fills)
    n_risk_on_fills = sum(1 for f in res.fills if f.role == "RISK_ON")
    closed_all = list(res.round_trips)
    closed_risk_on = [t for t in closed_all if t.role == "RISK_ON"]

    eval_trips = list(closed_all)
    if include_open_in_win_rate and res.open_position_at_end is not None:
        eval_trips.append(res.open_position_at_end)

    eval_risk_on = [t for t in eval_trips if t.role == "RISK_ON"]

    win_rate_all = (
        float(np.mean([t.net_pnl_eur > 0 for t in eval_trips])) if eval_trips else float("nan")
    )
    win_rate_risk_on = (
        float(np.mean([t.net_pnl_eur > 0 for t in eval_risk_on])) if eval_risk_on else float("nan")
    )

    total_friction = res.total_commissions_eur + res.total_slippage_eur + res.hosting_fees_eur
    avg_invested_pct = float((curve["market_value"] / curve["equity"].replace(0.0, np.nan)).fillna(0.0).mean())
    risk_on_exposure = float(curve["risk_on_exposure"].mean())
    any_invested_exposure = float((curve["open_positions"] > 0).mean())

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
        "max_drawdown": max_drawdown(eq, init),
        "total_orders": n_fills,
        "orders_per_year": float(n_fills / years),
        "risk_on_orders": n_risk_on_fills,
        "risk_on_orders_per_year": float(n_risk_on_fills / years),
        "closed_round_trips": len(closed_all),
        "closed_risk_on_trips": len(closed_risk_on),
        "evaluated_trips": len(eval_trips),
        "win_rate": win_rate_all,
        "win_rate_risk_on": win_rate_risk_on,
        "risk_on_exposure": risk_on_exposure,
        "any_invested_exposure": any_invested_exposure,
        "avg_capital_deployed_pct": avg_invested_pct,
        "commissions_eur": res.total_commissions_eur,
        "slippage_eur": res.total_slippage_eur,
        "hosting_eur": res.hosting_fees_eur,
        "total_friction_eur": total_friction,
        "friction_pct_initial": total_friction / init,
        "regime_switches_us": res.regime_switches_us,
        "skipped_zero_shares": res.skipped_zero_shares,
        "settlement_wait_days": res.settlement_wait_days,
        "min_settled_cash_eur": res.min_settled_cash_eur,
    }


def compute_theoretical_bh_metrics(
    bars: BarSeries,
    start_date: date,
    end_date: date,
    initial_capital: float = 1000.0,
    metrics_cfg: MetricsConfig | None = None,
) -> dict[str, float]:
    """Fractional (zero-cash-drag, zero-fee) Buy & Hold reference directly from split-adjusted close."""
    mcfg = metrics_cfg or MetricsConfig()
    df = bars.frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
    if df.empty:
        raise ValueError(f"No bars for {bars.symbol} in [{start_date}, {end_date}]")
    # Enter at first bar's open to match Strategy v5 open-entry timing
    open0 = float(df["open"].iloc[0])
    eq = df["close"] / open0 * initial_capital
    r = daily_returns(eq, initial_capital)
    return {
        "initial_eur": initial_capital,
        "final_eur": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial_capital - 1.0),
        "cagr": cagr(eq, initial_capital),
        "sharpe": sharpe(r, mcfg.periods_per_year, mcfg.risk_free_rate),
        "sortino": sortino(r, mcfg.periods_per_year, mcfg.risk_free_rate),
        "deflated_sharpe": deflated_sharpe(
            r, mcfg.periods_per_year, mcfg.dsr_n_trials, mcfg.dsr_trial_sharpe_std
        ),
        "max_drawdown": max_drawdown(eq, initial_capital),
    }
