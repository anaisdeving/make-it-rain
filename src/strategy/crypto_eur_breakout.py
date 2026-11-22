"""Strategy v8b-clean: Bias-Corrected, Low-Frequency EUR Spot Crypto Trend Breakout.

Designed for Bitvavo EUR Spot (Tier 1: 0.25% taker fee + 0.10% slippage = 0.35%/side,
0.70% round-trip, fractional EUR sizing with EUR 10 min order, T+0 instant settlement,
EUR 0.00/mo hosting), with Kraken Pro Tier 1 fee sensitivity (0.80% taker + 0.10% slippage).

Locked Canonical Trend-Following Rule Set (zero grid search, zero OOS tuning):
- **Macro Regime Filter:** ``BTC-EUR`` daily close must be strictly above ``SMA(100)``
  (``Close_t > SMA100_t``) to open any new position.
- **Entry Trigger (day ``t`` close -> execute day ``t+1`` open):**
  ``Close_t > Prior_20d_High`` (``max(High_{t-20..t-1})``, strictly excluding day ``t``)
  AND ``Close_t > SMA(100)``.
  Tie-breaker when more candidates trigger than open slots: rank descending by
  ``Return_20d / (ATR14 / Close_t)`` (deterministic alphabetical symbol tie-breaker).
- **Exit Trigger (day ``t`` close -> execute day ``t+1`` open):**
  Exit if ``Close_t < Prior_10d_Low`` (``min(Low_{t-10..t-1})``, strictly excluding day ``t``)
  OR ``Close_t < HighestCloseSinceEntry - 2.5 * ATR14``
  OR ``BTC-EUR Close_t < BTC-EUR SMA(100)_t``.
- **Intra-Day Hard Catastrophe Stop:**
  ``EntryFillPrice - 3.0 * EntryATR14`` evaluated on every held bar (including entry bar
  after open): if ``Open <= stop`` (on held bars after entry day), fill at ``Open``
  (honoring overnight/next-bar gap-through-stop, NEVER assuming fills inside a gap);
  else if ``Low <= stop``, fill at ``stop``. Apply sell fee + slippage to every stop fill.

Variants:
- ``v8b_majors``: Universe A (``BTC-EUR``, ``ETH-EUR``), max 2 concurrent slots (50% equity each).
- ``v8b_top10_slots2``: Universe B (Nov 2021 Top 10), max 2 concurrent slots (50% equity each).
- ``v8b_top10_slots3``: Universe B (Nov 2021 Top 10), max 3 concurrent slots (33.3% equity each).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

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
from src.strategy.indicators import wilder_atr

CRYPTO_PERIODS_PER_YEAR = 365

ExitCategory = Literal["HARD_STOP", "SIGNAL_EXIT", "OPEN_MTM"]
ExitReason = Literal[
    "HARD_STOP_GAP",
    "HARD_STOP_INTRADAY",
    "BTC_MACRO_EXIT",
    "DONCHIAN_10D_LOW",
    "ATR_TRAILING_STOP",
    "OPEN_MTM",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CryptoBreakoutConfig(_Frozen):
    """Frozen configuration for Strategy v8b-clean."""

    name: str = "v8b_majors"
    universe: tuple[str, ...] = ("BTC-EUR", "ETH-EUR")
    btc_symbol: str = "BTC-EUR"
    max_slots: int = Field(2, ge=1, le=20)
    sma_window: int = Field(100, ge=2)
    entry_high_window: int = Field(20, ge=2)
    exit_low_window: int = Field(10, ge=2)
    return_rank_window: int = Field(20, ge=2)
    atr_window: int = Field(14, ge=2)
    trail_atr_mult: float = Field(2.5, gt=0.0)
    hard_stop_atr_mult: float = Field(3.0, gt=0.0)
    taker_fee_pct: float = Field(0.0025, ge=0.0, lt=0.1)
    slippage_pct: float = Field(0.0010, ge=0.0, lt=0.1)
    min_order_eur: float = Field(10.0, ge=0.0)
    monthly_hosting_eur: float = Field(0.0, ge=0.0)
    fee_tier_label: str = "Bitvavo EUR Tier 1 (0.25% taker + 0.10% slip)"


class CryptoOrderFill(_Frozen):
    """One executed EUR spot crypto order fill."""

    date: date
    signal_date: date
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: float = Field(gt=0.0)
    raw_price: float = Field(gt=0.0)
    fill_price: float = Field(gt=0.0)
    notional_eur: float = Field(gt=0.0)
    fee_eur: float = Field(ge=0.0)
    slippage_eur: float = Field(ge=0.0)
    reason: str


class CryptoRoundTrip(_Frozen):
    """Completed (or end-of-window open MTM) spot crypto position lifecycle in EUR."""

    symbol: str
    signal_entry_date: date
    entry_date: date
    exit_date: date
    qty: float = Field(gt=0.0)
    entry_raw_price: float = Field(gt=0.0)
    entry_fill_price: float = Field(gt=0.0)
    entry_atr14: float = Field(gt=0.0)
    hard_stop_price: float = Field(ge=0.0)
    exit_raw_price: float = Field(gt=0.0)
    exit_fill_price: float = Field(gt=0.0)
    highest_close: float = Field(gt=0.0)
    bars_held: int = Field(ge=0)
    cost_basis_eur: float = Field(gt=0.0)
    gross_pnl_eur: float
    fees_eur: float
    slippage_eur: float
    net_pnl_eur: float
    return_pct: float
    exit_reason: ExitReason
    exit_category: ExitCategory
    is_open_at_end: bool = False


class CoinAttribution(_Frozen):
    """Exact additive P&L, trade, and exit-type attribution for one coin."""

    symbol: str
    days_held: int
    pct_days_held: float
    avg_weight_pct: float
    closed_trades: int
    open_at_end: int
    evaluated_trades: int
    hard_stop_exits: int
    signal_exits: int
    win_rate: float | None
    profit_factor: float | None
    avg_bars_held: float | None
    gross_pnl_eur: float
    fees_eur: float
    slippage_eur: float
    net_pnl_eur: float
    return_contribution_pct: float


@dataclass
class _OpenCryptoPos:
    symbol: str
    signal_entry_date: date
    entry_date: date
    qty: float
    entry_raw_price: float
    entry_fill_price: float
    entry_atr14: float
    hard_stop_price: float
    entry_notional_eur: float
    entry_fee_eur: float
    entry_slippage_eur: float
    highest_close: float = 0.0
    bars_held: int = 0


@dataclass(frozen=True)
class CryptoBacktestResult:
    """Full output of a Strategy v8b-clean EUR spot crypto simulation."""

    config: CryptoBreakoutConfig
    initial_capital_eur: float
    start_date: date
    end_date: date
    equity_curve: pd.DataFrame
    fills: list[CryptoOrderFill]
    round_trips: list[CryptoRoundTrip]
    open_positions_at_end: list[CryptoRoundTrip]
    total_fees_eur: float
    total_slippage_eur: float
    hosting_fees_eur: float
    total_buy_notional_eur: float
    total_sell_notional_eur: float
    skipped_min_order: int
    min_cash_eur: float


# ---------------------------------------------------------------------------
# Strictly Causal Indicator Computation
# ---------------------------------------------------------------------------


def compute_coin_indicators(bars: BarSeries, cfg: CryptoBreakoutConfig) -> pd.DataFrame:
    """Compute strictly causal daily indicators for one crypto pair.

    Every value on row ``t`` depends strictly on bars dated ``<= t``.
    ``prior_high`` is ``max(High_{t-entry_high_window..t-1})`` (strictly excluding day ``t``).
    ``prior_low`` is ``min(Low_{t-exit_low_window..t-1})`` (strictly excluding day ``t``).
    """
    df = bars.frame
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    sma100 = close.rolling(cfg.sma_window, min_periods=cfg.sma_window).mean()
    prior_high = high.shift(1).rolling(cfg.entry_high_window, min_periods=cfg.entry_high_window).max()
    prior_low = low.shift(1).rolling(cfg.exit_low_window, min_periods=cfg.exit_low_window).min()
    atr14 = wilder_atr(df, cfg.atr_window)
    ret_20d = close / close.shift(cfg.return_rank_window) - 1.0

    norm_atr = atr14 / close.replace(0.0, np.nan)
    rank_score = ret_20d / norm_atr.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "open": df["open"].astype(float),
            "high": high,
            "low": low,
            "close": close,
            "sma100": sma100,
            "prior_high": prior_high,
            "prior_low": prior_low,
            "atr14": atr14,
            "ret_20d": ret_20d,
            "rank_score": rank_score,
        },
        index=df.index,
    )


def _build_crypto_calendar(
    btc_frame: pd.DataFrame,
    universe_frames: dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
) -> pd.DatetimeIndex:
    idx = btc_frame.index
    for df in universe_frames.values():
        idx = idx.union(df.index)
    idx = idx.sort_values().unique()
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    cal = idx[(idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))]
    if len(cal) == 0:
        raise ValueError(f"No crypto sessions in [{start_date}, {end_date}]")
    return cal


def _months_between_crossed(prev_date: date | None, cur_date: date) -> int:
    if prev_date is None:
        return 0
    return (cur_date.year - prev_date.year) * 12 + (cur_date.month - prev_date.month)


def _exit_category(reason: ExitReason) -> ExitCategory:
    if reason in ("HARD_STOP_GAP", "HARD_STOP_INTRADAY"):
        return "HARD_STOP"
    if reason == "OPEN_MTM":
        return "OPEN_MTM"
    return "SIGNAL_EXIT"


# ---------------------------------------------------------------------------
# Strategy v8b-clean Simulator (T+0 Instant Settlement, Fractional EUR Sizing)
# ---------------------------------------------------------------------------


def run_crypto_breakout_backtest(
    bars_by_symbol: dict[str, BarSeries],
    cfg: CryptoBreakoutConfig,
    *,
    start_date: date,
    end_date: date,
    initial_capital_eur: float = 1000.0,
) -> CryptoBacktestResult:
    """Simulate Strategy v8b-clean on Bitvavo (or Kraken sensitivity) EUR Spot."""
    if initial_capital_eur <= 0.0:
        raise ValueError("initial_capital_eur must be positive")
    if cfg.btc_symbol not in bars_by_symbol:
        raise ValueError(f"Macro anchor {cfg.btc_symbol!r} missing from bars_by_symbol")

    active_universe = [sym for sym in cfg.universe if sym in bars_by_symbol]
    if not active_universe:
        raise ValueError("None of cfg.universe symbols are present in bars_by_symbol")

    end_ts = pd.Timestamp(end_date)

    # 1. Strictly truncate all input bars to <= end_date BEFORE computing any indicator
    ind_by_sym: dict[str, pd.DataFrame] = {}
    all_needed = set(active_universe) | {cfg.btc_symbol}
    for sym in sorted(all_needed):
        cut_frame = bars_by_symbol[sym].frame.loc[:end_ts]
        cut_series = BarSeries(symbol=sym, frame=cut_frame)
        ind_by_sym[sym] = compute_coin_indicators(cut_series, cfg)

    btc_ind = ind_by_sym[cfg.btc_symbol]
    univ_frames = {sym: ind_by_sym[sym] for sym in active_universe}
    calendar = _build_crypto_calendar(btc_ind, univ_frames, start_date, end_date)

    # Causal last-known close initialized strictly < start_date
    last_close: dict[str, float] = {}
    for sym in active_universe:
        prior = ind_by_sym[sym].loc[ind_by_sym[sym].index < calendar[0], "close"]
        if len(prior):
            last_close[sym] = float(prior.iloc[-1])

    def _btc_regime_on(ts: pd.Timestamp) -> tuple[bool | None, float, float]:
        """Return (btc_above_sma100, btc_close, btc_sma100) using latest BTC bar <= ts."""
        sub = btc_ind.loc[:ts]
        if sub.empty:
            return None, float("nan"), float("nan")
        row = sub.iloc[-1]
        c = float(row["close"])
        m = float(row["sma100"])
        if not (math.isfinite(c) and math.isfinite(m)):
            return None, c, m
        return (c > m), c, m

    def _evaluate_close_signals(
        sig_ts: pd.Timestamp,
        current_positions: dict[str, _OpenCryptoPos],
    ) -> tuple[dict[str, tuple[ExitReason, date]], list[tuple[str, float, float, date]], bool]:
        """Evaluate day ``sig_ts`` close signals for execution on the next bar's Open."""
        sig_date = sig_ts.date()
        btc_above, _, _ = _btc_regime_on(sig_ts)
        btc_bull = bool(btc_above) if btc_above is not None else False
        btc_bear_exit = (btc_above is False)

        exits: dict[str, tuple[ExitReason, date]] = {}
        for sym, pos in current_positions.items():
            df_s = ind_by_sym[sym]
            if btc_bear_exit:
                exits[sym] = ("BTC_MACRO_EXIT", sig_date)
                continue
            if sig_ts not in df_s.index:
                continue
            row = df_s.loc[sig_ts]
            c = float(row["close"])
            pl = float(row["prior_low"])
            atr = float(row["atr14"])
            if math.isfinite(pl) and c < pl:
                exits[sym] = ("DONCHIAN_10D_LOW", sig_date)
            elif math.isfinite(atr) and pos.highest_close > 0.0 and c < pos.highest_close - cfg.trail_atr_mult * atr:
                exits[sym] = ("ATR_TRAILING_STOP", sig_date)

        candidates: list[tuple[str, float, float, date]] = []
        if btc_bull:
            for sym in active_universe:
                if sym in current_positions:
                    continue
                df_s = ind_by_sym[sym]
                if sig_ts not in df_s.index:
                    continue
                row = df_s.loc[sig_ts]
                c = float(row["close"])
                ph = float(row["prior_high"])
                m = float(row["sma100"])
                atr = float(row["atr14"])
                score = float(row["rank_score"])
                if (
                    math.isfinite(c)
                    and math.isfinite(ph)
                    and math.isfinite(m)
                    and math.isfinite(atr)
                    and math.isfinite(score)
                    and atr > 0.0
                    and c > ph
                    and c > m
                ):
                    candidates.append((sym, score, atr, sig_date))
            candidates.sort(key=lambda x: (-x[1], x[0]))

        return exits, candidates, btc_bull

    cash = float(initial_capital_eur)
    min_cash = cash
    positions: dict[str, _OpenCryptoPos] = {}
    fills: list[CryptoOrderFill] = []
    round_trips: list[CryptoRoundTrip] = []

    total_fees = 0.0
    total_slippage = 0.0
    hosting_fees = 0.0
    total_buy_notional = 0.0
    total_sell_notional = 0.0
    skipped_min_order = 0

    def _close_position(
        pos: _OpenCryptoPos,
        *,
        exec_date: date,
        sig_date: date,
        raw_exit_px: float,
        reason: ExitReason,
    ) -> None:
        nonlocal cash, min_cash, total_fees, total_slippage, total_sell_notional
        fill_px = raw_exit_px * (1.0 - cfg.slippage_pct)
        gross_proceeds = pos.qty * fill_px
        fee = gross_proceeds * cfg.taker_fee_pct
        slip = pos.qty * raw_exit_px * cfg.slippage_pct
        net_proceeds = gross_proceeds - fee

        cash += net_proceeds
        min_cash = min(min_cash, cash)
        total_fees += fee
        total_slippage += slip
        total_sell_notional += gross_proceeds

        fills.append(
            CryptoOrderFill(
                date=exec_date,
                signal_date=sig_date,
                symbol=pos.symbol,
                side="SELL",
                qty=pos.qty,
                raw_price=raw_exit_px,
                fill_price=fill_px,
                notional_eur=gross_proceeds,
                fee_eur=fee,
                slippage_eur=slip,
                reason=reason,
            )
        )

        cost_basis = pos.entry_notional_eur + pos.entry_fee_eur
        gross_pnl = pos.qty * (raw_exit_px - pos.entry_raw_price)
        tot_fees = pos.entry_fee_eur + fee
        tot_slip = pos.entry_slippage_eur + slip
        net_pnl = net_proceeds - cost_basis
        ret_pct = net_pnl / cost_basis if cost_basis > 0.0 else 0.0

        round_trips.append(
            CryptoRoundTrip(
                symbol=pos.symbol,
                signal_entry_date=pos.signal_entry_date,
                entry_date=pos.entry_date,
                exit_date=exec_date,
                qty=pos.qty,
                entry_raw_price=pos.entry_raw_price,
                entry_fill_price=pos.entry_fill_price,
                entry_atr14=pos.entry_atr14,
                hard_stop_price=pos.hard_stop_price,
                exit_raw_price=raw_exit_px,
                exit_fill_price=fill_px,
                highest_close=max(pos.highest_close, pos.entry_raw_price),
                bars_held=pos.bars_held,
                cost_basis_eur=cost_basis,
                gross_pnl_eur=gross_pnl,
                fees_eur=tot_fees,
                slippage_eur=tot_slip,
                net_pnl_eur=net_pnl,
                return_pct=ret_pct,
                exit_reason=reason,
                exit_category=_exit_category(reason),
                is_open_at_end=False,
            )
        )

    # Seed Day-0 pending entries strictly from the latest bar < start_date (if any)
    pending_exits: dict[str, tuple[ExitReason, date]] = {}
    pending_entries: list[tuple[str, float, float, date]] = []
    prior_btc_dates = btc_ind.index[btc_ind.index < calendar[0]]
    if len(prior_btc_dates) > 0:
        prev_sig_ts = prior_btc_dates[-1]
        pending_exits, pending_entries, _ = _evaluate_close_signals(prev_sig_ts, positions)

    rows: list[dict[str, object]] = []
    prev_date: date | None = None

    for ts in calendar:
        cur_date = ts.date()

        # Monthly hosting fee (EUR 0.00 for Bitvavo/Kraken spot)
        if cfg.monthly_hosting_eur > 0.0:
            months = _months_between_crossed(prev_date, cur_date)
            if months > 0:
                hfee = months * cfg.monthly_hosting_eur
                cash = max(0.0, cash - hfee)
                hosting_fees += hfee

        # ------------------------------------------------------------------
        # Step 1 (Day d Open): Execute queued signal exits from day t close.
        # T+0 instant settlement -> net EUR proceeds immediately available.
        # ------------------------------------------------------------------
        still_pending_exits: dict[str, tuple[ExitReason, date]] = {}
        for sym, (reason, sig_dt) in list(pending_exits.items()):
            pos = positions.get(sym)
            if pos is None:
                continue
            df_s = ind_by_sym[sym]
            if ts not in df_s.index:
                still_pending_exits[sym] = (reason, sig_dt)
                continue
            raw_open = float(df_s.loc[ts, "open"])
            _close_position(
                pos,
                exec_date=cur_date,
                sig_date=sig_dt,
                raw_exit_px=raw_open,
                reason=reason,
            )
            del positions[sym]

        # ------------------------------------------------------------------
        # Step 2 (Day d Open): Execute queued entries from day t close.
        # Decision uses ONLY day t signals (never peeks at day d stops).
        # ------------------------------------------------------------------
        entered_today: set[str] = set()
        open_slots = cfg.max_slots - len(positions)
        if open_slots > 0 and pending_entries:
            # Causal pre-entry equity estimate using settled cash + prior closes of held coins
            held_val_est = sum(
                p.qty * last_close.get(s, p.entry_fill_price) for s, p in positions.items()
            )
            slot_target_eur = (cash + held_val_est) / float(cfg.max_slots)

            for sym, _score, entry_atr, sig_dt in pending_entries:
                if open_slots <= 0:
                    break
                if sym in positions:
                    continue
                df_s = ind_by_sym[sym]
                if ts not in df_s.index:
                    continue
                raw_open = float(df_s.loc[ts, "open"])
                if not (math.isfinite(raw_open) and raw_open > 0.0):
                    continue

                # Allocate target slot size, capped by available EUR cash (no margin/borrowing)
                budget = min(cash, slot_target_eur)
                # Avoid tiny floating-point dust when filling the final slot with 100% of cash
                if cash - budget < 1e-6:
                    budget = cash
                if budget < cfg.min_order_eur:
                    skipped_min_order += 1
                    continue

                fill_px = raw_open * (1.0 + cfg.slippage_pct)
                notional = budget / (1.0 + cfg.taker_fee_pct)
                fee = budget - notional
                qty = notional / fill_px
                slip = qty * raw_open * cfg.slippage_pct

                cash = max(0.0, cash - budget)
                min_cash = min(min_cash, cash)
                total_fees += fee
                total_slippage += slip
                total_buy_notional += notional

                hard_stop = max(0.0, fill_px - cfg.hard_stop_atr_mult * entry_atr)

                fills.append(
                    CryptoOrderFill(
                        date=cur_date,
                        signal_date=sig_dt,
                        symbol=sym,
                        side="BUY",
                        qty=qty,
                        raw_price=raw_open,
                        fill_price=fill_px,
                        notional_eur=notional,
                        fee_eur=fee,
                        slippage_eur=slip,
                        reason="BREAKOUT_20D_HIGH",
                    )
                )
                positions[sym] = _OpenCryptoPos(
                    symbol=sym,
                    signal_entry_date=sig_dt,
                    entry_date=cur_date,
                    qty=qty,
                    entry_raw_price=raw_open,
                    entry_fill_price=fill_px,
                    entry_atr14=entry_atr,
                    hard_stop_price=hard_stop,
                    entry_notional_eur=notional,
                    entry_fee_eur=fee,
                    entry_slippage_eur=slip,
                    highest_close=0.0,
                    bars_held=0,
                )
                entered_today.add(sym)
                open_slots -= 1

        # ------------------------------------------------------------------
        # Step 3 (Day d Intra-Day): Hard Catastrophe Stop on EVERY held bar
        # (including entry bar after open).
        #   - Held from prior bar & Open <= stop -> fill at Open (gap-through-stop)
        #   - Else if Low <= stop -> fill at stop (intraday breach)
        # Apply sell fee + slippage to every stop fill.
        # ------------------------------------------------------------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            df_s = ind_by_sym[sym]
            if ts not in df_s.index or pos.hard_stop_price <= 0.0:
                continue
            bar = df_s.loc[ts]
            bar_open = float(bar["open"])
            bar_low = float(bar["low"])

            if sym not in entered_today and bar_open <= pos.hard_stop_price:
                _close_position(
                    pos,
                    exec_date=cur_date,
                    sig_date=cur_date,
                    raw_exit_px=bar_open,
                    reason="HARD_STOP_GAP",
                )
                del positions[sym]
                still_pending_exits.pop(sym, None)
            elif bar_low <= pos.hard_stop_price:
                _close_position(
                    pos,
                    exec_date=cur_date,
                    sig_date=cur_date,
                    raw_exit_px=pos.hard_stop_price,
                    reason="HARD_STOP_INTRADAY",
                )
                del positions[sym]
                still_pending_exits.pop(sym, None)

        # ------------------------------------------------------------------
        # Step 4 (Day d Close): Update last_close, HighestCloseSinceEntry,
        # Mark-to-Market equity, and evaluate Day d close signals for Day d+1 Open.
        # ------------------------------------------------------------------
        for sym in active_universe:
            df_s = ind_by_sym[sym]
            if ts in df_s.index:
                last_close[sym] = float(df_s.loc[ts, "close"])

        mv_total = 0.0
        pos_weights: dict[str, float] = {}
        for sym, pos in positions.items():
            pos.bars_held += 1
            cur_c = last_close.get(sym, pos.entry_fill_price)
            if sym in ind_by_sym and ts in ind_by_sym[sym].index:
                if pos.highest_close <= 0.0:
                    pos.highest_close = cur_c
                else:
                    pos.highest_close = max(pos.highest_close, cur_c)
            mv_s = pos.qty * cur_c
            mv_total += mv_s
            pos_weights[sym] = mv_s

        equity = cash + mv_total
        if cash < -1e-6:
            raise AssertionError(f"Negative cash violation on {cur_date}: cash={cash}")

        btc_above_d, btc_c_d, btc_sma_d = _btc_regime_on(ts)

        row_dict: dict[str, object] = {
            "date": ts,
            "equity": equity,
            "cash": cash,
            "market_value": mv_total,
            "open_positions": len(positions),
            "deployed_weight": (mv_total / equity) if equity > 0.0 else 0.0,
            "cash_weight": (cash / equity) if equity > 0.0 else 1.0,
            "btc_close": btc_c_d,
            "btc_sma100": btc_sma_d,
            "btc_bull": bool(btc_above_d) if btc_above_d is not None else False,
        }
        for sym in active_universe:
            row_dict[f"w_{sym}"] = (pos_weights.get(sym, 0.0) / equity) if equity > 0.0 else 0.0
        rows.append(row_dict)

        # Compute Day d close signals strictly for Day d+1 Open
        next_exits, next_entries, _ = _evaluate_close_signals(ts, positions)
        # Preserve any exit that couldn't execute today due to missing coin bar
        for sym, ex_val in still_pending_exits.items():
            if sym in positions and sym not in next_exits:
                next_exits[sym] = ex_val
        pending_exits = next_exits
        pending_entries = next_entries
        prev_date = cur_date

    # Snapshot any open positions at end_date (mark-to-market, NOT an executed sell fill)
    open_at_end: list[CryptoRoundTrip] = []
    last_date = calendar[-1].date()
    for sym in sorted(positions.keys()):
        pos = positions[sym]
        end_close = last_close.get(sym, pos.entry_fill_price)
        mv_end = pos.qty * end_close
        cost_basis = pos.entry_notional_eur + pos.entry_fee_eur
        net_pnl_mtm = mv_end - cost_basis
        gross_pnl_mtm = pos.qty * (end_close - pos.entry_raw_price)
        open_at_end.append(
            CryptoRoundTrip(
                symbol=sym,
                signal_entry_date=pos.signal_entry_date,
                entry_date=pos.entry_date,
                exit_date=last_date,
                qty=pos.qty,
                entry_raw_price=pos.entry_raw_price,
                entry_fill_price=pos.entry_fill_price,
                entry_atr14=pos.entry_atr14,
                hard_stop_price=pos.hard_stop_price,
                exit_raw_price=end_close,
                exit_fill_price=end_close,
                highest_close=max(pos.highest_close, pos.entry_raw_price),
                bars_held=pos.bars_held,
                cost_basis_eur=cost_basis,
                gross_pnl_eur=gross_pnl_mtm,
                fees_eur=pos.entry_fee_eur,
                slippage_eur=pos.entry_slippage_eur,
                net_pnl_eur=net_pnl_mtm,
                return_pct=net_pnl_mtm / cost_basis if cost_basis > 0.0 else 0.0,
                exit_reason="OPEN_MTM",
                exit_category="OPEN_MTM",
                is_open_at_end=True,
            )
        )

    curve = pd.DataFrame(rows).set_index("date")
    return CryptoBacktestResult(
        config=cfg,
        initial_capital_eur=float(initial_capital_eur),
        start_date=start_date,
        end_date=end_date,
        equity_curve=curve,
        fills=fills,
        round_trips=round_trips,
        open_positions_at_end=open_at_end,
        total_fees_eur=float(total_fees),
        total_slippage_eur=float(total_slippage),
        hosting_fees_eur=float(hosting_fees),
        total_buy_notional_eur=float(total_buy_notional),
        total_sell_notional_eur=float(total_sell_notional),
        skipped_min_order=skipped_min_order,
        min_cash_eur=float(min_cash),
    )


# ---------------------------------------------------------------------------
# Standardized 365-Day Crypto Metrics, Attribution & Benchmarks
# ---------------------------------------------------------------------------


def _profit_factor(pnls: list[float]) -> float | None:
    if not pnls:
        return None
    gross_win = sum(p for p in pnls if p > 0.0)
    gross_loss = -sum(p for p in pnls if p <= 0.0)
    if gross_loss <= 0.0:
        return float("inf") if gross_win > 0.0 else None
    return float(gross_win / gross_loss)


def compute_coin_attribution(res: CryptoBacktestResult) -> dict[str, CoinAttribution]:
    """Compute exact additive P&L, holding-time, and exit breakdown per coin."""
    curve = res.equity_curve
    n_days = len(curve)
    init = res.initial_capital_eur

    all_trips = list(res.round_trips) + list(res.open_positions_at_end)
    symbols = [s for s in res.config.universe if f"w_{s}" in curve.columns]

    out: dict[str, CoinAttribution] = {}
    for sym in symbols:
        w_col = f"w_{sym}"
        days_held = int((curve[w_col] > 0.0).sum()) if n_days > 0 else 0
        avg_w = float(curve[w_col].mean()) if n_days > 0 else 0.0
        closed_s = [t for t in res.round_trips if t.symbol == sym]
        open_s = [t for t in res.open_positions_at_end if t.symbol == sym]
        eval_s = [t for t in all_trips if t.symbol == sym]

        hard_exits = sum(1 for t in closed_s if t.exit_category == "HARD_STOP")
        sig_exits = sum(1 for t in closed_s if t.exit_category == "SIGNAL_EXIT")
        pnls = [t.net_pnl_eur for t in eval_s]

        gross_pnl = float(sum(t.gross_pnl_eur for t in eval_s))
        fees = float(sum(t.fees_eur for t in eval_s))
        slip = float(sum(t.slippage_eur for t in eval_s))
        net_pnl = float(sum(pnls))

        out[sym] = CoinAttribution(
            symbol=sym,
            days_held=days_held,
            pct_days_held=float(days_held / n_days) if n_days > 0 else 0.0,
            avg_weight_pct=avg_w,
            closed_trades=len(closed_s),
            open_at_end=len(open_s),
            evaluated_trades=len(eval_s),
            hard_stop_exits=hard_exits,
            signal_exits=sig_exits,
            win_rate=float(np.mean([p > 0.0 for p in pnls])) if pnls else None,
            profit_factor=_profit_factor(pnls),
            avg_bars_held=float(np.mean([t.bars_held for t in eval_s])) if eval_s else None,
            gross_pnl_eur=gross_pnl,
            fees_eur=fees,
            slippage_eur=slip,
            net_pnl_eur=net_pnl,
            return_contribution_pct=net_pnl / init if init > 0.0 else 0.0,
        )
    return out


def compute_crypto_metrics(
    res: CryptoBacktestResult,
    metrics_cfg: MetricsConfig | None = None,
    *,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
    cumulative_trials_n: int = 40,
    cumulative_trials_conservative_n: int = 60,
    include_open_in_trade_stats: bool = True,
) -> dict[str, float | int | None]:
    """Compute 365-day annualized crypto performance, cumulative DSR (N=10/40/60), Calmar, and exit stats."""
    mcfg = metrics_cfg or MetricsConfig()
    curve = res.equity_curve
    eq = curve["equity"]
    init = res.initial_capital_eur
    r = daily_returns(eq, init)

    days = (curve.index[-1] - curve.index[0]).days + 1
    years = max(days / 365.25, 1e-6)

    cagr_val = cagr(eq, init)
    mdd_val = max_drawdown(eq, init)
    calmar_val = (cagr_val / abs(mdd_val)) if (mdd_val < -1e-9 and math.isfinite(cagr_val)) else float("nan")

    eval_trips = list(res.round_trips)
    if include_open_in_trade_stats:
        eval_trips.extend(res.open_positions_at_end)

    pnls = [t.net_pnl_eur for t in eval_trips]
    win_rate = float(np.mean([p > 0.0 for p in pnls])) if pnls else float("nan")
    pf = _profit_factor(pnls)

    closed = res.round_trips
    hard_gap = [t for t in closed if t.exit_reason == "HARD_STOP_GAP"]
    hard_intra = [t for t in closed if t.exit_reason == "HARD_STOP_INTRADAY"]
    sig_donchian = [t for t in closed if t.exit_reason == "DONCHIAN_10D_LOW"]
    sig_trail = [t for t in closed if t.exit_reason == "ATR_TRAILING_STOP"]
    sig_btc = [t for t in closed if t.exit_reason == "BTC_MACRO_EXIT"]

    hard_all = hard_gap + hard_intra
    sig_all = sig_donchian + sig_trail + sig_btc

    total_friction = res.total_fees_eur + res.total_slippage_eur + res.hosting_fees_eur
    n_fills = len(res.fills)

    return {
        "initial_eur": init,
        "final_eur": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / init - 1.0),
        "cagr": cagr_val,
        "periods_per_year": periods_per_year,
        "daily_sample_length": int(len(r)),
        "sharpe": sharpe(r, periods_per_year, mcfg.risk_free_rate),
        "sortino": sortino(r, periods_per_year, mcfg.risk_free_rate),
        "deflated_sharpe_n10": deflated_sharpe(
            r, periods_per_year, mcfg.dsr_n_trials, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum40": deflated_sharpe(
            r, periods_per_year, cumulative_trials_n, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum60": deflated_sharpe(
            r, periods_per_year, cumulative_trials_conservative_n, mcfg.dsr_trial_sharpe_std
        ),
        "max_drawdown": mdd_val,
        "calmar": calmar_val,
        "total_orders": n_fills,
        "orders_per_year": float(n_fills / years),
        "closed_round_trips": len(closed),
        "open_positions_at_end": len(res.open_positions_at_end),
        "evaluated_trips": len(eval_trips),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_bars_held": float(np.mean([t.bars_held for t in eval_trips])) if eval_trips else float("nan"),
        "avg_deployed_pct": float(curve["deployed_weight"].mean()),
        "cash_only_days_pct": float((curve["open_positions"] == 0).mean()),
        "btc_bull_days_pct": float(curve["btc_bull"].mean()),
        "fees_eur": res.total_fees_eur,
        "slippage_eur": res.total_slippage_eur,
        "hosting_eur": res.hosting_fees_eur,
        "total_friction_eur": total_friction,
        "friction_pct_initial": total_friction / init,
        "hard_stop_exits": len(hard_all),
        "hard_stop_gap_exits": len(hard_gap),
        "hard_stop_intraday_exits": len(hard_intra),
        "hard_stop_net_pnl_eur": float(sum(t.net_pnl_eur for t in hard_all)),
        "signal_exits": len(sig_all),
        "signal_donchian_exits": len(sig_donchian),
        "signal_trail_exits": len(sig_trail),
        "signal_btc_macro_exits": len(sig_btc),
        "signal_exit_net_pnl_eur": float(sum(t.net_pnl_eur for t in sig_all)),
        "skipped_min_order": res.skipped_min_order,
        "min_cash_eur": res.min_cash_eur,
    }


def compute_crypto_bh_benchmark(
    bars_map: dict[str, BarSeries],
    weights: dict[str, float],
    start_date: date,
    end_date: date,
    initial_capital_eur: float = 1000.0,
    metrics_cfg: MetricsConfig | None = None,
    *,
    periods_per_year: int = CRYPTO_PERIODS_PER_YEAR,
    cumulative_trials_n: int = 40,
    cumulative_trials_conservative_n: int = 60,
) -> tuple[pd.Series, dict[str, float | int]]:
    """Compute pure Buy & Hold (no rebalancing, zero fee) EUR crypto benchmark curve & 365-day metrics."""
    mcfg = metrics_cfg or MetricsConfig()
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Benchmark weights must sum to 1.0, got {weights}")

    idx: pd.DatetimeIndex | None = None
    legs: dict[str, pd.DataFrame] = {}
    for sym in weights:
        df = bars_map[sym].frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
        if df.empty:
            raise ValueError(f"No bars for benchmark leg {sym} in [{start_date}, {end_date}]")
        legs[sym] = df
        idx = df.index if idx is None else idx.union(df.index)
    assert idx is not None
    idx = idx.sort_values().unique()

    eq = pd.Series(0.0, index=idx, dtype=float)
    for sym, w in weights.items():
        df = legs[sym]
        open0 = float(df["open"].iloc[0])
        aligned_close = df["close"].reindex(idx).ffill().bfill()
        eq += (initial_capital_eur * w) * (aligned_close / open0)

    r = daily_returns(eq, initial_capital_eur)
    cagr_val = cagr(eq, initial_capital_eur)
    mdd_val = max_drawdown(eq, initial_capital_eur)
    calmar_val = (cagr_val / abs(mdd_val)) if (mdd_val < -1e-9 and math.isfinite(cagr_val)) else float("nan")

    metrics: dict[str, float | int] = {
        "initial_eur": float(initial_capital_eur),
        "final_eur": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / initial_capital_eur - 1.0),
        "cagr": cagr_val,
        "periods_per_year": periods_per_year,
        "daily_sample_length": int(len(r)),
        "sharpe": sharpe(r, periods_per_year, mcfg.risk_free_rate),
        "sortino": sortino(r, periods_per_year, mcfg.risk_free_rate),
        "deflated_sharpe_n10": deflated_sharpe(
            r, periods_per_year, mcfg.dsr_n_trials, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum40": deflated_sharpe(
            r, periods_per_year, cumulative_trials_n, mcfg.dsr_trial_sharpe_std
        ),
        "deflated_sharpe_cum60": deflated_sharpe(
            r, periods_per_year, cumulative_trials_conservative_n, mcfg.dsr_trial_sharpe_std
        ),
        "max_drawdown": mdd_val,
        "calmar": calmar_val,
    }
    return eq, metrics
