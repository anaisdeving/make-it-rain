"""Real 5Y+ EUR spot crypto OHLCV loader with causal EURUSD=X conversion & integrity audits.

Downloads and caches daily 24/7 crypto bars from Yahoo Finance under ``data/cache/yfinance/``
(Parquet + SQLite ``fetch_log.sqlite``) with zero synthetic fallback:

- **Universe A (``majors_btc_eth``):** ``BTC-EUR``, ``ETH-EUR``.
- **Universe B (``nov2021_top10``):** ``BTC-EUR``, ``ETH-EUR``, ``BNB-EUR``, ``SOL-EUR``,
  ``ADA-EUR``, ``XRP-EUR``, ``DOT-EUR``, ``DOGE-EUR``, ``AVAX-EUR``, ``SHIB-EUR``.
- **Terra Classic Stress Test:** Inspects ``LUNC-EUR``, ``LUNC-USD``, and ``LUNA1-USD``
  to check whether Yahoo Finance retains usable pre-May-2022 history through the -99.99%
  May 2022 collapse.

Strict anti-bias & integrity rules:
- Uses native ``-EUR`` Yahoo Finance pairs whenever full warmup (>= 120 days prior to
  ``2021-11-22``) and 5Y coverage exist without corruption.
- If a coin lacks clean ``-EUR`` history, logs the exact reason and converts its ``-USD``
  pair via daily ``EURUSD=X`` (strictly causal forward-fill over weekends/holidays via
  ``asof_series`` — NEVER back-filled; ``EUR_price = USD_price / EURUSD``).
- Never silently drops any coin: emits a complete ``CryptoCoinAudit`` for every requested
  ticker covering exact date ranges, warmup/IS/OOS/5Y bar counts, missing calendar days,
  consecutive flat closes, extreme 1-day moves, and warnings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.data.bar_utils import normalise_ohlcv
from src.data.cache import DiskCache
from src.data.cross_asset_eu_loader import (
    HistoryFetcher,
    _default_history_fetcher,
    _max_flat_close_run,
)
from src.domain.models import BarSeries
from src.strategy.indicators import asof_series

log = logging.getLogger(__name__)

UNIVERSE_MAJORS: tuple[str, ...] = ("BTC-EUR", "ETH-EUR")

UNIVERSE_NOV2021_TOP10: tuple[str, ...] = (
    "BTC-EUR",
    "ETH-EUR",
    "BNB-EUR",
    "SOL-EUR",
    "ADA-EUR",
    "XRP-EUR",
    "DOT-EUR",
    "DOGE-EUR",
    "AVAX-EUR",
    "SHIB-EUR",
)

LUNC_STRESS_CANDIDATES: tuple[str, ...] = ("LUNC-EUR", "LUNC-USD", "LUNA1-USD")

FX_SYMBOL = "EURUSD=X"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CryptoCoinAudit(_Frozen):
    """Complete data-coverage and integrity audit record for one crypto pair."""

    canonical_symbol: str
    source_symbol: str
    conversion_used: bool
    native_eur_status: str
    fx_ffilled_days: int
    currency: Literal["EUR"] = "EUR"
    rows_total: int
    rows_warmup: int
    rows_is: int
    rows_oos: int
    rows_5y: int
    first_date: str | None
    last_date: str | None
    missing_calendar_days_5y: int
    max_gap_calendar_days_5y: int
    zero_vol_days_5y: int
    max_flat_close_run_5y: int
    max_1d_gain_pct_5y: float
    max_1d_drop_pct_5y: float
    close_is_start: float | None
    close_oos_start: float | None
    close_end: float | None
    min_close_5y: float | None
    max_close_5y: float | None
    integrity_warnings: list[str]
    usable: bool
    synthetic: bool = False


class LuncCandidateInspection(_Frozen):
    """Transparent inspection record for each Terra Classic candidate ticker."""

    candidate_symbol: str
    conversion_used: bool
    rows_total: int
    rows_warmup: int
    rows_pre_crash: int
    rows_may2022_crash: int
    rows_5y: int
    first_date: str | None
    last_date: str | None
    pre_crash_max_eur: float | None
    post_crash_min_eur: float | None
    may2022_drawdown_pct: float | None
    usable_for_stress_test: bool
    status_note: str


@dataclass(frozen=True)
class CryptoEurDataset:
    """Validated real-data bundle for Strategy v8b-clean."""

    bars: dict[str, BarSeries]
    audits: dict[str, CryptoCoinAudit]
    eurusd_bars: BarSeries
    eurusd_audit: dict[str, object]
    lunc_inspections: list[LuncCandidateInspection]
    lunc_bars: BarSeries | None
    lunc_audit: CryptoCoinAudit | None
    lunc_selected_source: str | None


def convert_usd_bars_to_eur(
    usd_series: BarSeries,
    eurusd_series: BarSeries,
    target_symbol: str,
) -> tuple[BarSeries, int]:
    """Convert daily USD crypto bars to EUR using strictly causal forward-filled ``EURUSD=X``.

    ``EURUSD=X`` quotes USD per 1 EUR (e.g. 1.08 USD/EUR), so ``EUR_price = USD_price / EURUSD``.
    Weekend and holiday crypto bars inherit the latest FX close dated ``<= crypto_date``
    (never back-filled).
    """
    if usd_series.frame.empty:
        raise ValueError(f"Empty USD bar series for {usd_series.symbol}")
    if eurusd_series.frame.empty:
        raise ValueError("Empty EURUSD=X bar series")

    df = usd_series.frame.copy()
    fx_close = eurusd_series.frame["close"].astype(float)
    fx_aligned = asof_series(fx_close, df.index)

    valid_mask = fx_aligned.notna() & (fx_aligned > 0.0)
    if not valid_mask.any():
        raise ValueError(f"No causal EURUSD=X rates available on or before {usd_series.symbol} dates")

    df = df.loc[valid_mask].copy()
    fx_valid = fx_aligned.loc[valid_mask]

    fx_min = float(fx_valid.min())
    fx_max = float(fx_valid.max())
    if fx_min <= 0.5 or fx_max >= 2.0:
        raise ValueError(
            f"Anomalous EURUSD=X range [{fx_min:.4f}, {fx_max:.4f}] — expected USD per 1 EUR in (0.5, 2.0)"
        )

    for col in ("open", "high", "low", "close", "raw_close"):
        if col in df.columns:
            df[col] = df[col].astype(float) / fx_valid

    # Preserve OHLC invariants cleanly against floating-point rounding
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    fx_ffilled_days = int((~df.index.isin(eurusd_series.frame.index)).sum())
    return BarSeries(symbol=target_symbol, frame=df), fx_ffilled_days


def _max_calendar_gap_days(idx: pd.DatetimeIndex) -> int:
    if len(idx) <= 1:
        return 0
    diffs = np.diff(idx.to_numpy(dtype="datetime64[D]")).astype(int)
    max_missing = int(np.max(diffs) - 1)
    return max(0, max_missing)


def _base_from_eur_symbol(symbol: str) -> str:
    if symbol.endswith("-EUR"):
        return symbol[:-4]
    if symbol.endswith("-USD"):
        return symbol[:-4]
    return symbol


def _audit_series(
    *,
    canonical_symbol: str,
    source_symbol: str,
    conversion_used: bool,
    native_eur_status: str,
    fx_ffilled_days: int,
    series: BarSeries | None,
    eval_start: date,
    is_end: date,
    oos_start: date,
    eval_end: date,
    min_warmup_bars: int,
    min_eval_bars: int,
    max_end_lag_days: int = 7,
    allow_death_spiral: bool = False,
) -> CryptoCoinAudit:
    expected_5y_days = (eval_end - eval_start).days + 1
    if series is None or series.frame.empty:
        return CryptoCoinAudit(
            canonical_symbol=canonical_symbol,
            source_symbol=source_symbol,
            conversion_used=conversion_used,
            native_eur_status=native_eur_status,
            fx_ffilled_days=fx_ffilled_days,
            rows_total=0,
            rows_warmup=0,
            rows_is=0,
            rows_oos=0,
            rows_5y=0,
            first_date=None,
            last_date=None,
            missing_calendar_days_5y=expected_5y_days,
            max_gap_calendar_days_5y=expected_5y_days,
            zero_vol_days_5y=0,
            max_flat_close_run_5y=0,
            max_1d_gain_pct_5y=0.0,
            max_1d_drop_pct_5y=0.0,
            close_is_start=None,
            close_oos_start=None,
            close_end=None,
            min_close_5y=None,
            max_close_5y=None,
            integrity_warnings=["empty OHLCV history"],
            usable=False,
            synthetic=False,
        )

    df = series.frame
    pre_df = df.loc[df.index < pd.Timestamp(eval_start)]
    is_df = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(is_end))]
    oos_df = df.loc[(df.index >= pd.Timestamp(oos_start)) & (df.index <= pd.Timestamp(eval_end))]
    df5 = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(eval_end))]

    first_dt = df.index[0].date()
    last_dt = df.index[-1].date()
    missing_5y = max(0, expected_5y_days - len(df5))
    max_gap_5y = _max_calendar_gap_days(df5.index) if not df5.empty else expected_5y_days
    zero_vol_5y = int((df5["volume"] == 0).sum()) if not df5.empty else 0
    flat_run_5y = _max_flat_close_run(df5["close"]) if not df5.empty else 0

    rets = df5["close"].pct_change().dropna() if len(df5) > 1 else pd.Series(dtype=float)
    max_gain_pct = float(rets.max() * 100.0) if len(rets) else 0.0
    max_drop_pct = float(rets.min() * 100.0) if len(rets) else 0.0

    warnings: list[str] = []
    usable = True

    if len(pre_df) < min_warmup_bars:
        warnings.append(f"insufficient warmup bars ({len(pre_df)} < {min_warmup_bars}; first={first_dt})")
        usable = False
    if len(df5) < min_eval_bars:
        warnings.append(f"insufficient 5Y bars ({len(df5)} < {min_eval_bars}; range {first_dt}..{last_dt})")
        usable = False
    if (eval_end - last_dt).days > max_end_lag_days:
        warnings.append(f"history ends early ({last_dt} vs {eval_end})")
        usable = False
    if max_gap_5y > 14:
        warnings.append(f"large calendar gap ({max_gap_5y} consecutive missing days)")
        usable = False
    elif max_gap_5y > 2:
        warnings.append(f"minor calendar gap ({max_gap_5y} missing days)")

    if flat_run_5y >= 30:
        warnings.append(f"corrupted frozen flatline ({flat_run_5y} consecutive identical closes)")
        usable = False
    elif flat_run_5y >= 5:
        warnings.append(f"flat close run of {flat_run_5y} days (check micro-price tick rounding)")

    if not allow_death_spiral and max_gain_pct > 500.0:
        warnings.append(f"extreme +1d jump ({max_gain_pct:+.1f}% — check unadjusted rebranding/redenomination)")
        usable = False
    elif max_gain_pct > 100.0:
        warnings.append(f"high 1d gain ({max_gain_pct:+.1f}%)")

    if max_drop_pct < -80.0 and not allow_death_spiral:
        warnings.append(f"extreme -1d drop ({max_drop_pct:+.1f}%)")

    return CryptoCoinAudit(
        canonical_symbol=canonical_symbol,
        source_symbol=source_symbol,
        conversion_used=conversion_used,
        native_eur_status=native_eur_status,
        fx_ffilled_days=fx_ffilled_days,
        rows_total=len(df),
        rows_warmup=len(pre_df),
        rows_is=len(is_df),
        rows_oos=len(oos_df),
        rows_5y=len(df5),
        first_date=str(first_dt),
        last_date=str(last_dt),
        missing_calendar_days_5y=missing_5y,
        max_gap_calendar_days_5y=max_gap_5y,
        zero_vol_days_5y=zero_vol_5y,
        max_flat_close_run_5y=flat_run_5y,
        max_1d_gain_pct_5y=round(max_gain_pct, 2),
        max_1d_drop_pct_5y=round(max_drop_pct, 2),
        close_is_start=float(df5["close"].iloc[0]) if not df5.empty else None,
        close_oos_start=float(oos_df["close"].iloc[0]) if not oos_df.empty else None,
        close_end=float(df5["close"].iloc[-1]) if not df5.empty else None,
        min_close_5y=float(df5["close"].min()) if not df5.empty else None,
        max_close_5y=float(df5["close"].max()) if not df5.empty else None,
        integrity_warnings=warnings,
        usable=usable,
        synthetic=False,
    )


class CryptoEurLoader:
    """Download, cache, validate, and audit 5Y+ EUR daily crypto OHLCV bars."""

    def __init__(
        self,
        cache_root: Path,
        *,
        history_fetcher: HistoryFetcher | None = None,
    ) -> None:
        self.cache = DiskCache(cache_root, "yfinance")
        self._history_fetcher = history_fetcher or _default_history_fetcher

    def get_bars(self, symbol: str, start: date, end: date) -> BarSeries | None:
        key = f"{symbol}_{start}_{end}"
        status = self.cache.status("bars", key)
        if status == "ok":
            df = self.cache.read("bars", key)
            return BarSeries(symbol=symbol, frame=df) if df is not None else None
        if status == "empty":
            return None

        hist, splits = self._history_fetcher(symbol, start, end)
        if hist is None or hist.empty:
            self.cache.write("bars", key, None, status="empty")
            return None

        series = normalise_ohlcv(symbol, hist, splits)
        if series is None:
            self.cache.write("bars", key, None, status="empty")
            return None

        self.cache.write("bars", key, series.frame, status="ok")
        return series

    def load_eurusd(self, warmup_start: date, eval_end: date) -> tuple[BarSeries, dict[str, object]]:
        """Load ``EURUSD=X`` with a 14-day pre-warmup buffer for strictly causal weekend ffill."""
        fx_start = warmup_start - timedelta(days=14)
        fx = self.get_bars(FX_SYMBOL, fx_start, eval_end)
        if fx is None or fx.frame.empty:
            raise RuntimeError(f"Failed to download required FX reference series {FX_SYMBOL}")
        df = fx.frame
        audit = {
            "symbol": FX_SYMBOL,
            "rows_total": int(len(df)),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "min_rate": round(float(df["close"].min()), 5),
            "max_rate": round(float(df["close"].max()), 5),
            "start_rate": round(float(df["close"].iloc[0]), 5),
            "end_rate": round(float(df["close"].iloc[-1]), 5),
        }
        log.info(
            "Loaded FX %s | rows=%d [%s..%s] range=[%.4f..%.4f]",
            FX_SYMBOL,
            audit["rows_total"],
            audit["first_date"],
            audit["last_date"],
            audit["min_rate"],
            audit["max_rate"],
        )
        return fx, audit

    def load_coin(
        self,
        canonical_eur_symbol: str,
        eurusd: BarSeries,
        *,
        warmup_start: date,
        eval_start: date,
        is_end: date,
        oos_start: date,
        eval_end: date,
        min_warmup_bars: int = 120,
        min_eval_bars: int = 1750,
    ) -> tuple[BarSeries | None, CryptoCoinAudit]:
        """Load a single coin in EUR, falling back to ``-USD / EURUSD=X`` if native ``-EUR`` is unusable."""
        base = _base_from_eur_symbol(canonical_eur_symbol)
        native_sym = f"{base}-EUR"
        usd_sym = f"{base}-USD"

        native_series = self.get_bars(native_sym, warmup_start, eval_end)
        native_audit = _audit_series(
            canonical_symbol=canonical_eur_symbol,
            source_symbol=native_sym,
            conversion_used=False,
            native_eur_status="ok",
            fx_ffilled_days=0,
            series=native_series,
            eval_start=eval_start,
            is_end=is_end,
            oos_start=oos_start,
            eval_end=eval_end,
            min_warmup_bars=min_warmup_bars,
            min_eval_bars=min_eval_bars,
        )

        # Prefer native -EUR when fully usable without excessive micro-price flatlining
        if native_audit.usable and native_series is not None and native_audit.max_flat_close_run_5y < 15:
            log.info(
                "Loaded %-9s (native EUR) | rows=%d (warm=%d, 5Y=%d) [%s..%s] miss=%d flat=%d",
                canonical_eur_symbol,
                native_audit.rows_total,
                native_audit.rows_warmup,
                native_audit.rows_5y,
                native_audit.first_date,
                native_audit.last_date,
                native_audit.missing_calendar_days_5y,
                native_audit.max_flat_close_run_5y,
            )
            return native_series, native_audit

        native_reason = (
            "; ".join(native_audit.integrity_warnings)
            if native_audit.integrity_warnings
            else f"native flat run {native_audit.max_flat_close_run_5y} >= 15"
        )
        log.warning(
            "Native %s failed/degraded (%s) -> attempting causal %s / %s conversion",
            native_sym,
            native_reason,
            usd_sym,
            FX_SYMBOL,
        )

        usd_series = self.get_bars(usd_sym, warmup_start, eval_end)
        if usd_series is None or usd_series.frame.empty:
            failed = native_audit.model_copy(
                update={
                    "native_eur_status": f"failed: {native_reason}; fallback {usd_sym} also empty",
                    "usable": False,
                }
            )
            return None, failed

        conv_series, ffill_days = convert_usd_bars_to_eur(usd_series, eurusd, canonical_eur_symbol)
        conv_audit = _audit_series(
            canonical_symbol=canonical_eur_symbol,
            source_symbol=f"{usd_sym} / {FX_SYMBOL}",
            conversion_used=True,
            native_eur_status=f"fallback ({native_reason})",
            fx_ffilled_days=ffill_days,
            series=conv_series,
            eval_start=eval_start,
            is_end=is_end,
            oos_start=oos_start,
            eval_end=eval_end,
            min_warmup_bars=min_warmup_bars,
            min_eval_bars=min_eval_bars,
        )

        # If native -EUR was actually usable (only flagged for mild flat run) and USD fallback is worse, keep native
        if not conv_audit.usable and native_audit.usable and native_series is not None:
            log.warning(
                "Keeping native %s over unusable %s fallback (%s)",
                native_sym,
                usd_sym,
                "; ".join(conv_audit.integrity_warnings),
            )
            return native_series, native_audit

        log.info(
            "Loaded %-9s (%s) | rows=%d (warm=%d, 5Y=%d) [%s..%s] ffill_fx=%d miss=%d flat=%d usable=%s",
            canonical_eur_symbol,
            conv_audit.source_symbol,
            conv_audit.rows_total,
            conv_audit.rows_warmup,
            conv_audit.rows_5y,
            conv_audit.first_date,
            conv_audit.last_date,
            conv_audit.fx_ffilled_days,
            conv_audit.missing_calendar_days_5y,
            conv_audit.max_flat_close_run_5y,
            conv_audit.usable,
        )
        return (conv_series if conv_audit.usable else None), conv_audit

    def inspect_lunc_stress_candidates(
        self,
        eurusd: BarSeries,
        *,
        warmup_start: date,
        eval_start: date,
        is_end: date,
        oos_start: date,
        eval_end: date,
        candidates: tuple[str, ...] = LUNC_STRESS_CANDIDATES,
        min_warmup_bars: int = 120,
    ) -> tuple[list[LuncCandidateInspection], BarSeries | None, CryptoCoinAudit | None, str | None]:
        """Inspect Terra Classic tickers for pre-May-2022 history + -99.99% death-spiral usability."""
        inspections: list[LuncCandidateInspection] = []
        selected_series: BarSeries | None = None
        selected_audit: CryptoCoinAudit | None = None
        selected_source: str | None = None

        crash_start = pd.Timestamp("2022-05-01")
        crash_end = pd.Timestamp("2022-05-31")

        for sym in candidates:
            raw = self.get_bars(sym, warmup_start, eval_end)
            is_usd = sym.endswith("-USD")
            if raw is None or raw.frame.empty:
                inspections.append(
                    LuncCandidateInspection(
                        candidate_symbol=sym,
                        conversion_used=is_usd,
                        rows_total=0,
                        rows_warmup=0,
                        rows_pre_crash=0,
                        rows_may2022_crash=0,
                        rows_5y=0,
                        first_date=None,
                        last_date=None,
                        pre_crash_max_eur=None,
                        post_crash_min_eur=None,
                        may2022_drawdown_pct=None,
                        usable_for_stress_test=False,
                        status_note="no OHLCV bars returned by Yahoo Finance",
                    )
                )
                continue

            ffill_days = 0
            if is_usd:
                eur_series, ffill_days = convert_usd_bars_to_eur(raw, eurusd, "LUNC-EUR")
            else:
                eur_series = BarSeries(symbol="LUNC-EUR", frame=raw.frame.copy())

            df = eur_series.frame
            pre_warm = df.loc[df.index < pd.Timestamp(eval_start)]
            pre_crash = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index < crash_start)]
            crash_df = df.loc[(df.index >= crash_start) & (df.index <= crash_end)]
            post_crash = df.loc[df.index >= crash_start]
            df5 = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(eval_end))]

            pre_max = float(pre_crash["close"].max()) if not pre_crash.empty else None
            post_min = float(post_crash["close"].min()) if not post_crash.empty else None
            dd_pct: float | None = None
            if pre_max is not None and pre_max > 0 and post_min is not None:
                dd_pct = round((post_min / pre_max - 1.0) * 100.0, 4)

            reasons: list[str] = []
            if len(pre_warm) < min_warmup_bars:
                reasons.append(f"warmup={len(pre_warm)}<{min_warmup_bars}")
            if len(pre_crash) < 120:
                reasons.append(f"pre_crash_bars={len(pre_crash)}<120")
            if len(crash_df) < 15:
                reasons.append(f"may2022_bars={len(crash_df)}<15")
            if pre_max is None or pre_max < 10.0:
                # Real Terra LUNA traded ~EUR 35 - EUR 105 between Nov 2021 and Apr 2022;
                # values << EUR 10 indicate Yahoo retroactively flattened/corrupted pre-crash prices.
                reasons.append(
                    f"pre_crash_max_eur={pre_max} (< EUR 10; pre-crash history missing or corrupted)"
                )
            if dd_pct is None or dd_pct > -95.0:
                reasons.append(f"may2022_dd={dd_pct}% (does not reflect -99.99% collapse)")

            usable = len(reasons) == 0
            note = (
                "usable: contains real pre-May-2022 history and -99.99% May 2022 collapse"
                if usable
                else "unusable: " + "; ".join(reasons)
            )

            inspections.append(
                LuncCandidateInspection(
                    candidate_symbol=sym,
                    conversion_used=is_usd,
                    rows_total=len(df),
                    rows_warmup=len(pre_warm),
                    rows_pre_crash=len(pre_crash),
                    rows_may2022_crash=len(crash_df),
                    rows_5y=len(df5),
                    first_date=str(df.index[0].date()),
                    last_date=str(df.index[-1].date()),
                    pre_crash_max_eur=pre_max,
                    post_crash_min_eur=post_min,
                    may2022_drawdown_pct=dd_pct,
                    usable_for_stress_test=usable,
                    status_note=note,
                )
            )
            log.info(
                "LUNC inspect %-10s | rows=%d warm=%d pre_crash=%d may22=%d 5Y=%d pre_max=%s post_min=%s dd=%s | %s",
                sym,
                len(df),
                len(pre_warm),
                len(pre_crash),
                len(crash_df),
                len(df5),
                f"EUR {pre_max:.4f}" if pre_max is not None else "None",
                f"EUR {post_min:.8f}" if post_min is not None else "None",
                f"{dd_pct:.4f}%" if dd_pct is not None else "None",
                note,
            )

            if usable and selected_series is None:
                selected_series = eur_series
                selected_source = f"{sym} / {FX_SYMBOL}" if is_usd else sym
                selected_audit = _audit_series(
                    canonical_symbol="LUNC-EUR",
                    source_symbol=selected_source,
                    conversion_used=is_usd,
                    native_eur_status="ok" if not is_usd else f"stress fallback via {sym}",
                    fx_ffilled_days=ffill_days,
                    series=eur_series,
                    eval_start=eval_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    eval_end=eval_end,
                    min_warmup_bars=min_warmup_bars,
                    min_eval_bars=300,  # allow delisting/truncation after death spiral if it happened
                    max_end_lag_days=2000,
                    allow_death_spiral=True,
                )

        return inspections, selected_series, selected_audit, selected_source

    def load_dataset(
        self,
        *,
        symbols: tuple[str, ...] = UNIVERSE_NOV2021_TOP10,
        warmup_start: date = date(2021, 6, 1),
        eval_start: date = date(2021, 11, 22),
        is_end: date = date(2024, 11, 21),
        oos_start: date = date(2024, 11, 22),
        eval_end: date = date(2026, 11, 20),
        min_warmup_bars: int = 120,
        min_eval_bars: int = 1750,
        inspect_lunc: bool = True,
    ) -> CryptoEurDataset:
        """Load and audit all requested EUR spot crypto pairs plus optional LUNC stress test."""
        eurusd, eurusd_audit = self.load_eurusd(warmup_start, eval_end)

        bars: dict[str, BarSeries] = {}
        audits: dict[str, CryptoCoinAudit] = {}

        for sym in symbols:
            series, audit = self.load_coin(
                sym,
                eurusd,
                warmup_start=warmup_start,
                eval_start=eval_start,
                is_end=is_end,
                oos_start=oos_start,
                eval_end=eval_end,
                min_warmup_bars=min_warmup_bars,
                min_eval_bars=min_eval_bars,
            )
            audits[sym] = audit
            if series is not None and audit.usable:
                bars[sym] = series
            else:
                log.error(
                    "Coin %s UNUSABLE (not silently dropped — recorded in audits): %s",
                    sym,
                    audit.integrity_warnings,
                )

        if "BTC-EUR" not in bars:
            raise RuntimeError(f"Required macro anchor BTC-EUR failed integrity checks: {audits.get('BTC-EUR')}")

        if inspect_lunc:
            lunc_insp, lunc_bars, lunc_audit, lunc_src = self.inspect_lunc_stress_candidates(
                eurusd,
                warmup_start=warmup_start,
                eval_start=eval_start,
                is_end=is_end,
                oos_start=oos_start,
                eval_end=eval_end,
                min_warmup_bars=min_warmup_bars,
            )
        else:
            lunc_insp, lunc_bars, lunc_audit, lunc_src = [], None, None, None

        return CryptoEurDataset(
            bars=bars,
            audits=audits,
            eurusd_bars=eurusd,
            eurusd_audit=eurusd_audit,
            lunc_inspections=lunc_insp,
            lunc_bars=lunc_bars,
            lunc_audit=lunc_audit,
            lunc_selected_source=lunc_src,
        )
