"""Real VIX term-structure & EU UCITS volatility/equity loader with integrity guards.

Loads and caches real daily OHLCV bars from Yahoo Finance (zero synthetic fallback)
reusing ``data/cache/yfinance/`` (Parquet + SQLite ``fetch_log.sqlite``):

- **US Signal Series:**
  - ``QQQ`` (Invesco QQQ Trust — reuses existing cache)
  - ``^VIX`` (CBOE 30-day Volatility Index)
  - ``^VIX3M`` (CBOE S&P 500 3-Month Volatility Index)
- **EU Tradable EUR Instruments (strict ``EUR`` currency + integrity verification):**
  - Tail Hedge: tries ``LVO.PA`` first (Amundi/Lyxor S&P 500 VIX Futures Enhanced Roll
    UCITS ETF, ISIN ``LU0832435464``); falls back cleanly to same-fund native-EUR listings
    ``LVO.MI`` (Borsa Italiana) / ``VOOL.DE`` (Xetra) before ``VIXL.MI`` / ``VIXL.DE``,
    rejecting delisted tickers, non-EUR listings, stale zero-volume flatlines, and
    unadjusted reverse-split price jumps.
  - 2x Equity: ``CL2.PA`` (Amundi MSCI USA Daily 2x UCITS ETF, Euronext Paris; reuses cache).
  - 3x Equity: ``QQQ3.MI`` (WisdomTree Nasdaq-100 3x Daily ETP, Borsa Italiana; reuses cache).
  - Benchmark: ``CSPX.AS`` (iShares Core S&P 500 UCITS ETF EUR; reuses cache).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.data.bar_utils import normalise_ohlcv
from src.data.cache import DiskCache
from src.data.cross_asset_eu_loader import (
    HistoryFetcher,
    MetaFetcher,
    TickerMetadata,
    _default_history_fetcher,
    _default_meta_fetcher,
    _extract_splits_from_frame,
    _max_flat_close_run,
)
from src.domain.models import BarSeries

log = logging.getLogger(__name__)

InstrumentRole = Literal["us_signal", "tail_hedge", "equity_2x", "equity_3x", "benchmark"]

# Primary prompt candidate tried first, then exact same Amundi/Lyxor VIX Enhanced Roll
# UCITS ETF (ISIN LU0832435464) on Borsa Italiana (LVO.MI) and Xetra (VOOL.DE) in native
# EUR, followed by WisdomTree 2.25x VIX ETP fallbacks (VIXL.MI / VIXL.DE).
DEFAULT_TAIL_HEDGE_CANDIDATES: tuple[str, ...] = (
    "LVO.PA",
    "LVO.MI",
    "VOOL.DE",
    "VIXL.MI",
    "VIXL.DE",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VixInstrumentAudit(_Frozen):
    """Complete ticker resolution and data-integrity audit record."""

    role: InstrumentRole
    label: str
    symbol: str
    currency: str
    exchange: str
    name: str
    candidates_tried: list[str]
    rejected_candidates: dict[str, str]
    rows_total: int
    rows_warmup: int
    rows_5y: int
    first_date: str
    last_date: str
    splits_all: dict[str, float]
    splits_in_5y: dict[str, float]
    adj_close_min_5y: float
    adj_close_max_5y: float
    raw_close_min_5y: float
    raw_close_max_5y: float
    raw_close_start: float
    raw_close_oos_start: float | None
    raw_close_end: float
    zero_vol_days: int
    max_flat_close_run: int
    max_abs_1d_return_pct: float
    synthetic: bool = False


@dataclass(frozen=True)
class VixTermDataset:
    """Validated real-data bundle for Strategy v7."""

    qqq: BarSeries
    vix: BarSeries
    vix3m: BarSeries
    tail_hedge: BarSeries
    equity_2x: BarSeries
    equity_3x: BarSeries
    benchmark: BarSeries
    tail_hedge_symbol: str
    audits: dict[str, VixInstrumentAudit]


class VixTermLoader:
    """Download, validate, and cache VIX term-structure signals and EUR UCITS instruments."""

    def __init__(
        self,
        cache_root: Path,
        *,
        meta_fetcher: MetaFetcher | None = None,
        history_fetcher: HistoryFetcher | None = None,
    ) -> None:
        self.cache = DiskCache(cache_root, "yfinance")
        self._meta_fetcher = meta_fetcher or _default_meta_fetcher
        self._history_fetcher = history_fetcher or _default_history_fetcher

    def get_metadata(self, symbol: str) -> TickerMetadata:
        status = self.cache.status("ticker_meta", symbol)
        if status == "ok":
            df = self.cache.read("ticker_meta", symbol)
            if df is not None and not df.empty:
                row = df.iloc[0]
                return TickerMetadata(
                    symbol=symbol,
                    currency=None if pd.isna(row["currency"]) else str(row["currency"]),
                    exchange=None if pd.isna(row["exchange"]) else str(row["exchange"]),
                    name=None if pd.isna(row["name"]) else str(row["name"]),
                )
        if status == "empty":
            return TickerMetadata(symbol=symbol, currency=None, exchange=None, name=None)

        meta = self._meta_fetcher(symbol)
        if meta.currency is None and meta.exchange is None and meta.name is None:
            self.cache.write("ticker_meta", symbol, None, status="empty")
            return meta

        df_meta = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "currency": meta.currency,
                    "exchange": meta.exchange,
                    "name": meta.name,
                }
            ]
        )
        self.cache.write("ticker_meta", symbol, df_meta, status="ok")
        return meta

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

    def validate_eu_candidate(
        self,
        symbol: str,
        *,
        warmup_start: date,
        eval_start: date,
        eval_end: date,
        min_warmup_bars: int = 200,
        min_eval_bars: int = 1200,
        max_end_lag_days: int = 7,
        max_flat_run: int = 20,
        max_unsplit_jump: float = 1.00,
    ) -> tuple[BarSeries | None, TickerMetadata, str | None]:
        """Validate native EUR currency, coverage, and integrity guards for an EU ticker."""
        meta = self.get_metadata(symbol)
        if meta.currency != "EUR":
            return None, meta, f"non-EUR currency ({meta.currency!r})"

        series = self.get_bars(symbol, warmup_start, eval_end)
        if series is None or series.frame.empty:
            return None, meta, "no OHLCV bars returned (empty/delisted)"

        df = series.frame
        first_dt = df.index[0].date()
        last_dt = df.index[-1].date()
        pre_df = df.loc[df.index < pd.Timestamp(eval_start)]
        eval_df = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(eval_end))]

        if len(eval_df) < min_eval_bars:
            return (
                None,
                meta,
                f"insufficient 5Y bars ({len(eval_df)} < {min_eval_bars}; range {first_dt}..{last_dt})",
            )
        if (eval_end - last_dt).days > max_end_lag_days:
            return None, meta, f"history ends too early ({last_dt} vs {eval_end})"
        if len(pre_df) < min_warmup_bars:
            return (
                None,
                meta,
                f"insufficient warmup bars before {eval_start} ({len(pre_df)} < {min_warmup_bars}; first={first_dt})",
            )

        flat_run = _max_flat_close_run(eval_df["close"])
        if flat_run >= max_flat_run:
            return (
                None,
                meta,
                f"corrupted stale flatline detected ({flat_run} consecutive identical closes)",
            )

        max_ret = float(eval_df["close"].pct_change().dropna().abs().max()) if len(eval_df) > 1 else 0.0
        if max_ret > max_unsplit_jump:
            return (
                None,
                meta,
                f"unexplained single-day jump ({max_ret * 100:.1f}% > {max_unsplit_jump * 100:.0f}%)",
            )

        return series, meta, None

    def validate_us_signal(
        self,
        symbol: str,
        *,
        warmup_start: date,
        eval_start: date,
        eval_end: date,
        min_warmup_bars: int = 200,
        min_eval_bars: int = 1200,
        max_end_lag_days: int = 7,
        max_flat_run: int = 10,
        max_1d_jump: float = 1.50,
    ) -> tuple[BarSeries | None, TickerMetadata, str | None]:
        """Validate a US signal series (QQQ, ^VIX, ^VIX3M)."""
        meta = self.get_metadata(symbol)
        series = self.get_bars(symbol, warmup_start, eval_end)
        if series is None or series.frame.empty:
            return None, meta, "no OHLCV bars returned (empty)"

        df = series.frame
        first_dt = df.index[0].date()
        last_dt = df.index[-1].date()
        pre_df = df.loc[df.index < pd.Timestamp(eval_start)]
        eval_df = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(eval_end))]

        if len(eval_df) < min_eval_bars:
            return (
                None,
                meta,
                f"insufficient 5Y bars ({len(eval_df)} < {min_eval_bars}; range {first_dt}..{last_dt})",
            )
        if (eval_end - last_dt).days > max_end_lag_days:
            return None, meta, f"history ends too early ({last_dt} vs {eval_end})"
        if len(pre_df) < min_warmup_bars:
            return (
                None,
                meta,
                f"insufficient warmup bars before {eval_start} ({len(pre_df)} < {min_warmup_bars}; first={first_dt})",
            )

        flat_run = _max_flat_close_run(eval_df["close"])
        if flat_run >= max_flat_run:
            return (
                None,
                meta,
                f"stale flatline detected ({flat_run} consecutive identical closes)",
            )

        max_ret = float(eval_df["close"].pct_change().dropna().abs().max()) if len(eval_df) > 1 else 0.0
        if max_ret > max_1d_jump:
            return (
                None,
                meta,
                f"anomalous single-day jump ({max_ret * 100:.1f}% > {max_1d_jump * 100:.0f}%)",
            )

        return series, meta, None

    @staticmethod
    def _build_audit(
        *,
        role: InstrumentRole,
        label: str,
        symbol: str,
        meta: TickerMetadata,
        series: BarSeries,
        candidates_tried: list[str],
        rejected_candidates: dict[str, str],
        eval_start: date,
        eval_end: date,
        oos_start: date | None,
        default_currency: str,
    ) -> VixInstrumentAudit:
        df = series.frame
        pre_df = df.loc[df.index < pd.Timestamp(eval_start)]
        df5 = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(eval_end))]
        splits_all = _extract_splits_from_frame(df)
        splits_5y = {
            dt: ratio for dt, ratio in splits_all.items() if str(eval_start) <= dt <= str(eval_end)
        }
        oos_px: float | None = None
        if oos_start is not None:
            oos_sub = df5.loc[df5.index >= pd.Timestamp(oos_start)]
            if not oos_sub.empty:
                oos_px = round(float(oos_sub["raw_close"].iloc[0]), 4)
        max_ret = float(df5["close"].pct_change().dropna().abs().max()) if len(df5) > 1 else 0.0

        return VixInstrumentAudit(
            role=role,
            label=label,
            symbol=symbol,
            currency=meta.currency or default_currency,
            exchange=meta.exchange or "UNKNOWN",
            name=meta.name or symbol,
            candidates_tried=candidates_tried,
            rejected_candidates=rejected_candidates,
            rows_total=len(df),
            rows_warmup=len(pre_df),
            rows_5y=len(df5),
            first_date=str(df.index[0].date()),
            last_date=str(df.index[-1].date()),
            splits_all=splits_all,
            splits_in_5y=splits_5y,
            adj_close_min_5y=round(float(df5["close"].min()), 4),
            adj_close_max_5y=round(float(df5["close"].max()), 4),
            raw_close_min_5y=round(float(df5["raw_close"].min()), 4),
            raw_close_max_5y=round(float(df5["raw_close"].max()), 4),
            raw_close_start=round(float(df5["raw_close"].iloc[0]), 4),
            raw_close_oos_start=oos_px,
            raw_close_end=round(float(df5["raw_close"].iloc[-1]), 4),
            zero_vol_days=int((df5["volume"] == 0).sum()),
            max_flat_close_run=_max_flat_close_run(df5["close"]),
            max_abs_1d_return_pct=round(max_ret * 100.0, 2),
            synthetic=False,
        )

    def load_dataset(
        self,
        *,
        warmup_start: date = date(2020, 10, 18),
        eval_start: date = date(2021, 11, 22),
        eval_end: date = date(2026, 11, 20),
        oos_start: date | None = date(2024, 11, 22),
        tail_hedge_candidates: tuple[str, ...] = DEFAULT_TAIL_HEDGE_CANDIDATES,
        equity_2x_symbol: str = "CL2.PA",
        equity_3x_symbol: str = "QQQ3.MI",
        benchmark_symbol: str = "CSPX.AS",
    ) -> VixTermDataset:
        """Load and audit all US signal series and EUR tradable instruments."""
        audits: dict[str, VixInstrumentAudit] = {}

        # 1. US Signals: QQQ, ^VIX, ^VIX3M
        us_loaded: dict[str, BarSeries] = {}
        for key, sym, lbl in (
            ("qqq", "QQQ", "Invesco QQQ Trust (US Signal)"),
            ("vix", "^VIX", "CBOE 30-Day VIX Index (US Signal)"),
            ("vix3m", "^VIX3M", "CBOE 3-Month VIX Index (US Signal)"),
        ):
            series, meta, reason = self.validate_us_signal(
                sym,
                warmup_start=warmup_start,
                eval_start=eval_start,
                eval_end=eval_end,
            )
            if reason is not None or series is None:
                raise RuntimeError(f"US signal '{sym}' failed integrity check: {reason}")
            us_loaded[key] = series
            audit = self._build_audit(
                role="us_signal",
                label=lbl,
                symbol=sym,
                meta=meta,
                series=series,
                candidates_tried=[sym],
                rejected_candidates={},
                eval_start=eval_start,
                eval_end=eval_end,
                oos_start=oos_start,
                default_currency="USD",
            )
            audits[key] = audit
            log.info(
                "Loaded US signal %-7s (%s, %s) | rows=%d (warmup=%d, 5Y=%d) [%s..%s]",
                sym,
                audit.currency,
                audit.exchange,
                audit.rows_total,
                audit.rows_warmup,
                audit.rows_5y,
                audit.first_date,
                audit.last_date,
            )

        # 2. EU Tail-Hedge VIX UCITS ETF (candidate fallback with strict EUR + integrity guards)
        tried: list[str] = []
        rejected: dict[str, str] = {}
        tail_series: BarSeries | None = None
        tail_meta: TickerMetadata | None = None
        tail_sym: str | None = None

        for cand in tail_hedge_candidates:
            tried.append(cand)
            series, meta, reason = self.validate_eu_candidate(
                cand,
                warmup_start=warmup_start,
                eval_start=eval_start,
                eval_end=eval_end,
            )
            if reason is not None or series is None:
                rejected[cand] = reason or "validation failed"
                log.info("Tail-hedge candidate %s rejected -> %s", cand, rejected[cand])
                continue
            tail_series = series
            tail_meta = meta
            tail_sym = cand
            break

        if tail_series is None or tail_meta is None or tail_sym is None:
            raise RuntimeError(f"No valid EUR VIX tail-hedge ETF resolved; rejected={rejected}")

        audits["tail_hedge"] = self._build_audit(
            role="tail_hedge",
            label="EUR VIX Tail-Hedge UCITS ETF",
            symbol=tail_sym,
            meta=tail_meta,
            series=tail_series,
            candidates_tried=tried,
            rejected_candidates=rejected,
            eval_start=eval_start,
            eval_end=eval_end,
            oos_start=oos_start,
            default_currency="EUR",
        )
        log.info(
            "Resolved tail_hedge -> %-8s (%s, %s) | rows=%d (warmup=%d, 5Y=%d) [%s..%s] splits=%s rejected=%s",
            tail_sym,
            audits["tail_hedge"].currency,
            audits["tail_hedge"].exchange,
            audits["tail_hedge"].rows_total,
            audits["tail_hedge"].rows_warmup,
            audits["tail_hedge"].rows_5y,
            audits["tail_hedge"].first_date,
            audits["tail_hedge"].last_date,
            audits["tail_hedge"].splits_all or "none",
            rejected or "none",
        )

        # 3. EU Equity & Benchmark ETFs (strict EUR + integrity verification; reuses cache)
        eu_loaded: dict[str, BarSeries] = {}
        for key, role, sym, lbl in (
            ("equity_2x", "equity_2x", equity_2x_symbol, "2x MSCI USA UCITS ETF"),
            ("equity_3x", "equity_3x", equity_3x_symbol, "3x Nasdaq-100 ETP"),
            ("benchmark", "benchmark", benchmark_symbol, "1x S&P 500 UCITS Benchmark"),
        ):
            series, meta, reason = self.validate_eu_candidate(
                sym,
                warmup_start=warmup_start,
                eval_start=eval_start,
                eval_end=eval_end,
            )
            if reason is not None or series is None:
                raise RuntimeError(f"EU instrument '{sym}' ({key}) failed integrity check: {reason}")
            eu_loaded[key] = series
            audits[key] = self._build_audit(
                role=role,
                label=lbl,
                symbol=sym,
                meta=meta,
                series=series,
                candidates_tried=[sym],
                rejected_candidates={},
                eval_start=eval_start,
                eval_end=eval_end,
                oos_start=oos_start,
                default_currency="EUR",
            )
            log.info(
                "Loaded EU %-10s -> %-8s (%s, %s) | rows=%d (warmup=%d, 5Y=%d) [%s..%s] splits=%s",
                key,
                sym,
                audits[key].currency,
                audits[key].exchange,
                audits[key].rows_total,
                audits[key].rows_warmup,
                audits[key].rows_5y,
                audits[key].first_date,
                audits[key].last_date,
                audits[key].splits_all or "none",
            )

        return VixTermDataset(
            qqq=us_loaded["qqq"],
            vix=us_loaded["vix"],
            vix3m=us_loaded["vix3m"],
            tail_hedge=tail_series,
            equity_2x=eu_loaded["equity_2x"],
            equity_3x=eu_loaded["equity_3x"],
            benchmark=eu_loaded["benchmark"],
            tail_hedge_symbol=tail_sym,
            audits=audits,
        )
