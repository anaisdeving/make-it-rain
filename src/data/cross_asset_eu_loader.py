"""Real EUR cross-asset ETP/ETC data loader with currency & integrity verification.

Resolves European-listed PRIIPs-compliant ETPs/ETCs trading natively in EUR on
Xetra (``.DE``), Euronext (``.AS`` / ``.PA``), and Borsa Italiana (``.MI``) from
Yahoo Finance with zero synthetic fallbacks.

Key guarantees:
- Strict ``EUR`` currency verification (never mixes USD, GBP/GBp, or CHF into EUR
  portfolios).
- Strict data-integrity verification: rejects empty/delisted tickers, insufficient
  history, severe stale zero-volume flatlines (e.g. ``PHAG.AS`` had 905 consecutive
  frozen zero-volume closes at EUR 18.849 followed by a bogus +215.8% jump on
  2026-01-08), or unrecorded >100% single-day price jumps.
- Split-adjusted OHLCV + reconstructed ``raw_close`` via ``normalise_ohlcv`` so
  indicators/returns/peak tracking use split-adjusted prices while IBKR whole-share
  sizing uses historical as-traded prices (matching Strategy v5).
- Reuses ``data/cache/yfinance/`` Parquet + SQLite ``fetch_log.sqlite`` layout so
  existing cached bars (``QQQ3.MI``, ``CL2.PA``, ``CSPX.AS``) and subsequent runs
  execute instantaneously with zero network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.bar_utils import normalise_ohlcv
from src.data.cache import DiskCache
from src.data.yfinance_provider import _Transient, _with_retry
from src.domain.models import BarSeries

log = logging.getLogger(__name__)

AssetClass = Literal["equity", "crypto", "metal"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BucketSpec(_Frozen):
    """Candidate resolution spec for one cross-asset bucket."""

    bucket: str
    label: str
    asset_class: AssetClass
    vol_cap: float = Field(gt=0.0)
    candidates: tuple[str, ...]


DEFAULT_CROSS_ASSET_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec(
        bucket="equities_3x",
        label="3x Nasdaq-100 ETP",
        asset_class="equity",
        vol_cap=0.45,
        candidates=("QQQ3.MI",),
    ),
    BucketSpec(
        bucket="btc",
        label="Bitcoin ETP",
        asset_class="crypto",
        vol_cap=0.85,
        candidates=("BTCE.DE", "21XB.DE", "VBTC.DE"),
    ),
    BucketSpec(
        bucket="eth",
        label="Ethereum ETP",
        asset_class="crypto",
        vol_cap=0.85,
        candidates=("ZETH.DE", "21XE.DE", "VETH.DE"),
    ),
    BucketSpec(
        bucket="silver",
        label="Silver ETC",
        asset_class="metal",
        vol_cap=0.45,
        # Primary prompt candidates tried first; PHAG.MI (same WisdomTree Physical Silver
        # ETC ISIN JE00B1VS3333 on Borsa Italiana in EUR) and XAD6.DE added as clean EUR
        # fallbacks because Yahoo's PHAG.AS feed is corrupted (905 flatlined bars),
        # SSLN.MI has 1 bar, and VZLC.DE is delisted/empty.
        candidates=("PHAG.AS", "SSLN.MI", "VZLC.DE", "PHAG.MI", "XAD6.DE"),
    ),
    BucketSpec(
        bucket="gold",
        label="Gold ETC",
        asset_class="metal",
        vol_cap=0.45,
        candidates=("SGLD.MI", "PHAU.AS", "4GLD.DE"),
    ),
)

EQUITIES_2X_BUCKET = BucketSpec(
    bucket="equities_2x",
    label="2x MSCI USA UCITS",
    asset_class="equity",
    vol_cap=0.45,
    candidates=("CL2.PA",),
)


class ResolvedAssetAudit(_Frozen):
    """Complete resolution & data-quality audit record for one resolved bucket."""

    bucket: str
    label: str
    asset_class: AssetClass
    vol_cap: float
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
    coverage_caveat: str | None = None
    synthetic: bool = False


@dataclass(frozen=True)
class CrossAssetUniverse:
    """Resolved EUR universe with real bars and audit metadata."""

    bars: dict[str, BarSeries]
    audits: dict[str, ResolvedAssetAudit]
    bucket_to_symbol: dict[str, str]
    symbol_to_bucket: dict[str, str]
    vol_caps: dict[str, float]
    asset_classes: dict[str, AssetClass]


@dataclass(frozen=True)
class TickerMetadata:
    symbol: str
    currency: str | None
    exchange: str | None
    name: str | None


MetaFetcher = Callable[[str], TickerMetadata]
HistoryFetcher = Callable[[str, date, date], tuple[pd.DataFrame | None, pd.Series | None]]


def _default_meta_fetcher(symbol: str) -> TickerMetadata:
    import yfinance as yf

    t = yf.Ticker(symbol)
    curr: str | None = None
    exch: str | None = None
    name: str | None = None

    def _get_fi() -> dict[str, object]:
        fi = t.fast_info
        out: dict[str, object] = {}
        for k in ("currency", "exchange"):
            try:
                out[k] = fi.get(k)
            except Exception:
                out[k] = None
        return out

    try:
        fi_data = _with_retry(_get_fi, f"fast_info {symbol}", attempts=4) or {}
        curr = str(fi_data["currency"]) if fi_data.get("currency") else None
        exch = str(fi_data["exchange"]) if fi_data.get("exchange") else None
    except _Transient as exc:
        log.warning("fast_info failed for %s: %s", symbol, exc)

    try:
        info = _with_retry(lambda: t.info or {}, f"info {symbol}", attempts=3) or {}
        if not curr and info.get("currency"):
            curr = str(info["currency"])
        if not exch and info.get("exchange"):
            exch = str(info["exchange"])
        raw_name = info.get("shortName") or info.get("longName")
        name = str(raw_name).strip() if raw_name else None
    except _Transient as exc:
        log.warning("info lookup failed for %s: %s", symbol, exc)

    return TickerMetadata(symbol=symbol, currency=curr, exchange=exch, name=name)


def _default_history_fetcher(symbol: str, start: date, end: date) -> tuple[pd.DataFrame | None, pd.Series | None]:
    import yfinance as yf

    yf.config.debug.hide_exceptions = False
    try:
        hist = _with_retry(
            lambda: yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
            ),
            f"bars {symbol}",
        )
    except _Transient as exc:
        log.warning("bars fetch failed (not cached) for %s: %s", symbol, exc)
        return None, None
    if hist is None or hist.empty:
        return None, None
    splits = hist["Stock Splits"] if "Stock Splits" in hist.columns else None
    return hist, splits


def _extract_splits_from_frame(df: pd.DataFrame) -> dict[str, float]:
    """Infer split dates & ratios directly from cached ``raw_close / close`` step changes."""
    if df.empty or "raw_close" not in df.columns:
        return {}
    ratio = (df["raw_close"] / df["close"]).round(6)
    # On split date t, ratio drops from R_{t-1} to R_t where R_{t-1} / R_t == split_ratio.
    prev = ratio.shift(1)
    splits: dict[str, float] = {}
    for ts in df.index[1:]:
        r_prev = float(prev.loc[ts])
        r_cur = float(ratio.loc[ts])
        if r_cur > 0 and r_prev > 0 and abs(r_prev / r_cur - 1.0) > 1e-3:
            splits[str(ts.date())] = round(r_prev / r_cur, 4)
    return splits


def _max_flat_close_run(close: pd.Series) -> int:
    """Return longest run of consecutive identical closing prices."""
    if len(close) <= 1:
        return 0
    same = close.eq(close.shift(1)).to_numpy()
    best = 0
    cur = 0
    for flag in same:
        if flag:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


class CrossAssetEULoader:
    """Resolve, validate, and cache EUR-denominated cross-asset ETP/ETC daily bars."""

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

    def validate_candidate(
        self,
        symbol: str,
        *,
        warmup_start: date,
        eval_start: date,
        eval_end: date,
        min_warmup_bars: int = 100,
        min_eval_bars: int = 1200,
        max_end_lag_days: int = 7,
        max_flat_run: int = 20,
        max_unsplit_jump: float = 1.00,
    ) -> tuple[BarSeries | None, TickerMetadata, str | None]:
        """Validate currency (strictly EUR), coverage, and data integrity for one symbol.

        Returns ``(bar_series, metadata, rejection_reason)``. ``rejection_reason`` is
        ``None`` iff the candidate passes all checks.
        """
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

    def resolve_universe(
        self,
        buckets: tuple[BucketSpec, ...] = DEFAULT_CROSS_ASSET_BUCKETS,
        *,
        warmup_start: date = date(2020, 10, 18),
        eval_start: date = date(2021, 11, 22),
        eval_end: date = date(2026, 11, 20),
        oos_start: date | None = date(2024, 11, 22),
        min_warmup_bars: int = 100,
        ideal_warmup_bars: int = 200,
        min_eval_bars: int = 1200,
    ) -> CrossAssetUniverse:
        """Resolve the first clean EUR ticker with full 5Y + warmup history per bucket."""
        bars: dict[str, BarSeries] = {}
        audits: dict[str, ResolvedAssetAudit] = {}
        bucket_to_symbol: dict[str, str] = {}
        symbol_to_bucket: dict[str, str] = {}
        vol_caps: dict[str, float] = {}
        asset_classes: dict[str, AssetClass] = {}

        for spec in buckets:
            tried: list[str] = []
            rejected: dict[str, str] = {}
            resolved_series: BarSeries | None = None
            resolved_meta: TickerMetadata | None = None
            resolved_sym: str | None = None

            for sym in spec.candidates:
                tried.append(sym)
                series, meta, reason = self.validate_candidate(
                    sym,
                    warmup_start=warmup_start,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    min_warmup_bars=min_warmup_bars,
                    min_eval_bars=min_eval_bars,
                )
                if reason is not None:
                    rejected[sym] = reason
                    log.info("Bucket '%s': rejected %s -> %s", spec.bucket, sym, reason)
                    continue
                resolved_series = series
                resolved_meta = meta
                resolved_sym = sym
                break

            if resolved_series is None or resolved_meta is None or resolved_sym is None:
                raise RuntimeError(
                    f"No valid EUR ticker resolved for bucket '{spec.bucket}'; rejected={rejected}"
                )

            df = resolved_series.frame
            pre_df = df.loc[df.index < pd.Timestamp(eval_start)]
            df5 = df.loc[(df.index >= pd.Timestamp(eval_start)) & (df.index <= pd.Timestamp(eval_end))]
            splits_all = _extract_splits_from_frame(df)
            splits_5y = {
                dt: ratio
                for dt, ratio in splits_all.items()
                if str(eval_start) <= dt <= str(eval_end)
            }

            oos_px: float | None = None
            if oos_start is not None:
                oos_sub = df5.loc[df5.index >= pd.Timestamp(oos_start)]
                if not oos_sub.empty:
                    oos_px = round(float(oos_sub["raw_close"].iloc[0]), 4)

            caveat: str | None = None
            if len(pre_df) < ideal_warmup_bars:
                caveat = (
                    f"Listed {df.index[0].date()}: {len(pre_df)} pre-IS warmup bars "
                    f"(< {ideal_warmup_bars} requested, >= {min_warmup_bars} required for SMA(100))"
                )

            audit = ResolvedAssetAudit(
                bucket=spec.bucket,
                label=spec.label,
                asset_class=spec.asset_class,
                vol_cap=spec.vol_cap,
                symbol=resolved_sym,
                currency=resolved_meta.currency or "EUR",
                exchange=resolved_meta.exchange or "UNKNOWN",
                name=resolved_meta.name or resolved_sym,
                candidates_tried=tried,
                rejected_candidates=rejected,
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
                coverage_caveat=caveat,
                synthetic=False,
            )

            bars[resolved_sym] = resolved_series
            audits[spec.bucket] = audit
            bucket_to_symbol[spec.bucket] = resolved_sym
            symbol_to_bucket[resolved_sym] = spec.bucket
            vol_caps[resolved_sym] = spec.vol_cap
            asset_classes[resolved_sym] = spec.asset_class

            log.info(
                "Resolved %-12s -> %-8s (%s, %s) | rows=%d (warmup=%d, 5Y=%d) [%s..%s] splits=%s%s",
                spec.bucket,
                resolved_sym,
                audit.currency,
                audit.exchange,
                audit.rows_total,
                audit.rows_warmup,
                audit.rows_5y,
                audit.first_date,
                audit.last_date,
                audit.splits_all or "none",
                f" | CAVEAT: {caveat}" if caveat else "",
            )

        return CrossAssetUniverse(
            bars=bars,
            audits=audits,
            bucket_to_symbol=bucket_to_symbol,
            symbol_to_bucket=symbol_to_bucket,
            vol_caps=vol_caps,
            asset_classes=asset_classes,
        )
