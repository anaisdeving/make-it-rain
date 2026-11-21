"""Bundled offline dataset: deterministic, seeded SYNTHETIC market data.

Used when no keys are configured and public endpoints are unreachable, and by the
test-suite. Symbols are prefixed ``SYN`` and every artefact produced from this
provider is labelled synthetic: it exercises the pipeline end-to-end but says
nothing about real-world strategy performance.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.data.provider_base import MarketDataProvider
from src.domain.models import AnnouncementTiming, BarSeries, EarningsEvent, UniverseMember


def _business_days(start: date, end: date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator, vol: float) -> tuple[np.ndarray, ...]:
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, vol * 0.3, len(close)))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol * 0.5, len(close))))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol * 0.5, len(close))))
    return open_, hi, lo


class SampleProvider(MarketDataProvider):
    name = "sample"
    is_synthetic = True

    def __init__(self, start: date, end: date, n_symbols: int = 60, seed: int = 7) -> None:
        self.start, self.end, self.n, self.seed = start, end, n_symbols, seed
        self._bars: dict[str, BarSeries] = {}
        self._events: list[EarningsEvent] = []
        self._shares: dict[str, pd.Series] = {}
        self._spy: BarSeries | None = None
        self._generate()

    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        idx = _business_days(self.start, self.end)
        n = len(idx)

        # Benchmark with an embedded drawdown + high-vol stretch to exercise all regimes.
        mu = np.full(n, 0.0004)
        sig = np.full(n, 0.009)
        a, b = int(n * 0.35), int(n * 0.50)
        mu[a:b], sig[a:b] = -0.0025, 0.026
        rets = rng.normal(mu, sig)
        spy_close = 400 * np.exp(np.cumsum(rets))
        o, h, l = _ohlc_from_close(spy_close, rng, 0.006)
        spy = pd.DataFrame({"open": o, "high": h, "low": l, "close": spy_close,
                            "volume": rng.uniform(6e7, 1.2e8, n), "raw_close": spy_close}, index=idx)
        self._spy = BarSeries(symbol="SPY", frame=spy)

        for i in range(self.n):
            sym = f"SYN{i:03d}"
            daily_vol = rng.uniform(0.015, 0.03)
            drift = rng.normal(0.0003, 0.0004)
            r = rng.normal(drift, daily_vol, n) + 0.8 * (rets - rets.mean())
            base_vol = rng.uniform(4e5, 2e6)
            volume = base_vol * np.exp(rng.normal(0, 0.25, n))
            first = int(rng.integers(5, 60))
            for k in range(first, n - 2, 63):
                timing = AnnouncementTiming.BMO if rng.random() < 0.5 else AnnouncementTiming.AMC
                react = k if timing == AnnouncementTiming.BMO else k + 1
                surprise = rng.normal(4.0, 12.0)
                jump = 0.012 * surprise + rng.normal(0, 0.03)
                r[react] += jump
                volume[react] *= rng.uniform(1.5, 6.0)
                drift_days = min(30, n - react - 1)
                if drift_days > 0:
                    r[react + 1 : react + 1 + drift_days] += np.sign(surprise) * 0.0015
                hour = 7 if timing == AnnouncementTiming.BMO else 16
                est = rng.uniform(0.2, 2.0)
                self._events.append(EarningsEvent(
                    symbol=sym,
                    announced_at=datetime.combine(idx[k].date(), datetime.min.time()).replace(hour=hour, minute=5),
                    timing=timing, eps_estimate=round(est, 2),
                    eps_actual=round(est * (1 + surprise / 100), 2), eps_surprise_pct=round(surprise, 2),
                    source="sample:synthetic",
                ))
            close = rng.uniform(15, 90) * np.exp(np.cumsum(r))
            o, h, l = _ohlc_from_close(close, rng, daily_vol)
            last = len(r) - 1
            for k in range(1, last + 1):
                if abs(r[k]) > 0.04:  # event days: realistic gap open, strong close location
                    o[k] = close[k - 1] * np.exp(r[k] * 0.7)
                    h[k] = max(o[k], close[k]) * (1 + abs(rng.normal(0, 0.006)))
                    l[k] = min(o[k], close[k]) * (1 - abs(rng.normal(0, 0.012)))
            df = pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                               "volume": volume, "raw_close": close}, index=idx)
            self._bars[sym] = BarSeries(symbol=sym, frame=df)
            self._shares[sym] = pd.Series([rng.uniform(1e7, 1.5e8)], index=[idx[0]])

    def get_universe(self, as_of: date | None = None) -> list[UniverseMember]:
        return [UniverseMember(symbol=s, exchange="SYNTH", source="sample:synthetic") for s in self._bars]

    def get_daily_bars(self, symbols: list[str], start: date, end: date) -> dict[str, BarSeries]:
        out = {}
        for s in symbols:
            if s in self._bars:
                f = self._bars[s].frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
                out[s] = BarSeries(symbol=s, frame=f)
        return out

    def get_earnings_calendar(self, symbols: list[str], start: date, end: date) -> list[EarningsEvent]:
        wanted = set(symbols)
        return [e for e in self._events if e.symbol in wanted and start <= e.announced_at.date() <= end]

    def get_benchmark_bars(self, start: date, end: date, symbol: str = "SPY") -> BarSeries:
        assert self._spy is not None
        return BarSeries(symbol=symbol, frame=self._spy.frame.loc[pd.Timestamp(start) : pd.Timestamp(end)])

    def get_shares_outstanding(self, symbols: list[str], start: date, end: date) -> dict[str, pd.Series]:
        return {s: self._shares[s] for s in symbols if s in self._shares}

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.name,
            "synthetic": "True",
            "bars": f"SYNTHETIC seeded GBM (seed={self.seed}) - NOT market data",
            "earnings": "SYNTHETIC earnings events - NOT market data",
            "universe": f"{self.n} synthetic symbols",
            "market_cap": "SYNTHETIC constant share counts",
        }


def sample_window(end: date, years: int = 6) -> tuple[date, date]:
    return end - timedelta(days=365 * years), end
