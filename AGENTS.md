# AGENTS.md: rules for humans and AI agents working in this repo

## Golden rule

**Never introduce look-ahead bias, and never alter risk math outside `src/risk/risk_manager.py`.**

- A decision made at the close of session *t* may only use bars dated <= *t* and events
  whose reaction session is <= *t*. Entries fill at the open of *t+1* at the earliest.
- Stop distance, risk fraction, share count and settled-cash rules live only in
  `risk_manager.py`. Other modules call it; they do not re-derive or tweak it.
- Any change touching timing must keep `tests/test_lookahead.py` green (it scrambles all
  data after a cutoff and asserts every pre-cutoff decision is unchanged).

## Layout and ownership

| Path | Responsibility | May depend on |
|---|---|---|
| `config/settings.yaml`, `src/config.py` | every tunable (thresholds, fees, risk), pydantic-settings | - |
| `src/domain/models.py` | frozen pydantic contracts (`EarningsEvent`, `DailyBar`, `BarSeries`, `MarketRegime`, `TradeSignal`, `OrderIntent`, `Position`, `ClosedTrade`, ...) | - |
| `src/data/provider_base.py` | `MarketDataProvider` ABC | domain |
| `src/data/fmp_massive_provider.py` | FMP + Massive (Polygon) keyed sources | data base, domain |
| `src/data/yfinance_provider.py`, `universe.py` | free fallback + Wikipedia S&P 400/600 universe | data base, domain |
| `src/data/sample_provider.py` | bundled offline SYNTHETIC dataset | data base, domain |
| `src/data/factory.py` | fallback chain: keys -> yfinance -> sample | data, config |
| `src/data/cache.py` | Parquet + SQLite cache | - |
| `src/strategy/indicators.py` | causal features (SMA, ATR, 52w high, point-in-time market cap, SUE) | config, domain |
| `src/strategy/regime_filter.py` | pure regime classification | config, domain |
| `src/strategy/pead_screener.py` | pure signal generation | config, domain |
| `src/risk/risk_manager.py` | sizing, stops, T+1 settlement ledger | config, domain |
| `src/execution/broker_base.py` | `ExecutionBroker` ABC (backtest now, IBKR `ib_async` later) | domain |
| `src/execution/backtest_broker.py` | simulated fills, frictions, cash account | risk, domain |
| `src/backtest/engine.py` | daily event loop, stop management | everything above via interfaces |
| `src/backtest/metrics.py`, `reporting.py` | statistics and exports | domain |
| `scripts/run_backtest.py` | CLI entry point | backtest |

Modules communicate only through `src/domain/models.py` types and the two ABCs. Swap a
data source by implementing `MarketDataProvider`; swap execution by implementing
`ExecutionBroker`. Nothing in `strategy/` or `risk/` performs I/O.

## Coding standards

- Python 3.11+, full type hints, `from __future__ import annotations`.
- Pydantic v2 models are frozen; evolve with `model_copy(update=...)`.
- New tunables go in `settings.yaml` + `src/config.py`, never as literals in logic.
- Every behaviour change ships with pytest coverage (`python -m pytest`).
- Never present synthetic data as real: `SampleProvider.is_synthetic` must propagate to
  every report/plot title.
- Never commit `.env`, virtualenvs, `data/cache/` or `outputs/`.
