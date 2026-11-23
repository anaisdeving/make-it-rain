# make-it-rain

Phase 0 of a PEAD (post-earnings-announcement drift) + regime-filtered momentum system
for US equities: a modular Python codebase and a zero-look-ahead daily backtester sized
for a small IBKR Pro cash account (EUR 1,000 to 5,000). Read [`AGENTS.md`](AGENTS.md)
before changing anything.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add FMP_API_KEY / MASSIVE_API_KEY
python -m pytest              # test-suite
python scripts/run_backtest.py --capital-eur 1000 5000 --vps both
python scripts/run_backtest.py --provider sample   # fully offline, SYNTHETIC data
```

Outputs land in `outputs/` (git-ignored): `equity-curve-<tag>.png/.html`,
`trade-log-<tag>.csv`, `skipped-entries-<tag>.csv`, `equity-<tag>.csv`,
`metrics-<tag>.json`, `summary.json`, and `trade_log.csv` (first capital level, no VPS).

## Data fallback chain (`src/data/factory.py`)

| Capability | `FMP_API_KEY` set | `MASSIVE_API_KEY` set | no keys, Yahoo reachable | offline |
|---|---|---|---|---|
| Daily bars | yfinance | Massive aggregates + splits | yfinance | synthetic sample |
| Earnings (timing, surprise) | FMP (BMO/AMC, EPS + revenue) | yfinance | yfinance (EPS only) | synthetic |
| Market cap (point-in-time) | FMP historical market cap | shares x price (yfinance) | yfinance shares x as-traded close | synthetic |
| Universe | FMP screener + delisted list | Wikipedia S&P 400/600 | Wikipedia S&P 400/600 + removals | synthetic |

Everything is cached under `data/cache/` (Parquet frames + SQLite fetch log, including
negative results), so repeat runs are instant and stay inside free rate limits.
Rate-limit errors are retried with backoff and never cached as "no data".
The FMP and Massive adapters follow the vendors' documented schemas but were not
exercised against live keys in Phase 0.

## Strategy summary

- **Regime** (`regime_filter.py`): SPY vs SMA200 and 20-day annualised realised vol.
  NORMAL (SPY > SMA200 and vol < 22%): full risk, min event-day gap +5%, max 3 positions.
  CAUTION (otherwise): risk x0.5, min gap +8%, max 2 positions.
  PANIC (SPY < SMA200 and vol > 35%): no new entries.
- **Screener** (`pead_screener.py`): price $5-$150 (as-traded), market cap $300M-$10B
  (point-in-time), 20-day avg dollar volume >= $5M, EPS surprise > 0 (and revenue > 0 when
  known) or SUE > 1.5, event-day return >= regime gap, volume >= 2.5x the prior 20-day
  average, close in the top 35% of the day's range, close > SMA50, within 20% of the
  52-week high.
- **Timing**: BMO/intraday releases react the same session, AMC/unknown the next session.
  Signals are evaluated at the reaction-session close; entry is a limit at close x 1.003
  on the next open, skipped if the open gaps above it.
- **Risk** (`risk_manager.py`): d = min(max(1.5 ATR14, entry - event-day low), 15% entry);
  shares = floor(min(equity x f / d, 50% equity / entry, settled cash / entry)), f = 5%.
  T+1 settled cash only (no good-faith violations).
- **Exits**: hard stop (gap-through fills at the open), breakeven at +1.5R, trail at
  highest high - 2.5 ATR14 from +2R, time stop after 45 sessions if below +1R.
- **Friction**: $0.38 commission per side, 0.25% slippage per side, optional EUR 6.50/month VPS.

All parameters are in `config/settings.yaml`.
