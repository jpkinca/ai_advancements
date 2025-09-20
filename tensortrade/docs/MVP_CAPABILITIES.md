# TensorTrade MVP Capabilities

Updated: 2025-08-17

## Overview

A lean, reproducible daily US equities ingestion + risk-aware RL training scaffold built on IBKR historical data, PostgreSQL persistence, and TensorTrade environment abstractions. Emphasis is on reliability, transparency, and minimal moving parts while enabling incremental extension.

## Data Ingestion

- **Source**: IBKR (ib_insync) only (no yfinance, no demo data).
- **Universe**: Watchlist tickers from `data/tensorwatchlist.csv` (subset via CLI `--limit`).
- **Intervals**: Daily bars (base). Intraday placeholders reserved for future.
- **Lean Mode (default)**: SMART exchange, `whatToShow=TRADES` with automatic fallback to `MIDPOINT`.
- **Advanced Mode**: Optional (`--advanced-fetch`) adds exchange fallback sequence, listing-date adjustment, retry & classification of no-data scenarios.
- **Rate Limiting**: Configurable per-request delay (`--rate-limit`) + batching to respect IBKR pacing.

### Fetch Enhancements

- Contract qualification (primary exchange detection) for lean path.
- Retry logic with exponential backoff (advanced path).
- Primary exchange fallback if SMART returns no bars.
- Listing date heuristic to avoid impossible windows.

## Persistence (PostgreSQL)

Auto-created minimal schema:

- `tt_prices(instrument, timestamp, open, high, low, close, volume)` with unique index (instrument, timestamp).
- `tt_episode(start_time, end_time, stop_reason, max_drawdown, sharpe_ratio, turnover)`.
- `tt_action(episode_id, timestamp, instrument, action_value)`.
- `tt_reward(episode_id, timestamp, reward_value)`.

Idempotent bar insertion (ON CONFLICT DO NOTHING) prevents duplicates. Indexes support fast training/evaluation queries.

## Feature Engineering

Pipeline features (computed post-ingestion):

- Daily return: `return_1`.
- Rolling volatility windows: `vol_10`, `vol_20`.

Training feed optional engineered streams (`--with-features`):

- Close price.
- 1-step return.
- Rolling volatility (5 & 10 periods).
- Normalized volume (z-score style).

Design keeps feature expansion decoupled and easily extendable.

## Environment & Schemes

Action schemes:

- **VolatilityTargetedAction**: Scales position exposures to hit a target daily risk (`--risk-target`) with leverage clamp (`--max-leverage`).
- **SimpleDiscreteAction**: Equal-weight per-instrument allocations (FLAT/LONG/(SHORT optional)), baseline/debug scheme.

Reward schemes:

- **RiskAwareReward**: Sharpe-like risk-adjusted return minus drawdown & turnover penalties.
- **Returns (fallback)**: Uses TensorTrade default RiskAdjustedReturns if selected.

Stopper:

- **DrawdownStopper**: Early episode termination on exceeding max drawdown (`--max-dd`) or equity floor (`--min-equity`).

## Training & Evaluation

- **Trainer**: Stable-Baselines3 PPO (`src/train_mvp.py`).
- **Configurable**: months of history, steps, symbol limit, schemes, evaluation episode count, advanced fetch, feature inclusion.
- **Outputs**: PPO model (zip), per-episode metrics (Sharpe proxy, max drawdown, turnover), per-step action & reward logs (evaluation; optional training logging).

## Diagnostics & Instrumentation

- Day-level data availability tool (classifies pre-listing vs. genuine no data).
- Verbose fetch summaries (success/no-data/failed counts).
- Action/reward DB logging wrapper (optional) for training introspection.

## Actionable Insights Enabled Now

| Insight | Source | Use Case |
|---------|--------|----------|
| Volatility profile (vol_10/20, engineered vol5/vol10) | Ingestion + features | Symbol stability screening, position sizing heuristics |
| Daily return distribution | `return_1`, reward logs | Return dispersion & anomaly detection |
| Drawdown behavior | `tt_episode.max_drawdown` | Risk management tuning, early stop calibration |
| Risk-adjusted performance | Sharpe proxy (evaluation) | Model selection & reward shaping iteration |
| Turnover / churn | `tt_episode.turnover` | Strategy efficiency & cost sensitivity |
| Allocation patterns | `tt_action` rows | Attribution analysis, exposure drift detection |
| Data coverage / gaps | Diagnostics script output | Universe curation, data quality SLA |

## Security & Data Integrity Notes

- Live data only (no synthetic fill).
- Centralized bar store prevents redundant IBKR calls (future caching extension).
- Idempotent writes reduce duplication risk.

## Current Limitations

| Area | Gap |
|------|-----|
| Asset classes | Equities only (no FX, futures) |
| Frequencies | Daily only (intraday pending) |
| Execution | No live order routing (paper/live staging not wired) |
| Analytics | No dashboard yet for visual equity curves |
| Feature depth | Limited set (no advanced TA, fundamentals, alternative data) |
| Strategy | Single PPO baseline (no ensemble / model selection) |

## Suggested Next Enhancements

1. Intraday bar ingestion + resampling logic.
2. Extended feature library (factor signals, regime indicators, cross-sectional ranks).
3. Portfolio analytics dashboard (time-series equity, risk attribution, heatmaps).
4. Paper trading adapter (order simulation with slippage & fees) then live adaptor gating.
5. CI integration (pytest + coverage + lint) across modules.
6. Config-driven watchlist segmentation (core vs. experimental symbols).
7. Model monitoring: drift detection on volatility & reward distributions.
8. Exportable data quality report (HTML/JSON) from diagnostics.

## Execution Quick Reference

```bash
# Ingestion (lean)
python -m src.mvp_pipeline --start 2025-05-01 --end 2025-08-01 --interval 1d --limit 10

# Training (with features & evaluation)
python -m src.train_mvp --months 3 --limit 10 --steps 50000 --with-features --action-scheme vol_target --eval-episodes 2

# Advanced fetch
python -m src.train_mvp --months 6 --advanced-fetch --fallback-wts MIDPOINT
```

## Architecture Snapshot

1. Fetch Layer (ib_insync + connect helper) → DataFrame of OHLCV.
2. Persistence Layer (PostgreSQL) → `tt_prices` & episodic tables.
3. Feature Layer → returns & vol metrics (expandable streams).
4. Environment Layer → Action/Reward/Stopper composition.
5. Trainer → PPO model training & evaluation log back into DB.
6. Diagnostics → Data quality & symbol availability classification.

## Glossary

- **Lean Fetch**: Minimal, fast path using SMART + limited fallback.
- **Advanced Fetch**: Robust path with exchange heuristics & listing awareness.
- **Turnover**: Mean absolute change in action vector across steps (proxy for trading intensity).

---
_This document will evolve as new capabilities (intraday, execution adapters, analytics dashboards) are added._
