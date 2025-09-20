# End-to-End Development Plan: Watchlist → Ingestion → Processing → Signals → Analytics

## 1. Watchlist & Configuration Management
- Source: `data/tensorwatchlist.csv` (single symbol column). Future: profiles (core, experimental).
- Validation: dedupe, uppercase normalization, IBKR contract resolution (exchange, currency, primary listing).
- Config: YAML/JSON (interval, start/end dates, capital, fees, leverage, risk params, mode backtest|paper|live).
- Versioning: snapshot hash stored in `tt_episode` metadata or a future `tt_watchlist_version` table.

## 2. Data Ingestion (IBKR-Centric)
Historical:
- Function: `fetch_price_history_ibkr(symbols, start, end, interval)` using `connect_me` + `ib_insync`.
- Pacing: sequential loop (Phase 1); Phase 2 add batching + retry (exponential backoff on pacing violations).
- Store raw bars into `tt_prices` (daily/hourly/minute). Add integrity audit (expected vs actual bar count).

Real-Time:
- Use `MarketDataStreamer` to subscribe to ticks; build in-memory bar aggregator (tick → interim OHLCV) per symbol.
- Warm start: last N bars pulled historically, then append new live bars.
- Graceful reconnection and missing tick gap detection.

Corporate Actions / Adjustments (Future):
- Dividends & splits table; adjusted vs unadjusted feed toggle.

## 3. Persistence Layer
- Operational store: existing SQLite inside `ibkr_api` for ticks (short retention window, e.g., 5 trading days).
- Analytical store: PostgreSQL (`tt_` schema) for bars, episodes, actions, rewards, orders, observations.
- Indexing: `(instrument, timestamp)` on `tt_prices`; `(episode_id, timestamp)` on action/reward/portfolio tables.
- Optional materialized views: episode summaries, daily performance rollups.

## 4. Data Processing & Feature Engineering
Feature Groups:
- Price: returns (1,5,20 bars), log returns, rolling mean/volatility, ATR, true range %, gap %, range compression.
- Volume: rolling z-score, volume spike indicators.
- Cross-Sectional: relative strength vs SPY or median of watchlist, momentum ranks.
- Regime: volatility regime label, trend classification (EMA slope / ADX).
Implementation:
- TensorTrade Streams per feature; dependency graph to avoid recomputation.
- Normalization: rolling z-score window (e.g., 60 bars) per feature; mask until window filled.
- Caching: optional feature store (persist computed features keyed by (symbol, timestamp, feature_set_hash)).

## 5. Environment Assembly (TensorTrade)
Components:
- ActionScheme: multi-asset `VolatilityTargetedAction` (existing, vector exposures; future: add exposure cap matrix & sector constraints).
- RewardScheme: `RiskAwareReward` (Sharpe-like + DD + turnover penalties).
- Stopper: `DrawdownStopper` (max drawdown + equity floor + warmup steps).
- Observer: builds stacked feature tensor (shape: [symbols, features, window]) or flattened vector.
- Portfolio: generated from watchlist; base currency USD; initial cash configurable.
- Configurable parameters: window_size, episode_length, long_only flag, leverage cap, risk target.

## 6. Training & Evaluation Pipeline
- Data Split: chronological (train/validation/test) or walk-forward (rolling windows).
- Algorithm: Stable Baselines3 PPO baseline (Phase 1). Optionally A2C, SAC later.
- Hyperparameter Tuning: Ray Tune integration (Phase 3).
- Checkpointing: model weights + environment config + feature set hash + seed.
- Metrics logged each episode: reward sum, Sharpe, Sortino, max DD, turnover, exposure utilization, win ratio.
- Reproducibility: fixed random seeds & deterministic settings where possible.

## 7. Signal & Order Lifecycle
- Policy Output: continuous exposures per symbol ([-1,1]) → scaled by volatility target → orders (proportion rebalances).
- Execution Simulation: fill at bar close (Phase 1). Future: slippage model (mid ± half-spread), partial fills.
- Logging: `tt_action` (raw action & target exposure), `tt_order` (simulated size, price), `tt_reward` (per step), `tt_portfolio` & `tt_holdings` (post-step state).
- Additional risk filter layer (Phase 2): block orders violating exposure or liquidity constraints.

## 8. Risk Management & Guardrails
Current:
- Drawdown stopper per episode.
- Volatility-based scaling of exposures.
Planned Enhancements:
- Hard per-symbol max allocation (e.g., ≤ 20%).
- Gross/net leverage caps.
- Circuit breaker: terminate paper/live session if runtime drawdown > threshold.
- Scenario stress test harness (apply synthetic price shocks to current state).

## 9. Analytics & Insights
Dashboards / Reports:
- Equity curve + drawdown chart.
- Rolling Sharpe & volatility.
- Allocation heatmap (symbol vs time).
- Action distribution (histogram of exposures; turnover analysis).
- Feature importance proxy (perturbation or SHAP on trained policy network – later phase if model introspection feasible).
Derived Views / Tables:
- `tt_episode_summary` (episode_id, start/end, sharpe, pnl, max_dd, turnover, avg_exposure).
- `tt_daily_performance` (date, pnl, returns, vol, drawdown running values).
Monitoring Metrics:
- Data coverage %, bar latency (live mode), drift metrics (feature mean/std vs training baseline).

## 10. Monitoring, Scheduling & Orchestration
- Backfill Job: nightly historical sync (post-close) for any missing bars.
- Training Job: configurable cadence (weekly or on-demand).
- Evaluation Job: runs latest model on rolling out-of-sample slice.
- Live/Paper Loop: event-driven (bar close triggers step) or scheduled (cron at interval boundary).
- Alerting (Phase 3): Telegram/email for data gaps, connection loss, exceed risk thresholds.

## 11. Deployment Modes
- Backtest: fully offline; deterministic sequence of historical bars.
- Paper: live IBKR data; orders simulated (no real execution) with real-time guardrails.
- Live (Phase 5+): integrate IBKR Order Manager for real submissions; maintain shadow simulation for divergence metrics.
- CLI Flags: `--mode`, `--config path`, `--interval`, `--resume-checkpoint`.

## 12. Testing & QA
Unit Tests:
- Reward calculations (edge: zero vol, large drawdown, high turnover).
- Action scaling (vol=0 fallback, leverage clamp, multi-symbol vector mapping).
- Feature transformations (rolling windows, NaN handling until warmup complete).
Integration Tests:
- End-to-end single-episode simulation with synthetic deterministic price series.
- IBKR historical fetch mock (simulate pacing/empty responses).
Data Quality Tests:
- Validate monotonic timestamps per symbol.
- Missing bar detector (expected vs present counts).
Performance Benchmarks:
- Steps/sec with N symbols and F features; memory usage check.
Regression Harness:
- Golden seed baseline metrics with tolerance bands.

## 13. Security & Secrets
- Credentials (.env): IBKR connection parameters, DB URL.
- Secret loading via centralized utility; never committed.
- Future: Hash integrity checks on feature config & model files; audit logs for live trades.

## 14. Roadmap (Incremental Delivery)
Phase 1 (MVP):
- IBKR historical fetch → store `tt_prices` → feature streams (basic returns & volatility) → environment → PPO training → episode metrics persisted.
Phase 2:
- Real-time bar builder, paper mode loop, extended features, order & action logging to DB.
Phase 3:
- Walk-forward evaluation, Ray Tune integration, monitoring & alerting skeleton, drift metrics.
Phase 4:
- Advanced risk (allocation caps, circuit breaker), slippage model, feature store caching.
Phase 5:
- Live order routing prototype with divergence tracking & scenario stress tests.
Phase 6:
- Analytics dashboards, attribution, model introspection (SHAP/perturb), automated report generation.

## 15. Key Early KPIs
- Historical ingestion completeness ≥ 99% bars.
- Training loop throughput ≥ 50 steps/sec (target baseline) given current feature set.
- Stable reward variance reduction across successive training runs.
- Drawdown containment: Max DD within configured threshold during validation episodes.

## 16. Edge Cases & Handling Strategy
- Missing Bars: skip or forward-fill flag; log discrepancy.
- Delisted Symbols: auto-prune & mark inactive.
- IBKR Pacing Violations: backoff + retry counter; escalate after threshold.
- Data Gaps Live: trigger alert if no update > 2× interval.
- Extreme Volatility: dynamic risk target adaptation (future enhancement).

## 17. Future Enhancements (Backlog)
- Portfolio optimization hybrid (RL action blended with mean-variance overlay).
- Factor exposure constraints (beta neutrality, sector balance).
- Multi-horizon policies (separate short/long term heads).
- Ensemble of policies (vote or weighted blend).
- On-line learning / incremental fine-tuning for regime shifts.

---
This document will evolve; each phase completion should append a short changelog section summarizing scope, metrics, and decisions.
