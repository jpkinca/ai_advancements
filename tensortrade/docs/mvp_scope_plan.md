# MVP Scope Plan: TensorTrade + IBKR Integration

Purpose: Define the minimal viable product (MVP) path from watchlist ingestion to a runnable backtest-ready RL environment with risk-aware components. Clearly separate IN-SCOPE vs OUT-OF-SCOPE items for the first delivery.

## 1. Scope Summary (High Level)
IN-SCOPE (MVP):
- Watchlist ingestion (single CSV) and symbol normalization
- Historical bar ingestion via IBKR (daily interval; fallback yfinance only if explicitly chosen)
- Persistence of OHLCV bars to PostgreSQL (`tt_prices`)
- Basic feature derivation (simple returns + rolling volatility)
- Minimal environment assembly with:
  - Portfolio (cash + symbols, no shorts optional toggle)
  - Multi-asset volatility-targeted action scheme
  - Risk-aware reward scheme
  - Drawdown stopper
- One baseline RL training run (e.g., PPO) over historical slice
- Logging of episode metrics (reward sum, net worth progression) to stdout and DB (partial: actions & rewards)
- CLI pipeline scripts (`mvp_pipeline.py` + future `train_mvp.py` placeholder)
- Basic configuration via command-line flags / .env for DB + dates
- Documentation: development plan, MVP scope, initial parameter settings

OUT-OF-SCOPE (Phase > MVP):
- Real-time streaming / bar aggregation
- Paper / live trading execution routing
- Advanced feature library (cross-sectional, factor, regime labeling)
- Hyperparameter tuning (Ray Tune)
- Walk-forward / rolling out-of-sample evaluation
- Slippage, commissions modeling (beyond flat fee placeholder)
- Advanced risk constraints (sector caps, gross/net dynamic limits)
- Alerting / monitoring / dashboards
- Corporate action adjustments & split handling
- Feature store caching layer
- Model introspection (SHAP, perturbation)
- Stress test harness & scenario simulation
- Multi-process distributed training
- Live order submission to IBKR

## 2. Detailed Step Breakdown
| Step | Description | In Scope | Deliverable |
|------|-------------|----------|-------------|
| 1 | Load & validate watchlist CSV | Yes | `tensorwatchlist.csv` loader (done) |
| 2 | IBKR historical fetch (daily bars) | Yes | Updated `watchlist_loader.fetch_price_history_ibkr` |
| 3 | DB schema & connection | Yes | `create_tt_tables.py`, `db_utils.py` |
| 4 | Insert bars into `tt_prices` with upsert | Yes | `upsert_price_bars` (done) |
| 5 | Basic feature calc (return, vol) | Yes | In `mvp_pipeline.compute_basic_features` |
| 6 | Environment assembly skeleton | Yes | Placeholder now; to implement `train_mvp.py` |
| 7 | Action & reward integration | Yes | `tensortrade_risk_module.py` (extended) |
| 8 | Episode training loop (single run) | Yes | Future `train_mvp.py` |
| 9 | Logging actions, rewards to DB | Partial (reward later) | Extend pipeline after env build |
| 10 | CLI configuration flags | Yes | `mvp_pipeline.py` args |
| 11 | Documentation (plan, scope) | Yes | This file + `development_plan.md` |
| 12 | Unit tests for core utilities | No (Phase 2) | Deferred |
| 13 | Real-time ticker streaming | No | Deferred |
| 14 | Paper trading mode | No | Deferred |
| 15 | Hyperparameter tuning | No | Deferred |
| 16 | Advanced feature sets | No | Deferred |
| 17 | Walk-forward evaluation | No | Deferred |
| 18 | Slippage/commission modeling | No | Deferred |
| 19 | Risk circuit breaker | No | Deferred |
| 20 | Alerting & monitoring | No | Deferred |
| 21 | Stress testing harness | No | Deferred |
| 22 | Live execution integration | No | Deferred |
| 23 | Dashboards / analytics UI | No | Deferred |

## 3. MVP Success Criteria
- Able to run pipeline script to ingest data and persist ≥ 95% expected daily bars for watchlist period.
- Environment can step through at least one full episode without exceptions using constructed features.
- PPO (or comparable) training produces model checkpoints and prints episodic reward & net worth changes.
- Reward and action logging accessible from DB (at least aggregated or last 100 steps) for inspection.

## 4. Assumptions
- IBKR Gateway / TWS available for historical data; if not, temporary yfinance fallback acceptable with explicit flag.
- Daily bars sufficient for initial learning demonstration.
- No corporate action adjustment needed for MVP evaluation horizon.
- Simple flat trading cost (optionally zero) acceptable.

## 5. Risks & Mitigations (MVP)
| Risk | Impact | Mitigation |
|------|--------|------------|
| IBKR pacing limits on many symbols | Slow fetch / incomplete dataset | Batch throttling, simple sleep between requests |
| Missing bars for illiquid symbols | Inconsistent feature windows | Drop symbols with < X% coverage |
| Volatility scaling unstable with short history | Erratic actions | Enforce minimum warmup length; clamp vol floor |
| DB credential misconfiguration | Pipeline failure | Early engine connection test before long operations |

## 6. Immediate Next Actions
1. Implement `train_mvp.py` to: load symbols → fetch (or reuse) bars → build minimal features → construct TensorTrade environment → run N episodes of PPO.
2. Extend DB logging: insert rewards & actions each step (append-only).
3. Add simple config file (optional) for run parameters (JSON/YAML parse if provided).

## 7. Deferred Work Log (Post-MVP Backlog Extract)
- Add minute/hourly interval support (resample, multi-timeframe features).
- Build feature registry & hashing for reproducibility.
- Implement walk-forward evaluator module.
- Add real-time ingestion & paper loop orchestrator.

## 8. Change Control
Any additions to IN-SCOPE must displace equal or lower-priority current items or shift timeline. Update this file with a dated CHANGELOG section at bottom.

---
Status: INITIAL VERSION
Date: 2025-08-17
Author: MVP Scope Generator
