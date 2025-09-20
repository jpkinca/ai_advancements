# XGBoost Development Plan

Date: 2025-09-19
Status: Active
Owner: AI Advancements Engineering

## Overview
This document formalizes the staged development and integration plan for XGBoost-based predictive analytics inside the `ai_advancements` trading research and execution stack. It extends current capabilities beyond offline experimentation into production-aligned, monitored, and extensible alpha generation infrastructure.

## Phase Summary
| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core module + feature engineering + training + backtest | Complete |
| 2 | Real-time inference & streaming signals | In Progress |
| 3 | Custom objectives & eval metrics | Pending |
| 4 | Portfolio & ensemble layer | Pending |
| 5 | Regime & anomaly detection | Pending |
| 6 | Survival / time-to-event modeling | Optional / Pending |
| 7 | Monitoring & drift detection | Pending |
| 8 | Dependency resolution & stability | Pending |
| 9 | Expanded testing | Pending |
| 10 | Deployment & registry tooling | Pending |
| 11 | Advanced documentation & examples | Pending |

## Completed (Phase 1)
- `xgboost_trading_model.py`: Unified regression (next close) + classification (directional signal) implementation
- Feature set: Returns, log returns, SMA/EMA (5/10/20/50), volatility bands, RSI, MACD, Bollinger width/position, momentum ladders, volume ratio
- Training script: `train_xgboost_models.py` (optionally with grid search + time series CV)
- Backtesting script: `backtest_xgboost.py` including: total return, Sharpe ratio approximation, max drawdown, buy vs hold comparison, optional walk-forward analysis
- Model persistence (pickle) + scaler saving

## Phase 2: Real-Time Inference (In Progress)
### Goals
Provide low-latency model inference on streaming OHLCV bars or aggregated ticks, generating standardized trading signals with risk filters.

### Components
- `realtime/feature_buffer.py`: Rolling window store with incremental indicator updates (avoids full recompute)
- `realtime/xgb_realtime_engine.py`: Loads models, ingests new bars, updates features, outputs signals
- `realtime/signal_dispatcher.py`: Pluggable sink (stdout, DB, Redis, WebSocket placeholder)
- Risk gating: volatility threshold, cooldown, max open signals

### Interface Contract
Input event schema (bar): `{ symbol, timestamp, open, high, low, close, volume }`
Output signal schema: `{ symbol, timestamp, action: BUY|SELL|HOLD, confidence, features_snapshot_hash, model_version }`

### Edge Handling
- Insufficient history: engine stays in WARMING state
- NaN indicator values: skip inference until stable
- Model missing: raise recoverable error & log

## Phase 3: Custom Objectives & Metrics
### Rationale
Plain MAE / accuracy do not fully reflect trading edge. Introduce:
- Pseudo-Sharpe eval metric (mean(pred)/std) on validation return deltas
- Downside-penalized objective emphasizing false-positive sell/buy cost asymmetry

### Deliverables
- `xgboost_custom_objectives.py` with: `sharpe_eval(preds, dtrain)` and `asymmetric_objective(preds, dtrain)`
- Training flags: `--custom-objective`, `--eval-metric sharpe`
- Documentation examples

## Phase 4: Portfolio & Ensemble Layer
### Objectives
Shift from single-symbol optimization to cross-asset alpha combination.

### Features
- Correlation matrix + rolling covariance loader from `AIDataAccessor`
- Risk parity weight suggestion (inverse volatility)
- Simple stacking pipeline (e.g., combine RL model probability + XGBoost directional probability)

### Deliverables
- `portfolio/portfolio_helper.py`
- `ensemble/stacking_manager.py`

## Phase 5: Regime & Anomaly Detection
### Use Cases
- Volatility regime switching (stable → expansion)
- Volume or spread anomalies
- Gap detection pre-open

### Methods
- Rolling statistical thresholds (z-score approach)
- Auxiliary XGBoost classifier trained on labeled historical regimes
- IsolationForest (optional fallback)

### Deliverables
- `regime/regime_detector.py`
- `regime/anomaly_signals.py`

## Phase 6: Survival / Time-To-Event Modeling (Optional)
Predict probability of hitting stop-loss/target within horizon.
- Frame as discrete horizon classification (T+1 … T+N return bins)
- Alternative: Cox-style approximation via gradient boosted ranking objective

Deliverable: `experimental/survival_model.py`

## Phase 7: Monitoring & Drift Detection
### Metrics
- Feature distribution shift: mean/std drift, PSI (population stability index)
- Prediction distribution tracking (class entropy, probability collapse)
- Data quality anomalies (zero volume ratio, missing bars)
- Latency metrics (inference time)

### Deliverables
- `monitoring/model_monitor.py`
- `monitoring/drift_metrics.py`
- Log artifacts under `monitoring_logs/`

## Phase 8: Dependency Conflict Resolution
Issue: Pydantic version mismatch (installed 2.5.0; dependencies require >=2.5.3 or >=2.7.0). Action plan:
1. Pin: `pydantic>=2.7.4,<3.0.0`
2. Update `requirements.txt`
3. Reinstall & validate import of `trade-app-components` and internal modules
4. Record resolution in `INTEGRATION_STATUS_RESOLVED.md`

## Phase 9: Testing Expansion
| Layer | Tests |
|-------|-------|
| Unit | Indicator correctness, custom objective gradient shape |
| Integration | Train small model, ensure non-empty metrics, walk-forward slice works |
| Real-Time | Mock bar feed -> signal emission |
| Monitoring | Synthetic drift triggers warning |

Deliverables:
- `tests/test_features.py`
- `tests/test_training_smoke.py`
- `tests/test_backtest_metrics.py`
- `tests/test_realtime_engine.py`
- `tests/test_monitoring_drift.py`

## Phase 10: Deployment & Registry
Add reproducibility + traceability.
- Model registry JSON: model id, symbol, timeframe, training window, hash of params
- Version tagging: timestamp + optional git commit
- CLI: list / inspect / promote model

Deliverables:
- `deploy/model_registry.py`
- Extend `train_xgboost_models.py` with `--register`

## Phase 11: Advanced Documentation & Examples
Documents to maintain:
- `docs/XGBOOST_MODULE.md` (user guide)
- `docs/REALTIME_PIPELINE.md` (stream architecture)
- `docs/MONITORING_AND_DRIFT.md`
- Notebook examples (optional future)

## Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting via grid search | Poor live performance | Walk-forward validation + early stopping |
| Feature drift | Degrading edge | Monitoring + retrain trigger thresholds |
| Data gaps | Invalid indicators | Warming state + integrity checks |
| Class imbalance | Biased signals | Class weights / focal loss variant |
| Latency spikes | Missed fills | Pre-compute rolling state, async I/O |

## Implementation Order (Recommended Next)
1. Finish real-time engine scaffolding (Phase 2)
2. Dependency pin & reinstall (Phase 8)
3. Baseline tests (Phase 9 core subset)
4. Monitoring skeleton (Phase 7 initial subset)
5. Custom objective integration (Phase 3)
6. Portfolio helper (Phase 4 minimal)
7. Docs batch update (Phase 11 wave 1)

## Acceptance Criteria
- Real-time: <50ms inference per bar (single symbol) locally
- Backtest reproducible: same seed -> stable metrics variance <1%
- Registry entries generated on every training run (post Phase 10)
- Monitoring logs show rolling PSI values for top 10 features
- Documentation covers: training, backtesting, realtime, tuning, monitoring

## Appendix: File Inventory (Current Relevant)
- `xgboost_trading_model.py`
- `train_xgboost_models.py`
- `backtest_xgboost.py`
- `ai_data_accessor.py` (data retrieval base)

Planned new directories will follow this structure:
```
ai_advancements/
  realtime/
    feature_buffer.py
    xgb_realtime_engine.py
    signal_dispatcher.py
  monitoring/
    model_monitor.py
    drift_metrics.py
  portfolio/
    portfolio_helper.py
  ensemble/
    stacking_manager.py
  regime/
    regime_detector.py
    anomaly_signals.py
  experimental/
    survival_model.py
  deploy/
    model_registry.py
  docs/
    XGBOOST_DEVELOPMENT_PLAN.md (this file)
```

---
Maintained as a living document; update phases and statuses as modules ship.
