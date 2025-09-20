# TensorTrade Complete Table Population Guide

**Database**: PostgreSQL with `tt_` prefixed tables  
**Last Updated**: August 17, 2025

## 📊 Complete Data Model Overview

Based on comprehensive source code analysis, here's how each table in the TensorTrade data model is populated:

---

## 🔄 **Core Data Flow Tables**

### 1. **tt_prices** ✅ POPULATED (1082 records)

**Purpose**: Historical OHLCV price data from IBKR  
**Populated by**: `src/mvp_pipeline.py` → `src/watchlist_loader.py` → `src/db_utils.py`

#### Schema:
```sql
CREATE TABLE tt_prices (
    id SERIAL PRIMARY KEY,
    instrument TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION
);
```

#### Population Process:
1. **Trigger**: `python -m src.mvp_pipeline --start YYYY-MM-DD --end YYYY-MM-DD --db-url <URL>`
2. **Data Source**: Interactive Brokers via `ib_insync`
3. **Function Call Chain**:
   ```python
   fetch_price_history() → upsert_price_bars() → INSERT INTO tt_prices
   ```
4. **Rate Limiting**: 0.15s delay between IBKR requests
5. **Deduplication**: Unique index on (instrument, timestamp)

#### Field Details:
- **instrument**: Symbol (e.g., 'AGX', 'CLS', 'IREN')
- **timestamp**: Date of trading session (daily bars)
- **open/high/low/close**: Price data from IBKR
- **volume**: Trading volume from IBKR

---

### 2. **tt_episode** ❌ EMPTY (0 records)

**Purpose**: Training episode metadata and performance metrics  
**Populated by**: `src/train_mvp.py` during RL training

#### Schema:
```sql
CREATE TABLE tt_episode (
    id SERIAL PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    stop_reason TEXT,
    max_drawdown DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    turnover DOUBLE PRECISION,
    config_json TEXT
);
```

#### Population Process:
1. **Trigger**: `python -m src.train_mvp --months 3 --limit 10 --steps 10000 --db-url <URL>`
2. **Episode Start**: `create_episode()` called at training start
3. **Episode End**: `finalize_episode()` called when training completes

#### Field Details:
- **start_time**: UTC timestamp when episode begins
- **end_time**: UTC timestamp when episode ends
- **stop_reason**: "evaluation_complete", "terminated", "truncated", "reset_called", "env_closed"
- **max_drawdown**: Maximum portfolio drawdown during episode (0.0-1.0)
- **sharpe_ratio**: Annualized Sharpe-like ratio calculated from equity curve
- **turnover**: Average absolute change in portfolio weights per step
- **config_json**: Training hyperparameters and environment settings (JSON string)

#### Population Code:
```python
# Episode creation
episode_id = create_episode(engine, datetime.utcnow())

# Episode finalization  
finalize_episode(engine, episode_id, datetime.utcnow(), 
                stop_reason="evaluation_complete",
                max_drawdown=0.05, sharpe_ratio=1.2, turnover=0.15)
```

---

### 3. **tt_action** ❌ EMPTY (0 records)

**Purpose**: Agent actions at each timestep (portfolio weights)  
**Populated by**: `src/train_mvp.py` and `src/env_wrappers.py` during RL training

#### Schema:
```sql
CREATE TABLE tt_action (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER REFERENCES tt_episode(id),
    timestamp TIMESTAMP,
    instrument TEXT,
    action_value DOUBLE PRECISION
);
```

#### Population Process:
1. **Trigger**: RL agent making decisions during training
2. **Frequency**: Every environment step
3. **Source**: PPO model predictions converted to portfolio weights

#### Field Details:
- **episode_id**: Foreign key to tt_episode
- **timestamp**: UTC timestamp of action
- **instrument**: Symbol this action applies to
- **action_value**: Portfolio weight (0.0-1.0) or allocation signal

#### Population Code:
```python
# Multi-asset action logging
action, _ = model.predict(obs, deterministic=True)
arr = np.asarray(action).reshape(-1)
for i, sym in enumerate(symbols[:len(arr)]):
    insert_action(engine, episode_id, timestamp, sym, float(arr[i]))
```

---

### 4. **tt_reward** ❌ EMPTY (0 records)

**Purpose**: Reward values from the reward scheme  
**Populated by**: `src/train_mvp.py` and `src/env_wrappers.py` during RL training

#### Schema:
```sql
CREATE TABLE tt_reward (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER REFERENCES tt_episode(id),
    timestamp TIMESTAMP,
    reward_value DOUBLE PRECISION
);
```

#### Population Process:
1. **Trigger**: Environment step evaluation
2. **Source**: `RiskAwareReward` scheme in `src/tensortrade_risk_module.py`
3. **Frequency**: Every environment step

#### Field Details:
- **episode_id**: Foreign key to tt_episode
- **timestamp**: UTC timestamp of reward calculation
- **reward_value**: Scalar reward (can be negative)

#### Population Code:
```python
obs, reward, done, truncated, info = env.step(action)
insert_reward(engine, episode_id, timestamp, float(reward))
```

---

## 🧠 **Feature Engineering Tables**

### 5. **tt_features** ❌ EMPTY (0 records)

**Purpose**: Technical features in long-form storage  
**Populated by**: Feature computation processes with `--with-features` flag

#### Schema:
```sql
CREATE TABLE tt_features (
    id SERIAL PRIMARY KEY,
    instrument TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature TEXT NOT NULL,
    value DOUBLE PRECISION,
    UNIQUE (instrument, timestamp, feature)
);
```

#### Population Process:
1. **Trigger**: Training with `--with-features` or manual feature computation
2. **Source**: `compute_basic_features()` in `src/mvp_pipeline.py`
3. **Storage**: Long-form (one row per instrument-timestamp-feature combination)

#### Available Features:
- **return_1**: Daily returns (`close.pct_change()`)
- **vol_10**: 10-period rolling volatility of returns
- **vol_20**: 20-period rolling volatility of returns

#### Field Details:
- **instrument**: Symbol
- **timestamp**: Date of feature calculation
- **feature**: Feature name ('return_1', 'vol_10', 'vol_20')
- **value**: Calculated feature value

#### Population Code:
```python
# Compute features
feat_df = compute_basic_features(price_df)

# Persist to database
upsert_features(engine, feat_df, ['return_1', 'vol_10', 'vol_20'])
```

#### Manual Population Command:
```python
python -c "
from src.mvp_pipeline import compute_basic_features
from src.db_utils import get_engine, fetch_prices, upsert_features
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
df = fetch_prices(engine, ['AGX', 'CLS', 'IREN'], '2023-12-26', '2024-04-02')
feat_df = compute_basic_features(df)
upsert_features(engine, feat_df, ['return_1', 'vol_10', 'vol_20'])
print('Features populated!')
"
```

---

### 6. **tt_signal_scores** ❌ EMPTY (0 records)

**Purpose**: Trading signal scores (momentum, mean reversion, etc.)  
**Populated by**: Signal generation modules (manual or automated)

#### Schema:
```sql
CREATE TABLE tt_signal_scores (
    id SERIAL PRIMARY KEY,
    instrument TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    signal_name TEXT NOT NULL,
    score DOUBLE PRECISION,
    meta_json TEXT,
    UNIQUE (instrument, timestamp, signal_name)
);
```

#### Population Process:
1. **Trigger**: Manual signal generation or automated signal modules
2. **Current Status**: Framework exists but signal generation not implemented
3. **Example Usage**: In `tests/test_integration_pipeline.py`

#### Field Details:
- **instrument**: Symbol
- **timestamp**: Date of signal calculation
- **signal_name**: Signal identifier ('momentum', 'mean_reversion', etc.)
- **score**: Signal strength (-1.0 to 1.0 typical range)
- **meta_json**: Additional metadata as JSON string

#### Population Code:
```python
# Example signal generation
signals = [
    {
        'instrument': 'AAPL',
        'timestamp': datetime.now(),
        'signal_name': 'momentum',
        'score': 0.75,
        'meta_json': json.dumps({'lookback': 20, 'source': 'return_1'})
    }
]
upsert_signal_scores(engine, signals)
```

---

## 📈 **Portfolio & Trading Tables**

### 7. **tt_target_weights** ❌ EMPTY (0 records)

**Purpose**: Target portfolio weights with rationale  
**Populated by**: `src/train_mvp.py` during RL training (future implementation)

#### Schema:
```sql
CREATE TABLE tt_target_weights (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER REFERENCES tt_episode(id),
    timestamp TIMESTAMP NOT NULL,
    instrument TEXT NOT NULL,
    target_weight DOUBLE PRECISION,
    strategy TEXT,
    rationale TEXT
);
```

#### Population Process:
1. **Trigger**: Portfolio allocation decisions during training
2. **Current Status**: Function exists but not called in training loop
3. **Future Use**: Record portfolio target weights before execution

#### Field Details:
- **episode_id**: Foreign key to tt_episode
- **timestamp**: When target weights were determined
- **instrument**: Symbol
- **target_weight**: Desired portfolio allocation (0.0-1.0)
- **strategy**: Strategy name ('ppo_momentum', 'volatility_targeted', etc.)
- **rationale**: Human-readable explanation of weight decision

#### Population Code:
```python
weights = {'AAPL': 0.3, 'MSFT': 0.2, 'GOOGL': 0.25, 'CASH': 0.25}
insert_target_weights(engine, episode_id, timestamp, weights, 
                     strategy='ppo_momentum', 
                     rationale='High momentum signals on tech stocks')
```

---

### 8. **tt_equity_curve** ❌ EMPTY (0 records)

**Purpose**: Portfolio net worth over time  
**Populated by**: `src/train_mvp.py` during RL training (future implementation)

#### Schema:
```sql
CREATE TABLE tt_equity_curve (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER REFERENCES tt_episode(id),
    timestamp TIMESTAMP NOT NULL,
    net_worth DOUBLE PRECISION
);
```

#### Population Process:
1. **Trigger**: Portfolio value tracking during training episodes
2. **Current Status**: Function exists but net worth extraction not implemented
3. **Frequency**: Every environment step

#### Field Details:
- **episode_id**: Foreign key to tt_episode
- **timestamp**: When portfolio value was recorded
- **net_worth**: Total portfolio value (cash + positions)

#### Population Code:
```python
# Extract portfolio value from environment
equity = env.portfolio.net_worth  # or env.net_worth
insert_equity_point(engine, episode_id, timestamp, float(equity))
```

---

## 🏗️ **Extended Schema Tables** (from `create_tt_tables.py`)

### 9. **tt_portfolio** ❌ NOT USED

**Purpose**: Portfolio snapshots (alternative to tt_equity_curve)  
**Status**: Defined in schema but not used by current codebase  
**Schema**: Contains net_worth, cash_balance, episode_id

### 10. **tt_holdings** ❌ NOT USED

**Purpose**: Detailed position holdings per portfolio  
**Status**: Defined in schema but not used by current codebase  
**Schema**: Contains portfolio_id, instrument, quantity

### 11. **tt_order** ❌ NOT USED

**Purpose**: Order execution details  
**Status**: Defined in schema but not used by current codebase  
**Schema**: Contains order_type, size, price, instrument

### 12. **tt_observation** ❌ NOT USED

**Purpose**: Environment observations (state vectors)  
**Status**: Defined in schema but not used by current codebase  
**Schema**: Contains features (JSON text field)

---

## 🚀 **Population Commands Summary**

### Populate All Tables (Complete Training Run):
```bash
# 1. Ensure price data exists
python -m src.mvp_pipeline --start 2023-12-01 --end 2024-04-30 --limit 10 --db-url postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway

# 2. Run training with features (populates episode, action, reward tables)
python -m src.train_mvp --months 3 --limit 10 --steps 10000 --with-features --eval-episodes 2 --db-url postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway
```

### Populate Features Only:
```python
python -c "
from src.mvp_pipeline import compute_basic_features
from src.db_utils import get_engine, fetch_prices, upsert_features
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
df = fetch_prices(engine)  # All symbols and dates
feat_df = compute_basic_features(df)
upsert_features(engine, feat_df, ['return_1', 'vol_10', 'vol_20'])
print('All features populated!')
"
```

### Populate Sample Signals:
```python
python -c "
import json
from datetime import datetime
from src.db_utils import get_engine, upsert_signal_scores
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
signals = [
    {'instrument': 'AGX', 'timestamp': datetime(2024,4,1), 'signal_name': 'momentum', 'score': 0.7, 'meta_json': json.dumps({'source': 'manual'})},
    {'instrument': 'CLS', 'timestamp': datetime(2024,4,1), 'signal_name': 'momentum', 'score': -0.3, 'meta_json': json.dumps({'source': 'manual'})}
]
upsert_signal_scores(engine, signals)
print('Sample signals populated!')
"
```

---

## 📊 **Data Dependencies**

```mermaid
graph TD
    A[tt_prices] --> B[Features Computation]
    B --> C[tt_features]
    C --> D[Signal Generation]
    D --> E[tt_signal_scores]
    A --> F[RL Training]
    C --> F
    E --> F
    F --> G[tt_episode]
    F --> H[tt_action]
    F --> I[tt_reward]
    F --> J[tt_target_weights]
    F --> K[tt_equity_curve]
```

**Population Order**:
1. **tt_prices** (Data ingestion)
2. **tt_features** (Feature engineering)
3. **tt_signal_scores** (Signal generation)
4. **tt_episode, tt_action, tt_reward, tt_target_weights, tt_equity_curve** (RL training)

---

## 🔧 **Implementation Status**

| Table | Schema ✅ | Population Code ✅ | Actually Used ✅ | Data Present ✅ |
|-------|-----------|-------------------|-----------------|----------------|
| tt_prices | ✅ | ✅ | ✅ | ✅ (1082 rows) |
| tt_episode | ✅ | ✅ | ✅ | ❌ (training needed) |
| tt_action | ✅ | ✅ | ✅ | ❌ (training needed) |
| tt_reward | ✅ | ✅ | ✅ | ❌ (training needed) |
| tt_features | ✅ | ✅ | ✅ | ❌ (feature run needed) |
| tt_signal_scores | ✅ | ✅ | ⚠️ | ❌ (manual generation) |
| tt_target_weights | ✅ | ✅ | ⚠️ | ❌ (training enhancement needed) |
| tt_equity_curve | ✅ | ✅ | ⚠️ | ❌ (training enhancement needed) |
| tt_portfolio | ✅ | ❌ | ❌ | ❌ (unused) |
| tt_holdings | ✅ | ❌ | ❌ | ❌ (unused) |
| tt_order | ✅ | ❌ | ❌ | ❌ (unused) |
| tt_observation | ✅ | ❌ | ❌ | ❌ (unused) |

**Legend**:
- ✅ = Fully implemented and working
- ⚠️ = Implemented but needs enhancement
- ❌ = Not implemented or not used

---

This comprehensive guide documents every table, field, and population process in the TensorTrade data model based on complete source code analysis.
