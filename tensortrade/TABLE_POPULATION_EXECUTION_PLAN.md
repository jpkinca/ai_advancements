# TensorTrade Table Population Execution Plan

**Created**: August 17, 2025  
**Database**: PostgreSQL (Railway)  
**Connection**: `postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway`

## 📊 Current Status Summary

### ✅ POPULATED
- **tt_prices**: 1082 records (12 symbols, 2023-12-26 to 2024-04-02)

### ❌ EMPTY (Ready for Population)
- **tt_features**: 0 records - Feature computation needed
- **tt_episode**: 0 records - Training execution needed
- **tt_action**: 0 records - Training execution needed  
- **tt_reward**: 0 records - Training execution needed
- **tt_signal_scores**: 0 records - Manual signal generation needed
- **tt_target_weights**: 0 records - Training enhancement needed
- **tt_equity_curve**: 0 records - Training enhancement needed

### ❌ UNUSED (Schema Only)
- **tt_portfolio, tt_holdings, tt_order, tt_observation**: Not used by current codebase

---

## 🎯 Execution Plan Overview

### Phase 1: Feature Engineering (Immediate)
- Populate `tt_features` with technical indicators
- **Duration**: 5 minutes
- **Dependencies**: tt_prices (✅ available)

### Phase 2: Signal Generation (Manual)
- Populate `tt_signal_scores` with sample trading signals
- **Duration**: 10 minutes
- **Dependencies**: tt_features

### Phase 3: RL Training Preparation (Infrastructure)
- Fix import conflicts and validate training environment
- **Duration**: 15 minutes
- **Dependencies**: Features + Signals

### Phase 4: Training Execution (Core)
- Execute RL training to populate episode/action/reward tables
- **Duration**: 30-60 minutes
- **Dependencies**: All previous phases

### Phase 5: Enhanced Logging (Advanced)
- Add target weights and equity curve logging to training
- **Duration**: 20 minutes
- **Dependencies**: Working training pipeline

---

## 🛠️ Detailed Step-by-Step Execution

### **PHASE 1: Feature Engineering** 🔧

#### Step 1.1: Validate Price Data
```bash
cd "C:\Users\nzcon\VSPython\tensortrade"

python -c "
from src.db_utils import get_engine, fetch_prices
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
df = fetch_prices(engine)
print(f'Price data: {len(df)} records, {df.symbol.nunique()} symbols')
print(f'Date range: {df.datetime.min()} to {df.datetime.max()}')
print(f'Symbols: {sorted(df.symbol.unique())}')
"
```

#### Step 1.2: Compute and Persist Features
```bash
python -c "
from src.mvp_pipeline import compute_basic_features
from src.db_utils import get_engine, fetch_prices, upsert_features

print('🔧 Computing technical features...')
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Fetch all price data
df = fetch_prices(engine)
print(f'✅ Loaded {len(df)} price records')

# Compute features
feat_df = compute_basic_features(df)
print(f'✅ Computed features for {len(feat_df)} records')

# Persist features
feature_cols = ['return_1', 'vol_10', 'vol_20']
n_inserted = upsert_features(engine, feat_df, feature_cols)
print(f'✅ Inserted {n_inserted} feature records')

# Validate
from sqlalchemy import text
with engine.connect() as conn:
    count = conn.execute(text('SELECT COUNT(*) FROM tt_features')).fetchone()[0]
    features = conn.execute(text('SELECT DISTINCT feature FROM tt_features')).fetchall()
    print(f'✅ Total features in DB: {count}')
    print(f'✅ Feature types: {[f[0] for f in features]}')
"
```

#### Step 1.3: Verify Feature Population
```bash
python -c "
from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📊 tt_features Sample Data:')
with engine.connect() as conn:
    df = pd.read_sql(text('SELECT * FROM tt_features LIMIT 10'), conn)
    print(df.to_string(index=False))
    
    print('\n📈 Feature Summary by Symbol:')
    summary = pd.read_sql(text('''
        SELECT instrument, feature, COUNT(*) as records, 
               ROUND(AVG(value)::numeric, 4) as avg_value,
               ROUND(MIN(value)::numeric, 4) as min_value,
               ROUND(MAX(value)::numeric, 4) as max_value
        FROM tt_features 
        GROUP BY instrument, feature 
        ORDER BY instrument, feature
    '''), conn)
    print(summary.to_string(index=False))
"
```

**Expected Output**: ~3,246 feature records (1082 prices × 3 features)

---

### **PHASE 2: Signal Generation** 📈

#### Step 2.1: Generate Momentum Signals
```bash
python -c "
import json
from datetime import datetime, timedelta
from src.db_utils import get_engine, upsert_signal_scores, fetch_prices
from sqlalchemy import text
import pandas as pd
import numpy as np

print('📈 Generating momentum signals...')
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Get latest features for signal generation
with engine.connect() as conn:
    features_df = pd.read_sql(text('''
        SELECT f.instrument, f.timestamp, f.feature, f.value,
               p.close
        FROM tt_features f
        JOIN tt_prices p ON f.instrument = p.instrument AND f.timestamp = p.timestamp
        WHERE f.feature = 'return_1'
        ORDER BY f.instrument, f.timestamp
    '''), conn)

print(f'✅ Loaded {len(features_df)} return records for signal generation')

# Generate momentum signals (20-day rolling return)
signals = []
for symbol in features_df.instrument.unique():
    sym_data = features_df[features_df.instrument == symbol].sort_values('timestamp')
    if len(sym_data) >= 20:
        # Use last 20 days for momentum calculation
        recent_returns = sym_data.tail(20)['value']
        momentum_score = recent_returns.mean() * np.sqrt(252)  # Annualized
        
        # Normalize to [-1, 1] range
        momentum_score = np.tanh(momentum_score * 2)  # tanh for bounded output
        
        latest_date = sym_data.timestamp.max()
        signals.append({
            'instrument': symbol,
            'timestamp': latest_date,
            'signal_name': 'momentum_20d',
            'score': float(momentum_score),
            'meta_json': json.dumps({
                'lookback_days': 20,
                'method': 'rolling_return_mean',
                'annualized': True,
                'normalization': 'tanh'
            })
        })

print(f'✅ Generated {len(signals)} momentum signals')

# Persist signals
n_inserted = upsert_signal_scores(engine, signals)
print(f'✅ Inserted {n_inserted} signal records')
"
```

#### Step 2.2: Generate Mean Reversion Signals
```bash
python -c "
import json
import numpy as np
import pandas as pd
from src.db_utils import get_engine, upsert_signal_scores
from sqlalchemy import text

print('📉 Generating mean reversion signals...')
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Get volatility features for mean reversion signals
with engine.connect() as conn:
    vol_df = pd.read_sql(text('''
        SELECT instrument, timestamp, value as volatility
        FROM tt_features 
        WHERE feature = 'vol_20'
        ORDER BY instrument, timestamp
    '''), conn)
    
    price_df = pd.read_sql(text('''
        SELECT instrument, timestamp, close
        FROM tt_prices
        ORDER BY instrument, timestamp
    '''), conn)

# Merge price and volatility data
merged = pd.merge(price_df, vol_df, on=['instrument', 'timestamp'], how='inner')
print(f'✅ Loaded {len(merged)} price-volatility records')

# Generate mean reversion signals
signals = []
for symbol in merged.instrument.unique():
    sym_data = merged[merged.instrument == symbol].sort_values('timestamp')
    if len(sym_data) >= 20:
        # Calculate price deviation from 20-day moving average
        sym_data = sym_data.copy()
        sym_data['ma_20'] = sym_data['close'].rolling(20).mean()
        sym_data['price_deviation'] = (sym_data['close'] - sym_data['ma_20']) / sym_data['ma_20']
        
        # Latest deviation and volatility
        latest = sym_data.iloc[-1]
        if pd.notna(latest['price_deviation']) and pd.notna(latest['volatility']) and latest['volatility'] > 0:
            # Normalized deviation (z-score like)
            reversion_score = -latest['price_deviation'] / latest['volatility']  # Negative for mean reversion
            reversion_score = np.tanh(reversion_score)  # Bounded [-1, 1]
            
            signals.append({
                'instrument': symbol,
                'timestamp': latest['timestamp'],
                'signal_name': 'mean_reversion_20d',
                'score': float(reversion_score),
                'meta_json': json.dumps({
                    'lookback_days': 20,
                    'method': 'price_deviation_normalized',
                    'price_deviation': float(latest['price_deviation']),
                    'volatility': float(latest['volatility'])
                })
            })

print(f'✅ Generated {len(signals)} mean reversion signals')

# Persist signals
n_inserted = upsert_signal_scores(engine, signals)
print(f'✅ Inserted {n_inserted} signal records')
"
```

#### Step 2.3: Verify Signal Population
```bash
python -c "
from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📊 tt_signal_scores Summary:')
with engine.connect() as conn:
    # Signal counts
    counts = pd.read_sql(text('''
        SELECT signal_name, COUNT(*) as records,
               ROUND(AVG(score)::numeric, 4) as avg_score,
               ROUND(MIN(score)::numeric, 4) as min_score,
               ROUND(MAX(score)::numeric, 4) as max_score
        FROM tt_signal_scores 
        GROUP BY signal_name
    '''), conn)
    print(counts.to_string(index=False))
    
    print('\n📈 Sample Signal Data:')
    sample = pd.read_sql(text('''
        SELECT instrument, signal_name, 
               ROUND(score::numeric, 4) as score,
               timestamp
        FROM tt_signal_scores 
        ORDER BY instrument, signal_name
        LIMIT 10
    '''), conn)
    print(sample.to_string(index=False))
"
```

**Expected Output**: ~24 signal records (12 symbols × 2 signal types)

---

### **PHASE 3: Training Environment Validation** 🔧

#### Step 3.1: Fix Import Conflicts
```bash
# Check for TensorTrade import issues
python -c "
import sys
print('🔍 Python path:')
for p in sys.path[:5]:
    print(f'  {p}')

print('\n🔍 Testing TensorTrade imports...')
try:
    import site
    site_packages = site.getsitepackages()
    print(f'Site packages: {site_packages}')
    
    # Test import with site packages priority
    sys.path = site_packages + sys.path
    import tensortrade
    print(f'✅ TensorTrade imported successfully: {tensortrade.__file__}')
except Exception as e:
    print(f'❌ TensorTrade import failed: {e}')
    print('ℹ️  Will use fallback SimpleMultiAssetEnv')
"
```

#### Step 3.2: Validate Training Dependencies
```bash
python -c "
print('🔍 Checking training dependencies...')

# Test core dependencies
deps = ['pandas', 'numpy', 'sqlalchemy', 'stable_baselines3', 'gymnasium']
for dep in deps:
    try:
        __import__(dep)
        print(f'✅ {dep}')
    except ImportError as e:
        print(f'❌ {dep}: {e}')

# Test database connection
from src.db_utils import get_engine
try:
    engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('✅ Database connection')
except Exception as e:
    print(f'❌ Database connection: {e}')

# Test training modules
try:
    from src.train_mvp import main
    print('✅ Training module imports')
except Exception as e:
    print(f'❌ Training module: {e}')
"
```

#### Step 3.3: Test Environment Creation
```bash
python -c "
print('🔍 Testing environment creation...')

from src.db_utils import get_engine, fetch_prices
from src.mvp_pipeline import compute_basic_features

# Test data loading
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
symbols = ['AGX', 'CLS', 'IREN']  # Test with 3 symbols
df = fetch_prices(engine, symbols, '2024-01-01', '2024-04-01')
print(f'✅ Loaded test data: {len(df)} records, {df.symbol.nunique()} symbols')

# Test feature computation
feat_df = compute_basic_features(df)
print(f'✅ Computed features: {len(feat_df)} records')

# Test environment imports
try:
    from src.simple_env import SimpleMultiAssetEnv
    print('✅ Fallback environment available')
except Exception as e:
    print(f'❌ Fallback environment: {e}')

print('🎯 Environment validation complete')
"
```

---

### **PHASE 4: RL Training Execution** 🚀

#### Step 4.1: Short Training Test (100 steps)
```bash
python -c "
print('🚀 Starting short training test...')

# Set environment variable for database
import os
os.environ['DATABASE_URL'] = 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'

# Import and run training
from src.train_mvp import main
import sys

# Simulate command line args for short test
sys.argv = [
    'train_mvp.py',
    '--months', '3',
    '--limit', '3',  # Use only 3 symbols for speed
    '--steps', '100',  # Very short training
    '--with-features',
    '--eval-episodes', '1',
    '--db-url', 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'
]

print('📋 Training args:', sys.argv[1:])
main()
"
```

#### Step 4.2: Validate Training Data Population
```bash
python -c "
from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📊 Training Data Validation:')
with engine.connect() as conn:
    # Episode data
    episodes = pd.read_sql(text('SELECT COUNT(*) as count FROM tt_episode'), conn)
    print(f'Episodes: {episodes.iloc[0]["count"]}')
    
    # Action data
    actions = pd.read_sql(text('SELECT COUNT(*) as count FROM tt_action'), conn)
    print(f'Actions: {actions.iloc[0]["count"]}')
    
    # Reward data
    rewards = pd.read_sql(text('SELECT COUNT(*) as count FROM tt_reward'), conn)
    print(f'Rewards: {rewards.iloc[0]["count"]}')
    
    if episodes.iloc[0]['count'] > 0:
        print('\n📈 Episode Details:')
        episode_details = pd.read_sql(text('''
            SELECT id, start_time, end_time, stop_reason,
                   ROUND(max_drawdown::numeric, 4) as max_dd,
                   ROUND(sharpe_ratio::numeric, 4) as sharpe,
                   ROUND(turnover::numeric, 4) as turnover
            FROM tt_episode 
            ORDER BY id DESC 
            LIMIT 5
        '''), conn)
        print(episode_details.to_string(index=False))
"
```

#### Step 4.3: Full Training Run (If Test Successful)
```bash
# Only run if Step 4.1 and 4.2 were successful
python -m src.train_mvp \
    --months 3 \
    --limit 10 \
    --steps 10000 \
    --with-features \
    --eval-episodes 2 \
    --db-url postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway
```

**Expected Output**: 
- 1-2 episodes in tt_episode
- ~200-20,000 action records (depending on episode length)
- ~100-10,000 reward records
- Performance metrics in episode table

---

### **PHASE 5: Enhanced Training Logging** 📊

#### Step 5.1: Add Target Weights Logging
```bash
# This requires code modification in train_mvp.py
python -c "
print('📝 Target weights logging enhancement needed')
print('Location: src/train_mvp.py line ~328')
print('Add after action logging:')
print('''
# Log target weights
weights = {sym: float(arr[i]) for i, sym in enumerate(symbols[:len(arr)])}
insert_target_weights(engine, episode_id, ts, weights, 
                     strategy=\"ppo_training\", 
                     rationale=f\"Step {step_count} portfolio allocation\")
''')
"
```

#### Step 5.2: Add Equity Curve Logging
```bash
# This requires code modification in train_mvp.py
python -c "
print('📈 Equity curve logging enhancement needed')
print('Location: src/train_mvp.py line ~350')
print('Modify equity tracking section to call:')
print('''
if eq is not None:
    eq_f = float(eq)
    insert_equity_point(engine, episode_id, ts, eq_f)
    equity_hist.append(eq_f)
''')
"
```

#### Step 5.3: Test Enhanced Logging (Manual)
```bash
python -c "
from src.db_utils import get_engine, insert_target_weights, insert_equity_point
from datetime import datetime

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Test target weights logging
test_weights = {'AGX': 0.3, 'CLS': 0.4, 'IREN': 0.3}
n_weights = insert_target_weights(
    engine, 999, datetime.now(), test_weights,
    strategy='test_strategy', rationale='Manual test'
)
print(f'✅ Test target weights logged: {n_weights}')

# Test equity curve logging  
insert_equity_point(engine, 999, datetime.now(), 100000.0)
print('✅ Test equity point logged')

# Verify
from sqlalchemy import text
with engine.connect() as conn:
    weights_count = conn.execute(text('SELECT COUNT(*) FROM tt_target_weights')).fetchone()[0]
    equity_count = conn.execute(text('SELECT COUNT(*) FROM tt_equity_curve')).fetchone()[0]
    print(f'Target weights records: {weights_count}')
    print(f'Equity curve records: {equity_count}')
"
```

---

## 📋 **Execution Checklist**

### Pre-Execution Checklist:
- [ ] Database connection confirmed
- [ ] Python environment activated
- [ ] Navigate to tensortrade directory
- [ ] Backup database (optional but recommended)

### Phase 1 - Features ✅:
- [ ] Step 1.1: Validate price data
- [ ] Step 1.2: Compute and persist features
- [ ] Step 1.3: Verify feature population
- [ ] **Target**: ~3,246 feature records

### Phase 2 - Signals ✅:
- [ ] Step 2.1: Generate momentum signals
- [ ] Step 2.2: Generate mean reversion signals  
- [ ] Step 2.3: Verify signal population
- [ ] **Target**: ~24 signal records

### Phase 3 - Validation ✅:
- [ ] Step 3.1: Fix import conflicts
- [ ] Step 3.2: Validate training dependencies
- [ ] Step 3.3: Test environment creation
- [ ] **Target**: All systems green

### Phase 4 - Training ✅:
- [ ] Step 4.1: Short training test (100 steps)
- [ ] Step 4.2: Validate training data population
- [ ] Step 4.3: Full training run (10,000 steps)
- [ ] **Target**: Episode/action/reward tables populated

### Phase 5 - Enhancement ⚠️:
- [ ] Step 5.1: Add target weights logging (code modification)
- [ ] Step 5.2: Add equity curve logging (code modification)
- [ ] Step 5.3: Test enhanced logging
- [ ] **Target**: Complete training data capture

---

## 🎯 **Success Metrics**

### Data Population Targets:
| Table | Current | Target | Success Criteria |
|-------|---------|--------|------------------|
| tt_prices | 1,082 | 1,082 | ✅ Already populated |
| tt_features | 0 | ~3,246 | 3 features × 1,082 records |
| tt_signal_scores | 0 | ~24 | 2 signals × 12 symbols |
| tt_episode | 0 | 2-3 | Training + eval episodes |
| tt_action | 0 | 1,000+ | Actions per training step |
| tt_reward | 0 | 500+ | Rewards per training step |
| tt_target_weights | 0 | 1,000+ | Portfolio weights per step |
| tt_equity_curve | 0 | 500+ | Equity points per step |

### Performance Targets:
- **Feature computation**: <5 minutes
- **Signal generation**: <10 minutes  
- **Training (10K steps)**: 30-60 minutes
- **Total execution time**: <90 minutes

### Quality Checks:
- [ ] No NULL values in critical fields
- [ ] Timestamp continuity in training data
- [ ] Portfolio weights sum to ~1.0
- [ ] Sharpe ratio calculated correctly
- [ ] All foreign key relationships intact

---

## 🚨 **Troubleshooting Guide**

### Common Issues:

**1. TensorTrade Import Errors**
```bash
# Solution: Use fallback environment
export USE_TENSORTRADE=false
```

**2. Database Connection Timeouts**
```bash
# Solution: Retry with connection pooling
engine = create_engine(url, pool_timeout=30, pool_recycle=3600)
```

**3. Memory Issues During Training**
```bash
# Solution: Reduce batch size and symbols
--limit 5 --steps 5000
```

**4. Feature Computation Errors**
```bash
# Solution: Check for missing price data
df.dropna(subset=['close'], inplace=True)
```

**5. Signal Generation NaN Values**
```bash
# Solution: Add NaN filtering
if pd.notna(momentum_score) and np.isfinite(momentum_score):
```

---

## 📞 **Execution Support**

**Estimated Total Time**: 90 minutes  
**Difficulty**: Intermediate  
**Prerequisites**: Working Python environment, database access  
**Next Steps**: After completion, run analysis queries to validate all table populations

This plan provides a complete roadmap to populate all TensorTrade tables through their appropriate processes, with validation at each step.
