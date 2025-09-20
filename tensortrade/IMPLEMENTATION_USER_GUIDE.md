# TensorTrade Implementation & User Guide

**Version**: 1.0  
**Created**: August 17, 2025  
**Purpose**: Complete step-by-step guide to populate all TensorTrade database tables

---

## 📋 **Quick Start Guide**

### Prerequisites
- ✅ Python 3.8+ environment
- ✅ PostgreSQL database access
- ✅ TensorTrade project downloaded
- ✅ Required dependencies installed

### Time Commitment
- **Total Time**: 90-120 minutes
- **Active Work**: 30 minutes (rest is automated)
- **Difficulty**: Beginner to Intermediate

### What You'll Accomplish
- Populate 8 database tables with real trading data
- Generate technical features and trading signals
- Execute RL training pipeline
- Validate complete data pipeline

---

## 🚀 **Getting Started**

### Step 0: Environment Setup

#### 0.1 Navigate to Project Directory
```bash
# Open PowerShell and navigate to project
cd "C:\Users\nzcon\VSPython\tensortrade"
pwd  # Verify you're in the right directory
```

#### 0.2 Verify Database Connection
```bash
python -c "
print('🔍 Testing database connection...')
from src.db_utils import get_engine
try:
    engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
    with engine.connect() as conn:
        result = conn.execute('SELECT current_database(), current_user, now()')
        row = result.fetchone()
        print(f'✅ Connected to database: {row[0]}')
        print(f'✅ User: {row[1]}')
        print(f'✅ Server time: {row[2]}')
except Exception as e:
    print(f'❌ Connection failed: {e}')
    print('Please check your database URL and network connection')
"
```

#### 0.3 Check Current Table Status
```bash
python -c "
from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
print('📊 Current Table Status:')
print('=' * 50)

tables = ['tt_prices', 'tt_episode', 'tt_action', 'tt_reward', 'tt_features', 'tt_signal_scores', 'tt_target_weights', 'tt_equity_curve']

with engine.connect() as conn:
    for table in tables:
        try:
            count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).fetchone()[0]
            status = '✅ POPULATED' if count > 0 else '❌ EMPTY'
            print(f'{table:20} | {count:8} rows | {status}')
        except Exception as e:
            print(f'{table:20} | ERROR: {str(e)[:30]}...')
"
```

**Expected Output**: Only `tt_prices` should show populated (~1082 rows)

---

## 📊 **PHASE 1: Feature Engineering**

*Duration: 5 minutes*

### Step 1.1: Validate Price Data

```bash
python -c "
print('🔍 PHASE 1: Feature Engineering')
print('=' * 50)
print('Step 1.1: Validating price data...')

from src.db_utils import get_engine, fetch_prices
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Load all price data
df = fetch_prices(engine)
print(f'✅ Price records loaded: {len(df):,}')
print(f'✅ Unique symbols: {df.symbol.nunique()}')
print(f'✅ Date range: {df.datetime.min().date()} to {df.datetime.max().date()}')
print(f'✅ Data quality: {(~df.close.isna()).mean():.1%} non-null closes')

print('\n📈 Symbol Summary:')
symbol_counts = df.groupby('symbol').size().sort_values(ascending=False)
for symbol, count in symbol_counts.head(12).items():
    print(f'  {symbol}: {count} records')

if len(df) == 0:
    print('❌ ERROR: No price data found!')
    print('Run data ingestion pipeline first.')
else:
    print('\n✅ Price data validation complete')
"
```

### Step 1.2: Compute Technical Features

```bash
python -c "
print('\nStep 1.2: Computing technical features...')

from src.mvp_pipeline import compute_basic_features
from src.db_utils import get_engine, fetch_prices, upsert_features
import time

start_time = time.time()
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Fetch all price data
print('📊 Loading price data...')
df = fetch_prices(engine)

# Compute features
print('🧮 Computing features (return_1, vol_10, vol_20)...')
feat_df = compute_basic_features(df)

# Show sample of computed features
print('\n📋 Sample computed features:')
sample = feat_df[['symbol', 'datetime', 'close', 'return_1', 'vol_10', 'vol_20']].dropna().head(10)
print(sample.to_string(index=False))

# Persist to database
print('\n💾 Persisting features to database...')
feature_cols = ['return_1', 'vol_10', 'vol_20']
n_inserted = upsert_features(engine, feat_df, feature_cols)

elapsed = time.time() - start_time
print(f'\n✅ Feature computation complete!')
print(f'✅ Records processed: {len(feat_df):,}')
print(f'✅ Features inserted: {n_inserted:,}')
print(f'✅ Processing time: {elapsed:.1f} seconds')
"
```

### Step 1.3: Validate Feature Population

```bash
python -c "
print('\nStep 1.3: Validating feature population...')

from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

with engine.connect() as conn:
    # Count total features
    total_count = conn.execute(text('SELECT COUNT(*) FROM tt_features')).fetchone()[0]
    print(f'✅ Total feature records: {total_count:,}')
    
    # Count by feature type
    print('\n📊 Features by type:')
    feature_counts = pd.read_sql(text('''
        SELECT feature, COUNT(*) as records,
               ROUND(AVG(value)::numeric, 6) as avg_value,
               ROUND(STDDEV(value)::numeric, 6) as std_value
        FROM tt_features 
        GROUP BY feature 
        ORDER BY feature
    '''), conn)
    print(feature_counts.to_string(index=False))
    
    # Count by symbol
    print('\n📈 Features by symbol (top 10):')
    symbol_counts = pd.read_sql(text('''
        SELECT instrument, COUNT(*) as records
        FROM tt_features 
        GROUP BY instrument 
        ORDER BY records DESC 
        LIMIT 10
    '''), conn)
    print(symbol_counts.to_string(index=False))
    
    # Sample recent features
    print('\n🔍 Recent feature samples:')
    recent = pd.read_sql(text('''
        SELECT instrument, timestamp::date as date, feature, 
               ROUND(value::numeric, 6) as value
        FROM tt_features 
        WHERE timestamp >= (SELECT MAX(timestamp) - INTERVAL '5 days' FROM tt_features)
        ORDER BY timestamp DESC, instrument, feature
        LIMIT 15
    '''), conn)
    print(recent.to_string(index=False))

expected_records = total_count
if expected_records > 3000:
    print(f'\n✅ PHASE 1 COMPLETE: {expected_records:,} features generated')
else:
    print(f'\n⚠️  Warning: Expected ~3,000+ features, got {expected_records}')
"
```

**Expected Results**:
- ~3,246 total feature records
- 3 feature types: return_1, vol_10, vol_20
- Features distributed across all symbols
- No major NaN issues

---

## 📈 **PHASE 2: Signal Generation**

*Duration: 10 minutes*

### Step 2.1: Generate Momentum Signals

```bash
python -c "
print('\n🔍 PHASE 2: Signal Generation')
print('=' * 50)
print('Step 2.1: Generating momentum signals...')

import json
import numpy as np
import pandas as pd
from datetime import datetime
from src.db_utils import get_engine, upsert_signal_scores
from sqlalchemy import text
import time

start_time = time.time()
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Get return features for momentum calculation
print('📊 Loading return data for momentum signals...')
with engine.connect() as conn:
    returns_df = pd.read_sql(text('''
        SELECT f.instrument, f.timestamp, f.value as daily_return
        FROM tt_features f
        WHERE f.feature = 'return_1'
        AND f.value IS NOT NULL
        ORDER BY f.instrument, f.timestamp
    '''), conn)

print(f'✅ Loaded {len(returns_df):,} return observations')

# Generate momentum signals
print('📈 Computing 20-day momentum signals...')
signals = []
processed_symbols = 0

for symbol in returns_df.instrument.unique():
    sym_data = returns_df[returns_df.instrument == symbol].sort_values('timestamp')
    
    if len(sym_data) >= 20:
        # Calculate 20-day rolling momentum
        recent_returns = sym_data.tail(20)['daily_return']
        
        # Momentum score = annualized average return
        momentum_raw = recent_returns.mean() * 252  # Annualized
        
        # Normalize using tanh for bounded [-1, 1] output
        momentum_score = np.tanh(momentum_raw * 2)
        
        # Use most recent date for signal
        latest_date = sym_data.timestamp.max()
        
        signals.append({
            'instrument': symbol,
            'timestamp': latest_date,
            'signal_name': 'momentum_20d',
            'score': float(momentum_score),
            'meta_json': json.dumps({
                'lookback_days': 20,
                'method': 'rolling_return_mean',
                'raw_momentum': float(momentum_raw),
                'annualized': True,
                'normalization': 'tanh'
            })
        })
        processed_symbols += 1

print(f'✅ Generated {len(signals)} momentum signals for {processed_symbols} symbols')

# Persist signals
print('💾 Saving momentum signals to database...')
n_inserted = upsert_signal_scores(engine, signals)

elapsed = time.time() - start_time
print(f'✅ Momentum signals complete!')
print(f'✅ Signals generated: {n_inserted}')
print(f'✅ Processing time: {elapsed:.1f} seconds')

# Show signal distribution
scores = [s['score'] for s in signals]
if scores:
    print(f'\n📊 Momentum signal distribution:')
    print(f'  Mean: {np.mean(scores):.4f}')
    print(f'  Std:  {np.std(scores):.4f}')
    print(f'  Min:  {np.min(scores):.4f}')
    print(f'  Max:  {np.max(scores):.4f}')
"
```

### Step 2.2: Generate Mean Reversion Signals

```bash
python -c "
print('\nStep 2.2: Generating mean reversion signals...')

import json
import numpy as np
import pandas as pd
from src.db_utils import get_engine, upsert_signal_scores
from sqlalchemy import text
import time

start_time = time.time()
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Get price and volatility data
print('📊 Loading price and volatility data...')
with engine.connect() as conn:
    data_df = pd.read_sql(text('''
        SELECT p.instrument, p.timestamp, p.close,
               f.value as volatility
        FROM tt_prices p
        JOIN tt_features f ON p.instrument = f.instrument 
                           AND p.timestamp = f.timestamp
        WHERE f.feature = 'vol_20'
        AND f.value IS NOT NULL
        AND f.value > 0
        ORDER BY p.instrument, p.timestamp
    '''), conn)

print(f'✅ Loaded {len(data_df):,} price-volatility observations')

# Generate mean reversion signals
print('📉 Computing mean reversion signals...')
signals = []
processed_symbols = 0

for symbol in data_df.instrument.unique():
    sym_data = data_df[data_df.instrument == symbol].sort_values('timestamp').copy()
    
    if len(sym_data) >= 20:
        # Calculate 20-day moving average
        sym_data['ma_20'] = sym_data['close'].rolling(20).mean()
        
        # Price deviation from moving average
        sym_data['price_deviation'] = (sym_data['close'] - sym_data['ma_20']) / sym_data['ma_20']
        
        # Get latest observation
        latest = sym_data.iloc[-1]
        
        if pd.notna(latest['price_deviation']) and latest['volatility'] > 0:
            # Mean reversion score = negative normalized deviation
            raw_deviation = latest['price_deviation']
            volatility = latest['volatility']
            
            # Normalize by volatility (like z-score)
            reversion_raw = -raw_deviation / volatility  # Negative for mean reversion
            
            # Bound to [-1, 1]
            reversion_score = np.tanh(reversion_raw)
            
            signals.append({
                'instrument': symbol,
                'timestamp': latest['timestamp'],
                'signal_name': 'mean_reversion_20d',
                'score': float(reversion_score),
                'meta_json': json.dumps({
                    'lookback_days': 20,
                    'method': 'price_deviation_normalized',
                    'price_deviation': float(raw_deviation),
                    'volatility': float(volatility),
                    'current_price': float(latest['close']),
                    'moving_average': float(latest['ma_20'])
                })
            })
            processed_symbols += 1

print(f'✅ Generated {len(signals)} mean reversion signals for {processed_symbols} symbols')

# Persist signals
print('💾 Saving mean reversion signals to database...')
n_inserted = upsert_signal_scores(engine, signals)

elapsed = time.time() - start_time
print(f'✅ Mean reversion signals complete!')
print(f'✅ Signals generated: {n_inserted}')
print(f'✅ Processing time: {elapsed:.1f} seconds')

# Show signal distribution
scores = [s['score'] for s in signals]
if scores:
    print(f'\n📊 Mean reversion signal distribution:')
    print(f'  Mean: {np.mean(scores):.4f}')
    print(f'  Std:  {np.std(scores):.4f}')
    print(f'  Min:  {np.min(scores):.4f}')
    print(f'  Max:  {np.max(scores):.4f}')
"
```

### Step 2.3: Validate Signal Population

```bash
python -c "
print('\nStep 2.3: Validating signal population...')

from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

with engine.connect() as conn:
    # Signal summary
    print('📊 Signal Summary:')
    signal_summary = pd.read_sql(text('''
        SELECT signal_name, 
               COUNT(*) as signal_count,
               ROUND(AVG(score)::numeric, 4) as avg_score,
               ROUND(MIN(score)::numeric, 4) as min_score,
               ROUND(MAX(score)::numeric, 4) as max_score,
               ROUND(STDDEV(score)::numeric, 4) as std_score
        FROM tt_signal_scores 
        GROUP BY signal_name
        ORDER BY signal_name
    '''), conn)
    print(signal_summary.to_string(index=False))
    
    # Signals by symbol
    print('\n📈 Signals by Symbol:')
    symbol_signals = pd.read_sql(text('''
        SELECT instrument,
               COUNT(*) as total_signals,
               AVG(CASE WHEN signal_name = 'momentum_20d' THEN score END)::numeric(6,4) as momentum,
               AVG(CASE WHEN signal_name = 'mean_reversion_20d' THEN score END)::numeric(6,4) as mean_reversion
        FROM tt_signal_scores
        GROUP BY instrument
        ORDER BY instrument
    '''), conn)
    print(symbol_signals.to_string(index=False))
    
    # Recent signals sample
    print('\n🔍 Recent Signal Samples:')
    recent_signals = pd.read_sql(text('''
        SELECT instrument, signal_name, 
               ROUND(score::numeric, 4) as score,
               timestamp::date as signal_date
        FROM tt_signal_scores
        ORDER BY timestamp DESC, instrument, signal_name
        LIMIT 20
    '''), conn)
    print(recent_signals.to_string(index=False))

    # Total count
    total_signals = conn.execute(text('SELECT COUNT(*) FROM tt_signal_scores')).fetchone()[0]
    expected_signals = signal_summary['signal_count'].sum()
    
    if total_signals >= 20:  # Expect ~24 signals (12 symbols × 2 signal types)
        print(f'\n✅ PHASE 2 COMPLETE: {total_signals} signals generated')
    else:
        print(f'\n⚠️  Warning: Expected ~24 signals, got {total_signals}')
"
```

**Expected Results**:
- ~24 total signals (12 symbols × 2 signal types)
- 2 signal types: momentum_20d, mean_reversion_20d
- Signals bounded between -1 and 1
- All symbols have both signal types

---

## 🔧 **PHASE 3: Training Environment Validation**

*Duration: 15 minutes*

### Step 3.1: Dependency Check

```bash
python -c "
print('\n🔍 PHASE 3: Training Environment Validation')
print('=' * 50)
print('Step 3.1: Checking dependencies...')

import sys
print(f'Python version: {sys.version}')
print(f'Python path: {sys.executable}')

# Core dependencies
dependencies = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing', 
    'sqlalchemy': 'Database ORM',
    'stable_baselines3': 'Reinforcement learning',
    'gymnasium': 'RL environments',
    'psycopg2': 'PostgreSQL adapter'
}

print('\n📦 Core Dependencies:')
for package, description in dependencies.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {package:20} {version:15} - {description}')
    except ImportError as e:
        print(f'❌ {package:20} {"MISSING":15} - {description}')
        print(f'   Install with: pip install {package}')

# TensorTrade check
print('\n🔍 TensorTrade Import Test:')
try:
    import site
    site_packages = site.getsitepackages()
    sys.path = site_packages + sys.path
    import tensortrade
    print(f'✅ TensorTrade: {tensortrade.__version__} at {tensortrade.__file__}')
    USE_TENSORTRADE = True
except Exception as e:
    print(f'⚠️  TensorTrade: Import failed - {e}')
    print('   Will use fallback SimpleMultiAssetEnv')
    USE_TENSORTRADE = False

# Test fallback environment
print('\n🔍 Fallback Environment Test:')
try:
    from src.simple_env import SimpleMultiAssetEnv
    print('✅ SimpleMultiAssetEnv: Available')
except Exception as e:
    print(f'❌ SimpleMultiAssetEnv: {e}')

print(f'\n✅ Environment validation complete. TensorTrade mode: {USE_TENSORTRADE}')
"
```

### Step 3.2: Database & Data Validation

```bash
python -c "
print('\nStep 3.2: Validating database and data readiness...')

from src.db_utils import get_engine, fetch_prices
from src.mvp_pipeline import compute_basic_features
from sqlalchemy import text

# Database connection test
print('🔍 Testing database connection...')
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

with engine.connect() as conn:
    # Test tables exist
    tables = ['tt_prices', 'tt_features', 'tt_signal_scores', 'tt_episode', 'tt_action', 'tt_reward']
    print('\n📋 Table Existence Check:')
    for table in tables:
        try:
            count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).fetchone()[0]
            print(f'✅ {table:20} - {count:,} records')
        except Exception as e:
            print(f'❌ {table:20} - ERROR: {e}')

# Data readiness for training
print('\n📊 Training Data Readiness:')
symbols = ['AGX', 'CLS', 'IREN']  # Use subset for testing
try:
    df = fetch_prices(engine, symbols, '2024-01-01', '2024-04-01')
    print(f'✅ Price data: {len(df)} records for {df.symbol.nunique()} symbols')
    
    feat_df = compute_basic_features(df)
    print(f'✅ Feature computation: {len(feat_df)} feature records')
    
    # Check for NaN issues
    nan_close = feat_df['close'].isna().sum()
    nan_return = feat_df['return_1'].isna().sum()
    print(f'✅ Data quality: {nan_close} NaN closes, {nan_return} NaN returns')
    
    if len(df) > 100 and nan_close / len(df) < 0.1:
        print('✅ Training data is ready')
    else:
        print('⚠️  Training data may have quality issues')
        
except Exception as e:
    print(f'❌ Data validation failed: {e}')

print('\n✅ Database validation complete')
"
```

### Step 3.3: Training Module Test

```bash
python -c "
print('\nStep 3.3: Testing training module imports...')

# Test training module imports
print('🔍 Testing training module components...')

try:
    from src.train_mvp import main
    print('✅ Training main function')
except Exception as e:
    print(f'❌ Training main: {e}')

try:
    from src.db_utils import create_episode, finalize_episode, insert_action, insert_reward
    print('✅ Database logging functions')
except Exception as e:
    print(f'❌ DB logging: {e}')

try:
    from src.tensortrade_risk_module import RiskAwareReward, DrawdownStopper
    print('✅ Risk management modules')
except Exception as e:
    print(f'❌ Risk modules: {e}')

try:
    from src.extra_action_schemes import VolatilityTargetedAction, SimpleDiscreteAction
    print('✅ Action schemes')
except Exception as e:
    print(f'❌ Action schemes: {e}')

# Test argument parsing
print('\n🔍 Testing argument parsing...')
try:
    import sys
    from src.train_mvp import parse_args
    
    # Save original argv
    original_argv = sys.argv.copy()
    
    # Test with minimal args
    sys.argv = [
        'train_mvp.py',
        '--months', '1',
        '--limit', '3',
        '--steps', '100',
        '--db-url', 'test'
    ]
    
    args = parse_args()
    print(f'✅ Argument parsing: months={args.months}, limit={args.limit}, steps={args.steps}')
    
    # Restore original argv
    sys.argv = original_argv
    
except Exception as e:
    print(f'❌ Argument parsing: {e}')

print('\n✅ Training module validation complete')
"
```

**Expected Results**:
- All core dependencies available
- TensorTrade may fail (fallback environment ready)
- Database connection successful
- Training data quality good
- Training module imports successful

---

## 🚀 **PHASE 4: RL Training Execution**

*Duration: 30-60 minutes*

### Step 4.1: Short Training Test (5 minutes)

```bash
python -c "
print('\n🔍 PHASE 4: RL Training Execution')
print('=' * 50)
print('Step 4.1: Short training test (100 steps)...')

import os
import sys
import time
from datetime import datetime

# Set database URL in environment
os.environ['DATABASE_URL'] = 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'

print(f'🕐 Training start time: {datetime.now()}')
start_time = time.time()

try:
    # Import training function
    from src.train_mvp import main
    
    # Set up arguments for short test
    sys.argv = [
        'train_mvp.py',
        '--months', '3',
        '--limit', '3',          # Only 3 symbols for speed
        '--steps', '100',        # Very short training
        '--with-features',       # Use computed features
        '--eval-episodes', '1',  # Single evaluation episode
        '--db-url', 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'
    ]
    
    print('📋 Training configuration:')
    print(f'  Symbols: 3 (limited for testing)')
    print(f'  Training steps: 100')
    print(f'  Evaluation episodes: 1')
    print(f'  Features enabled: Yes')
    print()
    
    # Execute training
    print('🚀 Starting training...')
    main()
    
    elapsed = time.time() - start_time
    print(f'\n✅ Short training test completed!')
    print(f'✅ Total time: {elapsed:.1f} seconds')
    
except KeyboardInterrupt:
    print('\n⚠️  Training interrupted by user')
except Exception as e:
    print(f'\n❌ Training failed: {e}')
    print('Check error details above and troubleshoot if needed')
    elapsed = time.time() - start_time
    print(f'Time before failure: {elapsed:.1f} seconds')
"
```

### Step 4.2: Validate Training Results

```bash
python -c "
print('\nStep 4.2: Validating training results...')

from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📊 Training Data Population Check:')
print('=' * 50)

with engine.connect() as conn:
    # Episode data
    episodes = pd.read_sql(text('''
        SELECT COUNT(*) as episodes,
               MIN(start_time) as first_start,
               MAX(end_time) as last_end
        FROM tt_episode
    '''), conn)
    print(f'Episodes created: {episodes.iloc[0]["episodes"]}')
    if episodes.iloc[0]["episodes"] > 0:
        print(f'First episode: {episodes.iloc[0]["first_start"]}')
        print(f'Last episode: {episodes.iloc[0]["last_end"]}')
    
    # Action data
    actions = pd.read_sql(text('''
        SELECT COUNT(*) as total_actions,
               COUNT(DISTINCT episode_id) as episodes_with_actions,
               COUNT(DISTINCT instrument) as instruments_traded
        FROM tt_action
    '''), conn)
    print(f'\\nActions logged: {actions.iloc[0]["total_actions"]:,}')
    print(f'Episodes with actions: {actions.iloc[0]["episodes_with_actions"]}')
    print(f'Instruments traded: {actions.iloc[0]["instruments_traded"]}')
    
    # Reward data
    rewards = pd.read_sql(text('''
        SELECT COUNT(*) as total_rewards,
               ROUND(AVG(reward_value)::numeric, 6) as avg_reward,
               ROUND(MIN(reward_value)::numeric, 6) as min_reward,
               ROUND(MAX(reward_value)::numeric, 6) as max_reward
        FROM tt_reward
    '''), conn)
    print(f'\\nRewards logged: {rewards.iloc[0]["total_rewards"]:,}')
    if rewards.iloc[0]["total_rewards"] > 0:
        print(f'Average reward: {rewards.iloc[0]["avg_reward"]}')
        print(f'Reward range: [{rewards.iloc[0]["min_reward"]}, {rewards.iloc[0]["max_reward"]}]')

# Episode performance details
if episodes.iloc[0]["episodes"] > 0:
    print('\n📈 Episode Performance:')
    with engine.connect() as conn:
        episode_perf = pd.read_sql(text('''
            SELECT id, 
                   start_time::timestamp(0) as start_time,
                   end_time::timestamp(0) as end_time,
                   stop_reason,
                   ROUND(max_drawdown::numeric, 4) as max_dd,
                   ROUND(sharpe_ratio::numeric, 4) as sharpe,
                   ROUND(turnover::numeric, 4) as turnover
            FROM tt_episode 
            ORDER BY id DESC
        '''), conn)
        print(episode_perf.to_string(index=False))

# Validation summary
total_episodes = episodes.iloc[0]["episodes"]
total_actions = actions.iloc[0]["total_actions"]
total_rewards = rewards.iloc[0]["total_rewards"]

print(f'\n🎯 Training Validation Summary:')
if total_episodes > 0:
    print(f'✅ Episodes: {total_episodes} (expected: 1+)')
else:
    print(f'❌ Episodes: {total_episodes} (expected: 1+)')

if total_actions > 0:
    print(f'✅ Actions: {total_actions:,} (expected: 100+)')
else:
    print(f'❌ Actions: {total_actions} (expected: 100+)')
    
if total_rewards > 0:
    print(f'✅ Rewards: {total_rewards:,} (expected: 50+)')
else:
    print(f'❌ Rewards: {total_rewards} (expected: 50+)')

if total_episodes > 0 and total_actions > 0 and total_rewards > 0:
    print('\\n🎉 SHORT TRAINING TEST SUCCESSFUL!')
    print('Ready to proceed with full training run')
else:
    print('\\n⚠️  Training test had issues. Check logs above.')
"
```

### Step 4.3: Full Training Run (30-60 minutes)

**⚠️ Only run this if Step 4.1 and 4.2 were successful**

```bash
python -c "
print('\nStep 4.3: Full training run...')
print('⚠️  This will take 30-60 minutes')
print('⏸️  You can interrupt with Ctrl+C if needed')

response = input('Do you want to proceed with full training? (y/N): ')
if response.lower() != 'y':
    print('Skipping full training run')
    exit()

import os
import sys
import time
from datetime import datetime

# Set database URL
os.environ['DATABASE_URL'] = 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'

print(f'🕐 Full training start time: {datetime.now()}')
start_time = time.time()

try:
    from src.train_mvp import main
    
    # Full training configuration
    sys.argv = [
        'train_mvp.py',
        '--months', '3',
        '--limit', '10',         # Use 10 symbols
        '--steps', '10000',      # Full training steps
        '--with-features',       # Use computed features
        '--eval-episodes', '2',  # Two evaluation episodes
        '--db-url', 'postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway'
    ]
    
    print('📋 Full Training Configuration:')
    print(f'  Symbols: 10')
    print(f'  Training steps: 10,000')
    print(f'  Evaluation episodes: 2')
    print(f'  Features enabled: Yes')
    print()
    
    print('🚀 Starting full training run...')
    print('📊 Progress will be shown during training')
    main()
    
    elapsed = time.time() - start_time
    print(f'\n🎉 FULL TRAINING COMPLETED!')
    print(f'✅ Total training time: {elapsed/60:.1f} minutes')
    
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f'\n⚠️  Training interrupted after {elapsed/60:.1f} minutes')
    print('Partial results may be available in database')
except Exception as e:
    elapsed = time.time() - start_time
    print(f'\n❌ Training failed after {elapsed/60:.1f} minutes: {e}')
"
```

### Step 4.4: Final Training Validation

```bash
python -c "
print('\nStep 4.4: Final training validation...')

from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📊 FINAL TRAINING RESULTS')
print('=' * 50)

with engine.connect() as conn:
    # Complete summary
    summary = pd.read_sql(text('''
        SELECT 
            (SELECT COUNT(*) FROM tt_episode) as episodes,
            (SELECT COUNT(*) FROM tt_action) as actions,
            (SELECT COUNT(*) FROM tt_reward) as rewards,
            (SELECT COUNT(DISTINCT instrument) FROM tt_action) as instruments,
            (SELECT MAX(end_time) - MIN(start_time) FROM tt_episode) as total_duration
        '''), conn)
    
    print('🎯 Training Summary:')
    print(f'  Episodes completed: {summary.iloc[0]["episodes"]}')
    print(f'  Actions logged: {summary.iloc[0]["actions"]:,}')
    print(f'  Rewards logged: {summary.iloc[0]["rewards"]:,}')
    print(f'  Instruments traded: {summary.iloc[0]["instruments"]}')
    if summary.iloc[0]["total_duration"]:
        print(f'  Training duration: {summary.iloc[0]["total_duration"]}')
    
    # Performance metrics
    if summary.iloc[0]["episodes"] > 0:
        print('\n📈 Episode Performance Metrics:')
        performance = pd.read_sql(text('''
            SELECT 
                ROUND(AVG(max_drawdown)::numeric, 4) as avg_drawdown,
                ROUND(AVG(sharpe_ratio)::numeric, 4) as avg_sharpe,
                ROUND(AVG(turnover)::numeric, 4) as avg_turnover,
                COUNT(*) as total_episodes
            FROM tt_episode 
            WHERE max_drawdown IS NOT NULL
        '''), conn)
        print(f'  Average Drawdown: {performance.iloc[0]["avg_drawdown"]}')
        print(f'  Average Sharpe: {performance.iloc[0]["avg_sharpe"]}')
        print(f'  Average Turnover: {performance.iloc[0]["avg_turnover"]}')
    
    # Action distribution
    if summary.iloc[0]["actions"] > 0:
        print('\n📊 Action Distribution by Instrument:')
        action_dist = pd.read_sql(text('''
            SELECT instrument,
                   COUNT(*) as action_count,
                   ROUND(AVG(action_value)::numeric, 4) as avg_weight,
                   ROUND(STDDEV(action_value)::numeric, 4) as weight_volatility
            FROM tt_action
            GROUP BY instrument
            ORDER BY action_count DESC
            LIMIT 10
        '''), conn)
        print(action_dist.to_string(index=False))

# Success criteria
episodes = summary.iloc[0]["episodes"]
actions = summary.iloc[0]["actions"]
rewards = summary.iloc[0]["rewards"]

print(f'\n🎯 PHASE 4 SUCCESS CHECK:')
success_criteria = [
    (episodes >= 1, f'Episodes: {episodes} ≥ 1'),
    (actions >= 1000, f'Actions: {actions:,} ≥ 1,000'),
    (rewards >= 500, f'Rewards: {rewards:,} ≥ 500'),
]

all_success = True
for criterion, description in success_criteria:
    status = '✅' if criterion else '❌'
    print(f'{status} {description}')
    if not criterion:
        all_success = False

if all_success:
    print(f'\n🎉 PHASE 4 COMPLETE: Training pipeline successfully populated all core tables!')
else:
    print(f'\n⚠️  Some training targets not met. Check individual results above.')
"
```

**Expected Results**:
- 2-3 episodes completed
- 1,000+ action records
- 500+ reward records  
- Performance metrics calculated
- All 10 symbols have action records

---

## 📊 **PHASE 5: Enhanced Logging (Optional)**

*Duration: 20 minutes*

### Step 5.1: Test Target Weights Logging

```bash
python -c "
print('\n🔍 PHASE 5: Enhanced Logging (Optional)')
print('=' * 50)
print('Step 5.1: Testing target weights logging...')

from src.db_utils import get_engine, insert_target_weights
from datetime import datetime
import json

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

# Test with sample weights
test_weights = {
    'AGX': 0.15,
    'CLS': 0.20,
    'IREN': 0.25,
    'CASH': 0.40
}

print('📊 Testing target weights insertion...')
try:
    n_inserted = insert_target_weights(
        engine=engine,
        episode_id=999,  # Test episode
        timestamp=datetime.now(),
        weights=test_weights,
        strategy='test_strategy',
        rationale='Manual test of target weights logging'
    )
    print(f'✅ Inserted {n_inserted} target weight records')
    
    # Verify insertion
    from sqlalchemy import text
    with engine.connect() as conn:
        test_data = conn.execute(text('''
            SELECT instrument, target_weight, strategy, rationale 
            FROM tt_target_weights 
            WHERE episode_id = 999
            ORDER BY instrument
        ''')).fetchall()
        
        print('\n📋 Test target weights:')
        for row in test_data:
            print(f'  {row[0]}: {row[1]} ({row[2]})')
        
        # Clean up test data
        conn.execute(text('DELETE FROM tt_target_weights WHERE episode_id = 999'))
        conn.commit()
        print('✅ Test data cleaned up')
        
except Exception as e:
    print(f'❌ Target weights test failed: {e}')

print('✅ Target weights logging test complete')
"
```

### Step 5.2: Test Equity Curve Logging

```bash
python -c "
print('\nStep 5.2: Testing equity curve logging...')

from src.db_utils import get_engine, insert_equity_point
from datetime import datetime, timedelta
import numpy as np

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📈 Testing equity curve insertion...')
try:
    # Generate sample equity curve
    base_time = datetime.now()
    base_equity = 100000.0
    
    equity_points = []
    for i in range(10):
        # Simulate random walk equity curve
        timestamp = base_time + timedelta(hours=i)
        equity_value = base_equity * (1 + np.cumsum(np.random.normal(0, 0.01, i+1))[-1])
        
        insert_equity_point(engine, 999, timestamp, equity_value)
        equity_points.append((timestamp, equity_value))
    
    print(f'✅ Inserted {len(equity_points)} equity curve points')
    
    # Verify insertion
    from sqlalchemy import text
    with engine.connect() as conn:
        test_data = conn.execute(text('''
            SELECT timestamp, net_worth
            FROM tt_equity_curve 
            WHERE episode_id = 999
            ORDER BY timestamp
        ''')).fetchall()
        
        print('\n📈 Test equity curve:')
        for i, (timestamp, net_worth) in enumerate(test_data[:5]):
            print(f'  {timestamp}: ${net_worth:,.2f}')
        if len(test_data) > 5:
            print(f'  ... and {len(test_data) - 5} more points')
        
        # Calculate performance metrics
        values = [float(row[1]) for row in test_data]
        if len(values) > 1:
            total_return = (values[-1] - values[0]) / values[0]
            max_value = max(values)
            min_drawdown = min(v/max_value - 1 for v in values)
            print(f'\n📊 Test Performance:')
            print(f'  Total Return: {total_return:.2%}')
            print(f'  Max Drawdown: {min_drawdown:.2%}')
        
        # Clean up test data
        conn.execute(text('DELETE FROM tt_equity_curve WHERE episode_id = 999'))
        conn.commit()
        print('✅ Test data cleaned up')
        
except Exception as e:
    print(f'❌ Equity curve test failed: {e}')

print('✅ Equity curve logging test complete')
"
```

### Step 5.3: Implementation Guide for Enhanced Logging

```bash
python -c "
print('\nStep 5.3: Enhanced logging implementation guide...')

print('''
📝 ENHANCED LOGGING IMPLEMENTATION GUIDE
=====================================

To add target weights and equity curve logging to the training pipeline,
you need to modify the src/train_mvp.py file:

🎯 TARGET WEIGHTS LOGGING:
-------------------------
Location: Around line 328 in src/train_mvp.py
Add after the action logging loop:

```python
# Log target weights
if len(arr) > 0:
    weights = {sym: float(arr[i]) for i, sym in enumerate(symbols[:len(arr)])}
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
        
    insert_target_weights(
        engine, episode_id, ts, weights,
        strategy=f\"ppo_step_{step_count}\",
        rationale=f\"Portfolio allocation at training step {step_count}\"
    )
```

📈 EQUITY CURVE LOGGING:
-----------------------
Location: Around line 350 in src/train_mvp.py
Modify the equity tracking section:

```python
if eq is not None:
    eq_f = float(eq)
    # Log equity point
    insert_equity_point(engine, episode_id, ts, eq_f)
    equity_hist.append(eq_f)
    # ... rest of existing code
```

🔧 REQUIRED IMPORTS:
------------------
Add to imports at top of train_mvp.py:

```python
from db_utils import insert_target_weights, insert_equity_point
```

⚠️  PERFORMANCE IMPACT:
---------------------
These enhancements will:
- Increase database writes significantly
- Add ~10-20% overhead to training time
- Generate large amounts of data for long training runs
- Provide detailed portfolio tracking for analysis

🎯 TESTING:
----------
After implementation:
1. Run short training test (100 steps)
2. Verify tt_target_weights has records
3. Verify tt_equity_curve has records
4. Check data quality and performance impact

📊 EXPECTED DATA VOLUME:
-----------------------
For 10,000 training steps with 10 symbols:
- Target weights: ~100,000 records (10 symbols × 10,000 steps)
- Equity curve: ~10,000 records (1 per step)
- Additional storage: ~50MB
''')

print('✅ Enhanced logging implementation guide complete')
print('\\n🎯 OPTIONAL: You can implement these enhancements for more detailed tracking')
"
```

---

## 🎉 **COMPLETION & VALIDATION**

### Final System Validation

```bash
python -c "
print('\n🎉 FINAL SYSTEM VALIDATION')
print('=' * 50)

from src.db_utils import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

print('📊 COMPLETE TABLE POPULATION STATUS:')
print('=' * 50)

tables_info = [
    ('tt_prices', 'Historical OHLCV price data', '1,082+'),
    ('tt_features', 'Technical features (return, volatility)', '3,000+'),
    ('tt_signal_scores', 'Trading signals (momentum, mean reversion)', '24+'),
    ('tt_episode', 'Training episode metadata', '1+'),
    ('tt_action', 'Agent portfolio actions', '1,000+'),
    ('tt_reward', 'Environment rewards', '500+'),
    ('tt_target_weights', 'Portfolio target weights', '0+ (optional)'),
    ('tt_equity_curve', 'Portfolio equity over time', '0+ (optional)')
]

with engine.connect() as conn:
    total_populated = 0
    for table, description, target in tables_info:
        try:
            count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).fetchone()[0]
            status = '✅ POPULATED' if count > 0 else '❌ EMPTY'
            print(f'{table:20} | {count:8,} | {target:8} | {status}')
            if count > 0:
                total_populated += 1
        except Exception as e:
            print(f'{table:20} | ERROR   | {target:8} | ❌ ERROR')

print(f'\n📋 POPULATION SUMMARY:')
print(f'Tables populated: {total_populated}/8')
print(f'Core tables (required): {min(total_populated, 6)}/6')
print(f'Enhanced tables (optional): {max(0, total_populated-6)}/2')

# Data quality checks
print(f'\n🔍 DATA QUALITY CHECKS:')

# Check for foreign key integrity
with engine.connect() as conn:
    # Episode-Action consistency
    orphan_actions = conn.execute(text('''
        SELECT COUNT(*) FROM tt_action a 
        LEFT JOIN tt_episode e ON a.episode_id = e.id 
        WHERE e.id IS NULL
    ''')).fetchone()[0]
    
    # Episode-Reward consistency  
    orphan_rewards = conn.execute(text('''
        SELECT COUNT(*) FROM tt_reward r
        LEFT JOIN tt_episode e ON r.episode_id = e.id
        WHERE e.id IS NULL
    ''')).fetchone()[0]
    
    print(f'Orphaned actions: {orphan_actions} (should be 0)')
    print(f'Orphaned rewards: {orphan_rewards} (should be 0)')

# Performance summary
if total_populated >= 6:
    print(f'\n🎉 SUCCESS: Core TensorTrade pipeline is fully populated!')
    print(f'✅ Data ingestion: COMPLETE')
    print(f'✅ Feature engineering: COMPLETE') 
    print(f'✅ Signal generation: COMPLETE')
    print(f'✅ RL training: COMPLETE')
    print(f'\n🚀 System ready for:')
    print(f'  - Advanced feature engineering')
    print(f'  - Signal strategy development')
    print(f'  - Model performance analysis')
    print(f'  - Paper trading simulation')
else:
    print(f'\n⚠️  INCOMPLETE: {6-min(total_populated,6)} core tables still empty')
    print(f'Review the execution steps above to complete population')

print(f'\n📅 Completion timestamp: {pd.Timestamp.now()}')
"
```

---

## 📋 **Quick Reference Commands**

### Check Table Status
```bash
python show_table_data.py
```

### Repopulate Features
```bash
python -c "
from src.mvp_pipeline import compute_basic_features
from src.db_utils import get_engine, fetch_prices, upsert_features
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
df = fetch_prices(engine)
feat_df = compute_basic_features(df)
upsert_features(engine, feat_df, ['return_1', 'vol_10', 'vol_20'])
"
```

### Quick Training Test
```bash
python -m src.train_mvp --months 1 --limit 3 --steps 100 --with-features --db-url postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway
```

### View Recent Results
```bash
python -c "
from src.db_utils import get_engine
from sqlalchemy import text
engine = get_engine('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')
with engine.connect() as conn:
    episodes = conn.execute(text('SELECT * FROM tt_episode ORDER BY id DESC LIMIT 5')).fetchall()
    for ep in episodes:
        print(f'Episode {ep[0]}: {ep[3]} (Sharpe: {ep[5]}, DD: {ep[4]})')
"
```

---

## 🚨 **Troubleshooting Guide**

### Common Issues & Solutions

**❌ Database Connection Failed**
```bash
# Check network connectivity
ping crossover.proxy.rlwy.net

# Test direct connection
python -c "import psycopg2; psycopg2.connect('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')"
```

**❌ TensorTrade Import Errors**
```bash
# Use fallback environment
export USE_TENSORTRADE=false
```

**❌ Training Crashes**
```bash
# Reduce complexity
python -m src.train_mvp --months 1 --limit 2 --steps 50 --db-url <URL>
```

**❌ Feature Computation Errors**
```bash
# Check for missing data
python -c "from src.db_utils import fetch_prices, get_engine; df=fetch_prices(get_engine('...')); print(df.isna().sum())"
```

**❌ Out of Memory**
```bash
# Process in smaller batches
python -c "
engine = get_engine('...')
for symbol in ['AGX', 'CLS']:  # Process one at a time
    df = fetch_prices(engine, [symbol])
    # ... process
"
```

---

## 📞 **Support & Next Steps**

### Completion Checklist
- [ ] All 8 tables populated (6 core + 2 optional)
- [ ] Data quality validation passed
- [ ] Training pipeline executed successfully
- [ ] Performance metrics calculated
- [ ] System ready for advanced development

### Next Development Steps
1. **Advanced Features**: Add more technical indicators
2. **Signal Strategies**: Implement automated signal generation
3. **Model Optimization**: Hyperparameter tuning and A/B testing
4. **Paper Trading**: Live simulation with real market data
5. **Performance Analytics**: Comprehensive backtesting framework

### Documentation References
- `COMPLETE_TABLE_POPULATION_GUIDE.md` - Detailed table documentation
- `TABLE_POPULATION_EXECUTION_PLAN.md` - Phase-by-phase execution plan
- `PROJECT_STATUS_REPORT.md` - Overall system status and roadmap

**Estimated Total Execution Time**: 90-120 minutes  
**Success Rate**: High (with proper dependency management)  
**Next Review**: After successful completion, review performance metrics and plan advanced features

---

*This guide provides complete step-by-step instructions to populate all TensorTrade database tables through their appropriate processes, with validation and troubleshooting at every step.*
