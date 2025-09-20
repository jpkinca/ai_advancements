# TensorTrade MVP Program Architecture Guide

**Date:** August 21, 2025  
**Purpose:** Complete overview of TensorTrade MVP programs and their functions

## 🏗️ **TensorTrade MVP Architecture Overview**

### **📊 Data Pipeline Programs:**

#### 1. **`watchlist_loader.py`** - **Data Ingestion Hub**
- **Purpose**: Loads 150 stock symbols from CSV and fetches historical data from IBKR
- **Functions**:
  - `load_watchlist()` - Reads symbols from `data/input/tensorwatchlist.csv`
  - `fetch_price_history_ibkr()` - Downloads OHLCV data from IBKR with rate limiting
  - `build_price_streams()` - Converts data to TensorTrade format
  - **NEW**: `fetch_multi_timeframe_data()` - Fetches 3 timeframes + stores to DB
    - **3 years** of daily data (`1d` interval)
    - **2 years** of hourly data (`1h` interval) 
    - **1 year** of 15-minute data (`15m` interval)

#### 2. **`mvp_pipeline.py`** - **Data Processing Pipeline**
- **Purpose**: End-to-end data pipeline (ingestion → processing → persistence)
- **Functions**:
  - Orchestrates watchlist loading → IBKR fetch → feature computation → DB storage
  - `compute_basic_features()` - Generates returns and rolling volatility features
  - CLI interface for configurable date ranges and intervals

#### 3. **`create_tt_tables.py`** - **Database Schema**
- **Purpose**: Creates PostgreSQL database schema
- **Tables Created**:
  - `tt_prices` - Historical OHLCV data
  - `tt_episode` - Training episode metadata
  - `tt_action`, `tt_reward` - RL logging tables
  - `tt_portfolio`, `tt_holdings` - Portfolio state tracking

#### 4. **`db_utils.py`** - **Database Operations**
- **Purpose**: Database utilities for CRUD operations
- **Key Functions**:
  - `upsert_price_bars()` - Store historical data with conflict handling
  - `get_engine()` - Database connection management
  - `insert_action()`, `insert_reward()` - RL step logging

### **🤖 Machine Learning Programs:**

#### 5. **`train_mvp.py`** - **Complete Training Pipeline**
- **Purpose**: Full end-to-end RL training system
- **Functions**:
  - Loads symbols and historical data
  - Builds TensorTrade environment with risk-aware components
  - Trains PPO models using Stable-Baselines3
  - Logs episodes, actions, and rewards to database
  - Saves trained models as `.zip` files

#### 6. **`tensortrade_risk_module.py`** - **Risk-Aware RL Components**
- **Purpose**: Custom TensorTrade components for risk management
- **Components**:
  - `VolatilityTargetedAction` - Position sizing based on volatility
  - `RiskAwareReward` - Sharpe-like reward with drawdown penalties
  - `DrawdownStopper` - Early episode termination on max drawdown

#### 7. **`env_wrappers.py`** - **Database Logging Wrapper**
- **Purpose**: Wraps TensorTrade environment to log all steps to database
- **Functions**:
  - Records every action and reward to `tt_action`/`tt_reward` tables
  - Tracks episode lifecycle and portfolio state

### **🔧 Utility Programs:**

#### 8. **`pnl_dashboard.py`** - **Performance Analytics Dashboard**
- **Purpose**: Streamlit dashboard for P&L analysis and performance tracking
- **Features**:
  - Real-time P&L visualization
  - Trade-level analytics
  - Performance attribution and risk metrics

#### 9. **`show_table_data.py`** - **Database Inspector**
- **Purpose**: Query and display data from database tables
- **Functions**: Browse stored prices, episodes, actions, and rewards

### **📋 Program Execution Flow:**

```
tensorwatchlist.csv (150 symbols)
        ↓
watchlist_loader.py (Data Ingestion)
        ↓
IBKR Data Fetch (Multi-timeframe)
        ↓
mvp_pipeline.py (Processing & Features)
        ↓
Database Storage (PostgreSQL)
        ↓
train_mvp.py (RL Training)
        ↓
TensorTrade Environment (Risk-aware)
        ↓
PPO Training (Stable-Baselines3)
        ↓
Model + Logs (Performance tracking)
        ↓
pnl_dashboard.py (Analytics)
```

### **🚀 Command Line Usage:**

```bash
# 1. Data Pipeline (Fetch & Store Historical Data)
python -m mvp_pipeline --start 2022-01-01 --end 2025-08-20 --limit 25

# 2. Multi-timeframe Data Fetch (Enhanced)
python watchlist_loader.py --multi-timeframe

# 3. Full RL Training Pipeline
python -m train_mvp --months 6 --steps 10000 --limit 25 --source ibkr

# 4. View Results Dashboard
streamlit run pnl_dashboard.py

# 5. Inspect Database Tables
python show_table_data.py
```

### **📊 Data Volume Expectations:**

For 150 symbols with multi-timeframe approach:
- **Daily data (3 years)**: ~113,400 data points
- **Hourly data (2 years)**: ~1,470,600 data points  
- **15-minute data (1 year)**: ~982,800 data points
- **Total**: ~2.5+ million data points

### **🔧 Recent Enhancements (August 2025):**

#### Multi-timeframe Data Fetching:
- **Duration Format Fix**: Proper IBKR format (`3 Y` vs `1095 D`) for >365 day requests
- **Event Loop Handling**: `nest_asyncio` integration to resolve async conflicts
- **Conservative Rate Limiting**: Progressive delays (0.3s, 0.4s, 0.5s) for different timeframes
- **Automatic Database Storage**: Real-time upsert to `tt_prices` table during fetch
- **Error Recovery**: Individual symbol failures don't stop entire process

#### Key Files Updated:
- `watchlist_loader.py` - Enhanced with `fetch_multi_timeframe_data()`
- `_duration_str_from_days()` - IBKR-compliant duration formatting
- Database integration for automatic storage during fetch

### **💡 Best Practices:**

1. **Testing**: Start with `--limit 25` for first 25 symbols before full runs
2. **Rate Limiting**: IBKR requires conservative pacing - built into system
3. **Database**: Ensure PostgreSQL connection before running data pipelines
4. **Storage**: Multi-timeframe fetch automatically stores to database
5. **Monitoring**: Check logs for fetch progress and storage confirmation

### **🎯 Recommended Workflow:**

1. **Initial Setup**: Run `mvp_pipeline.py` to fetch and store historical data
2. **Training**: Use `train_mvp.py` for RL model development
3. **Analysis**: Monitor performance with `pnl_dashboard.py`
4. **Inspection**: Use `show_table_data.py` to verify stored data

This architecture provides a **complete algorithmic trading system** from data ingestion through model training to performance analysis, all integrated with a PostgreSQL database for persistence and tracking.

---

**Status**: Updated with multi-timeframe enhancements  
**Last Modified**: August 21, 2025  
**Author**: TensorTrade Development Team
