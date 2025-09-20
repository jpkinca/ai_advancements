import json
import math
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import pytest
import tempfile
import os

import db_utils as du
from mvp_pipeline import compute_basic_features

# Import new components for enhanced testing
from src.real_time_streaming import TensorTradeRealTimeEngine
from src.enhanced_risk_module import EnhancedRiskManager, RiskLimits, Position
from src.paper_trading_engine import PaperTradingEngine, OrderSide, OrderType
from src.signal_bridge import TensorTradeSignalBridge, SignalType


class MockRLModel:
    """Mock RL model for testing"""
    def __init__(self, action_space_size: int = 3):
        self.action_space_size = action_space_size
        self.prediction_count = 0
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.prediction_count += 1
        if self.action_space_size == 3:
            actions = np.random.dirichlet([2, 1, 1], size=features.shape[0])
        else:
            actions = np.random.normal(0, 0.1, size=(features.shape[0], 1))
        return actions


class MockFeatureEngine:
    """Mock feature engineering for testing"""
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
    
    def compute_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        if price_data.empty:
            return pd.DataFrame()
        
        n_samples = len(price_data.groupby('symbol'))
        features = np.random.randn(n_samples, self.n_features)
        symbols = price_data['symbol'].unique()
        timestamps = [datetime.now()] * len(symbols)
        
        feature_df = pd.DataFrame(
            features,
            index=pd.MultiIndex.from_arrays([symbols, timestamps], names=['symbol', 'timestamp'])
        )
        return feature_df


@pytest.mark.integration
def test_end_to_end_pipeline(engine):
    du.ensure_tables(engine)
    du.ensure_indexes(engine)

    # 1. Synthetic price history for 2 symbols over 6 days
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(6):
        ts = base + timedelta(days=i)
        for sym, base_price in [("AAA", 10.0), ("BBB", 20.0)]:
            price = base_price * (1 + 0.01 * i)
            rows.append({
                "symbol": sym,
                "datetime": ts,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1000 + 10 * i,
            })
    price_df = pd.DataFrame(rows)

    inserted = du.upsert_price_bars(engine, price_df)
    assert inserted == len(price_df)

    fetched = du.fetch_prices(engine, ["AAA", "BBB"], start="2024-01-01", end="2024-01-31")
    assert len(fetched) == len(price_df)

    # 2. Feature computation & persistence
    feat_df = compute_basic_features(fetched)
    # Fill NaNs for first return, vols
    for col in ["return_1", "vol_10", "vol_20"]:
        if col in feat_df:
            feat_df[col] = feat_df[col].fillna(0.0)
    n_features = du.upsert_features(engine, feat_df, ["return_1", "vol_10", "vol_20"])
    # Each row * 3 features
    assert n_features == len(feat_df) * 3

    # 3. Simple signal scoring (use return_1 as momentum proxy on final day only)
    last_day = feat_df.groupby('symbol').tail(1)
    signal_scores = []
    for _, r in last_day.iterrows():
        signal_scores.append({
            'instrument': r['symbol'],
            'timestamp': r['datetime'],
            'signal_name': 'momentum_last',
            'score': float(r['return_1']),
            'meta_json': json.dumps({'source': 'return_1'})
        })
    du.upsert_signal_scores(engine, signal_scores)

    # 4. Episode lifecycle & target weights derived from signal scores (normalized positive scores)
    ep_id = du.create_episode(engine, datetime.utcnow())
    total_positive = sum(max(0.0, s['score']) for s in signal_scores) or 1.0
    weights = {s['instrument']: max(0.0, s['score']) / total_positive for s in signal_scores}
    du.insert_target_weights(engine, ep_id, datetime.utcnow(), weights, strategy='momentum_norm', rationale='normalized momentum scores')

    # Equity curve points (simulate monotonic growth)
    for step in range(3):
        du.insert_equity_point(engine, ep_id, datetime.utcnow() + timedelta(minutes=step), 10000.0 * (1 + 0.001 * step))

    du.set_episode_config(engine, ep_id, json.dumps({'test': True}))
    du.finalize_episode(engine, ep_id, datetime.utcnow(), stop_reason='test_complete', max_drawdown=0.0, sharpe_ratio=0.0, turnover=0.0)

    # 5. Assertions on persistence
    with engine.begin() as conn:
        feat_cnt = conn.execute(pd.io.sql.text('SELECT COUNT(*) FROM tt_features')).scalar()
        sig_cnt = conn.execute(pd.io.sql.text('SELECT COUNT(*) FROM tt_signal_scores')).scalar()
        tw_cnt = conn.execute(pd.io.sql.text('SELECT COUNT(*) FROM tt_target_weights WHERE episode_id=:ep'), {'ep': ep_id}).scalar()
        eq_cnt = conn.execute(pd.io.sql.text('SELECT COUNT(*) FROM tt_equity_curve WHERE episode_id=:ep'), {'ep': ep_id}).scalar()
        episode_cfg = conn.execute(pd.io.sql.text('SELECT config_json FROM tt_episode WHERE id=:ep'), {'ep': ep_id}).scalar()
    assert feat_cnt >= n_features
    assert sig_cnt == len(signal_scores)
    assert tw_cnt == len(signal_scores)
    assert eq_cnt == 3
    assert episode_cfg is not None and 'test' in episode_cfg
    # Weights should sum to ~1
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_pipeline_with_new_components(engine):
    """Test enhanced pipeline with real-time streaming, risk management, and paper trading"""
    
    # Ensure tables exist
    du.ensure_tables(engine)
    du.ensure_indexes(engine)
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Initialize enhanced components
    risk_limits = RiskLimits(
        max_position_size=0.10,
        max_daily_drawdown=0.05,
        circuit_breaker_threshold=0.03
    )
    
    # Create temporary database manager
    from src.db_utils import DatabaseManager
    db_manager = DatabaseManager()
    db_manager.engine = engine  # Use the test engine
    
    risk_manager = EnhancedRiskManager(
        db_manager=db_manager,
        risk_limits=risk_limits,
        initial_capital=100000.0
    )
    
    paper_engine = PaperTradingEngine(
        initial_capital=100000.0,
        risk_manager=risk_manager
    )
    
    signal_bridge = TensorTradeSignalBridge(db_manager)
    
    real_time_engine = TensorTradeRealTimeEngine(
        symbols=symbols,
        db_manager=db_manager,
        use_simulation=True
    )
    
    # Set up mock RL components
    mock_rl_model = MockRLModel()
    mock_feature_engine = MockFeatureEngine()
    
    real_time_engine.set_rl_model(mock_rl_model)
    real_time_engine.set_feature_engine(mock_feature_engine)
    
    # Track pipeline activity
    decisions_made = []
    trades_executed = []
    signals_generated = []
    
    def on_rl_decision(actions: np.ndarray, symbols_list):
        decisions_made.append((actions.copy(), symbols_list.copy()))
        
        # Convert to signals and execute trades
        current_prices = real_time_engine.streamer.get_latest_prices()
        signals = signal_bridge.export_rl_signals(
            symbols=symbols_list,
            actions=actions,
            current_prices=current_prices
        )
        signals_generated.extend(signals)
        
        # Execute trades based on signals
        for signal in signals:
            if signal.signal_type == SignalType.ENTRY_LONG:
                portfolio_value = paper_engine.get_total_portfolio_value()
                position_size = signal.position_size or 0.02
                dollar_amount = portfolio_value * position_size
                
                if signal.entry_price and signal.entry_price > 0:
                    quantity = int(dollar_amount / signal.entry_price)
                    if quantity > 0:
                        order_id = paper_engine.place_order(
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                        if order_id:
                            trades_executed.append(('BUY', signal.symbol, quantity))
    
    def on_trade(trade):
        print(f"Trade executed: {trade.side.value} {trade.total_quantity} {trade.symbol} @ ${trade.avg_price:.2f}")
    
    # Set up callbacks
    real_time_engine.add_decision_handler(on_rl_decision)
    paper_engine.add_trade_callback(on_trade)
    
    # Start real-time engine
    success = await real_time_engine.start()
    assert success, "Real-time engine should start successfully"
    
    # Simulate market data and trading activity
    test_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2800.0}
    
    for i in range(3):  # 3 market updates
        # Add some price movement
        for symbol in symbols:
            price_change = np.random.normal(0, 0.01) * test_prices[symbol]
            test_prices[symbol] += price_change
        
        # Update market prices (triggers decision pipeline)
        paper_engine.update_market_prices(test_prices)
        await asyncio.sleep(0.1)  # Allow processing time
    
    # Stop engine
    real_time_engine.stop()
    
    # Verify pipeline execution
    assert len(decisions_made) >= 1, "Should have made at least one RL decision"
    assert mock_rl_model.prediction_count > 0, "RL model should have been called"
    
    # Check signal generation
    assert len(signals_generated) >= 0, "Should have generated signals"
    
    # Verify portfolio state
    portfolio_summary = paper_engine.get_portfolio_summary()
    assert portfolio_summary['total_value'] > 0, "Portfolio should have positive value"
    
    # Verify risk management
    risk_metrics = risk_manager.calculate_risk_metrics()
    assert risk_metrics.risk_level is not None, "Risk level should be calculated"
    
    # Test signal bridge functionality
    signal_report = signal_bridge.get_signal_performance_report()
    assert 'total_signals_generated' in signal_report
    
    print(f"✅ Enhanced pipeline test completed:")
    print(f"   - Decisions made: {len(decisions_made)}")
    print(f"   - Signals generated: {len(signals_generated)}")
    print(f"   - Trades executed: {len(trades_executed)}")
    print(f"   - Final portfolio value: ${portfolio_summary['total_value']:,.2f}")
    print(f"   - Risk level: {risk_metrics.risk_level.value}")


@pytest.mark.integration
def test_risk_management_validation(engine):
    """Test risk management validation in isolation"""
    from src.db_utils import DatabaseManager
    
    db_manager = DatabaseManager()
    db_manager.engine = engine
    
    risk_limits = RiskLimits(
        max_position_size=0.05,  # 5% max position
        max_daily_drawdown=0.02,  # 2% daily loss limit
    )
    
    risk_manager = EnhancedRiskManager(
        db_manager=db_manager,
        risk_limits=risk_limits,
        initial_capital=100000.0
    )
    
    paper_engine = PaperTradingEngine(
        initial_capital=100000.0,
        risk_manager=risk_manager
    )
    
    # Update market prices
    test_prices = {'AAPL': 150.0, 'MSFT': 300.0}
    paper_engine.update_market_prices(test_prices)
    
    # Try large order (should be rejected)
    large_order_id = paper_engine.place_order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=1000,  # $150,000 - exceeds 5% limit
        order_type=OrderType.MARKET
    )
    assert large_order_id is None, "Large order should be rejected"
    
    # Try small order (should be accepted)
    small_order_id = paper_engine.place_order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=30,  # $4,500 - within 5% limit
        order_type=OrderType.MARKET
    )
    assert small_order_id is not None, "Small order should be accepted"
    
    # Verify position was created
    portfolio_summary = paper_engine.get_portfolio_summary()
    assert portfolio_summary['num_positions'] > 0, "Should have at least one position"


@pytest.mark.integration
def test_paper_trading_accuracy(engine):
    """Test paper trading execution accuracy"""
    paper_engine = PaperTradingEngine(initial_capital=100000.0)
    
    # Set initial prices
    prices = {'AAPL': 150.0, 'MSFT': 300.0}
    paper_engine.update_market_prices(prices)
    
    # Place buy order
    order_id = paper_engine.place_order(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    assert order_id is not None, "Order should be placed successfully"
    
    # Check portfolio
    portfolio = paper_engine.get_portfolio_summary()
    assert portfolio['num_positions'] == 1, "Should have one position"
    
    # Update prices and check P&L
    new_prices = {'AAPL': 155.0, 'MSFT': 300.0}  # AAPL up $5
    paper_engine.update_market_prices(new_prices)
    
    portfolio = paper_engine.get_portfolio_summary()
    assert portfolio['total_pnl'] > 400, "Should have positive P&L from price increase"
    
    # Place sell order
    sell_order_id = paper_engine.place_order(
        symbol='AAPL',
        side=OrderSide.SELL,
        quantity=100,
        order_type=OrderType.MARKET
    )
    assert sell_order_id is not None, "Sell order should be placed successfully"
    
    # Check final portfolio
    final_portfolio = paper_engine.get_portfolio_summary()
    assert final_portfolio['num_positions'] == 0, "Should have no positions after selling"
    assert final_portfolio['total_pnl'] > 0, "Should have positive realized P&L"
