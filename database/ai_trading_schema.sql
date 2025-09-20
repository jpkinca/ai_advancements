"""
PostgreSQL Database Schema for AI Trading Components

This module defines the database tables and relationships for storing:
- AI model training data and results
- Trading signals from all AI models
- Performance metrics and backtesting results
- Model configurations and metadata
"""

-- ============================================================================
-- AI MODELS AND CONFIGURATIONS
-- ============================================================================

-- Table to store AI model metadata and configurations
CREATE TABLE ai_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'reinforcement_learning', 'genetic_optimization', 'sparse_spectrum'
    model_subtype VARCHAR(50), -- 'ppo_advanced', 'multi_agent', 'parameter_optimizer', etc.
    version VARCHAR(20) NOT NULL,
    configuration JSONB NOT NULL, -- Store full model config as JSON
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    
    UNIQUE(model_name, version)
);

-- Table to store model training sessions and results
CREATE TABLE ai_training_sessions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ai_models(id) ON DELETE CASCADE,
    training_start TIMESTAMP WITH TIME ZONE NOT NULL,
    training_end TIMESTAMP WITH TIME ZONE,
    training_status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed', 'stopped'
    training_data_period_start DATE,
    training_data_period_end DATE,
    training_parameters JSONB,
    training_results JSONB, -- Store metrics like loss, reward, fitness scores
    validation_results JSONB,
    model_artifacts_path TEXT, -- Path to saved model files
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TRADING SIGNALS AND PREDICTIONS
-- ============================================================================

-- Table to store all AI-generated trading signals
CREATE TABLE ai_trading_signals (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ai_models(id) ON DELETE CASCADE,
    training_session_id INTEGER REFERENCES ai_training_sessions(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    price_target DECIMAL(12,4) NOT NULL,
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),
    signal_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    market_data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL, -- When the underlying market data was from
    metadata JSONB, -- Additional signal-specific data
    is_executed BOOLEAN DEFAULT FALSE,
    execution_price DECIMAL(12,4),
    execution_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_ai_signals_symbol_timestamp (symbol, signal_timestamp),
    INDEX idx_ai_signals_model_timestamp (model_id, signal_timestamp),
    INDEX idx_ai_signals_type_confidence (signal_type, confidence)
);

-- ============================================================================
-- REINFORCEMENT LEARNING SPECIFIC TABLES
-- ============================================================================

-- Table to store RL episodes and rewards
CREATE TABLE rl_episodes (
    id SERIAL PRIMARY KEY,
    training_session_id INTEGER REFERENCES ai_training_sessions(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    total_reward DECIMAL(12,6),
    episode_length INTEGER, -- Number of steps
    final_portfolio_value DECIMAL(12,2),
    max_drawdown DECIMAL(8,6),
    sharpe_ratio DECIMAL(8,4),
    episode_start TIMESTAMP WITH TIME ZONE,
    episode_end TIMESTAMP WITH TIME ZONE,
    episode_metadata JSONB, -- Actions taken, states visited, etc.
    
    UNIQUE(training_session_id, episode_number)
);

-- Table to store multi-agent system performance
CREATE TABLE multi_agent_performance (
    id SERIAL PRIMARY KEY,
    training_session_id INTEGER REFERENCES ai_training_sessions(id) ON DELETE CASCADE,
    agent_type VARCHAR(50) NOT NULL, -- 'trend_following', 'mean_reversion', 'volatility'
    agent_id VARCHAR(50) NOT NULL,
    performance_period_start TIMESTAMP WITH TIME ZONE,
    performance_period_end TIMESTAMP WITH TIME ZONE,
    total_return DECIMAL(8,6),
    win_rate DECIMAL(5,4),
    avg_confidence DECIMAL(5,4),
    signals_generated INTEGER,
    market_regime VARCHAR(30), -- 'trending', 'ranging', 'volatile', etc.
    
    INDEX idx_multi_agent_perf_session_agent (training_session_id, agent_type, agent_id)
);

-- ============================================================================
-- GENETIC OPTIMIZATION SPECIFIC TABLES
-- ============================================================================

-- Table to store genetic algorithm generations and populations
CREATE TABLE genetic_generations (
    id SERIAL PRIMARY KEY,
    training_session_id INTEGER REFERENCES ai_training_sessions(id) ON DELETE CASCADE,
    generation_number INTEGER NOT NULL,
    population_size INTEGER NOT NULL,
    best_fitness DECIMAL(12,8),
    avg_fitness DECIMAL(12,8),
    worst_fitness DECIMAL(12,8),
    diversity_metric DECIMAL(8,6), -- Population diversity measure
    generation_metadata JSONB,
    
    UNIQUE(training_session_id, generation_number)
);

-- Table to store individual solutions in genetic algorithm
CREATE TABLE genetic_individuals (
    id SERIAL PRIMARY KEY,
    generation_id INTEGER REFERENCES genetic_generations(id) ON DELETE CASCADE,
    individual_rank INTEGER,
    fitness_score DECIMAL(12,8),
    genes JSONB NOT NULL, -- The parameter values
    performance_metrics JSONB, -- Sharpe ratio, max drawdown, etc.
    is_elite BOOLEAN DEFAULT FALSE,
    
    INDEX idx_genetic_individuals_generation_fitness (generation_id, fitness_score DESC)
);

-- Table to store optimized portfolio allocations
CREATE TABLE portfolio_allocations (
    id SERIAL PRIMARY KEY,
    training_session_id INTEGER REFERENCES ai_training_sessions(id) ON DELETE CASCADE,
    allocation_name VARCHAR(100),
    symbols TEXT[] NOT NULL, -- Array of symbols
    weights DECIMAL(5,4)[] NOT NULL, -- Corresponding weights
    expected_return DECIMAL(8,6),
    portfolio_risk DECIMAL(8,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,6),
    allocation_date DATE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    CHECK (array_length(symbols, 1) = array_length(weights, 1))
);

-- ============================================================================
-- SPARSE SPECTRUM ANALYSIS SPECIFIC TABLES
-- ============================================================================

-- Table to store Fourier analysis results
CREATE TABLE fourier_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    data_period_start DATE NOT NULL,
    data_period_end DATE NOT NULL,
    dominant_frequencies DECIMAL(8,6)[] NOT NULL,
    frequency_powers DECIMAL(12,8)[] NOT NULL,
    spectral_centroid DECIMAL(8,6),
    spectral_rolloff DECIMAL(8,6),
    harmonic_patterns JSONB, -- Detected harmonic patterns
    noise_level DECIMAL(8,6),
    
    INDEX idx_fourier_symbol_timestamp (symbol, analysis_timestamp)
);

-- Table to store wavelet decomposition results
CREATE TABLE wavelet_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    data_period_start DATE NOT NULL,
    data_period_end DATE NOT NULL,
    wavelet_type VARCHAR(20) NOT NULL, -- 'db4', 'haar', etc.
    decomposition_levels INTEGER NOT NULL,
    level_energies DECIMAL(8,6)[] NOT NULL,
    trend_component DECIMAL(12,6)[] NOT NULL,
    detail_components JSONB NOT NULL, -- Multi-level detail coefficients
    denoising_threshold DECIMAL(8,6),
    
    INDEX idx_wavelet_symbol_timestamp (symbol, analysis_timestamp)
);

-- Table to store compressed sensing analysis
CREATE TABLE compressed_sensing_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    data_period_start DATE NOT NULL,
    data_period_end DATE NOT NULL,
    total_features INTEGER NOT NULL,
    active_features INTEGER NOT NULL,
    sparsity_level DECIMAL(5,4) NOT NULL,
    reconstruction_error DECIMAL(12,8),
    dictionary_components JSONB, -- Learned dictionary
    sparse_codes JSONB, -- Sparse representation
    anomalies_detected INTEGER DEFAULT 0,
    
    INDEX idx_cs_symbol_timestamp (symbol, analysis_timestamp)
);

-- Table to store detected anomalies
CREATE TABLE market_anomalies (
    id SERIAL PRIMARY KEY,
    cs_analysis_id INTEGER REFERENCES compressed_sensing_analysis(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    anomaly_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    anomaly_type VARCHAR(50), -- 'price_jump', 'volume_spike', 'pattern_break'
    severity_score DECIMAL(8,4) NOT NULL,
    anomalous_components INTEGER[],
    market_data_context JSONB, -- Surrounding market conditions
    follow_up_action VARCHAR(50), -- 'signal_generated', 'alert_sent', 'ignored'
    
    INDEX idx_anomalies_symbol_severity (symbol, severity_score DESC, anomaly_timestamp)
);

-- ============================================================================
-- PERFORMANCE TRACKING AND BACKTESTING
-- ============================================================================

-- Table to store backtesting results for AI models
CREATE TABLE ai_backtesting_results (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ai_models(id) ON DELETE CASCADE,
    backtest_name VARCHAR(100) NOT NULL,
    backtest_period_start DATE NOT NULL,
    backtest_period_end DATE NOT NULL,
    initial_capital DECIMAL(12,2) NOT NULL,
    final_capital DECIMAL(12,2) NOT NULL,
    total_return DECIMAL(8,6) NOT NULL,
    annualized_return DECIMAL(8,6),
    max_drawdown DECIMAL(8,6),
    sharpe_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_trade_return DECIMAL(8,6),
    avg_holding_period DECIMAL(8,2), -- In hours
    transaction_costs DECIMAL(10,2),
    backtest_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_id, backtest_name)
);

-- Table to store individual trade results from backtesting
CREATE TABLE ai_trade_results (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES ai_backtesting_results(id) ON DELETE CASCADE,
    signal_id INTEGER REFERENCES ai_trading_signals(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'LONG', 'SHORT'
    entry_price DECIMAL(12,4) NOT NULL,
    exit_price DECIMAL(12,4) NOT NULL,
    quantity DECIMAL(12,4) NOT NULL,
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    holding_period_hours DECIMAL(8,2),
    pnl DECIMAL(12,4) NOT NULL,
    pnl_percentage DECIMAL(8,6) NOT NULL,
    commission DECIMAL(8,4) DEFAULT 0,
    slippage DECIMAL(8,4) DEFAULT 0,
    exit_reason VARCHAR(50), -- 'take_profit', 'stop_loss', 'signal_reversal', 'time_limit'
    
    INDEX idx_trade_results_backtest_symbol (backtest_id, symbol),
    INDEX idx_trade_results_pnl (pnl DESC),
    INDEX idx_trade_results_timestamp (entry_timestamp, exit_timestamp)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Composite indexes for common queries
CREATE INDEX idx_signals_model_symbol_time ON ai_trading_signals(model_id, symbol, signal_timestamp DESC);
CREATE INDEX idx_training_sessions_model_status ON ai_training_sessions(model_id, training_status, training_start DESC);
CREATE INDEX idx_backtesting_model_period ON ai_backtesting_results(model_id, backtest_period_start, backtest_period_end);

-- Partial indexes for active records
CREATE INDEX idx_active_models ON ai_models(model_type, model_subtype) WHERE is_active = TRUE;
CREATE INDEX idx_unexecuted_signals ON ai_trading_signals(signal_timestamp DESC, confidence DESC) WHERE is_executed = FALSE;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for latest model performance summary
CREATE VIEW v_latest_model_performance AS
SELECT 
    m.id as model_id,
    m.model_name,
    m.model_type,
    m.model_subtype,
    ts.id as latest_session_id,
    ts.training_end,
    ts.training_results,
    br.total_return as latest_backtest_return,
    br.sharpe_ratio as latest_sharpe_ratio,
    br.max_drawdown as latest_max_drawdown,
    COUNT(ats.id) as total_signals_generated
FROM ai_models m
LEFT JOIN ai_training_sessions ts ON m.id = ts.model_id 
    AND ts.id = (SELECT MAX(id) FROM ai_training_sessions WHERE model_id = m.id AND training_status = 'completed')
LEFT JOIN ai_backtesting_results br ON m.id = br.model_id
    AND br.id = (SELECT MAX(id) FROM ai_backtesting_results WHERE model_id = m.id)
LEFT JOIN ai_trading_signals ats ON m.id = ats.model_id
WHERE m.is_active = TRUE
GROUP BY m.id, m.model_name, m.model_type, m.model_subtype, ts.id, ts.training_end, ts.training_results, br.total_return, br.sharpe_ratio, br.max_drawdown;

-- View for signal performance analysis
CREATE VIEW v_signal_performance AS
SELECT 
    ats.model_id,
    ats.symbol,
    ats.signal_type,
    COUNT(*) as total_signals,
    AVG(ats.confidence) as avg_confidence,
    COUNT(CASE WHEN ats.is_executed THEN 1 END) as executed_signals,
    AVG(CASE WHEN atr.pnl IS NOT NULL THEN atr.pnl END) as avg_pnl,
    COUNT(CASE WHEN atr.pnl > 0 THEN 1 END)::DECIMAL / NULLIF(COUNT(CASE WHEN atr.pnl IS NOT NULL THEN 1 END), 0) as win_rate
FROM ai_trading_signals ats
LEFT JOIN ai_trade_results atr ON ats.id = atr.signal_id
GROUP BY ats.model_id, ats.symbol, ats.signal_type;

-- ============================================================================
-- FUNCTIONS FOR DATA MANAGEMENT
-- ============================================================================

-- Function to update model updated_at timestamp
CREATE OR REPLACE FUNCTION update_ai_model_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update timestamp
CREATE TRIGGER tr_ai_models_update_timestamp
    BEFORE UPDATE ON ai_models
    FOR EACH ROW
    EXECUTE FUNCTION update_ai_model_timestamp();

-- Function to clean old data (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_ai_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Delete old trading signals (keep recent ones)
    DELETE FROM ai_trading_signals 
    WHERE signal_timestamp < CURRENT_DATE - INTERVAL '1 day' * retention_days
    AND is_executed = FALSE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up orphaned training sessions
    DELETE FROM ai_training_sessions 
    WHERE training_start < CURRENT_DATE - INTERVAL '1 day' * retention_days
    AND training_status = 'failed';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
