-- AI Trading Database Schema
-- PostgreSQL schema for AI trading modules integration
-- Compatible with Railway PostgreSQL deployment

-- Create AI trading schema for logical separation
CREATE SCHEMA IF NOT EXISTS ai_trading;

-- Set search path to include both schemas
-- SET search_path TO ai_trading, public;

-- ==============================================================================
-- CORE AI MODEL MANAGEMENT
-- ==============================================================================

-- AI Models registry - stores model configurations and metadata
CREATE TABLE IF NOT EXISTS ai_trading.ai_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'reinforcement_learning', 'genetic_optimization', 'sparse_spectrum'
    model_subtype VARCHAR(50), -- 'ppo', 'multi_agent', 'parameter_optimizer', 'fourier', etc.
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    description TEXT,
    configuration JSONB NOT NULL, -- Model configuration parameters
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    performance_metrics JSONB, -- Latest performance summary
    
    UNIQUE(model_name, version)
);

-- Training sessions - track model training history
CREATE TABLE IF NOT EXISTS ai_trading.training_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ai_trading.ai_models(model_id),
    session_name VARCHAR(100),
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed', 'stopped'
    training_data_period DATERANGE, -- Training data time range
    validation_data_period DATERANGE, -- Validation data time range
    hyperparameters JSONB,
    training_metrics JSONB, -- Loss curves, rewards, convergence metrics
    final_performance JSONB, -- Final validation metrics
    model_artifacts_path TEXT, -- Path to saved model files
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==============================================================================
-- AI TRADING SIGNALS
-- ==============================================================================

-- AI generated trading signals
CREATE TABLE IF NOT EXISTS ai_trading.ai_signals (
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ai_trading.ai_models(model_id),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    price_target DECIMAL(15,6) NOT NULL,
    stop_loss DECIMAL(15,6),
    take_profit DECIMAL(15,6),
    signal_timestamp TIMESTAMPTZ NOT NULL,
    market_data_timestamp TIMESTAMPTZ, -- Timestamp of market data used
    metadata JSONB, -- Model-specific signal metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_ai_signals_symbol_time (symbol, signal_timestamp DESC),
    INDEX idx_ai_signals_model_time (model_id, signal_timestamp DESC),
    INDEX idx_ai_signals_type_confidence (signal_type, confidence DESC)
);

-- Signal performance tracking
CREATE TABLE IF NOT EXISTS ai_trading.signal_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES ai_trading.ai_signals(signal_id),
    evaluation_timestamp TIMESTAMPTZ NOT NULL,
    actual_price DECIMAL(15,6),
    price_change_pct DECIMAL(8,4), -- Percentage change from signal time
    target_hit BOOLEAN, -- Whether price target was reached
    stop_loss_hit BOOLEAN, -- Whether stop loss was triggered
    holding_period_hours INTEGER, -- How long the signal was relevant
    pnl_if_executed DECIMAL(15,6), -- Theoretical P&L if signal was executed
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==============================================================================
-- FEATURE ENGINEERING AND ANALYSIS
-- ==============================================================================

-- Extracted features for ML models
CREATE TABLE IF NOT EXISTS ai_trading.feature_data (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_set_name VARCHAR(50) NOT NULL, -- 'technical_indicators', 'fourier_coefficients', etc.
    features JSONB NOT NULL, -- Feature vector as JSON
    raw_data_hash VARCHAR(64), -- Hash of input market data for versioning
    extraction_method VARCHAR(50), -- Method used for feature extraction
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(symbol, timestamp, feature_set_name),
    INDEX idx_feature_data_symbol_time (symbol, timestamp DESC),
    INDEX idx_feature_data_set_time (feature_set_name, timestamp DESC)
);

-- ==============================================================================
-- GENETIC OPTIMIZATION RESULTS
-- ==============================================================================

-- Genetic algorithm optimization runs
CREATE TABLE IF NOT EXISTS ai_trading.optimization_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ai_trading.ai_models(model_id),
    optimization_type VARCHAR(50) NOT NULL, -- 'parameter_optimization', 'portfolio_optimization'
    objective_function VARCHAR(50), -- 'sharpe_ratio', 'total_return', 'max_drawdown'
    population_size INTEGER,
    generations INTEGER,
    parameter_ranges JSONB, -- Search space definition
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running',
    best_individual JSONB, -- Best solution found
    best_fitness DECIMAL(15,6),
    convergence_data JSONB, -- Fitness evolution over generations
    final_population JSONB, -- Final population state
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual solutions from genetic optimization
CREATE TABLE IF NOT EXISTS ai_trading.optimization_individuals (
    individual_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES ai_trading.optimization_runs(run_id),
    generation INTEGER NOT NULL,
    individual_rank INTEGER, -- Rank within generation
    genes JSONB NOT NULL, -- Parameter values
    fitness_score DECIMAL(15,6),
    objectives JSONB, -- Multi-objective scores if applicable
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_optimization_individuals_run_gen (run_id, generation),
    INDEX idx_optimization_individuals_fitness (fitness_score DESC)
);

-- ==============================================================================
-- ANOMALY DETECTION
-- ==============================================================================

-- Anomalies detected by AI models
CREATE TABLE IF NOT EXISTS ai_trading.anomaly_detections (
    anomaly_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ai_trading.ai_models(model_id),
    symbol VARCHAR(20) NOT NULL,
    detection_timestamp TIMESTAMPTZ NOT NULL,
    anomaly_type VARCHAR(50), -- 'price_jump', 'volume_spike', 'pattern_break', etc.
    severity_score DECIMAL(8,4) NOT NULL, -- Anomaly severity (standard deviations from normal)
    confidence DECIMAL(5,4), -- Model confidence in anomaly detection
    market_data_context JSONB, -- Relevant market data at time of detection
    anomaly_details JSONB, -- Model-specific anomaly information
    resolved BOOLEAN DEFAULT FALSE,
    resolution_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_anomaly_detections_symbol_time (symbol, detection_timestamp DESC),
    INDEX idx_anomaly_detections_severity (severity_score DESC),
    INDEX idx_anomaly_detections_unresolved (resolved, detection_timestamp DESC) WHERE NOT resolved
);

-- ==============================================================================
-- MODEL PERFORMANCE METRICS
-- ==============================================================================

-- Model performance tracking over time
CREATE TABLE IF NOT EXISTS ai_trading.model_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ai_trading.ai_models(model_id),
    evaluation_period DATERANGE NOT NULL,
    total_signals INTEGER,
    successful_signals INTEGER,
    signal_accuracy DECIMAL(5,4), -- Percentage of correct signals
    avg_confidence DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    total_return DECIMAL(8,4),
    volatility DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    avg_holding_period_hours DECIMAL(8,2),
    risk_adjusted_return DECIMAL(8,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_model_performance_model_period (model_id, evaluation_period),
    INDEX idx_model_performance_sharpe (sharpe_ratio DESC NULLS LAST)
);

-- ==============================================================================
-- REINFORCEMENT LEARNING SPECIFIC TABLES
-- ==============================================================================

-- RL episode data for training tracking
CREATE TABLE IF NOT EXISTS ai_trading.rl_episodes (
    episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES ai_trading.training_sessions(session_id),
    episode_number INTEGER NOT NULL,
    total_reward DECIMAL(15,6),
    episode_length INTEGER, -- Number of steps
    final_portfolio_value DECIMAL(15,6),
    max_drawdown DECIMAL(8,4),
    actions_taken JSONB, -- Sequence of actions
    rewards JSONB, -- Reward sequence
    episode_end_reason VARCHAR(50), -- 'completed', 'early_stop', 'max_steps'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(session_id, episode_number),
    INDEX idx_rl_episodes_session_number (session_id, episode_number),
    INDEX idx_rl_episodes_reward (total_reward DESC)
);

-- ==============================================================================
-- SPECTRUM ANALYSIS RESULTS
-- ==============================================================================

-- Fourier and wavelet analysis results
CREATE TABLE IF NOT EXISTS ai_trading.spectrum_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ai_trading.ai_models(model_id),
    symbol VARCHAR(20) NOT NULL,
    analysis_timestamp TIMESTAMPTZ NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- 'fourier', 'wavelet', 'compressed_sensing'
    frequency_components JSONB, -- Dominant frequencies found
    spectral_features JSONB, -- Extracted spectral features
    pattern_detected VARCHAR(100), -- Named pattern if detected
    pattern_confidence DECIMAL(5,4),
    reconstruction_error DECIMAL(15,10), -- For compressed sensing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_spectrum_analysis_symbol_time (symbol, analysis_timestamp DESC),
    INDEX idx_spectrum_analysis_type_time (analysis_type, analysis_timestamp DESC)
);

-- ==============================================================================
-- DATA INTEGRATION VIEWS
-- ==============================================================================

-- View combining AI signals with market data (assumes market data in public schema)
CREATE OR REPLACE VIEW ai_trading.enriched_ai_signals AS
SELECT 
    s.signal_id,
    s.model_id,
    m.model_name,
    m.model_type,
    s.symbol,
    s.signal_type,
    s.confidence,
    s.price_target,
    s.stop_loss,
    s.take_profit,
    s.signal_timestamp,
    s.metadata,
    -- Performance data if available
    p.actual_price,
    p.price_change_pct,
    p.pnl_if_executed,
    p.target_hit,
    p.stop_loss_hit
FROM ai_trading.ai_signals s
JOIN ai_trading.ai_models m ON s.model_id = m.model_id
LEFT JOIN ai_trading.signal_performance p ON s.signal_id = p.signal_id;

-- View for latest model performance
CREATE OR REPLACE VIEW ai_trading.latest_model_performance AS
SELECT DISTINCT ON (mp.model_id)
    mp.model_id,
    m.model_name,
    m.model_type,
    mp.evaluation_period,
    mp.signal_accuracy,
    mp.sharpe_ratio,
    mp.total_return,
    mp.max_drawdown,
    mp.win_rate,
    mp.created_at
FROM ai_trading.model_performance mp
JOIN ai_trading.ai_models m ON mp.model_id = m.model_id
WHERE m.is_active = TRUE
ORDER BY mp.model_id, mp.created_at DESC;

-- ==============================================================================
-- INDEXES FOR PERFORMANCE
-- ==============================================================================

-- Additional composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_ai_signals_recent_by_model 
ON ai_trading.ai_signals (model_id, signal_timestamp DESC) 
WHERE signal_timestamp > NOW() - INTERVAL '30 days';

CREATE INDEX IF NOT EXISTS idx_feature_data_recent 
ON ai_trading.feature_data (symbol, feature_set_name, timestamp DESC)
WHERE timestamp > NOW() - INTERVAL '90 days';

CREATE INDEX IF NOT EXISTS idx_training_sessions_active 
ON ai_trading.training_sessions (model_id, start_time DESC)
WHERE status IN ('running', 'completed');

-- ==============================================================================
-- FUNCTIONS FOR DATA MANAGEMENT
-- ==============================================================================

-- Function to clean old feature data
CREATE OR REPLACE FUNCTION ai_trading.cleanup_old_feature_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ai_trading.feature_data 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update model performance metrics
CREATE OR REPLACE FUNCTION ai_trading.update_model_performance()
RETURNS VOID AS $$
BEGIN
    -- This would contain logic to calculate and update performance metrics
    -- Implementation depends on specific business logic
    RAISE NOTICE 'Model performance update function called';
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ==============================================================================

COMMENT ON SCHEMA ai_trading IS 'AI Trading modules database schema for ML models, signals, and performance tracking';
COMMENT ON TABLE ai_trading.ai_models IS 'Registry of AI trading models with configurations and metadata';
COMMENT ON TABLE ai_trading.ai_signals IS 'Trading signals generated by AI models';
COMMENT ON TABLE ai_trading.training_sessions IS 'Model training sessions with metrics and results';
COMMENT ON TABLE ai_trading.feature_data IS 'Extracted features for machine learning models';
COMMENT ON TABLE ai_trading.optimization_runs IS 'Genetic algorithm optimization results';
COMMENT ON TABLE ai_trading.anomaly_detections IS 'Market anomalies detected by AI models';
COMMENT ON TABLE ai_trading.model_performance IS 'Performance metrics for AI models over time';
