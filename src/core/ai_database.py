"""
PostgreSQL Data Access Layer for AI Trading Modules

This module provides database integration for the Week 2 AI trading implementations,
extending the existing PostgreSQL schema on Railway with AI-specific tables.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from decimal import Decimal
import asyncio
from contextlib import asynccontextmanager

import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool

from .data_structures import MarketData, TradingSignal, TradeResult, RiskMetrics

logger = logging.getLogger(__name__)

class AITradingDatabase:
    """
    PostgreSQL database manager for AI trading modules.
    
    Integrates with existing Railway PostgreSQL database, extending schema
    with AI-specific tables while maintaining referential integrity.
    """
    
    def __init__(self, database_url: str = None):
        """Initialize database connection."""
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable must be set")
        
        self.connection_pool = None
        self._setup_connection_pool()
        
        logger.info("[SUCCESS] AI Trading Database initialized")
    
    def _setup_connection_pool(self):
        """Set up connection pool for database operations."""
        try:
            self.connection_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                dsn=self.database_url
            )
            logger.info("[SUCCESS] Database connection pool created")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create connection pool: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async database connection."""
        conn = None
        try:
            conn = await asyncpg.connect(self.database_url)
            yield conn
        finally:
            if conn:
                await conn.close()
    
    def get_connection(self):
        """Get synchronous database connection from pool."""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool."""
        self.connection_pool.putconn(conn)
    
    def init_ai_schema(self):
        """Initialize AI-specific database tables."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Create AI models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_models (
                        id SERIAL PRIMARY KEY,
                        model_type VARCHAR(50) NOT NULL,
                        model_name VARCHAR(100) NOT NULL,
                        version VARCHAR(20) NOT NULL,
                        config JSONB NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        training_status VARCHAR(20) DEFAULT 'UNTRAINED',
                        performance_metrics JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(model_name, version)
                    )
                """)
                
                # Create AI trading signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_trading_signals (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER NOT NULL REFERENCES ai_models(id),
                        symbol VARCHAR(20) NOT NULL,
                        signal_type VARCHAR(10) NOT NULL,
                        confidence DECIMAL(5,3) NOT NULL,
                        price_target DECIMAL(15,4) NOT NULL,
                        stop_loss DECIMAL(15,4),
                        take_profit DECIMAL(15,4),
                        metadata JSONB,
                        
                        -- Integration with existing tables (if they exist)
                        pattern_id INTEGER,
                        trade_plan_id INTEGER,
                        
                        -- Status tracking
                        status VARCHAR(20) DEFAULT 'ACTIVE',
                        executed_at TIMESTAMPTZ,
                        execution_price DECIMAL(15,4),
                        pnl DECIMAL(12,2),
                        
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create RL training episodes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rl_training_episodes (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER NOT NULL REFERENCES ai_models(id),
                        episode_number INTEGER NOT NULL,
                        total_reward DECIMAL(12,4),
                        episode_length INTEGER,
                        average_loss DECIMAL(12,6),
                        portfolio_value DECIMAL(15,2),
                        sharpe_ratio DECIMAL(8,4),
                        actions_taken INTEGER,
                        exploration_rate DECIMAL(6,4),
                        training_data JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(model_id, episode_number)
                    )
                """)
                
                # Create genetic algorithm generations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS genetic_generations (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER NOT NULL REFERENCES ai_models(id),
                        generation_number INTEGER NOT NULL,
                        best_fitness DECIMAL(12,6),
                        average_fitness DECIMAL(12,6),
                        worst_fitness DECIMAL(12,6),
                        population_diversity DECIMAL(6,4),
                        mutation_rate DECIMAL(6,4),
                        crossover_rate DECIMAL(6,4),
                        best_individual JSONB,
                        population_stats JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(model_id, generation_number)
                    )
                """)
                
                # Create spectrum analysis results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS spectrum_analysis (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER NOT NULL REFERENCES ai_models(id),
                        symbol VARCHAR(20) NOT NULL,
                        analysis_type VARCHAR(30) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        dominant_frequencies JSONB,
                        pattern_confidence DECIMAL(5,3),
                        anomaly_score DECIMAL(8,4),
                        reconstruction_error DECIMAL(10,6),
                        analysis_data JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create AI model performance tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_model_performance (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER NOT NULL REFERENCES ai_models(id),
                        evaluation_date DATE NOT NULL,
                        
                        -- Trading Performance
                        total_signals INTEGER DEFAULT 0,
                        successful_signals INTEGER DEFAULT 0,
                        win_rate DECIMAL(5,2),
                        average_return DECIMAL(8,4),
                        sharpe_ratio DECIMAL(8,4),
                        max_drawdown DECIMAL(8,4),
                        
                        -- Model-specific metrics
                        model_accuracy DECIMAL(5,3),
                        prediction_confidence DECIMAL(5,3),
                        execution_rate DECIMAL(5,2),
                        
                        -- Risk metrics
                        value_at_risk DECIMAL(10,2),
                        expected_shortfall DECIMAL(10,2),
                        volatility DECIMAL(8,4),
                        
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(model_id, evaluation_date)
                    )
                """)
                
                # Create performance indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_signals_symbol_timestamp 
                    ON ai_trading_signals(symbol, created_at DESC);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_signals_model_status 
                    ON ai_trading_signals(model_id, status);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rl_episodes_model 
                    ON rl_training_episodes(model_id, episode_number DESC);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_genetic_generations_model 
                    ON genetic_generations(model_id, generation_number DESC);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_spectrum_analysis_symbol 
                    ON spectrum_analysis(symbol, analysis_type, created_at DESC);
                """)
                
                conn.commit()
                logger.info("[SUCCESS] AI database schema initialized")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to initialize AI schema: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    # ===========================
    # AI MODEL MANAGEMENT
    # ===========================
    
    def register_ai_model(self, model_type: str, model_name: str, version: str, 
                         config: Dict[str, Any]) -> int:
        """Register a new AI model."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ai_models (model_type, model_name, version, config)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (model_name, version) 
                    DO UPDATE SET 
                        config = EXCLUDED.config,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (model_type, model_name, version, Json(config)))
                
                model_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"[SUCCESS] Registered AI model: {model_name} v{version} (ID: {model_id})")
                return model_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to register AI model: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def update_model_training_status(self, model_id: int, status: str, 
                                   performance_metrics: Dict[str, Any] = None):
        """Update model training status and performance metrics."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE ai_models 
                    SET training_status = %s,
                        performance_metrics = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, Json(performance_metrics) if performance_metrics else None, model_id))
                
                conn.commit()
                logger.info(f"[SUCCESS] Updated model {model_id} status to {status}")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to update model status: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_active_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """Get list of active AI models."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if model_type:
                    cursor.execute("""
                        SELECT * FROM ai_models 
                        WHERE is_active = TRUE AND model_type = %s
                        ORDER BY created_at DESC
                    """, (model_type,))
                else:
                    cursor.execute("""
                        SELECT * FROM ai_models 
                        WHERE is_active = TRUE
                        ORDER BY created_at DESC
                    """)
                
                models = cursor.fetchall()
                return [dict(model) for model in models]
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get active models: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    # ===========================
    # TRADING SIGNALS
    # ===========================
    
    def store_trading_signal(self, signal: TradingSignal, model_id: int, 
                           pattern_id: int = None) -> int:
        """Store AI trading signal in database."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ai_trading_signals 
                    (model_id, symbol, signal_type, confidence, price_target,
                     stop_loss, take_profit, metadata, pattern_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    model_id, signal.symbol, signal.signal_type, 
                    signal.confidence, signal.price_target,
                    signal.stop_loss, signal.take_profit,
                    Json(signal.metadata) if signal.metadata else None,
                    pattern_id
                ))
                
                signal_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"[SUCCESS] Stored AI signal: {signal.signal_type} {signal.symbol} (ID: {signal_id})")
                return signal_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to store trading signal: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_active_signals(self, symbol: str = None, model_id: int = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get active trading signals."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT ats.*, am.model_name, am.model_type
                    FROM ai_trading_signals ats
                    JOIN ai_models am ON ats.model_id = am.id
                    WHERE ats.status = 'ACTIVE'
                """
                params = []
                
                if symbol:
                    query += " AND ats.symbol = %s"
                    params.append(symbol)
                
                if model_id:
                    query += " AND ats.model_id = %s"
                    params.append(model_id)
                
                query += " ORDER BY ats.created_at DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(query, params)
                signals = cursor.fetchall()
                
                return [dict(signal) for signal in signals]
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get active signals: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def update_signal_execution(self, signal_id: int, execution_price: Decimal, 
                              pnl: Decimal = None):
        """Update signal with execution details."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE ai_trading_signals 
                    SET status = 'EXECUTED',
                        executed_at = CURRENT_TIMESTAMP,
                        execution_price = %s,
                        pnl = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (execution_price, pnl, signal_id))
                
                conn.commit()
                logger.info(f"[SUCCESS] Updated signal {signal_id} execution")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to update signal execution: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    # ===========================
    # REINFORCEMENT LEARNING
    # ===========================
    
    def store_rl_episode(self, model_id: int, episode_data: Dict[str, Any]) -> int:
        """Store RL training episode data."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO rl_training_episodes 
                    (model_id, episode_number, total_reward, episode_length,
                     average_loss, portfolio_value, sharpe_ratio, actions_taken,
                     exploration_rate, training_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, episode_number)
                    DO UPDATE SET
                        total_reward = EXCLUDED.total_reward,
                        episode_length = EXCLUDED.episode_length,
                        average_loss = EXCLUDED.average_loss,
                        portfolio_value = EXCLUDED.portfolio_value,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        actions_taken = EXCLUDED.actions_taken,
                        exploration_rate = EXCLUDED.exploration_rate,
                        training_data = EXCLUDED.training_data
                    RETURNING id
                """, (
                    model_id,
                    episode_data['episode_number'],
                    episode_data.get('total_reward'),
                    episode_data.get('episode_length'),
                    episode_data.get('average_loss'),
                    episode_data.get('portfolio_value'),
                    episode_data.get('sharpe_ratio'),
                    episode_data.get('actions_taken'),
                    episode_data.get('exploration_rate'),
                    Json(episode_data.get('training_data', {}))
                ))
                
                episode_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"[SUCCESS] Stored RL episode {episode_data['episode_number']} for model {model_id}")
                return episode_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to store RL episode: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_rl_training_progress(self, model_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get RL training progress."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM rl_training_episodes 
                    WHERE model_id = %s
                    ORDER BY episode_number DESC
                    LIMIT %s
                """, (model_id, limit))
                
                episodes = cursor.fetchall()
                return [dict(episode) for episode in episodes]
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get RL training progress: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    # ===========================
    # GENETIC ALGORITHMS
    # ===========================
    
    def store_genetic_generation(self, model_id: int, generation_data: Dict[str, Any]) -> int:
        """Store genetic algorithm generation data."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO genetic_generations 
                    (model_id, generation_number, best_fitness, average_fitness,
                     worst_fitness, population_diversity, mutation_rate, crossover_rate,
                     best_individual, population_stats)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, generation_number)
                    DO UPDATE SET
                        best_fitness = EXCLUDED.best_fitness,
                        average_fitness = EXCLUDED.average_fitness,
                        worst_fitness = EXCLUDED.worst_fitness,
                        population_diversity = EXCLUDED.population_diversity,
                        mutation_rate = EXCLUDED.mutation_rate,
                        crossover_rate = EXCLUDED.crossover_rate,
                        best_individual = EXCLUDED.best_individual,
                        population_stats = EXCLUDED.population_stats
                    RETURNING id
                """, (
                    model_id,
                    generation_data['generation_number'],
                    generation_data.get('best_fitness'),
                    generation_data.get('average_fitness'),
                    generation_data.get('worst_fitness'),
                    generation_data.get('population_diversity'),
                    generation_data.get('mutation_rate'),
                    generation_data.get('crossover_rate'),
                    Json(generation_data.get('best_individual', {})),
                    Json(generation_data.get('population_stats', {}))
                ))
                
                generation_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"[SUCCESS] Stored generation {generation_data['generation_number']} for model {model_id}")
                return generation_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to store genetic generation: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    # ===========================
    # SPECTRUM ANALYSIS
    # ===========================
    
    def store_spectrum_analysis(self, model_id: int, analysis_data: Dict[str, Any]) -> int:
        """Store spectrum analysis results."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO spectrum_analysis 
                    (model_id, symbol, analysis_type, timeframe, dominant_frequencies,
                     pattern_confidence, anomaly_score, reconstruction_error, analysis_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    model_id,
                    analysis_data['symbol'],
                    analysis_data['analysis_type'],
                    analysis_data.get('timeframe', '1d'),
                    Json(analysis_data.get('dominant_frequencies', {})),
                    analysis_data.get('pattern_confidence'),
                    analysis_data.get('anomaly_score'),
                    analysis_data.get('reconstruction_error'),
                    Json(analysis_data.get('analysis_data', {}))
                ))
                
                analysis_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"[SUCCESS] Stored spectrum analysis for {analysis_data['symbol']}")
                return analysis_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to store spectrum analysis: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    # ===========================
    # PERFORMANCE ANALYTICS
    # ===========================
    
    def calculate_model_performance(self, model_id: int, evaluation_date: datetime.date = None):
        """Calculate and store model performance metrics."""
        if evaluation_date is None:
            evaluation_date = datetime.now(timezone.utc).date()
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Calculate signal performance
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as successful_signals,
                        AVG(CASE WHEN pnl IS NOT NULL THEN pnl END) as average_return,
                        STDDEV(CASE WHEN pnl IS NOT NULL THEN pnl END) as volatility,
                        AVG(confidence) as avg_confidence
                    FROM ai_trading_signals 
                    WHERE model_id = %s 
                    AND DATE(created_at) = %s
                    AND status IN ('EXECUTED', 'EXPIRED')
                """, (model_id, evaluation_date))
                
                performance_data = cursor.fetchone()
                
                if performance_data and performance_data[0] > 0:  # Has signals
                    total_signals, successful_signals, avg_return, volatility, avg_confidence = performance_data
                    
                    win_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0
                    sharpe_ratio = (avg_return / volatility) if volatility and volatility > 0 else 0
                    
                    # Store performance metrics
                    cursor.execute("""
                        INSERT INTO ai_model_performance 
                        (model_id, evaluation_date, total_signals, successful_signals,
                         win_rate, average_return, sharpe_ratio, model_accuracy, prediction_confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_id, evaluation_date)
                        DO UPDATE SET
                            total_signals = EXCLUDED.total_signals,
                            successful_signals = EXCLUDED.successful_signals,
                            win_rate = EXCLUDED.win_rate,
                            average_return = EXCLUDED.average_return,
                            sharpe_ratio = EXCLUDED.sharpe_ratio,
                            model_accuracy = EXCLUDED.model_accuracy,
                            prediction_confidence = EXCLUDED.prediction_confidence
                    """, (
                        model_id, evaluation_date, total_signals, successful_signals,
                        win_rate, avg_return, sharpe_ratio, win_rate, avg_confidence
                    ))
                    
                    conn.commit()
                    logger.info(f"[SUCCESS] Calculated performance for model {model_id} on {evaluation_date}")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to calculate model performance: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_model_performance_history(self, model_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get model performance history."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM ai_model_performance 
                    WHERE model_id = %s
                    AND evaluation_date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY evaluation_date DESC
                """, (model_id, days))
                
                performance = cursor.fetchall()
                return [dict(perf) for perf in performance]
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to get model performance history: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def close(self):
        """Close database connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("[SUCCESS] Database connection pool closed")

# Singleton instance for global access
_ai_db_instance = None

def get_ai_database(database_url: str = None) -> AITradingDatabase:
    """Get singleton instance of AI trading database."""
    global _ai_db_instance
    
    if _ai_db_instance is None:
        _ai_db_instance = AITradingDatabase(database_url)
        _ai_db_instance.init_ai_schema()
    
    return _ai_db_instance
