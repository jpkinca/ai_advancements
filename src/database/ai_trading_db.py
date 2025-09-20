"""
AI Trading Database Access Layer

This module provides database integration for AI trading components with PostgreSQL.
Designed to work with Railway PostgreSQL and the ai_trading schema.
"""

import os
import asyncio
import asyncpg
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import asdict
import uuid

from src.core.data_structures import MarketData, TradingSignal

logger = logging.getLogger(__name__)

class AITradingDatabase:
    """Database access layer for AI trading components."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database connection."""
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("Database connection string required (DATABASE_URL environment variable)")
        
        self.pool: Optional[asyncpg.Pool] = None
        logger.info("[SUCCESS] AI Trading Database initialized")
    
    async def connect(self) -> None:
        """Create database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60,
                server_settings={
                    'search_path': 'ai_trading,public'
                }
            )
            logger.info("[SUCCESS] Database connection pool created")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create database pool: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("[SUCCESS] Database connection pool closed")
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries."""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[ERROR] Query execution failed: {str(e)}")
            raise
    
    async def execute_command(self, command: str, *args) -> int:
        """Execute command and return number of affected rows."""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(command, *args)
                # Extract number from result string like "INSERT 0 1"
                return int(result.split()[-1]) if result else 0
        except Exception as e:
            logger.error(f"[ERROR] Command execution failed: {str(e)}")
            raise

class AIModelManager:
    """Manage AI model registry and metadata."""
    
    def __init__(self, db: AITradingDatabase):
        self.db = db
    
    async def register_model(self, model_name: str, model_type: str, 
                           model_subtype: str, configuration: Dict[str, Any],
                           description: Optional[str] = None,
                           version: str = "1.0.0") -> str:
        """Register a new AI model and return model_id."""
        model_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO ai_trading.ai_models 
        (model_id, model_name, model_type, model_subtype, version, description, configuration)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (model_name, version) 
        DO UPDATE SET 
            model_type = EXCLUDED.model_type,
            model_subtype = EXCLUDED.model_subtype,
            description = EXCLUDED.description,
            configuration = EXCLUDED.configuration,
            updated_at = NOW()
        RETURNING model_id
        """
        
        try:
            result = await self.db.execute_query(
                query, model_id, model_name, model_type, model_subtype, 
                version, description, json.dumps(configuration)
            )
            
            if result:
                actual_model_id = result[0]['model_id']
                logger.info(f"[SUCCESS] Model registered: {model_name} ({actual_model_id})")
                return str(actual_model_id)
            else:
                logger.error(f"[ERROR] Failed to register model: {model_name}")
                raise ValueError(f"Failed to register model: {model_name}")
                
        except Exception as e:
            logger.error(f"[ERROR] Model registration failed: {str(e)}")
            raise
    
    async def get_model_by_name(self, model_name: str, version: str = "1.0.0") -> Optional[Dict[str, Any]]:
        """Get model information by name and version."""
        query = """
        SELECT model_id, model_name, model_type, model_subtype, version, 
               description, configuration, performance_metrics, created_at, is_active
        FROM ai_trading.ai_models 
        WHERE model_name = $1 AND version = $2
        """
        
        try:
            result = await self.db.execute_query(query, model_name, version)
            if result:
                model_data = result[0]
                # Parse JSON fields
                if model_data['configuration']:
                    model_data['configuration'] = json.loads(model_data['configuration'])
                if model_data['performance_metrics']:
                    model_data['performance_metrics'] = json.loads(model_data['performance_metrics'])
                return model_data
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get model {model_name}: {str(e)}")
            raise
    
    async def update_model_performance(self, model_id: str, performance_metrics: Dict[str, Any]) -> None:
        """Update model performance metrics."""
        query = """
        UPDATE ai_trading.ai_models 
        SET performance_metrics = $2, updated_at = NOW()
        WHERE model_id = $1
        """
        
        try:
            await self.db.execute_command(query, model_id, json.dumps(performance_metrics))
            logger.info(f"[SUCCESS] Performance metrics updated for model: {model_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update performance for {model_id}: {str(e)}")
            raise

class TrainingSessionManager:
    """Manage AI model training sessions."""
    
    def __init__(self, db: AITradingDatabase):
        self.db = db
    
    async def start_training_session(self, model_id: str, session_name: str,
                                   training_period: tuple, validation_period: Optional[tuple] = None,
                                   hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """Start a new training session."""
        session_id = str(uuid.uuid4())
        
        # Format date ranges for PostgreSQL
        training_range = f"[{training_period[0]},{training_period[1]})"
        validation_range = f"[{validation_period[0]},{validation_period[1]})" if validation_period else None
        
        query = """
        INSERT INTO ai_trading.training_sessions 
        (session_id, model_id, session_name, training_data_period, validation_data_period, hyperparameters)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        try:
            await self.db.execute_command(
                query, session_id, model_id, session_name, training_range, 
                validation_range, json.dumps(hyperparameters) if hyperparameters else None
            )
            
            logger.info(f"[SUCCESS] Training session started: {session_name} ({session_id})")
            return session_id
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start training session: {str(e)}")
            raise
    
    async def update_training_metrics(self, session_id: str, training_metrics: Dict[str, Any]) -> None:
        """Update training metrics during training."""
        query = """
        UPDATE ai_trading.training_sessions 
        SET training_metrics = $2, updated_at = NOW()
        WHERE session_id = $1
        """
        
        try:
            await self.db.execute_command(query, session_id, json.dumps(training_metrics))
            logger.info(f"[SUCCESS] Training metrics updated for session: {session_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update training metrics: {str(e)}")
            raise
    
    async def complete_training_session(self, session_id: str, final_performance: Dict[str, Any],
                                      model_artifacts_path: Optional[str] = None) -> None:
        """Mark training session as completed."""
        query = """
        UPDATE ai_trading.training_sessions 
        SET status = 'completed', end_time = NOW(), final_performance = $2, model_artifacts_path = $3
        WHERE session_id = $1
        """
        
        try:
            await self.db.execute_command(
                query, session_id, json.dumps(final_performance), model_artifacts_path
            )
            logger.info(f"[SUCCESS] Training session completed: {session_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to complete training session: {str(e)}")
            raise

class SignalManager:
    """Manage AI trading signals."""
    
    def __init__(self, db: AITradingDatabase):
        self.db = db
    
    async def store_signal(self, signal: TradingSignal, model_id: str) -> str:
        """Store a trading signal in the database."""
        signal_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO ai_trading.ai_signals 
        (signal_id, model_id, symbol, signal_type, confidence, price_target, 
         stop_loss, take_profit, signal_timestamp, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        try:
            await self.db.execute_command(
                query, signal_id, model_id, signal.symbol, signal.signal_type,
                float(signal.confidence), float(signal.price_target),
                float(signal.stop_loss) if signal.stop_loss else None,
                float(signal.take_profit) if signal.take_profit else None,
                signal.timestamp, json.dumps(signal.metadata) if signal.metadata else None
            )
            
            logger.info(f"[SUCCESS] Signal stored: {signal.signal_type} {signal.symbol} ({signal_id})")
            return signal_id
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store signal: {str(e)}")
            raise
    
    async def store_signals_batch(self, signals: List[TradingSignal], model_id: str) -> List[str]:
        """Store multiple signals efficiently."""
        if not signals:
            return []
        
        signal_ids = []
        values = []
        
        for signal in signals:
            signal_id = str(uuid.uuid4())
            signal_ids.append(signal_id)
            
            values.extend([
                signal_id, model_id, signal.symbol, signal.signal_type,
                float(signal.confidence), float(signal.price_target),
                float(signal.stop_loss) if signal.stop_loss else None,
                float(signal.take_profit) if signal.take_profit else None,
                signal.timestamp, json.dumps(signal.metadata) if signal.metadata else None
            ])
        
        # Build bulk insert query
        placeholder_groups = []
        for i in range(len(signals)):
            start_idx = i * 10 + 1
            placeholders = ', '.join([f'${j}' for j in range(start_idx, start_idx + 10)])
            placeholder_groups.append(f'({placeholders})')
        
        query = f"""
        INSERT INTO ai_trading.ai_signals 
        (signal_id, model_id, symbol, signal_type, confidence, price_target, 
         stop_loss, take_profit, signal_timestamp, metadata)
        VALUES {', '.join(placeholder_groups)}
        """
        
        try:
            await self.db.execute_command(query, *values)
            logger.info(f"[SUCCESS] Stored {len(signals)} signals in batch")
            return signal_ids
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store signal batch: {str(e)}")
            raise
    
    async def get_recent_signals(self, model_id: Optional[str] = None, 
                               symbol: Optional[str] = None, 
                               hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent signals with optional filtering."""
        conditions = ["s.signal_timestamp > NOW() - INTERVAL '%s hours'"]
        params = [hours]
        
        if model_id:
            conditions.append("s.model_id = $%d")
            params.append(model_id)
        
        if symbol:
            conditions.append("s.symbol = $%d")
            params.append(symbol)
        
        # Update parameter placeholders
        for i, condition in enumerate(conditions[1:], 2):
            conditions[i] = condition % i
        
        query = f"""
        SELECT s.signal_id, s.model_id, m.model_name, s.symbol, s.signal_type,
               s.confidence, s.price_target, s.stop_loss, s.take_profit,
               s.signal_timestamp, s.metadata
        FROM ai_trading.ai_signals s
        JOIN ai_trading.ai_models m ON s.model_id = m.model_id
        WHERE {' AND '.join(conditions)}
        ORDER BY s.signal_timestamp DESC
        LIMIT 100
        """
        
        try:
            result = await self.db.execute_query(query, *params)
            
            # Parse JSON metadata
            for row in result:
                if row['metadata']:
                    row['metadata'] = json.loads(row['metadata'])
            
            logger.info(f"[SUCCESS] Retrieved {len(result)} recent signals")
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get recent signals: {str(e)}")
            raise

class FeatureManager:
    """Manage feature data for ML models."""
    
    def __init__(self, db: AITradingDatabase):
        self.db = db
    
    async def store_features(self, symbol: str, timestamp: datetime, 
                           feature_set_name: str, features: Dict[str, Any],
                           extraction_method: str = "auto") -> None:
        """Store extracted features."""
        # Create hash of features for versioning
        feature_hash = str(hash(json.dumps(features, sort_keys=True)))
        
        query = """
        INSERT INTO ai_trading.feature_data 
        (symbol, timestamp, feature_set_name, features, raw_data_hash, extraction_method)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (symbol, timestamp, feature_set_name)
        DO UPDATE SET 
            features = EXCLUDED.features,
            raw_data_hash = EXCLUDED.raw_data_hash,
            extraction_method = EXCLUDED.extraction_method,
            created_at = NOW()
        """
        
        try:
            await self.db.execute_command(
                query, symbol, timestamp, feature_set_name, 
                json.dumps(features), feature_hash, extraction_method
            )
            
            logger.info(f"[SUCCESS] Features stored: {symbol} - {feature_set_name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to store features: {str(e)}")
            raise
    
    async def get_features(self, symbol: str, feature_set_name: str,
                         start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get features for a time range."""
        query = """
        SELECT timestamp, features, extraction_method
        FROM ai_trading.feature_data 
        WHERE symbol = $1 AND feature_set_name = $2 
        AND timestamp BETWEEN $3 AND $4
        ORDER BY timestamp ASC
        """
        
        try:
            result = await self.db.execute_query(query, symbol, feature_set_name, start_time, end_time)
            
            # Parse JSON features
            for row in result:
                row['features'] = json.loads(row['features'])
            
            logger.info(f"[SUCCESS] Retrieved {len(result)} feature records for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get features: {str(e)}")
            raise

class PerformanceManager:
    """Manage model performance tracking."""
    
    def __init__(self, db: AITradingDatabase):
        self.db = db
    
    async def record_performance(self, model_id: str, evaluation_period: tuple,
                               metrics: Dict[str, Any]) -> None:
        """Record model performance metrics."""
        period_range = f"[{evaluation_period[0]},{evaluation_period[1]})"
        
        query = """
        INSERT INTO ai_trading.model_performance 
        (model_id, evaluation_period, total_signals, successful_signals, signal_accuracy,
         avg_confidence, sharpe_ratio, max_drawdown, total_return, volatility, win_rate,
         avg_holding_period_hours, risk_adjusted_return)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """
        
        try:
            await self.db.execute_command(
                query, model_id, period_range,
                metrics.get('total_signals', 0),
                metrics.get('successful_signals', 0),
                metrics.get('signal_accuracy', 0.0),
                metrics.get('avg_confidence', 0.0),
                metrics.get('sharpe_ratio', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('total_return', 0.0),
                metrics.get('volatility', 0.0),
                metrics.get('win_rate', 0.0),
                metrics.get('avg_holding_period_hours', 0.0),
                metrics.get('risk_adjusted_return', 0.0)
            )
            
            logger.info(f"[SUCCESS] Performance recorded for model: {model_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to record performance: {str(e)}")
            raise
    
    async def get_model_performance_history(self, model_id: str, 
                                          days_back: int = 30) -> List[Dict[str, Any]]:
        """Get performance history for a model."""
        query = """
        SELECT evaluation_period, total_signals, signal_accuracy, sharpe_ratio,
               max_drawdown, total_return, win_rate, created_at
        FROM ai_trading.model_performance 
        WHERE model_id = $1 
        AND created_at > NOW() - INTERVAL '%s days'
        ORDER BY created_at DESC
        """ % days_back
        
        try:
            result = await self.db.execute_query(query, model_id)
            logger.info(f"[SUCCESS] Retrieved {len(result)} performance records")
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get performance history: {str(e)}")
            raise

# Factory function for database integration
def create_ai_trading_database(connection_string: Optional[str] = None) -> AITradingDatabase:
    """Factory function to create AI trading database instance."""
    return AITradingDatabase(connection_string)
