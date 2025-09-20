"""
AI Trading Advancements Configuration Module

This module provides configuration management that integrates with the existing
TradeAppComponents infrastructure while maintaining modularity.

Author: AI Trading Development Team
Date: August 31, 2025
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal
import json
import logging

# Add parent directory to path for TradeAppComponents integration
current_dir = Path(__file__).parent
trade_app_root = current_dir.parent.parent
sys.path.append(str(trade_app_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging to follow TradeAppComponents standards (ASCII only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_advancements.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for PostgreSQL integration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class IBKRConfig:
    """IBKR integration configuration."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 10
    connection_timeout: int = 30


@dataclass
class AIModelConfig:
    """AI model configuration."""
    model_save_path: str
    training_data_path: str
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    max_position_size: Decimal
    stop_loss_percentage: Decimal
    take_profit_percentage: Decimal
    risk_per_trade: Decimal
    max_daily_trades: int
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    timezone: str = "America/New_York"


class AIAdvancementsConfig:
    """
    Central configuration manager for AI Trading Advancements.
    Integrates with existing TradeAppComponents infrastructure.
    """
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults."""
        self.logger = logger
        
        # Database configuration
        ai_db_url = os.getenv('DATABASE_URL', 'postgresql://localhost/trading_db')
        ai_db_url = self._normalize_db_url(ai_db_url)
        self.database = DatabaseConfig(
            url=ai_db_url,
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20'))
        )
        
        # IBKR configuration
        self.ibkr = IBKRConfig(
            host=os.getenv('IBKR_HOST', '127.0.0.1'),
            port=int(os.getenv('IBKR_PORT', '7497')),
            client_id=int(os.getenv('IBKR_CLIENT_ID', '1'))
        )
        
        # AI model configuration
        ai_models_path = trade_app_root / "ai_advancements" / "models"
        ai_data_path = trade_app_root / "ai_advancements" / "data"
        
        self.ai_models = AIModelConfig(
            model_save_path=str(ai_models_path),
            training_data_path=str(ai_data_path),
            validation_split=float(os.getenv('AI_VALIDATION_SPLIT', '0.2')),
            test_split=float(os.getenv('AI_TEST_SPLIT', '0.1')),
            random_seed=int(os.getenv('AI_RANDOM_SEED', '42')),
            batch_size=int(os.getenv('AI_BATCH_SIZE', '32')),
            epochs=int(os.getenv('AI_EPOCHS', '100')),
            learning_rate=float(os.getenv('AI_LEARNING_RATE', '0.001'))
        )
        
        # Trading configuration
        self.trading = TradingConfig(
            max_position_size=Decimal(os.getenv('MAX_POSITION_SIZE', '10000.00')),
            stop_loss_percentage=Decimal(os.getenv('STOP_LOSS_PCT', '0.02')),
            take_profit_percentage=Decimal(os.getenv('TAKE_PROFIT_PCT', '0.04')),
            risk_per_trade=Decimal(os.getenv('RISK_PER_TRADE', '0.01')),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', '10')),
            trading_hours_start=os.getenv('TRADING_HOURS_START', '09:30'),
            trading_hours_end=os.getenv('TRADING_HOURS_END', '16:00'),
            timezone=os.getenv('TRADING_TIMEZONE', 'America/New_York')
        )
        
        # API configurations
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'reddit_user_agent': os.getenv('REDDIT_USER_AGENT', 'AITradingBot/1.0')
        }
        
        # Feature flags
        self.features = {
            'enable_reinforcement_learning': os.getenv('ENABLE_RL', 'true').lower() == 'true',
            'enable_sentiment_analysis': os.getenv('ENABLE_SENTIMENT', 'true').lower() == 'true',
            'enable_genetic_algorithms': os.getenv('ENABLE_GENETIC', 'false').lower() == 'true',
            'enable_paper_trading': os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true',
            'enable_live_trading': os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
        }
        
        # Ensure required directories exist
        self._create_directories()
        
        logger.info("[STARTING] AI Advancements Configuration initialized")
        logger.info(f"[DATA] Database URL: {self.database.url[:20]}...")
        logger.info(f"[DATA] IBKR Host: {self.ibkr.host}:{self.ibkr.port}")
        logger.info(f"[DATA] Models path: {self.ai_models.model_save_path}")
    
    def _create_directories(self) -> None:
        """Create necessary directories for AI operations."""
        directories = [
            self.ai_models.model_save_path,
            self.ai_models.training_data_path,
            str(trade_app_root / "ai_advancements" / "logs"),
            str(trade_app_root / "ai_advancements" / "results"),
            str(trade_app_root / "ai_advancements" / "backtest_results")
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _normalize_db_url(self, url: str) -> str:
        """Ensure URL enforces SSL and disables GSS encryption.

        This avoids libpq defaulting to GSS (Kerberos) on some networks and
        aligns with Railway proxy requirements.
        """
        if not url:
            return url
        if 'sslmode=' not in url:
            url += ('&' if '?' in url else '?') + 'sslmode=require'
        if 'gssencmode=' not in url:
            url += ('&' if '?' in url else '?') + 'gssencmode=disable'
        return url
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.database.url
    
    def get_ibkr_config(self) -> Dict[str, Any]:
        """Get IBKR connection parameters."""
        return {
            'host': self.ibkr.host,
            'port': self.ibkr.port,
            'clientId': self.ibkr.client_id,
            'timeout': self.ibkr.timeout
        }
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for specified service."""
        return self.api_keys.get(service)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.features.get(feature, False)
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a feature is both enabled and properly configured."""
        if not self.is_feature_enabled(feature):
            return False
            
        # Check feature-specific requirements
        if feature == 'enable_sentiment_analysis':
            return bool(self.get_api_key('twitter_bearer_token'))
        elif feature == 'enable_live_trading':
            return bool(self.database.url and self.database.url != 'postgresql://localhost/trading_db')
        else:
            return True  # Default to available if enabled
    
    def get_feature_status(self) -> Dict[str, Dict[str, bool]]:
        """Get detailed status of all features."""
        status = {}
        for feature, enabled in self.features.items():
            status[feature] = {
                'enabled': enabled,
                'available': self.is_feature_available(feature) if enabled else False,
                'configured': True  # Will be updated below
            }
            
            # Check specific configurations
            if feature == 'enable_sentiment_analysis':
                status[feature]['configured'] = bool(self.get_api_key('twitter_bearer_token'))
            elif feature == 'enable_live_trading':
                status[feature]['configured'] = bool(
                    self.database.url and self.database.url != 'postgresql://localhost/trading_db'
                )
        
        return status
    
    def validate_configuration(self) -> bool:
        """Validate configuration completeness."""
        validation_errors = []
        validation_warnings = []
        
        # Check required API keys for enabled features
        if self.is_feature_enabled('enable_sentiment_analysis'):
            if not self.get_api_key('twitter_bearer_token'):
                validation_warnings.append("Twitter Bearer Token missing - sentiment analysis will be disabled")
                validation_warnings.append("To enable: Run 'python test_twitter_token.py' for setup help")
        
        # Check database URL  
        if not self.database.url or self.database.url == 'postgresql://localhost/trading_db':
            validation_warnings.append("Using default database URL - configure DATABASE_URL for production")
        
        # Check IBKR configuration
        if self.is_feature_enabled('enable_live_trading'):
            if self.ibkr.host == '127.0.0.1' and self.ibkr.port == 7497:
                validation_warnings.append("Using default IBKR configuration for live trading")
        
        # Log warnings (non-blocking)
        if validation_warnings:
            for warning in validation_warnings:
                logger.warning(f"[WARNING] Configuration: {warning}")
        
        # Log critical errors (blocking)
        if validation_errors:
            for error in validation_errors:
                logger.error(f"[ERROR] Configuration validation: {error}")
            return False
        
        logger.info("[SUCCESS] Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            'database': {
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow
            },
            'ibkr': {
                'host': self.ibkr.host,
                'port': self.ibkr.port,
                'client_id': self.ibkr.client_id
            },
            'ai_models': {
                'model_save_path': self.ai_models.model_save_path,
                'training_data_path': self.ai_models.training_data_path,
                'validation_split': self.ai_models.validation_split,
                'batch_size': self.ai_models.batch_size,
                'epochs': self.ai_models.epochs
            },
            'trading': {
                'max_position_size': str(self.trading.max_position_size),
                'stop_loss_percentage': str(self.trading.stop_loss_percentage),
                'take_profit_percentage': str(self.trading.take_profit_percentage),
                'max_daily_trades': self.trading.max_daily_trades
            },
            'features': self.features
        }


# Global configuration instance
config = AIAdvancementsConfig()


def get_config() -> AIAdvancementsConfig:
    """Get the global configuration instance."""
    return config


if __name__ == "__main__":
    # Configuration validation and display
    config = get_config()
    
    print("\n=== AI Trading Advancements Configuration ===")
    print(f"[DATA] Configuration loaded successfully")
    print(f"[DATA] Features enabled: {sum(config.features.values())} of {len(config.features)}")
    
    # Validate configuration
    is_valid = config.validate_configuration()
    print(f"[DATA] Configuration valid: {is_valid}")
    
    # Display non-sensitive configuration
    config_dict = config.to_dict()
    print(f"\n[DATA] Configuration summary:")
    for section, values in config_dict.items():
        print(f"  {section}: {len(values)} settings configured")
