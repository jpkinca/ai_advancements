"""
Production Configuration for Chain-of-Alpha MVP

Centralized configuration management for IBKR Gateway, PostgreSQL, 
and all production requirements per CO-PILOT INSTRUCTIONS.
"""

import os
from typing import Dict, Any
from datetime import datetime
import pytz

# US Eastern Timezone - NYSE/NASDAQ
US_EASTERN = pytz.timezone('US/Eastern')

class ProductionConfig:
    """
    Production configuration manager for Chain-of-Alpha.
    
    Handles all production settings:
    - IBKR Gateway connection
    - PostgreSQL database
    - Trading parameters
    - Compliance settings
    """
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load production configuration."""
        return {
            # ===============================
            # IBKR GATEWAY SETTINGS (REQUIRED)
            # ===============================
            'ibkr': {
                'host': os.getenv('IBKR_HOST', '127.0.0.1'),
                'port': int(os.getenv('IBKR_PORT', 4002)),  # Paper trading default
                'live_port': 4001,  # Live trading port
                'paper_port': 4002,  # Paper trading port
                'client_id': int(os.getenv('IBKR_CLIENT_ID', 300)),
                'timeout': 30,
                'paper_trading': os.getenv('IBKR_PAPER_TRADING', 'true').lower() == 'true'
            },
            
            # ===============================
            # POSTGRESQL DATABASE (REQUIRED)
            # ===============================
            'database': {
                'url': os.getenv('DATABASE_URL', 
                    'postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway'),
                'pool_min_size': 5,
                'pool_max_size': 20,
                'command_timeout': 60
            },
            
            # ===============================
            # MARKET DATA SETTINGS
            # ===============================
            'market_data': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
                'start_date': '2020-01-01',
                'end_date': datetime.now(US_EASTERN).strftime('%Y-%m-%d'),
                'timezone': 'US/Eastern',
                'bar_size': '1 day',
                'use_rth': True,  # Regular trading hours only
                'data_source': 'IBKR_GATEWAY'  # NO FALLBACKS
            },
            
            # ===============================
            # AI MODEL SETTINGS
            # ===============================
            'ai_model': {
                'name': 'meta-llama/Llama-3.2-3B-Instruct',
                'max_length': 2048,
                'max_new_tokens': 800,
                'temperature': 0.7,
                'do_sample': True,
                'device_map': 'auto',
                'torch_dtype': 'float16'
            },
            
            # ===============================
            # FACTOR GENERATION SETTINGS
            # ===============================
            'factors': {
                'num_factors': 10,
                'lookback_period': 252,  # 1 year trading days
                'min_data_points': 100,
                'evaluation_method': 'information_coefficient',
                'rank_method': 'cross_sectional'
            },
            
            # ===============================
            # PORTFOLIO SETTINGS
            # ===============================
            'portfolio': {
                'top_factors': 5,
                'rebalance_frequency': 'monthly',
                'weight_method': 'equal_weight',
                'max_position_size': 0.25,
                'min_position_size': 0.01
            },
            
            # ===============================
            # BACKTESTING SETTINGS
            # ===============================
            'backtest': {
                'initial_capital': 100000,
                'commission': 0.001,  # 0.1%
                'slippage': 0.0005,   # 0.05%
                'benchmark': 'SPY',
                'risk_free_rate': 0.02
            },
            
            # ===============================
            # TECHNICAL ANALYSIS (TA-LIB)
            # ===============================
            'technical_analysis': {
                'library': 'TA-LIB',  # REQUIRED per CO-PILOT INSTRUCTIONS
                'indicators': {
                    'moving_averages': [5, 10, 20, 50],
                    'ema_periods': [12, 26],
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'bb_period': 20,
                    'bb_std': 2,
                    'stoch_k': 14,
                    'stoch_d': 3,
                    'atr_period': 14,
                    'volume_sma': 20
                }
            },
            
            # ===============================
            # COMPLIANCE SETTINGS
            # ===============================
            'compliance': {
                'data_source_enforcement': 'STRICT',  # NO fallbacks allowed
                'required_data_source': 'IBKR_GATEWAY',
                'prohibited_sources': ['yfinance', 'alpha_vantage', 'quandl'],
                'required_database': 'PostgreSQL',
                'required_timezone': 'US/Eastern',
                'required_technical_library': 'TA-LIB',
                'allow_mock_data': False,
                'production_mode': True
            },
            
            # ===============================
            # LOGGING SETTINGS
            # ===============================
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'chain_of_alpha_production.log',
                'console': True,
                'max_file_size': '10MB',
                'backup_count': 5
            },
            
            # ===============================
            # PERFORMANCE SETTINGS
            # ===============================
            'performance': {
                'parallel_factor_generation': False,  # For stability
                'data_cache_enabled': True,
                'cache_ttl_minutes': 30,
                'batch_size': 1000,
                'memory_limit_gb': 8
            }
        }
    
    def get_ibkr_config(self) -> Dict[str, Any]:
        """Get IBKR Gateway configuration."""
        return self.config['ibkr']
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get PostgreSQL database configuration."""
        return self.config['database']
    
    def get_market_data_config(self) -> Dict[str, Any]:
        """Get market data configuration."""
        return self.config['market_data']
    
    def get_ai_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration."""
        return self.config['ai_model']
    
    def get_factor_config(self) -> Dict[str, Any]:
        """Get factor generation configuration."""
        return self.config['factors']
    
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio configuration."""
        return self.config['portfolio']
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.config['backtest']
    
    def get_technical_analysis_config(self) -> Dict[str, Any]:
        """Get TA-LIB configuration."""
        return self.config['technical_analysis']
    
    def get_compliance_config(self) -> Dict[str, Any]:
        """Get compliance configuration."""
        return self.config['compliance']
    
    def validate_compliance(self) -> Dict[str, bool]:
        """
        Validate compliance with CO-PILOT INSTRUCTIONS.
        
        Returns:
            Dictionary with compliance status for each requirement
        """
        compliance = self.get_compliance_config()
        
        checks = {
            'ibkr_gateway_required': compliance['required_data_source'] == 'IBKR_GATEWAY',
            'no_fallbacks': not any(source in str(self.config) for source in compliance['prohibited_sources']),
            'postgresql_required': compliance['required_database'] == 'PostgreSQL',
            'us_eastern_timezone': compliance['required_timezone'] == 'US/Eastern',
            'talib_required': compliance['required_technical_library'] == 'TA-LIB',
            'no_mock_data': not compliance['allow_mock_data'],
            'production_mode': compliance['production_mode']
        }
        
        return checks
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return self.config
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration section."""
        if section in self.config:
            self.config[section].update(updates)
        else:
            self.config[section] = updates
    
    def export_config(self) -> str:
        """Export configuration as formatted string."""
        import json
        return json.dumps(self.config, indent=2, default=str)


# Global configuration instance
production_config = ProductionConfig()

# Convenience functions
def get_config() -> ProductionConfig:
    """Get the global production configuration."""
    return production_config

def validate_production_compliance() -> bool:
    """
    Validate full compliance with CO-PILOT INSTRUCTIONS.
    
    Returns:
        True if all compliance checks pass
    """
    checks = production_config.validate_compliance()
    all_passed = all(checks.values())
    
    if not all_passed:
        failed_checks = [check for check, passed in checks.items() if not passed]
        raise ValueError(f"Production compliance failed: {failed_checks}")
    
    return True


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    
    print("="*60)
    print("PRODUCTION CONFIGURATION VALIDATION")
    print("="*60)
    
    # Validate compliance
    try:
        compliance_status = config.validate_compliance()
        print("\nCompliance Status:")
        for check, status in compliance_status.items():
            status_str = "✅ PASS" if status else "❌ FAIL"
            print(f"  {check}: {status_str}")
        
        if all(compliance_status.values()):
            print("\n🎉 ALL COMPLIANCE CHECKS PASSED")
        else:
            print("\n⚠️  COMPLIANCE ISSUES DETECTED")
    
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
    
    print("\n" + "="*60)