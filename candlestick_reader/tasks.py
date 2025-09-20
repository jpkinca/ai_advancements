#!/usr/bin/env python3
"""
Celery Tasks for Candlestick Analysis

This module provides Celery tasks for real-time candlestick analysis
and integration with the trading system.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from celery import Celery
from ai_data_accessor import AIDataAccessor
from candlestick_reader.integration import IntegratedCandlestickSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('candlestick_tasks')
celery_app.config_from_object('celeryconfig')

# Global system instance
_candlestick_system = None

def get_candlestick_system() -> IntegratedCandlestickSystem:
    """Get or create integrated candlestick system instance"""
    global _candlestick_system
    if _candlestick_system is None:
        _candlestick_system = IntegratedCandlestickSystem()
    return _candlestick_system

@celery_app.task(bind=True, name='candlestick.analyze_symbol')
def analyze_symbol_task(self, symbol: str, timeframe: str = '5min',
                       lookback_periods: int = 50) -> Dict[str, Any]:
    """
    Celery task for comprehensive symbol analysis

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        lookback_periods: Number of periods to analyze

    Returns:
        Analysis results
    """
    try:
        import asyncio

        # Get system instance
        system = get_candlestick_system()

        # Run analysis
        result = asyncio.run(system.analyze_symbol_comprehensive(
            symbol, timeframe, lookback_periods
        ))

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'results/candlestick_analysis_{symbol}_{timestamp}.json'

        os.makedirs('results', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"[TASK] Completed analysis for {symbol}")
        return result

    except Exception as e:
        logger.error(f"[TASK ERROR] Analysis failed for {symbol}: {e}")
        self.retry(countdown=60, max_retries=3)
        return {'error': str(e), 'symbol': symbol}

@celery_app.task(bind=True, name='candlestick.generate_signal')
def generate_signal_task(self, symbol: str, timeframe: str = '5min') -> Dict[str, Any]:
    """
    Celery task for trading signal generation

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe

    Returns:
        Trading signal
    """
    try:
        import asyncio

        # Get system instance
        system = get_candlestick_system()

        # Generate signal
        signal = asyncio.run(system.generate_trading_signal(symbol, timeframe))

        # Save signal
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'results/trading_signal_{symbol}_{timestamp}.json'

        os.makedirs('results', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(signal, f, indent=2, default=str)

        logger.info(f"[TASK] Generated signal for {symbol}: {signal.get('action', 'HOLD')}")
        return signal

    except Exception as e:
        logger.error(f"[TASK ERROR] Signal generation failed for {symbol}: {e}")
        self.retry(countdown=60, max_retries=3)
        return {'error': str(e), 'symbol': symbol}

@celery_app.task(bind=True, name='candlestick.batch_analyze')
def batch_analyze_task(self, symbols: List[str], timeframe: str = '5min',
                      lookback_periods: int = 50) -> Dict[str, Any]:
    """
    Celery task for batch symbol analysis

    Args:
        symbols: List of stock symbols
        timeframe: Data timeframe
        lookback_periods: Number of periods to analyze

    Returns:
        Batch analysis results
    """
    try:
        import asyncio

        # Get system instance
        system = get_candlestick_system()

        results = {}
        for symbol in symbols:
            try:
                analysis = asyncio.run(system.analyze_symbol_comprehensive(
                    symbol, timeframe, lookback_periods
                ))
                results[symbol] = analysis

                # Small delay to avoid overwhelming the system
                asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"[BATCH ERROR] Failed to analyze {symbol}: {e}")
                results[symbol] = {'error': str(e), 'symbol': symbol}

        # Save batch results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'results/batch_analysis_{timestamp}.json'

        os.makedirs('results', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"[BATCH TASK] Completed analysis for {len(symbols)} symbols")
        return results

    except Exception as e:
        logger.error(f"[BATCH TASK ERROR] Batch analysis failed: {e}")
        self.retry(countdown=120, max_retries=2)
        return {'error': str(e)}

@celery_app.task(bind=True, name='candlestick.real_time_monitor')
def real_time_monitor_task(self, symbols: List[str], timeframe: str = '1min',
                          interval_minutes: int = 5) -> Dict[str, Any]:
    """
    Celery task for real-time monitoring and signal generation

    Args:
        symbols: List of symbols to monitor
        timeframe: Data timeframe
        interval_minutes: Monitoring interval

    Returns:
        Monitoring results
    """
    try:
        import asyncio
        from datetime import datetime, timedelta

        # Get system instance
        system = get_candlestick_system()

        monitoring_results = {
            'start_time': datetime.now(),
            'symbols_monitored': symbols,
            'signals_generated': [],
            'errors': []
        }

        for symbol in symbols:
            try:
                # Generate signal
                signal = asyncio.run(system.generate_trading_signal(symbol, timeframe))

                if signal.get('action') != 'HOLD':
                    monitoring_results['signals_generated'].append(signal)
                    logger.info(f"[MONITOR] Signal generated for {symbol}: {signal['action']}")

            except Exception as e:
                error_info = {'symbol': symbol, 'error': str(e)}
                monitoring_results['errors'].append(error_info)
                logger.error(f"[MONITOR ERROR] Failed to monitor {symbol}: {e}")

        monitoring_results['end_time'] = datetime.now()
        monitoring_results['duration_seconds'] = (
            monitoring_results['end_time'] - monitoring_results['start_time']
        ).total_seconds()

        # Save monitoring results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'results/monitoring_{timestamp}.json'

        os.makedirs('results', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(monitoring_results, f, indent=2, default=str)

        logger.info(f"[MONITOR TASK] Completed monitoring for {len(symbols)} symbols, "
                   f"generated {len(monitoring_results['signals_generated'])} signals")

        return monitoring_results

    except Exception as e:
        logger.error(f"[MONITOR TASK ERROR] Real-time monitoring failed: {e}")
        self.retry(countdown=300, max_retries=3)  # Retry every 5 minutes
        return {'error': str(e)}

@celery_app.task(bind=True, name='candlestick.update_models')
def update_models_task(self) -> Dict[str, Any]:
    """
    Celery task for updating candlestick models with new data

    Returns:
        Model update results
    """
    try:
        from candlestick_reader.model_training import CandlestickModelTrainer

        # Initialize trainer
        trainer = CandlestickModelTrainer()

        # Update models
        results = trainer.update_models()

        # Save update results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'results/model_update_{timestamp}.json'

        os.makedirs('results', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("[MODEL UPDATE] Completed model updates")
        return results

    except Exception as e:
        logger.error(f"[MODEL UPDATE ERROR] Failed to update models: {e}")
        self.retry(countdown=3600, max_retries=2)  # Retry hourly
        return {'error': str(e)}

# Periodic tasks
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks"""
    # Update models daily at 2 AM
    sender.add_periodic_task(
        86400,  # 24 hours
        update_models_task.s(),
        name='update-candlestick-models'
    )

    # Monitor major symbols every 5 minutes
    sender.add_periodic_task(
        300,  # 5 minutes
        real_time_monitor_task.s(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'],
            timeframe='5min'
        ),
        name='monitor-major-symbols'
    )

# Task routing configuration
celery_app.conf.task_routes = {
    'candlestick.analyze_symbol': {'queue': 'candlestick'},
    'candlestick.generate_signal': {'queue': 'candlestick'},
    'candlestick.batch_analyze': {'queue': 'candlestick'},
    'candlestick.real_time_monitor': {'queue': 'candlestick'},
    'candlestick.update_models': {'queue': 'candlestick'},
}

# Task result backend
celery_app.conf.result_backend = 'redis://localhost:6379/0'
celery_app.conf.task_serializer = 'json'
celery_app.conf.result_serializer = 'json'
celery_app.conf.accept_content = ['json']

if __name__ == '__main__':
    celery_app.start()