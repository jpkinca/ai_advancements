#!/usr/bin/env python3
"""
XGBoost Training Script

This script trains XGBoost models for algorithmic trading using historical market data.
Supports both regression (price prediction) and classification (trading signals) models.

Usage:
    python train_xgboost_models.py --symbol AAPL --start-date 2020-01-01 --tune-params

Features:
- Automated data loading and feature engineering
- Hyperparameter tuning with grid search
- Cross-validation for robust evaluation
- Model persistence and loading
- Comprehensive logging and reporting
"""

import asyncio
import argparse
import os
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional

from xgboost_trading_model import XGBoostTradingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xgboost_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def train_models(symbol: str, start_date: str, end_date: Optional[str] = None,
                      tune_params: bool = True, save_models: bool = True) -> Dict[str, Any]:
    """
    Train XGBoost models for a given symbol

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date for training data
        end_date: End date for training data (default: today)
        tune_params: Whether to perform hyperparameter tuning
        save_models: Whether to save trained models

    Returns:
        Dictionary with training results and metrics
    """

    logger.info(f"Starting training for {symbol} from {start_date}")

    # Initialize model
    model = XGBoostTradingModel(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    # Load and prepare data
    logger.info("Loading data...")
    await model.load_data()

    if model.raw_data is None or model.raw_data.empty:
        logger.error(f"No data available for {symbol}")
        return {'error': 'No data available'}

    logger.info("Adding technical indicators...")
    model.add_technical_indicators()

    logger.info("Preparing targets...")
    X, y_reg, y_cls = model.prepare_targets()

    if X.empty:
        logger.error("No valid training data after preparation")
        return {'error': 'No valid training data'}

    # Train regression model
    logger.info("Training regression model...")
    reg_mae, reg_importance = model.train_regression_model(hyperparameter_tune=tune_params)

    # Train classification model
    logger.info("Training classification model...")
    cls_accuracy, cls_importance = model.train_classification_model(hyperparameter_tune=tune_params)

    # Backtest strategy
    logger.info("Running backtest...")
    total_return, trades = model.backtest_strategy()

    # Save models if requested
    if save_models:
        logger.info("Saving models...")
        model.save_models()

    # Compile results
    results = {
        'symbol': symbol,
        'training_period': {
            'start': start_date,
            'end': end_date or datetime.now().strftime('%Y-%m-%d')
        },
        'data_info': {
            'total_samples': len(X),
            'features': len(X.columns),
            'feature_names': list(X.columns)
        },
        'regression_model': {
            'mae': reg_mae,
            'top_features': sorted(reg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        },
        'classification_model': {
            'accuracy': cls_accuracy,
            'top_features': sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        },
        'backtest_results': {
            'total_return_percent': total_return,
            'total_trades': len(trades),
            'win_rate': len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades) if trades else 0
        },
        'training_timestamp': datetime.now().isoformat(),
        'hyperparameter_tuning': tune_params
    }

    logger.info(f"Training completed for {symbol}")
    logger.info(f"Regression MAE: {reg_mae:.4f}")
    logger.info(f"Classification Accuracy: {cls_accuracy:.4f}")
    logger.info(f"Backtest Return: {total_return:.2f}%")

    return results

async def batch_train(symbols: list, start_date: str, **kwargs) -> Dict[str, Dict[str, Any]]:
    """Train models for multiple symbols"""
    results = {}

    for symbol in symbols:
        logger.info(f"Training model for {symbol}...")
        try:
            result = await train_models(symbol, start_date, **kwargs)
            results[symbol] = result
        except Exception as e:
            logger.error(f"Failed to train model for {symbol}: {e}")
            results[symbol] = {'error': str(e)}

    return results

def save_results(results: Dict[str, Any], output_file: Optional[str] = None):
    """Save training results to JSON file"""
    if not output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'xgboost_training_results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

def print_summary(results: Dict[str, Any]):
    """Print training summary"""
    print("\n" + "="*80)
    print("XGBoost Training Summary")
    print("="*80)

    if isinstance(results, dict) and 'error' in results:
        print(f"Error: {results['error']}")
        return

    for symbol, result in results.items():
        if 'error' in result:
            print(f"\n{symbol}: ERROR - {result['error']}")
            continue

        print(f"\n{symbol}:")
        print(f"  Data: {result['data_info']['total_samples']} samples, {result['data_info']['features']} features")
        print(".4f")
        print(".4f")
        print(".2f")
        print(f"  Trades: {result['backtest_results']['total_trades']}")

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models for algorithmic trading')
    parser.add_argument('--symbol', '-s', required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--start-date', '-d', default='2020-01-01', help='Start date for training data')
    parser.add_argument('--end-date', '-e', help='End date for training data')
    parser.add_argument('--tune-params', '-t', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--no-save', action='store_true', help='Do not save trained models')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--batch', nargs='+', help='Train multiple symbols')

    args = parser.parse_args()

    # Configure logging level
    if args.tune_params:
        logger.info("Hyperparameter tuning enabled - this may take several minutes")

    try:
        if args.batch:
            # Batch training
            results = asyncio.run(batch_train(
                args.batch,
                args.start_date,
                end_date=args.end_date,
                tune_params=args.tune_params,
                save_models=not args.no_save
            ))
        else:
            # Single symbol training
            results = asyncio.run(train_models(
                args.symbol,
                args.start_date,
                end_date=args.end_date,
                tune_params=args.tune_params,
                save_models=not args.no_save
            ))
            results = {args.symbol: results}

        # Save and print results
        save_results(results, args.output)
        print_summary(results)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\nzcon\VSPython\ai_advancements\train_xgboost_models.py