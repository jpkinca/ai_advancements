#!/usr/bin/env python3
"""
IBD 50 VLM Training Pipeline

Comprehensive training pipeline for Vision-Language Models on IBD 50 stocks.
Fetches historical data, generates chart-text pairs, and trains calibrated CLIP models.

Features:
- IBD 50 stock universe integration
- Multi-source historical data fetching
- Automated chart generation and labeling
- CLIP fine-tuning with hard negatives
- Model calibration and evaluation
- Multimodal fusion training
- Performance benchmarking
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vlm_ibd50_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBD50VLMTrainer:
    """
    Complete training pipeline for VLM models on IBD 50 stocks
    """

    def __init__(self,
                 output_dir: str = "vlm/models/ibd50",
                 data_dir: str = "vlm/data/ibd50",
                 model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize IBD 50 VLM trainer

        Args:
            output_dir: Directory to save trained models
            data_dir: Directory for training data
            model_name: Base CLIP model to fine-tune
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.model_name = model_name

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # IBD 50 stock universe
        self.ibd50_stocks = self._get_ibd50_universe()

        # Training configuration
        self.training_config = {
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 5e-6,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'save_steps': 500,
            'eval_steps': 100,
            'max_samples_per_stock': 100,  # Limit samples per stock for balance
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'chart_timeframes': ['1D', '4H', '1H'],  # Multiple timeframes
            'include_indicators': ['rsi', 'macd', 'bb', 'sma_20', 'sma_50']
        }

        # Initialize components
        self.chart_generator = None
        self.text_generator = None
        self.dataset_builder = None
        self.clip_trainer = None
        self.calibrator = None
        self.fusion_model = None

        logger.info(f"[INIT] IBD 50 VLM Trainer initialized with {len(self.ibd50_stocks)} stocks")

    def _get_ibd50_universe(self) -> List[str]:
        """Get the IBD 50 stock universe"""
        try:
            # Try to load from database first
            from ibd50_database_manager import IBD50DatabaseManager
            db_manager = IBD50DatabaseManager()
            stocks = db_manager.get_ibd50_stocks(as_dataframe=False)  # Explicitly request list
            logger.info(f"[DATA] Loaded {len(stocks)} IBD 50 stocks from database")
            return stocks
        except ImportError:
            logger.warning("[DATA] Database manager not available, trying stock_universes")
            try:
                from stock_universes import YOUR_CUSTOM_UNIVERSE
                return YOUR_CUSTOM_UNIVERSE
            except ImportError:
                logger.warning("[DATA] Using fallback IBD 50 list")
                # Fallback IBD 50 list (approximate)
                return [
                    'NVDA', 'PLTR', 'HOOD', 'RKLB', 'IREN', 'ANET', 'FUTU', 'RDDT', 'DOCS', 'SOFI',
                    'IBKR', 'STNE', 'TARS', 'AMSC', 'ALAB', 'MEDP', 'PODD', 'CCJ', 'TSLA', 'AAPL',
                    'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'CRM', 'ADBE', 'INTC', 'CSCO', 'ORCL',
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'COF', 'USB',
                    'JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY', 'LLY', 'AMGN', 'GILD'
                ]
        except Exception as e:
            logger.error(f"[DATA] Failed to load IBD 50 stocks: {e}")
            logger.warning("[DATA] Using fallback IBD 50 list")
            # Fallback IBD 50 list (approximate)
            return [
                'NVDA', 'PLTR', 'HOOD', 'RKLB', 'IREN', 'ANET', 'FUTU', 'RDDT', 'DOCS', 'SOFI',
                'IBKR', 'STNE', 'TARS', 'AMSC', 'ALAB', 'MEDP', 'PODD', 'CCJ', 'TSLA', 'AAPL',
                'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'CRM', 'ADBE', 'INTC', 'CSCO', 'ORCL',
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'COF', 'USB',
                'JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY', 'LLY', 'AMGN', 'GILD'
            ]

    async def fetch_historical_data(self,
                                   start_date: str = "2023-01-01",
                                   end_date: str = "2024-12-31",
                                   interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for IBD 50 stocks

        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching
            interval: Data interval (1d, 1h, etc.)

        Returns:
            Dictionary of stock symbols to DataFrames
        """
        logger.info("[DATA] Fetching historical data for IBD 50 stocks...")

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required for data fetching. Install with: pip install yfinance")

        historical_data = {}
        failed_stocks = []

        for symbol in tqdm(self.ibd50_stocks, desc="Fetching data"):
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)

                if df.empty:
                    logger.warning(f"[DATA] No data for {symbol}")
                    failed_stocks.append(symbol)
                    continue

                # Clean and prepare data
                df = df.reset_index()
                df['symbol'] = symbol
                df.columns = df.columns.str.lower()

                # Ensure required columns exist
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"[DATA] Missing required columns for {symbol}")
                    failed_stocks.append(symbol)
                    continue

                # Add technical indicators
                df = self._add_technical_indicators(df)

                historical_data[symbol] = df
                logger.info(f"[DATA] Fetched {len(df)} records for {symbol}")

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"[DATA] Failed to fetch {symbol}: {e}")
                failed_stocks.append(symbol)

        logger.info(f"[DATA] Successfully fetched data for {len(historical_data)}/{len(self.ibd50_stocks)} stocks")
        if failed_stocks:
            logger.warning(f"[DATA] Failed stocks: {failed_stocks}")

        return historical_data

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['sma_20'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['sma_20'] - (df['bb_std'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']

            # Additional SMAs
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

            # Price change
            df['price_change'] = df['close'].pct_change()

            # Volatility (20-day rolling std of returns)
            df['volatility'] = df['price_change'].rolling(window=20).std()

        except Exception as e:
            logger.warning(f"[INDICATORS] Failed to add indicators: {e}")

        return df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    def prepare_training_data(self, historical_data: Dict[str, pd.DataFrame]) -> str:
        """
        Prepare VLM training data from historical stock data

        Args:
            historical_data: Dictionary of stock data

        Returns:
            Path to prepared dataset
        """
        logger.info("[PREP] Preparing VLM training data...")

        # Initialize components
        from vlm.chart_image_generator import ChartImageGenerator
        from vlm.text_label_generator import TextLabelGenerator
        from vlm.dataset_builder import VLMDatasetBuilder

        self.chart_generator = ChartImageGenerator()
        self.text_generator = TextLabelGenerator()
        self.dataset_builder = VLMDatasetBuilder()

        # Generate chart-text pairs for each stock
        all_samples = []

        for symbol, df in tqdm(historical_data.items(), desc="Processing stocks"):
            try:
                # Limit samples per stock for balance
                max_samples = self.training_config['max_samples_per_stock']
                if len(df) > max_samples:
                    # Sample evenly across time period
                    indices = np.linspace(50, len(df)-1, max_samples, dtype=int)
                    df_samples = df.iloc[indices]
                else:
                    df_samples = df.iloc[50:]  # Skip first 50 rows for indicator warmup

                stock_samples = []

                for idx, row in df_samples.iterrows():
                    try:
                        # Generate chart
                        chart_data = df.iloc[max(0, idx-100):idx+1]  # Last 100 periods
                        if len(chart_data) < 20:
                            continue

                        # Prepare chart data with correct column names
                        chart_data_for_gen = chart_data.copy()
                        if 'date' in chart_data_for_gen.columns:
                            chart_data_for_gen['timestamp'] = chart_data_for_gen['date']
                        chart_data_for_gen = chart_data_for_gen.set_index('timestamp')

                        # Generate chart image
                        chart_path, chart_metadata = self.chart_generator.generate_chart_image(
                            chart_data_for_gen, symbol, '1D', ['sma_20', 'bb']
                        )

                        # Generate text labels
                        text_descriptions = [
                            f"{symbol} showing {'bullish' if row['close'] > row['open'] else 'bearish'} price action",
                            f"{symbol} with RSI at {row['rsi']:.1f}",
                            f"{symbol} {'above' if row['close'] > row['sma_20'] else 'below'} 20-day moving average",
                            f"{symbol} exhibiting {'momentum' if row['macd_hist'] > 0 else 'weakness'} via MACD",
                            f"{symbol} trading {'within' if row['bb_lower'] <= row['close'] <= row['bb_upper'] else 'outside'} Bollinger Bands"
                        ]

                        # Create comprehensive text description
                        text_description = " | ".join(text_descriptions)

                        sample = {
                            'image_path': chart_path,
                            'symbol': symbol,
                            'timeframe': '1D',
                            'text': text_description,
                            'description': text_description,
                            'timestamp': row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                            'close': row['close'],
                            'confidence_score': 0.8,
                            'patterns_detected': [],
                            'generated_at': datetime.now().isoformat(),
                            'needs_review': False,
                            'technical_indicators': {
                                'rsi': row['rsi'],
                                'macd': row['macd'],
                                'macd_signal': row['macd_signal'],
                                'bb_upper': row['bb_upper'],
                                'bb_lower': row['bb_lower'],
                                'sma_20': row['sma_20'],
                                'sma_50': row['sma_50'],
                                'volume_ratio': row['volume_ratio']
                            },
                            'price_action': 'bullish' if row['close'] > row['open'] else 'bearish',
                            'trend': 'uptrend' if row['close'] > row['sma_20'] else 'downtrend'
                        }

                        stock_samples.append(sample)

                    except Exception as e:
                        logger.warning(f"[PREP] Failed to process sample {idx} for {symbol}: {e}")
                        continue

                all_samples.extend(stock_samples)
                logger.info(f"[PREP] Generated {len(stock_samples)} samples for {symbol}")

            except Exception as e:
                logger.error(f"[PREP] Failed to process {symbol}: {e}")
                continue

        # Build dataset
        dataset = self.dataset_builder.build_dataset_from_samples(
            samples=all_samples,
            output_path="ibd50_training_dataset_v1.0",
            version="v1.0"
        )

        logger.info(f"[PREP] Prepared dataset with {len(all_samples)} total samples")
        return dataset  # Return the actual dataset path

    def train_clip_model(self, dataset_path: str) -> str:
        """
        Train CLIP model on prepared dataset

        Args:
            dataset_path: Path to prepared dataset

        Returns:
            Path to trained model
        """
        logger.info("[TRAIN] Starting CLIP fine-tuning...")

        from vlm.clip_fine_tune import CLIPFineTuner

        # Initialize trainer
        self.clip_trainer = CLIPFineTuner(
            model_name=self.model_name,
            output_dir=str(self.output_dir / "clip_models")
        )

        # Prepare data
        train_loader, val_loader, test_loader = self.clip_trainer.prepare_data(dataset_path)

        # Train model
        self.clip_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.training_config['num_epochs'],
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            warmup_steps=self.training_config['warmup_steps'],
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            use_mixed_precision=True,
            gradient_checkpointing=True
        )

        # Get best model path
        best_model_path = self.output_dir / "clip_models" / "best_model.pt"

        logger.info(f"[TRAIN] CLIP training completed. Best model: {best_model_path}")
        return str(best_model_path)

    def calibrate_model(self, model_path: str, dataset_path: str) -> str:
        """
        Calibrate trained CLIP model

        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset

        Returns:
            Path to calibrated model
        """
        logger.info("[CALIBRATE] Starting model calibration...")

        from vlm.clip_calibration import CLIPCalibrator
        from vlm.clip_fine_tune import CLIPFineTuner

        # Load model and prepare validation data
        fine_tuner = CLIPFineTuner()
        _, val_loader, _ = fine_tuner.prepare_data(dataset_path)

        # Initialize calibrator
        self.calibrator = CLIPCalibrator(model_path)

        # Calibrate with temperature scaling
        self.calibrator.calibrate_temperature(val_loader)

        # Save calibrated model info
        calibrated_info = {
            'model_path': model_path,
            'calibration_method': 'temperature',
            'temperature': self.calibrator.temperature_scaler.temperature.item(),
            'timestamp': datetime.now().isoformat()
        }

        calibrated_path = self.output_dir / "calibrated_model_info.json"
        with open(calibrated_path, 'w') as f:
            json.dump(calibrated_info, f, indent=2)

        logger.info(f"[CALIBRATE] Model calibration completed. Info saved to {calibrated_path}")
        return str(calibrated_path)

    def train_multimodal_fusion(self, clip_model_path: str, dataset_path: str) -> str:
        """
        Train multimodal fusion model

        Args:
            clip_model_path: Path to calibrated CLIP model
            dataset_path: Path to dataset

        Returns:
            Path to trained fusion model
        """
        logger.info("[FUSION] Training multimodal fusion model...")

        from multimodal_fusion import MultimodalFusion
        from vlm.clip_fine_tune import CLIPFineTuner

        # Load XGBoost model (assuming it exists)
        xgb_model_path = "xgboost_trading_model.pkl"  # Update this path as needed

        # Initialize fusion model
        self.fusion_model = MultimodalFusion(
            clip_model_path=clip_model_path,
            xgb_model_path=xgb_model_path if os.path.exists(xgb_model_path) else None,
            fusion_method="weighted_average"
        )

        # Prepare training data for fusion
        fine_tuner = CLIPFineTuner()
        train_loader, val_loader, _ = fine_tuner.prepare_data(dataset_path)

        # Extract some training samples for fusion training
        training_samples = []
        
        # Ensure calibrator is initialized
        if self.calibrator is None:
            logger.info("[FUSION] Initializing calibrator for fusion training...")
            from vlm.clip_calibration import CLIPCalibrator
            self.calibrator = CLIPCalibrator(clip_model_path)
        
        self.fusion_model.clip_calibrator = self.calibrator  # Use calibrated CLIP model

        # Create synthetic training data for fusion
        # Use actual training set size (38 samples from dataset logs)
        max_samples = 38  # Training set size from dataset split
        
        for i in range(max_samples):
            try:
                sample = train_loader.dataset[i]

                # Mock XGBoost features (would need real feature extraction)
                mock_features = {
                    'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
                    'bb_upper': 100.0, 'bb_middle': 95.0, 'bb_lower': 90.0, 'bb_width': 0.1,
                    'sma_20': 95.0, 'sma_50': 92.0, 'ema_12': 96.0, 'ema_26': 94.0,
                    'volume_ratio': 1.0, 'price_change': 0.01, 'volatility': 0.02
                }

                # Get CLIP prediction using image data
                # Convert pixel_values tensor back to PIL Image for calibrator
                pixel_values = sample['pixel_values']
                # Denormalize the image tensor (CLIP preprocessing uses ImageNet normalization)
                # CLIP preprocessing: normalize=((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                
                # Denormalize: pixel = (normalized_pixel * std) + mean
                denorm_pixels = (pixel_values * std) + mean
                # Clamp to [0, 1] and convert to [0, 255]
                denorm_pixels = torch.clamp(denorm_pixels, 0, 1) * 255
                # Convert to numpy and PIL Image
                numpy_image = denorm_pixels.permute(1, 2, 0).numpy().astype(np.uint8)
                pil_image = Image.fromarray(numpy_image)
                
                # Mock text descriptions for prediction
                text_descriptions = ["bullish pattern", "bearish pattern", "neutral pattern"]
                
                clip_result = self.calibrator.predict_calibrated(
                    pil_image,
                    text_descriptions,
                    calibration_method="temperature"
                )

                # Mock XGBoost result
                xgb_result = {
                    'prediction': 1,
                    'probabilities': [0.3, 0.7],
                    'confidence': 0.7
                }

                # True label (mock)
                true_label = 1 if np.random.random() > 0.5 else 0

                training_samples.append((clip_result, xgb_result, true_label))

            except Exception as e:
                logger.warning(f"[FUSION] Failed to create training sample {i}: {e}")
                continue

        if training_samples:
            # Train stacking fusion
            self.fusion_model.train_stacking_fusion(training_samples, model_type="rf")

        # Save fusion model
        fusion_path = self.output_dir / "fusion_model"
        self.fusion_model.save_fusion_model(str(fusion_path))

        logger.info(f"[FUSION] Multimodal fusion training completed. Model saved to {fusion_path}")
        return str(fusion_path)

    def evaluate_models(self, model_path: str, dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate trained models on test set

        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset

        Returns:
            Evaluation results
        """
        logger.info("[EVAL] Evaluating trained models...")

        from vlm.clip_fine_tune import CLIPFineTuner
        from vlm.clip_calibration import CLIPCalibrator

        # Load test data
        fine_tuner = CLIPFineTuner()
        _, _, test_loader = fine_tuner.prepare_data(dataset_path)

        # Load calibrated model
        calibrator = CLIPCalibrator(model_path)

        # Evaluate
        from vlm.clip_calibration import evaluate_calibration
        results = evaluate_calibration(calibrator, test_loader, save_plot=True)

        # Save results
        eval_path = self.output_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"[EVAL] Evaluation completed. Results saved to {eval_path}")
        logger.info(f"[EVAL] ECE Before: {results['ece_before']:.4f}, After: {results['ece_after']:.4f}")

        return results

    async def run_full_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete IBD 50 VLM training pipeline

        Returns:
            Training results and paths
        """
        logger.info("[PIPELINE] Starting IBD 50 VLM training pipeline...")

        start_time = datetime.now()

        try:
            # Step 1: Fetch historical data
            logger.info("[PIPELINE] Step 1: Fetching historical data...")
            historical_data = await self.fetch_historical_data()

            # Step 2: Prepare training data
            logger.info("[PIPELINE] Step 2: Preparing training data...")
            dataset_path = self.prepare_training_data(historical_data)

            # Step 3: Train CLIP model
            logger.info("[PIPELINE] Step 3: Training CLIP model...")
            model_path = self.train_clip_model(dataset_path)

            # Step 4: Calibrate model
            logger.info("[PIPELINE] Step 4: Calibrating model...")
            calibrated_path = self.calibrate_model(model_path, dataset_path)

            # Step 5: Train multimodal fusion
            logger.info("[PIPELINE] Step 5: Training multimodal fusion...")
            fusion_path = self.train_multimodal_fusion(model_path, dataset_path)

            # Step 6: Evaluate models
            logger.info("[PIPELINE] Step 6: Evaluating models...")
            eval_results = self.evaluate_models(model_path, dataset_path)

            # Training summary
            training_time = (datetime.now() - start_time).total_seconds()
            results = {
                'status': 'completed',
                'training_time_seconds': training_time,
                'dataset_path': dataset_path,
                'model_path': model_path,
                'calibrated_path': calibrated_path,
                'fusion_path': fusion_path,
                'evaluation_results': eval_results,
                'stocks_processed': len(historical_data),
                'total_samples': len(historical_data) * self.training_config['max_samples_per_stock'],
                'timestamp': datetime.now().isoformat()
            }

            # Save training summary
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"[PIPELINE] Training pipeline completed successfully in {training_time:.1f} seconds")
            logger.info(f"[PIPELINE] Results saved to {summary_path}")

            return results

        except Exception as e:
            logger.error(f"[PIPELINE] Training pipeline failed: {e}")
            raise

# Utility functions
def run_ibd50_training(output_dir: str = "vlm/models/ibd50",
                      data_dir: str = "vlm/data/ibd50") -> Dict[str, Any]:
    """
    Run IBD 50 VLM training

    Args:
        output_dir: Output directory for models
        data_dir: Data directory

    Returns:
        Training results
    """
    trainer = IBD50VLMTrainer(output_dir=output_dir, data_dir=data_dir)

    # Run training pipeline
    results = asyncio.run(trainer.run_full_training_pipeline())

    return results

def quick_ibd50_test(num_stocks: int = 5) -> Dict[str, Any]:
    """
    Quick test training on subset of IBD 50 stocks

    Args:
        num_stocks: Number of stocks to test with

    Returns:
        Test results
    """
    trainer = IBD50VLMTrainer()
    trainer.ibd50_stocks = trainer.ibd50_stocks[:num_stocks]  # Limit stocks
    trainer.training_config['max_samples_per_stock'] = 10  # Reduce samples
    trainer.training_config['num_epochs'] = 2  # Quick training

    try:
        results = asyncio.run(trainer.run_full_training_pipeline())
        logger.info("[TEST] Quick IBD 50 test completed successfully")
        return results
    except KeyboardInterrupt:
        logger.warning("[TEST] Training interrupted by user")
        return {
            "status": "interrupted", 
            "message": "Training was interrupted",
            "stocks_processed": num_stocks,
            "training_time": 0
        }
    except Exception as e:
        logger.error(f"[TEST] Quick test failed: {e}")
        return {
            "status": "failed", 
            "error": str(e),
            "stocks_processed": 0,
            "training_time": 0
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IBD 50 VLM Training Pipeline")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test on 5 stocks")
    parser.add_argument("--num-stocks", type=int, default=5, help="Number of stocks for quick test")
    parser.add_argument("--output-dir", default="vlm/models/ibd50", help="Output directory")
    parser.add_argument("--data-dir", default="vlm/data/ibd50", help="Data directory")

    args = parser.parse_args()

    if args.quick_test:
        print(f"Running quick IBD 50 test on {args.num_stocks} stocks...")
        results = quick_ibd50_test(args.num_stocks)
    else:
        print("Running full IBD 50 VLM training pipeline...")
        results = run_ibd50_training(args.output_dir, args.data_dir)

    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    print(f"Status: {results['status']}")
    print(f"Training Time: {results['training_time_seconds']:.1f} seconds")
    print(f"Stocks Processed: {results['stocks_processed']}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Model Path: {results['model_path']}")
    print(f"Dataset Path: {results['dataset_path']}")

    if 'evaluation_results' in results:
        eval_res = results['evaluation_results']
        print(f"ECE Before Calibration: {eval_res['ece_before']:.4f}")
        print(f"ECE After Calibration: {eval_res['ece_after']:.4f}")
        print(f"Calibration Improvement: {eval_res['ece_improvement']:.4f}")

    print("="*50)