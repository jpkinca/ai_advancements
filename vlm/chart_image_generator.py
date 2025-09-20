#!/usr/bin/env python3
"""
Chart Image Generator for VLM Pipeline

This module generates deterministic chart images from OHLCV data for Vision-Language Model training and inference.
Supports candlestick charts with technical indicator overlays and metadata embedding.

Features:
- Deterministic rendering for reproducible images
- Multi-timeframe support
- Technical indicator overlays (SMA, EMA, RSI, MACD, Bollinger Bands)
- Metadata embedding in PNG files
- Batch processing capabilities
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartImageGenerator:
    """
    Generates deterministic chart images with technical overlays for VLM training
    """

    def __init__(self,
                 output_dir: str = 'vlm/charts',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 100):
        """
        Initialize the chart generator

        Args:
            output_dir: Directory to save generated charts
            figsize: Figure size in inches (width, height)
            dpi: Resolution for saved images
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set random seed for deterministic rendering
        np.random.seed(42)
        plt.style.use('default')

        logger.info(f"[INIT] Chart Image Generator (output: {output_dir}, size: {figsize}, dpi: {dpi})")

    def generate_chart_image(self,
                           ohlcv_data: pd.DataFrame,
                           symbol: str,
                           timeframe: str,
                           overlays: Optional[List[str]] = None,
                           save_metadata: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a single chart image with optional overlays

        Args:
            ohlcv_data: OHLCV data with datetime index
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1D', '1H')
            overlays: List of technical indicators to overlay
            save_metadata: Whether to embed metadata in PNG

        Returns:
            Tuple of (image_path, metadata_dict)
        """
        try:
            # Prepare data
            df = ohlcv_data.copy()
            df = df.reset_index()
            df['timestamp'] = df['timestamp'].map(mdates.date2num)

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                         gridspec_kw={'height_ratios': [3, 1]})

            # Main price chart
            self._plot_candlesticks(ax1, df)

            # Volume subplot
            self._plot_volume(ax2, df)

            # Add overlays
            if overlays:
                self._add_overlays(ax1, df, overlays)

            # Formatting
            self._format_chart(ax1, ax2, symbol, timeframe)

            # Generate deterministic filename
            chart_hash = self._generate_chart_hash(df, symbol, timeframe, overlays)
            filename = f"{symbol}_{timeframe}_{chart_hash}.png"
            image_path = os.path.join(self.output_dir, filename)

            # Save image
            plt.savefig(image_path, dpi=self.dpi, bbox_inches='tight')

            # Generate metadata
            metadata = self._generate_metadata(df, symbol, timeframe, overlays, chart_hash)

            # Embed metadata if requested
            if save_metadata:
                self._embed_metadata(image_path, metadata)

            plt.close(fig)

            logger.info(f"[SUCCESS] Generated chart: {image_path}")
            return image_path, metadata

        except Exception as e:
            logger.error(f"[ERROR] Failed to generate chart: {e}")
            plt.close('all')
            raise

    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart"""
        ohlc = df[['timestamp', 'open', 'high', 'low', 'close']].values
        candlestick_ohlc(ax, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume bars"""
        colors = ['green' if close >= open else 'red'
                 for open, close in zip(df['open'], df['close'])]
        ax.bar(df['timestamp'], df['volume'], color=colors, alpha=0.6, width=0.6)
        ax.set_ylabel('Volume')

    def _add_overlays(self, ax, df: pd.DataFrame, overlays: List[str]):
        """Add technical indicator overlays"""
        for overlay in overlays:
            if overlay.startswith('sma_'):
                period = int(overlay.split('_')[1])
                sma = df['close'].rolling(window=period).mean()
                ax.plot(df['timestamp'], sma, label=f'SMA {period}', linewidth=1.5)
            elif overlay.startswith('ema_'):
                period = int(overlay.split('_')[1])
                ema = df['close'].ewm(span=period).mean()
                ax.plot(df['timestamp'], ema, label=f'EMA {period}', linewidth=1.5)
            elif overlay == 'bb':
                self._add_bollinger_bands(ax, df)
            elif overlay == 'rsi':
                # RSI would need separate subplot, skip for now
                pass
            elif overlay == 'macd':
                # MACD would need separate subplot, skip for now
                pass

    def _add_bollinger_bands(self, ax, df: pd.DataFrame):
        """Add Bollinger Bands overlay"""
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        ax.plot(df['timestamp'], upper, 'r--', alpha=0.7, label='BB Upper')
        ax.plot(df['timestamp'], sma, 'b--', alpha=0.7, label='BB Middle')
        ax.plot(df['timestamp'], lower, 'r--', alpha=0.7, label='BB Lower')
        ax.fill_between(df['timestamp'], upper, lower, alpha=0.1, color='blue')

    def _format_chart(self, ax1, ax2, symbol: str, timeframe: str):
        """Format chart appearance"""
        ax1.set_title(f'{symbol} - {timeframe} Chart')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Link x-axes
        ax2.sharex(ax1)
        ax2.grid(True, alpha=0.3)

    def _generate_chart_hash(self, df: pd.DataFrame, symbol: str, timeframe: str, overlays: Optional[List[str]]) -> str:
        """Generate deterministic hash for chart filename"""
        # Include key data points and parameters
        hash_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'overlays': sorted(overlays or []),
            'first_timestamp': str(df['timestamp'].iloc[0]) if not df.empty else '',
            'last_timestamp': str(df['timestamp'].iloc[-1]) if not df.empty else '',
            'data_points': len(df),
            'last_close': float(df['close'].iloc[-1]) if not df.empty else 0.0
        }

        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]

    def _generate_metadata(self, df: pd.DataFrame, symbol: str, timeframe: str,
                          overlays: Optional[List[str]], chart_hash: str) -> Dict[str, Any]:
        """Generate metadata dictionary"""
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'overlays': overlays or [],
            'data_points': len(df),
            'date_range': {
                'start': str(df.index[0]) if not df.empty else None,
                'end': str(df.index[-1]) if not df.empty else None
            },
            'price_range': {
                'high': float(df['high'].max()) if not df.empty else None,
                'low': float(df['low'].min()) if not df.empty else None,
                'close': float(df['close'].iloc[-1]) if not df.empty else None
            },
            'volume_stats': {
                'total': float(df['volume'].sum()) if not df.empty else None,
                'average': float(df['volume'].mean()) if not df.empty else None
            },
            'chart_hash': chart_hash,
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0'
        }

    def _embed_metadata(self, image_path: str, metadata: Dict[str, Any]):
        """Embed metadata in PNG file (as text chunk)"""
        try:
            from PIL import Image
            import piexif

            # Load image
            img = Image.open(image_path)

            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata, indent=2)

            # Create EXIF data with metadata
            exif_dict = {"Exif": {piexif.ExifIFD.UserComment: metadata_json.encode('utf-8')}}
            exif_bytes = piexif.dump(exif_dict)

            # Save with EXIF
            img.save(image_path, exif=exif_bytes)
            img.close()

        except ImportError:
            logger.warning("[WARNING] PIL/piexif not available, skipping metadata embedding")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to embed metadata: {e}")

    def batch_generate_charts(self,
                            data_dict: Dict[str, pd.DataFrame],
                            symbols: List[str],
                            timeframes: List[str],
                            overlays_list: Optional[List[List[str]]] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate multiple charts in batch

        Args:
            data_dict: Dictionary of {symbol_timeframe: ohlcv_data}
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            overlays_list: List of overlay configurations

        Returns:
            List of (image_path, metadata) tuples
        """
        results = []
        overlays_list = overlays_list or [None] * len(symbols)

        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                if key not in data_dict:
                    logger.warning(f"[WARNING] No data for {key}")
                    continue

                overlays = overlays_list[min(len(results), len(overlays_list) - 1)]
                try:
                    image_path, metadata = self.generate_chart_image(
                        data_dict[key], symbol, timeframe, overlays
                    )
                    results.append((image_path, metadata))
                except Exception as e:
                    logger.error(f"[ERROR] Failed to generate {key}: {e}")

        logger.info(f"[BATCH] Generated {len(results)} charts")
        return results

# Utility functions
def create_sample_data(symbol: str = 'AAPL', days: int = 30) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

    # Generate realistic price data
    base_price = 150.0
    prices = [base_price]
    for _ in range(days - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Floor at $1

    # Create OHLCV
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = int(np.random.normal(1000000, 200000))

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(data).set_index('timestamp')

def example_usage():
    """Example usage of the chart generator"""
    generator = ChartImageGenerator()

    # Create sample data
    sample_data = create_sample_data('AAPL', 50)

    # Generate chart with overlays
    overlays = ['sma_20', 'ema_10', 'bb']
    image_path, metadata = generator.generate_chart_image(
        sample_data, 'AAPL', '1D', overlays
    )

    print(f"Generated chart: {image_path}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

if __name__ == "__main__":
    example_usage()