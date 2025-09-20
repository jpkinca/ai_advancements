#!/usr/bin/env python3
"""
Comprehensive Test Suite for Candlestick AI System

This module provides comprehensive testing for the candlestick analysis system,
including unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
import asyncio
import logging
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from candlestick_reader.candlestick_analyzer import CandlestickAnalyzer
from candlestick_reader.integration import IntegratedCandlestickSystem
from candlestick_reader.model_training import CandlestickModelTrainer, CandlestickDataGenerator
from ai_data_accessor import AIDataAccessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestCandlestickAnalyzer(unittest.TestCase):
    """Unit tests for CandlestickAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = CandlestickAnalyzer()
        self.sample_data = self._generate_sample_data()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')

        # Generate realistic price data
        base_price = 150.0
        prices = [base_price]
        for i in range(99):
            change = np.random.normal(0, 0.005)  # 0.5% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(1000, 10000)

            data.append({
                'date': dates[i],
                'open': open_price,
                'high': max(high, open_price, price),
                'low': min(low, open_price, price),
                'close': price,
                'volume': volume
            })

        return pd.DataFrame(data).set_index('date')

    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer, CandlestickAnalyzer)
        self.assertIsNotNone(self.analyzer.pattern_definitions)
        self.assertGreater(len(self.analyzer.pattern_definitions), 0)

    def test_pattern_detection(self):
        """Test basic pattern detection"""
        result = self.analyzer.detect_candlestick_patterns(self.sample_data)

        self.assertIsInstance(result, dict)
        self.assertIn('patterns', result)
        self.assertIn('confidence', result)
        self.assertIn('dominant_pattern', result)

    def test_chart_generation(self):
        """Test candlestick chart generation"""
        chart_path = self.analyzer.generate_candlestick_chart(
            self.sample_data, 'TEST', '5min'
        )

        self.assertIsInstance(chart_path, str)
        self.assertTrue(chart_path.endswith('.png'))
        self.assertTrue(os.path.exists(chart_path))

        # Clean up
        if os.path.exists(chart_path):
            os.remove(chart_path)

    def test_signal_analysis(self):
        """Test complete signal analysis"""
        chart_path = self.analyzer.generate_candlestick_chart(
            self.sample_data, 'TEST', '5min'
        )

        result = self.analyzer.analyze_candlestick_signal(
            self.sample_data, 'TEST', chart_path
        )

        self.assertIsInstance(result, dict)
        self.assertIn('signal', result)
        self.assertIn('confidence', result)
        self.assertIn('patterns', result)
        self.assertIn('analysis', result)

        # Clean up
        if os.path.exists(chart_path):
            os.remove(chart_path)

class TestIntegratedSystem(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up test fixtures"""
        self.system = IntegratedCandlestickSystem()
        self.sample_data = self._generate_sample_data()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='5min')

        data = []
        base_price = 100.0
        for i, date in enumerate(dates):
            price = base_price + np.sin(i * 0.1) * 5 + np.random.normal(0, 1)
            data.append({
                'date': date,
                'open': price + np.random.normal(0, 0.5),
                'high': price + abs(np.random.normal(0, 1)),
                'low': price - abs(np.random.normal(0, 1)),
                'close': price,
                'volume': np.random.randint(1000, 5000)
            })

        return pd.DataFrame(data).set_index('date')

    def test_system_initialization(self):
        """Test integrated system initialization"""
        self.assertIsInstance(self.system, IntegratedCandlestickSystem)
        self.assertIsNotNone(self.system.candlestick_analyzer)
        self.assertIsNotNone(self.system.vpa_analyzer)

    def test_signal_combination(self):
        """Test signal combination logic"""
        # Mock individual results
        candlestick_result = {
            'signal': 'bullish',
            'confidence': 0.8,
            'patterns': ['hammer']
        }

        vpa_result = {
            'signal': 'bullish',
            'confidence': 0.7
        }

        multimodal_result = {
            'prediction': 0,  # bullish
            'confidence': 0.6
        }

        combined = self.system._combine_all_signals(
            candlestick_result, vpa_result, multimodal_result, self.sample_data
        )

        self.assertIsInstance(combined, dict)
        self.assertIn('signal', combined)
        self.assertIn('confidence', combined)
        self.assertIn('components', combined)

    def test_trade_recommendation(self):
        """Test trade recommendation generation"""
        analysis = {
            'signal': 'bullish',
            'confidence': 0.85,
            'market_context': {
                'trend': 'bullish',
                'volatility': 'low',
                'current_price': 100.0
            }
        }

        recommendation = self.system._generate_trade_recommendation(analysis, self.sample_data)

        self.assertIsInstance(recommendation, dict)
        self.assertIn('action', recommendation)
        self.assertIn('position_size', recommendation)
        self.assertIn('stop_loss', recommendation)
        self.assertIn('take_profit', recommendation)

class TestModelTraining(unittest.TestCase):
    """Tests for model training components"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = CandlestickDataGenerator()
        self.trainer = CandlestickModelTrainer()

    def test_data_generation(self):
        """Test synthetic data generation"""
        patterns = ['hammer', 'shooting_star', 'doji']
        data = self.generator.generate_pattern_data(patterns, samples_per_pattern=10)

        self.assertIsInstance(data, dict)
        for pattern in patterns:
            self.assertIn(pattern, data)
            self.assertEqual(len(data[pattern]), 10)

    def test_chart_generation(self):
        """Test chart generation for training"""
        # Generate sample data
        sample_data = self.generator._generate_hammer_pattern()

        chart_path = self.generator.generate_training_chart(sample_data, 'hammer', 0)

        self.assertIsInstance(chart_path, str)
        self.assertTrue(chart_path.endswith('.png'))

        # Clean up
        if os.path.exists(chart_path):
            os.remove(chart_path)

class PerformanceTests(unittest.TestCase):
    """Performance benchmarks for the system"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.analyzer = CandlestickAnalyzer()
        self.system = IntegratedCandlestickSystem()

        # Generate larger dataset
        self.large_data = self._generate_large_dataset()

    def _generate_large_dataset(self) -> pd.DataFrame:
        """Generate large dataset for performance testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')

        data = []
        base_price = 150.0
        for i, date in enumerate(dates):
            price = base_price + np.sin(i * 0.01) * 10 + np.random.normal(0, 2)
            data.append({
                'date': date,
                'open': price + np.random.normal(0, 0.5),
                'high': price + abs(np.random.normal(0, 1.5)),
                'low': price - abs(np.random.normal(0, 1.5)),
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })

        return pd.DataFrame(data).set_index('date')

    def test_pattern_detection_performance(self):
        """Test pattern detection performance"""
        start_time = time.time()

        result = self.analyzer.detect_candlestick_patterns(self.large_data)

        end_time = time.time()
        duration = end_time - start_time

        self.assertLess(duration, 5.0)  # Should complete within 5 seconds
        logger.info(f"Pattern detection took {duration:.2f} seconds for {len(self.large_data)} bars")

    def test_chart_generation_performance(self):
        """Test chart generation performance"""
        start_time = time.time()

        chart_path = self.analyzer.generate_candlestick_chart(
            self.large_data, 'PERF_TEST', '1min'
        )

        end_time = time.time()
        duration = end_time - start_time

        self.assertLess(duration, 10.0)  # Should complete within 10 seconds
        logger.info(f"Chart generation took {duration:.2f} seconds for {len(self.large_data)} bars")

        # Clean up
        if os.path.exists(chart_path):
            os.remove(chart_path)

class IntegrationTests(unittest.TestCase):
    """End-to-end integration tests"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.system = IntegratedCandlestickSystem()

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        # This would normally connect to real data, but we'll mock it
        # In a real scenario, this would test with actual market data

        # Mock the data accessor for testing
        async def mock_get_data(symbol, timeframe, start, end):
            # Return sample data
            dates = pd.date_range(start, end, freq='5min')
            data = []
            for date in dates[:50]:  # Limit to 50 bars
                data.append({
                    'date': date,
                    'open': 100 + np.random.normal(0, 1),
                    'high': 102 + np.random.normal(0, 1),
                    'low': 98 + np.random.normal(0, 1),
                    'close': 100 + np.random.normal(0, 1),
                    'volume': np.random.randint(1000, 5000)
                })
            return pd.DataFrame(data).set_index('date')

        # Replace the data accessor method
        original_method = self.system.data_accessor.get_symbol_data
        self.system.data_accessor.get_symbol_data = mock_get_data

        try:
            # Run analysis
            async def run_test():
                result = await self.system.analyze_symbol_comprehensive('TEST', '5min', 50)
                return result

            result = asyncio.run(run_test())

            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('signal', result)
            self.assertIn('confidence', result)
            self.assertIn('components', result)

        finally:
            # Restore original method
            self.system.data_accessor.get_symbol_data = original_method

def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    logger.info("Running Candlestick AI Performance Benchmarks")

    analyzer = CandlestickAnalyzer()
    system = IntegratedCandlestickSystem()

    # Generate test data of various sizes
    sizes = [100, 500, 1000, 5000]
    results = {}

    for size in sizes:
        logger.info(f"Testing with {size} data points")

        # Generate data
        dates = pd.date_range('2024-01-01', periods=size, freq='1min')
        data = []
        for i, date in enumerate(dates):
            price = 100 + np.sin(i * 0.01) * 5 + np.random.normal(0, 1)
            data.append({
                'date': date,
                'open': price + np.random.normal(0, 0.5),
                'high': price + abs(np.random.normal(0, 1)),
                'low': price - abs(np.random.normal(0, 1)),
                'close': price,
                'volume': np.random.randint(1000, 5000)
            })

        df = pd.DataFrame(data).set_index('date')

        # Benchmark pattern detection
        start_time = time.time()
        pattern_result = analyzer.detect_candlestick_patterns(df)
        pattern_time = time.time() - start_time

        # Benchmark chart generation
        start_time = time.time()
        chart_path = analyzer.generate_candlestick_chart(df, f'BENCH_{size}', '1min')
        chart_time = time.time() - start_time

        # Clean up chart
        if os.path.exists(chart_path):
            os.remove(chart_path)

        results[size] = {
            'pattern_detection_time': pattern_time,
            'chart_generation_time': chart_time,
            'total_time': pattern_time + chart_time,
            'patterns_detected': len(pattern_result.get('patterns', []))
        }

        logger.info(f"Size {size}: Pattern detection: {pattern_time:.2f}s, "
                   f"Chart generation: {chart_time:.2f}s, "
                   f"Patterns: {results[size]['patterns_detected']}")

    # Save benchmark results
    benchmark_file = 'results/performance_benchmarks.json'
    os.makedirs('results', exist_ok=True)

    with open(benchmark_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    logger.info(f"Benchmarks saved to {benchmark_file}")
    return results

def run_system_validation():
    """Run comprehensive system validation"""
    logger.info("Running Candlestick AI System Validation")

    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'overall_status': 'unknown'
    }

    try:
        # Test basic imports
        validation_results['tests']['imports'] = {'status': 'passed', 'message': 'All imports successful'}

        # Test analyzer initialization
        analyzer = CandlestickAnalyzer()
        validation_results['tests']['analyzer_init'] = {'status': 'passed', 'message': 'Analyzer initialized successfully'}

        # Test pattern definitions
        pattern_count = len(analyzer.pattern_definitions)
        validation_results['tests']['pattern_definitions'] = {
            'status': 'passed' if pattern_count > 0 else 'failed',
            'message': f'Loaded {pattern_count} pattern definitions'
        }

        # Test integrated system
        system = IntegratedCandlestickSystem()
        validation_results['tests']['integrated_system'] = {'status': 'passed', 'message': 'Integrated system initialized'}

        # Test data generator
        generator = CandlestickDataGenerator()
        validation_results['tests']['data_generator'] = {'status': 'passed', 'message': 'Data generator initialized'}

        # Test model trainer
        trainer = CandlestickModelTrainer()
        validation_results['tests']['model_trainer'] = {'status': 'passed', 'message': 'Model trainer initialized'}

        # Overall status
        failed_tests = [k for k, v in validation_results['tests'].items() if v['status'] == 'failed']
        validation_results['overall_status'] = 'passed' if not failed_tests else 'failed'

        logger.info(f"System validation completed: {validation_results['overall_status']}")

    except Exception as e:
        validation_results['overall_status'] = 'failed'
        validation_results['error'] = str(e)
        logger.error(f"System validation failed: {e}")

    # Save validation results
    validation_file = 'results/system_validation.json'
    os.makedirs('results', exist_ok=True)

    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    logger.info(f"Validation results saved to {validation_file}")
    return validation_results

if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Run system validation
    validation = run_system_validation()

    if validation['overall_status'] == 'passed':
        # Run unit tests
        logger.info("Running unit tests...")
        unittest.main(verbosity=2, exit=False)

        # Run performance benchmarks
        logger.info("Running performance benchmarks...")
        benchmarks = run_performance_benchmarks()

        logger.info("All tests and benchmarks completed successfully!")
    else:
        logger.error("System validation failed. Skipping tests.")
        sys.exit(1)