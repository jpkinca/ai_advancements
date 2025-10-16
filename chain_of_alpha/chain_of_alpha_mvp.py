"""
Chain-of-Alpha MVP: AI-Driven Alpha Factor Discovery Framework

This MVP implements the Chain-of-Alpha framework for automated alpha factor generation
and optimization using large language models and quantitative backtesting.

Architecture:
1. Data Acquisition → 2. Factor Generation → 3. Backtesting → 4. Optimization → 5. Evaluation

Based on: https://arxiv.org/abs/2508.06312
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import warnings

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))

from src.data_acquisition import DataAcquisition
from src.llm_interface import LLMInterface
from src.factor_generation import FactorGenerationChain
from src.backtesting_engine import BacktestingEngine
from src.factor_optimization import FactorOptimizationChain
from src.evaluation_export import EvaluationExportModule

class ChainOfAlphaMVP:
    """
    Main orchestrator for the Chain-of-Alpha MVP framework
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Chain-of-Alpha MVP"""
        self.config = config or self._get_default_config()
        self.components = {}

        logger.info("Initializing Chain-of-Alpha MVP...")

        # Initialize components
        self._initialize_components()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        # Import configuration from config.py
        try:
            from config import CONFIG
            return CONFIG
        except ImportError:
            logger.warning("Could not import config.py, using fallback configuration")
            
        return {
            # Data settings
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC'],
            'start_date': '2019-01-01',
            'end_date': '2024-01-01',

            # LLM settings - using mock by default for easy testing
            'llm_model': 'mock',  # Change to 'llama-3-8b', 'grok', or 'openai' when ready
            'llm_api_key': None,  # Set for API models
            'huggingface_token': None,  # Set for Llama models
            'temperature': 0.7,
            'max_tokens': 1000,

            # Generation settings
            'num_candidate_factors': 20,
            'optimization_iterations': 3,
            'top_factors_to_keep': 5,

            # Backtesting settings
            'initial_capital': 100000,
            'commission': 0.001,
            'factor_threshold': 0.0,  # Signal threshold for long/short

            # Output settings
            'output_dir': 'outputs',
            'export_csv': True,
            'export_pine_script': True,
        }

    def _initialize_components(self):
        """Initialize all framework components"""
        try:
            # Data acquisition
            self.components['data'] = DataAcquisition(self.config)

            # LLM interface
            self.components['llm'] = LLMInterface(self.config)

            # Factor generation chain
            self.components['generation'] = FactorGenerationChain(self.config, self.components['llm'])

            # Backtesting engine
            self.components['backtesting'] = BacktestingEngine(self.config)

            # Factor optimization chain
            self.components['optimization'] = FactorOptimizationChain(self.config, self.components['llm'])

            # Evaluation and export
            self.components['evaluation'] = EvaluationExportModule(self.config)

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def run_mvp(self) -> Dict[str, Any]:
        """
        Run the complete Chain-of-Alpha MVP pipeline

        Returns:
            Dict containing results, metrics, and exported files
        """
        logger.info("Starting Chain-of-Alpha MVP execution...")

        start_time = datetime.now()
        results = {}

        try:
            # Step 1: Data Acquisition
            logger.info("Step 1: Acquiring market data...")
            market_data = self.components['data'].fetch_data()
            results['market_data'] = market_data
            logger.info(f"✓ Acquired data for {len(market_data)} stocks")

            # Step 2: Factor Generation
            logger.info("Step 2: Generating candidate factors...")
            candidate_factors = self.components['generation'].generate_factors(market_data, self.config.get('num_candidate_factors', 10))
            
            # Evaluate the factors to compute their values
            logger.info("Step 2b: Evaluating generated factors...")
            candidate_factors = self.components['generation'].evaluate_factors(candidate_factors, market_data)
            
            # Debug: Check evaluated factors
            logger.info(f"DEBUG: Evaluated factors count: {len(candidate_factors)}")
            for i, factor in enumerate(candidate_factors):
                factor_values = factor.get('values')
                logger.info(f"DEBUG: Factor {i+1}: ID={factor.get('id', 'N/A')}, expression='{factor.get('expression', 'N/A')}'")
                logger.info(f"DEBUG: Factor {i+1}: values_type={type(factor_values)}, is_series={isinstance(factor_values, pd.Series)}")
                if isinstance(factor_values, pd.Series):
                    logger.info(f"DEBUG: Factor {i+1}: values_shape={factor_values.shape}, values_sample={factor_values.head(3).tolist()}")
                else:
                    logger.info(f"DEBUG: Factor {i+1}: values={factor_values}")
            
            results['candidate_factors'] = candidate_factors
            logger.info(f"✓ Generated {len(candidate_factors)} candidate factors")

            # Step 3: Initial Backtesting
            logger.info("Step 3: Running initial backtests...")
            initial_results = self.components['backtesting'].backtest_factors(
                candidate_factors, market_data
            )
            results['initial_backtest'] = initial_results
            logger.info("✓ Completed initial backtesting")

            # Step 4: Factor Optimization
            logger.info("Step 4: Optimizing factors...")
            optimized_factors = self.components['optimization'].optimize_factors(
                initial_results, market_data, self.config.get('optimization_iterations', 3)
            )
            results['optimized_factors'] = optimized_factors
            logger.info(f"✓ Optimized to {len(optimized_factors)} top factors")

            # Step 5: Final Evaluation and Export
            logger.info("Step 5: Evaluating and exporting results...")
            evaluation = self.components['evaluation'].evaluate_pipeline_results(optimized_factors)
            self.components['evaluation'].export_results(optimized_factors, evaluation, ['json', 'html', 'csv'])
            results['final_evaluation'] = evaluation
            logger.info("✓ Evaluation and export completed")

            # Calculate execution time
            execution_time = datetime.now() - start_time
            results['execution_time'] = execution_time.total_seconds()
            results['execution_time_formatted'] = str(execution_time)

            logger.info(f"🎉 MVP execution completed successfully in {execution_time}")

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            logger.error(f"MVP execution failed: {e}")
            raise

    def _print_summary(self, results: Dict[str, Any]):
        """Print execution summary"""
        print("\n" + "="*60)
        print("CHAIN-OF-ALPHA MVP EXECUTION SUMMARY")
        print("="*60)

        # Performance metrics
        if 'final_evaluation' in results and 'metrics' in results['final_evaluation']:
            metrics = results['final_evaluation']['metrics']
            print(f"Top Factor Sharpe Ratio: {metrics.get('best_sharpe', 'N/A'):.3f}")
            print(f"Average Sharpe Ratio: {metrics.get('avg_sharpe', 'N/A'):.3f}")
            print(f"Top Factor Total Return: {metrics.get('best_return', 'N/A'):.2%}")
            print(f"Factors Generated: {len(results.get('candidate_factors', []))}")
            print(f"Factors Optimized: {len(results.get('optimized_factors', []))}")

        # Execution time
        if 'execution_time_formatted' in results:
            print(f"Execution Time: {results['execution_time_formatted']}")

        # Output files
        if 'final_evaluation' in results and 'exported_files' in results['final_evaluation']:
            files = results['final_evaluation']['exported_files']
            print(f"Exported Files: {len(files)}")
            for file in files:
                print(f"  - {file}")

        print("="*60)

def main():
    """Main entry point for the Chain-of-Alpha MVP"""
    print("Chain-of-Alpha MVP: AI-Driven Alpha Factor Discovery")
    print("="*60)

    # Initialize MVP
    try:
        mvp = ChainOfAlphaMVP()

        # Run the complete pipeline
        results = mvp.run_mvp()

        print("\n✅ MVP completed successfully!")
        print("Check the outputs directory for results and exports.")

    except Exception as e:
        logger.error(f"MVP failed: {e}")
        print(f"\n❌ MVP execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()