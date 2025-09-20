"""
Factor Optimization Chain Module for Chain-of-Alpha MVP

Implements the factor optimization chain that improves factors based on backtest results
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import re
from datetime import datetime

from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class FactorOptimizationChain:
    """
    Chain for optimizing alpha factors based on backtest performance
    """

    def __init__(self, config: Dict[str, Any], llm_interface: LLMInterface):
        self.config = config
        self.llm = llm_interface
        self.optimization_history = []

    def optimize_factors(self, factors: List[Dict[str, Any]], market_data: pd.DataFrame,
                        max_iterations: int = 3) -> List[Dict[str, Any]]:
        """
        Optimize factors through iterative improvement

        Args:
            factors: List of factors with backtest results
            market_data: Market data for re-evaluation
            max_iterations: Maximum optimization iterations per factor

        Returns:
            Optimized factors
        """
        logger.info(f"Optimizing {len(factors)} factors with max {max_iterations} iterations")

        optimized_factors = []

        for factor in factors:
            try:
                optimized = self._optimize_single_factor(factor, market_data, max_iterations)
                if optimized:
                    optimized_factors.append(optimized)

            except Exception as e:
                logger.error(f"Optimization failed for factor {factor['id']}: {e}")
                # Keep original factor if optimization fails
                optimized_factors.append(factor)

        logger.info(f"Successfully optimized {len(optimized_factors)} factors")
        return optimized_factors

    def _optimize_single_factor(self, factor: Dict[str, Any], market_data: pd.DataFrame,
                               max_iterations: int) -> Optional[Dict[str, Any]]:
        """Optimize a single factor through iterative refinement"""

        current_factor = factor.copy()
        best_factor = factor.copy()
        best_score = self._calculate_factor_score(factor)

        logger.info(f"Optimizing factor {factor['id']} (initial score: {best_score:.3f})")

        iteration = 0
        for iteration in range(max_iterations):
            try:
                # Generate optimization suggestions
                suggestions = self._generate_optimization_suggestions(current_factor)

                if not suggestions:
                    logger.info(f"No optimization suggestions for factor {factor['id']}")
                    break

                # Apply optimizations
                optimized_versions = self._apply_optimizations(current_factor, suggestions, market_data)

                if not optimized_versions:
                    logger.info(f"No valid optimizations generated for factor {factor['id']}")
                    break

                # Evaluate optimized versions
                best_optimized = self._evaluate_optimized_versions(optimized_versions, market_data)

                if best_optimized:
                    current_score = self._calculate_factor_score(best_optimized)

                    if current_score > best_score:
                        best_factor = best_optimized
                        best_score = current_score
                        current_factor = best_optimized
                        logger.info(f"Iteration {iteration+1}: Improved score to {best_score:.3f}")
                    else:
                        logger.info(f"Iteration {iteration+1}: No improvement (score: {current_score:.3f})")
                        break
                else:
                    break

            except Exception as e:
                logger.error(f"Error in optimization iteration {iteration+1}: {e}")
                break

        # Record optimization history
        self.optimization_history.append({
            'original_factor': factor,
            'optimized_factor': best_factor,
            'improvement': best_score - self._calculate_factor_score(factor),
            'iterations': iteration + 1
        })

        return best_factor

    def _calculate_factor_score(self, factor: Dict[str, Any]) -> float:
        """Calculate a composite score for factor performance"""

        try:
            metrics = factor.get('backtest_results', {}).get('metrics', {})

            if not metrics:
                return 0.0

            # Weighted combination of key metrics
            sharpe = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 1)
            max_dd = abs(metrics.get('max_drawdown', 0))  # Absolute value for penalty

            # Normalize and weight
            score = (
                0.4 * sharpe +  # Sharpe ratio (higher is better)
                0.2 * win_rate +  # Win rate (higher is better)
                0.2 * min(profit_factor, 3) / 3 +  # Profit factor (capped at 3)
                0.2 * (1 - min(max_dd, 0.5) / 0.5)  # Max drawdown penalty (lower is better)
            )

            return max(0, score)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error calculating factor score: {e}")
            return 0.0

    def _generate_optimization_suggestions(self, factor: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions using LLM"""

        try:
            metrics = factor.get('backtest_results', {}).get('metrics', {})

            prompt = f"""
Analyze this alpha factor's backtest performance and suggest specific improvements:

Factor: {factor['expression']}
Explanation: {factor.get('explanation', 'N/A')}

Performance Metrics:
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
- Annual Return: {metrics.get('annual_return', 'N/A')}
- Max Drawdown: {metrics.get('max_drawdown', 'N/A')}
- Win Rate: {metrics.get('win_rate', 'N/A')}
- Profit Factor: {metrics.get('profit_factor', 'N/A')}
- Total Trades: {metrics.get('total_trades', 'N/A')}

Please suggest 3-5 specific modifications to improve this factor. Focus on:
1. Adding risk management (volatility scaling, position sizing)
2. Including trend filters or market regime detection
3. Adding volume confirmation or liquidity filters
4. Implementing decay or momentum confirmation
5. Combining with other technical indicators

Format each suggestion as a specific code modification.
"""

            response = self.llm.generate_response(prompt)

            # Parse suggestions
            suggestions = self._parse_optimization_suggestions(response)

            return suggestions

        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return []

    def _parse_optimization_suggestions(self, response: str) -> List[str]:
        """Parse optimization suggestions from LLM response"""

        suggestions = []

        try:
            # Split by numbered items or bullet points
            lines = response.split('\n')
            current_suggestion = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this starts a new suggestion
                if re.match(r'^\d+\.|\*|-', line):
                    if current_suggestion:
                        suggestions.append(current_suggestion.strip())
                    current_suggestion = line
                else:
                    current_suggestion += " " + line

            # Add the last suggestion
            if current_suggestion:
                suggestions.append(current_suggestion.strip())

            # Clean up suggestions
            cleaned_suggestions = []
            for suggestion in suggestions:
                # Remove numbering/bullets
                clean_suggestion = re.sub(r'^\d+\.|\*|-', '', suggestion).strip()
                if len(clean_suggestion) > 10:  # Minimum length filter
                    cleaned_suggestions.append(clean_suggestion)

            return cleaned_suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            logger.error(f"Error parsing optimization suggestions: {e}")
            return []

    def _apply_optimizations(self, factor: Dict[str, Any], suggestions: List[str],
                           market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Apply optimization suggestions to create new factor versions"""

        optimized_factors = []

        for suggestion in suggestions:
            try:
                # Generate optimized expression based on suggestion
                optimized_expression = self._generate_optimized_expression(factor['expression'], suggestion)

                if optimized_expression and optimized_expression != factor['expression']:
                    # Validate the new expression
                    if self.llm.validate_factor_expression(optimized_expression):
                        # Create optimized factor
                        optimized_factor = factor.copy()
                        optimized_factor.update({
                            'expression': optimized_expression,
                            'optimization_suggestion': suggestion,
                            'parent_factor_id': factor['id'],
                            'id': f"{factor['id']}_opt_{len(optimized_factors) + 1}",
                            'optimization_time': datetime.now().isoformat()
                        })

                        optimized_factors.append(optimized_factor)

            except Exception as e:
                logger.error(f"Error applying optimization '{suggestion}': {e}")
                continue

        return optimized_factors

    def _generate_optimized_expression(self, original_expression: str, suggestion: str) -> Optional[str]:
        """Generate optimized expression based on suggestion"""

        try:
            # Common optimization patterns
            suggestion_lower = suggestion.lower()

            if 'volatility' in suggestion_lower and 'scal' in suggestion_lower:
                return f"({original_expression}) * (1 / df['volatility_20'])"

            elif 'volume' in suggestion_lower and 'confirm' in suggestion_lower:
                return f"({original_expression}) * df['volume_ratio']"

            elif 'trend' in suggestion_lower and 'filter' in suggestion_lower:
                return f"({original_expression}) * (df['sma_20'] / df['sma_50'] - 1)"

            elif 'decay' in suggestion_lower:
                return f"({original_expression}) * df['returns'].ewm(span=10).mean()"

            elif 'momentum' in suggestion_lower and 'confirm' in suggestion_lower:
                return f"({original_expression}) * df['momentum_5']"

            elif 'combine' in suggestion_lower and 'rsi' in suggestion_lower:
                return f"({original_expression}) * (df['rsi'] - 50) / 50"

            elif 'combine' in suggestion_lower and 'macd' in suggestion_lower:
                return f"({original_expression}) * df['macd']"

            else:
                # Use LLM to generate the optimized expression
                prompt = f"""
Original factor: {original_expression}

Optimization suggestion: {suggestion}

Generate a new factor expression that implements this suggestion.
Return only the Python expression, no explanation.
"""
                response = self.llm.generate_response(prompt)
                optimized = response.strip()

                # Clean up response
                optimized = re.sub(r'^```python\s*|\s*```$', '', optimized)
                optimized = optimized.strip()

                return optimized if optimized else None

        except Exception as e:
            logger.error(f"Error generating optimized expression: {e}")
            return None

    def _evaluate_optimized_versions(self, optimized_factors: List[Dict[str, Any]],
                                   market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Evaluate optimized factor versions and return the best one"""

        try:
            if not optimized_factors:
                return None

            # Simple evaluation based on factor statistics
            # In a full implementation, this would run backtests
            best_factor = None
            best_score = -1

            for factor in optimized_factors:
                try:
                    # Compute factor values
                    from .factor_generation import FactorGenerationChain
                    temp_chain = FactorGenerationChain({}, self.llm)
                    factor_values = temp_chain._compute_factor(market_data.copy(), factor['expression'])

                    if factor_values is not None:
                        # Calculate basic score based on factor properties
                        stats = temp_chain._calculate_factor_stats(factor_values)

                        # Simple scoring based on signal strength and stability
                        score = (
                            abs(stats.get('mean', 0)) * 0.5 +  # Signal strength
                            (1 - stats.get('missing_pct', 100) / 100) * 0.3 +  # Completeness
                            (1 - stats.get('outlier_pct', 100) / 100) * 0.2  # Stability
                        )

                        if score > best_score:
                            best_score = score
                            best_factor = factor

                except Exception as e:
                    logger.error(f"Error evaluating optimized factor: {e}")
                    continue

            return best_factor

        except Exception as e:
            logger.error(f"Error evaluating optimized versions: {e}")
            return None

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of factor optimizations"""
        return self.optimization_history.copy()

    def save_optimization_results(self, factors: List[Dict[str, Any]], filepath: str):
        """Save optimization results to file"""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(factors, f, indent=2, default=str)
            logger.info(f"Saved optimization results for {len(factors)} factors to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")