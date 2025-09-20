"""
Factor Generation Chain Module for Chain-of-Alpha MVP

Implements the factor generation chain that uses LLMs to create alpha factors
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class FactorGenerationChain:
    """
    Chain for generating alpha factors using LLM guidance
    """

    def __init__(self, config: Dict[str, Any], llm_interface: LLMInterface):
        self.config = config
        self.llm = llm_interface
        self.generated_factors = []
        self.factor_history = []

    def generate_factors(self, market_data: pd.DataFrame, num_factors: int = 10) -> List[Dict[str, Any]]:
        """
        Generate alpha factors using the LLM

        Args:
            market_data: Preprocessed market data
            num_factors: Number of factors to generate

        Returns:
            List of factor dictionaries with expressions and metadata
        """
        logger.info(f"Generating {num_factors} alpha factors")

        # Analyze market data to provide context
        data_context = self._analyze_market_data(market_data)

        factors = []

        for i in range(num_factors):
            try:
                factor = self._generate_single_factor(data_context, i + 1)
                if factor:
                    factors.append(factor)
                    self.generated_factors.append(factor)

                # Brief pause between generations
                import time
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to generate factor {i+1}: {e}")
                continue

        logger.info(f"Successfully generated {len(factors)} factors")
        return factors

    def _analyze_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data to provide context for factor generation"""
        # Handle MultiIndex case
        if isinstance(data.index, pd.MultiIndex):
            # Get date level (assuming it's the second level)
            date_level = 1 if 'ticker' in data.index.names and data.index.names[0] == 'ticker' else 0
            date_index = data.index.get_level_values(date_level)
            tickers = data.index.get_level_values('ticker').unique().tolist() if 'ticker' in data.index.names else ['unknown']
        else:
            date_index = data.index
            tickers = ['unknown']

        context = {
            'columns': list(data.columns),
            'date_range': {
                'start': date_index.min().strftime('%Y-%m-%d'),
                'end': date_index.max().strftime('%Y-%m-%d')
            },
            'tickers': tickers,
            'stats': {}
        }

        # Calculate basic statistics for key columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            try:
                context['stats'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                }
            except:
                pass

        return context

    def _generate_single_factor(self, data_context: Dict[str, Any], factor_num: int) -> Optional[Dict[str, Any]]:
        """Generate a single alpha factor"""

        prompt = self._build_factor_generation_prompt(data_context, factor_num)

        context = {
            'data_analysis': data_context,
            'factor_number': factor_num,
            'previous_factors': [str(factor.get('expression', '')) for factor in self.generated_factors[-3:]]  # Last 3 factors
        }

        response = self.llm.generate_response(prompt, context)

        # Parse the response to extract factor
        factor = self._parse_factor_response(response)

        if factor:
            factor['generation_time'] = datetime.now().isoformat()
            factor['data_context'] = data_context
            self.factor_history.append({
                'factor': factor,
                'prompt': prompt,
                'response': response
            })

        return factor

    def _build_factor_generation_prompt(self, data_context: Dict[str, Any], factor_num: int) -> str:
        """Build the prompt for factor generation"""

        available_columns = data_context['columns']

        prompt = f"""
Generate an innovative alpha factor for stock return prediction using the available market data.

Available data columns: {', '.join(available_columns)}

Data time period: {data_context['date_range']['start']} to {data_context['date_range']['end']}

Requirements:
1. Create a factor expression that can be evaluated on a pandas DataFrame
2. Use only the available columns listed above
3. Make it interpretable and economically meaningful
4. Focus on predicting future returns
5. Avoid factors that are too similar to previously generated ones

Previous factors generated (avoid similarity):
{chr(10).join(f"- {str(factor.get('expression', ''))}" for factor in self.generated_factors[-3:])}

Please provide:
1. The factor expression (Python code)
2. A brief explanation of the economic intuition
3. Expected relationship with future returns

Format your response as:
FACTOR: your_expression_here
EXPLANATION: brief explanation
"""

        return prompt

    def _parse_factor_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response to extract factor information"""

        try:
            # Extract factor expression
            factor_match = re.search(r'FACTOR:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if not factor_match:
                logger.warning("No FACTOR found in response")
                return None

            expression = factor_match.group(1).strip()

            # Extract explanation
            explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"

            # Validate the expression
            if not self.llm.validate_factor_expression(expression):
                logger.warning(f"Invalid factor expression: {expression}")
                return None

            factor = {
                'expression': expression,
                'explanation': explanation,
                'id': f"factor_{len(self.generated_factors) + 1}",
                'validated': True
            }

            return factor

        except Exception as e:
            logger.error(f"Failed to parse factor response: {e}")
            return None

    def evaluate_factors(self, factors: List[Dict[str, Any]], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evaluate generated factors by computing them on the data

        Args:
            factors: List of factor dictionaries
            data: Market data DataFrame

        Returns:
            Factors with computed values and basic statistics
        """
        logger.info(f"Evaluating {len(factors)} factors")

        evaluated_factors = []

        for factor in factors:
            try:
                # Compute factor values
                factor_values = self._compute_factor(data.copy(), factor['expression'])

                if factor_values is not None:
                    # Calculate basic statistics
                    stats = self._calculate_factor_stats(factor_values)

                    evaluated_factor = factor.copy()
                    evaluated_factor.update({
                        'values': factor_values,
                        'stats': stats,
                        'evaluation_time': datetime.now().isoformat()
                    })

                    evaluated_factors.append(evaluated_factor)
                else:
                    logger.warning(f"Failed to compute factor: {factor['expression']}")

            except Exception as e:
                logger.error(f"Error evaluating factor {factor['id']}: {e}")
                continue

        logger.info(f"Successfully evaluated {len(evaluated_factors)} factors")
        return evaluated_factors

    def _compute_factor(self, data: pd.DataFrame, expression: str) -> Optional[pd.Series]:
        """Compute a factor expression on the data"""

        try:
            # Create a safe evaluation environment
            safe_dict = {
                'df': data,
                'np': np,
                'pd': pd
            }

            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, safe_dict)

            # Ensure result is a pandas Series
            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, pd.DataFrame):
                # If DataFrame, take first column or flatten
                if result.shape[1] == 1:
                    return result.iloc[:, 0]
                else:
                    logger.warning("Factor expression returned DataFrame with multiple columns")
                    return None
            elif isinstance(result, (int, float)):
                # Constant factor
                return pd.Series(result, index=data.index)
            else:
                logger.warning(f"Factor expression returned unsupported type: {type(result)}")
                return None

        except Exception as e:
            logger.error(f"Error computing factor expression '{expression}': {e}")
            return None

    def _calculate_factor_stats(self, factor_values: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistics for a factor"""

        try:
            stats = {
                'mean': float(factor_values.mean()),
                'std': float(factor_values.std()),
                'min': float(factor_values.min()),
                'max': float(factor_values.max()),
                'median': float(factor_values.median()),
                'skewness': float(factor_values.skew()),
                'kurtosis': float(factor_values.kurtosis()),
                'missing_pct': float(factor_values.isnull().mean() * 100),
                'non_zero_pct': float((factor_values != 0).mean() * 100)
            }

            # Check for extreme values
            q99 = factor_values.quantile(0.99)
            q01 = factor_values.quantile(0.01)
            stats['outlier_pct'] = float(((factor_values > q99) | (factor_values < q01)).mean() * 100)

            return stats

        except Exception as e:
            logger.error(f"Error calculating factor stats: {e}")
            return {}

    def get_factor_history(self) -> List[Dict[str, Any]]:
        """Get the history of factor generation"""
        return self.factor_history.copy()

    def save_factors(self, filepath: str, factors: List[Dict[str, Any]]):
        """Save generated factors to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(factors, f, indent=2, default=str)
            logger.info(f"Saved {len(factors)} factors to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save factors: {e}")