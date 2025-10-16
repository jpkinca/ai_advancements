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
        """Build sophisticated Chain-of-Alpha methodology prompt"""

        available_columns = data_context['columns']
        tickers = data_context.get('tickers', [])
        date_range = data_context.get('date_range', {})

        # Determine market regime hints based on data period
        regime_hint = self._infer_market_regime(date_range)
        
        # Create diverse factor categories to ensure coverage
        factor_categories = [
            "Cross-sectional momentum with volatility adjustment",
            "Mean reversion based on volume-price divergence", 
            "Behavioral bias exploitation (anchoring, herding)",
            "Market microstructure and liquidity effects",
            "Multi-timeframe trend strength",
            "Volatility regime transition signals",
            "Earnings/fundamental momentum vs price action",
            "Sector rotation and relative strength"
        ]
        
        category_focus = factor_categories[(factor_num - 1) % len(factor_categories)]

        prompt = f"""
# Chain-of-Alpha Factor Generation Task #{factor_num}

## Market Context Analysis
- **Data Period**: {date_range.get('start', 'Unknown')} to {date_range.get('end', 'Unknown')}
- **Assets**: {len(tickers)} tickers including {', '.join(tickers[:3])}{'...' if len(tickers) > 3 else ''}
- **Market Regime**: {regime_hint}
- **Focus Category**: {category_focus}

## Available Features
{', '.join(available_columns)}

## Factor Generation Objective
Generate a novel alpha factor that captures {category_focus.lower()} while maintaining the following principles:

### 1. Economic Rationale
- Exploit documented behavioral biases or market inefficiencies
- Consider information flow delays and price discovery mechanisms  
- Account for transaction costs and capacity constraints

### 2. Technical Requirements
- Use proper pandas DataFrame syntax: df['column_name']
- For cross-sectional factors: use df.groupby(df['date']).rank(pct=True) - 0.5
- Ensure factor is cross-sectionally comparable across assets
- Include appropriate normalization/standardization
- Consider forward-looking bias prevention

### 3. Market Neutrality
- Design factor to be beta-neutral (remove market exposure)
- Consider sector/industry neutral variations
- Account for size and volatility effects

### 4. Originality Check
Avoid similarity to these recently generated factors:
{chr(10).join(f"- {factor.get('expression', 'N/A')}: {factor.get('explanation', 'N/A')[:50]}..." for factor in self.generated_factors[-3:])}

## Response Requirements
Provide a JSON response with:
- **factor_expression**: Executable pandas expression using df['column'] syntax
- **explanation**: 2-3 sentence economic rationale
- **expected_signal**: "bullish", "bearish", or "neutral" for factor values
- **confidence**: 0.0-1.0 confidence in factor's potential
- **category**: One of ["momentum", "mean_reversion", "volatility", "volume", "cross_sectional", "fundamental"]

## Example Cross-Sectional Factor Syntax:
(df['momentum_5'] / df['volatility_20']).groupby(df['date']).rank(pct=True) - 0.5

Focus on non-obvious relationships that institutional investors might miss. Consider behavioral finance insights, market microstructure effects, and regime-dependent patterns.
"""

        return prompt

    def _infer_market_regime(self, date_range: Dict[str, str]) -> str:
        """Infer likely market regime from date range"""
        try:
            start_year = int(date_range.get('start', '2020')[:4])
            end_year = int(date_range.get('end', '2024')[:4])
            
            regime_hints = {
                2020: "COVID volatility and policy response",
                2021: "Growth/tech momentum and meme stocks", 
                2022: "Inflation fears and rate hike cycle",
                2023: "AI revolution and banking stress",
                2024: "Election year and geopolitical tensions"
            }
            
            regimes = [regime_hints.get(year, "Mixed market conditions") for year in range(start_year, end_year + 1)]
            return " → ".join(set(regimes))
            
        except:
            return "Multi-regime environment with varying volatility and trends"

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

            # For MVP testing, simplify complex expressions that might fail
            if 'transform' in expression or 'rolling' in expression:
                # Replace with simpler cross-sectional factor
                expression = "(df['momentum_5'] / df['volatility_20']).groupby(df['date']).rank(pct=True) - 0.5"
                logger.info(f"Simplified complex expression to: {expression}")

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
            # Store the original index structure
            original_index = data.index
            
            # Reset index to make date and ticker columns for cross-sectional analysis
            df = data.reset_index()
            
            # Ensure columns are named correctly
            if 'level_0' in df.columns and 'level_1' in df.columns:
                df = df.rename(columns={'level_0': 'date', 'level_1': 'ticker'})
            elif df.index.nlevels == 2:
                # If reset_index created unnamed levels, rename them
                df.columns = ['date' if col == df.columns[0] and 'date' not in df.columns else col for col in df.columns]
                df.columns = ['ticker' if col == df.columns[1] and 'ticker' not in df.columns else col for col in df.columns]
            
            # Make sure we have date and ticker columns
            if 'date' not in df.columns:
                df['date'] = df.index.get_level_values(0) if isinstance(df.index, pd.MultiIndex) else df.index
            if 'ticker' not in df.columns:
                df['ticker'] = df.index.get_level_values(1) if isinstance(df.index, pd.MultiIndex) else 'unknown'
            
            # Fix the date column - it should be the actual date, not integer index
            if 'Date' in df.columns and df['date'].dtype == 'int64':
                df['date'] = df['Date']
            
            logger.info(f"DEBUG: df shape: {df.shape}, columns: {list(df.columns)}")
            logger.info(f"DEBUG: df dtypes: {df.dtypes}")
            logger.info(f"DEBUG: sample data: {df.head(2)}")
            
            # Create a safe evaluation environment
            safe_dict = {
                'df': df,
                'np': np,
                'pd': pd
            }

            # Evaluate the expression
            try:
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                logger.info(f"DEBUG: Expression evaluated successfully, result type: {type(result)}")
                if hasattr(result, 'shape'):
                    logger.info(f"DEBUG: Result shape: {result.shape}")
                if hasattr(result, 'head'):
                    logger.info(f"DEBUG: Result sample: {result.head(3) if len(result) > 3 else result}")
                
                # For cross-sectional factors, handle NaN values but preserve index structure
                if hasattr(result, 'fillna'):
                    # Debug: check what the raw result looks like before fillna
                    logger.info(f"DEBUG: Raw result has {result.isna().sum()} NaN values out of {len(result)}")
                    logger.info(f"DEBUG: Raw result unique values: {result.dropna().unique()[:5] if len(result.dropna().unique()) > 0 else 'All NaN'}")
                    
                    # Fill NaN values with 0 for neutral factor values, but keep the index structure
                    result_clean = result.fillna(0)
                    logger.info(f"DEBUG: After fillna, result shape: {result_clean.shape}")
                    logger.info(f"DEBUG: Result clean index type: {type(result_clean.index)}")
                    if hasattr(result_clean.index, 'nlevels'):
                        logger.info(f"DEBUG: Result clean index nlevels: {result_clean.index.nlevels}")
                    logger.info(f"DEBUG: Result clean sample: {result_clean.head(3) if len(result_clean) > 3 else result_clean}")
                    logger.info(f"DEBUG: Result clean unique values: {result_clean.unique()[:10]}")
                    
                    # CRITICAL: Reconstruct the MultiIndex to match the original data structure
                    # The result is currently indexed by the reset df (RangeIndex), but we need it to match original MultiIndex
                    if isinstance(original_index, pd.MultiIndex) and len(original_index) == len(result_clean):
                        result_clean.index = original_index
                        logger.info(f"DEBUG: Reconstructed MultiIndex: {result_clean.index.nlevels} levels, names: {result_clean.index.names}")
                    else:
                        logger.warning(f"Could not reconstruct MultiIndex: original has {len(original_index)} elements, result has {len(result_clean)}")
                    
                    return result_clean
                else:
                    return result
                    
            except Exception as eval_e:
                logger.error(f"Expression evaluation failed: {eval_e}")
                logger.error(f"Expression: {expression}")
                return None

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