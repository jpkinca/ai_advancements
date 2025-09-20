"""
Evaluation and Export Module for Chain-of-Alpha MVP

Provides comprehensive evaluation and export capabilities for the alpha discovery pipeline
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EvaluationExportModule:
    """
    Module for evaluating and exporting Chain-of-Alpha results
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = config.get('results_dir', 'results')
        self.ensure_results_directory()

    def ensure_results_directory(self):
        """Ensure results directory exists"""
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_pipeline_results(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of pipeline results

        Args:
            factors: List of optimized factors with backtest results

        Returns:
            Evaluation summary
        """
        logger.info("Evaluating pipeline results")

        evaluation = {
            'summary': self._generate_summary_stats(factors),
            'performance_analysis': self._analyze_performance_distribution(factors),
            'factor_characteristics': self._analyze_factor_characteristics(factors),
            'robustness_check': self._check_robustness(factors),
            'recommendations': self._generate_recommendations(factors),
            'evaluation_time': datetime.now().isoformat()
        }

        return evaluation

    def _generate_summary_stats(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""

        if not factors:
            return {'total_factors': 0, 'message': 'No factors to evaluate'}

        # Extract metrics
        metrics_list = []
        for factor in factors:
            metrics = factor.get('backtest_results', {}).get('metrics', {})
            if metrics:
                metrics_list.append(metrics)

        if not metrics_list:
            return {'total_factors': len(factors), 'message': 'No backtest results available'}

        # Calculate summary statistics
        summary = {
            'total_factors': len(factors),
            'factors_with_results': len(metrics_list),
            'avg_sharpe_ratio': float(np.mean([m.get('sharpe_ratio', 0) for m in metrics_list])),
            'avg_annual_return': float(np.mean([m.get('annual_return', 0) for m in metrics_list])),
            'avg_max_drawdown': float(np.mean([m.get('max_drawdown', 0) for m in metrics_list])),
            'avg_win_rate': float(np.mean([m.get('win_rate', 0) for m in metrics_list])),
            'best_sharpe': max([m.get('sharpe_ratio', 0) for m in metrics_list]),
            'worst_sharpe': min([m.get('sharpe_ratio', 0) for m in metrics_list]),
        }

        # Count factors meeting criteria
        summary.update({
            'profitable_factors': len([m for m in metrics_list if m.get('annual_return', 0) > 0]),
            'positive_sharpe_factors': len([m for m in metrics_list if m.get('sharpe_ratio', 0) > 0.5]),
            'low_drawdown_factors': len([m for m in metrics_list if abs(m.get('max_drawdown', 0)) < 0.2]),
        })

        return summary

    def _analyze_performance_distribution(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of factor performance"""

        metrics_list = []
        for factor in factors:
            metrics = factor.get('backtest_results', {}).get('metrics', {})
            if metrics:
                metrics_list.append({
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'return': metrics.get('annual_return', 0),
                    'drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'factor_id': factor['id']
                })

        if not metrics_list:
            return {}

        df = pd.DataFrame(metrics_list)

        analysis = {}

        # Sharpe ratio distribution
        analysis['sharpe_distribution'] = {
            'mean': float(df['sharpe'].mean()),
            'std': float(df['sharpe'].std()),
            'quartiles': {
                '25%': float(df['sharpe'].quantile(0.25)),
                '50%': float(df['sharpe'].quantile(0.50)),
                '75%': float(df['sharpe'].quantile(0.75))
            }
        }

        # Performance categories
        analysis['performance_categories'] = {
            'excellent': len(df[df['sharpe'] > 1.5]),
            'good': len(df[(df['sharpe'] > 0.5) & (df['sharpe'] <= 1.5)]),
            'poor': len(df[df['sharpe'] <= 0.5])
        }

        return analysis

    def _analyze_factor_characteristics(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of the generated factors"""

        characteristics = {
            'complexity_analysis': self._analyze_factor_complexity(factors),
            'feature_usage': self._analyze_feature_usage(factors),
            'correlation_analysis': self._analyze_factor_correlations(factors)
        }

        return characteristics

    def _analyze_factor_complexity(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the complexity of factor expressions"""

        complexities = []

        for factor in factors:
            expression = factor['expression']

            # Simple complexity metrics
            length = len(expression)
            operators = len(re.findall(r'[\+\-\*\/]', expression))
            functions = len(re.findall(r'df\[', expression))
            parentheses = len(re.findall(r'[\(\)]', expression))

            complexity_score = length * 0.1 + operators * 2 + functions * 3 + parentheses * 0.5

            complexities.append({
                'factor_id': factor['id'],
                'complexity_score': complexity_score,
                'expression_length': length,
                'operator_count': operators,
                'feature_count': functions
            })

        if complexities:
            df = pd.DataFrame(complexities)
            return {
                'avg_complexity': float(df['complexity_score'].mean()),
                'complexity_range': {
                    'min': float(df['complexity_score'].min()),
                    'max': float(df['complexity_score'].max())
                },
                'most_complex': df.loc[df['complexity_score'].idxmax()]['factor_id'],
                'least_complex': df.loc[df['complexity_score'].idxmin()]['factor_id']
            }

        return {}

    def _analyze_feature_usage(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze which features are most commonly used"""

        feature_counts = {}

        for factor in factors:
            expression = factor['expression']

            # Extract feature names
            features = re.findall(r"df\['([^']+)'\]", expression)

            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Sort by usage
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_unique_features': len(feature_counts),
            'most_used_features': sorted_features[:10],
            'feature_usage_distribution': feature_counts
        }

    def _analyze_factor_correlations(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between factors"""

        # Extract factor returns for correlation analysis
        factor_returns = []

        for factor in factors:
            metrics = factor.get('backtest_results', {}).get('metrics', {})
            if metrics and 'annual_return' in metrics:
                factor_returns.append(metrics['annual_return'])

        if len(factor_returns) < 2:
            return {'message': 'Insufficient data for correlation analysis'}

        # Calculate correlation matrix
        returns_df = pd.DataFrame(factor_returns)
        corr_matrix = returns_df.corr()

        return {
            'avg_correlation': float(corr_matrix.mean().mean()),
            'correlation_range': {
                'min': float(corr_matrix.min().min()),
                'max': float(corr_matrix.max().max())
            },
            'high_correlation_pairs': self._find_high_correlations(corr_matrix, 0.8)
        }

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float) -> List[Tuple[int, int, float]]:
        """Find pairs with high correlation"""

        high_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_value = corr_matrix.iloc[i, j]
                if isinstance(corr_value, (int, float, np.number)):
                    corr = float(corr_value)
                    if abs(corr) > threshold:
                        high_corr.append((i, j, corr))

        return high_corr

    def _check_robustness(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check robustness of results"""

        robustness = {
            'consistency_check': self._check_result_consistency(factors),
            'sensitivity_analysis': self._analyze_sensitivity(factors)
        }

        return robustness

    def _check_result_consistency(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency of results across factors"""

        if len(factors) < 2:
            return {'message': 'Need at least 2 factors for consistency check'}

        # Check if top performers are consistent
        metrics_list = []
        for factor in factors:
            metrics = factor.get('backtest_results', {}).get('metrics', {})
            if metrics:
                metrics_list.append((factor['id'], metrics.get('sharpe_ratio', 0)))

        if len(metrics_list) < 2:
            return {'message': 'Insufficient metrics for consistency check'}

        # Sort by performance
        sorted_factors = sorted(metrics_list, key=lambda x: x[1], reverse=True)

        # Check if top 3 are significantly better than bottom 3
        top_3_avg = np.mean([x[1] for x in sorted_factors[:3]])
        bottom_3_avg = np.mean([x[1] for x in sorted_factors[-3:]])

        return {
            'top_3_avg_sharpe': float(top_3_avg),
            'bottom_3_avg_sharpe': float(bottom_3_avg),
            'performance_gap': float(top_3_avg - bottom_3_avg),
            'consistent_results': abs(top_3_avg - bottom_3_avg) > 0.5  # Arbitrary threshold
        }

    def _analyze_sensitivity(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sensitivity of results to parameter changes"""

        # This is a placeholder for more sophisticated sensitivity analysis
        # In a full implementation, this would test factors under different market conditions

        return {
            'message': 'Sensitivity analysis not fully implemented in MVP',
            'recommendation': 'Consider testing factors across different time periods and market regimes'
        }

    def _generate_recommendations(self, factors: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on evaluation"""

        recommendations = []

        if not factors:
            return ["No factors generated - check pipeline configuration"]

        summary = self._generate_summary_stats(factors)

        # Performance-based recommendations
        if summary.get('positive_sharpe_factors', 0) == 0:
            recommendations.append("No factors achieved positive Sharpe ratio - consider adjusting factor generation parameters")

        if summary.get('avg_sharpe_ratio', 0) < 0.5:
            recommendations.append("Average Sharpe ratio is low - factors may need optimization or different market data")

        # Diversity recommendations
        feature_analysis = self._analyze_feature_usage(factors)
        if feature_analysis.get('total_unique_features', 0) < 5:
            recommendations.append("Limited feature diversity - consider expanding the feature set")

        # Complexity recommendations
        complexity = self._analyze_factor_complexity(factors)
        if complexity.get('avg_complexity', 0) > 50:
            recommendations.append("Factors are quite complex - consider simplifying for better interpretability")

        # General recommendations
        recommendations.extend([
            "Validate top-performing factors on out-of-sample data",
            "Consider ensemble methods combining multiple factors",
            "Implement proper risk management before live trading",
            "Monitor factor performance over time for decay"
        ])

        return recommendations

    def export_results(self, factors: List[Dict[str, Any]], evaluation: Dict[str, Any],
                      export_formats: Optional[List[str]] = None):
        """
        Export results in multiple formats

        Args:
            factors: List of factors with results
            evaluation: Evaluation summary
            export_formats: List of formats to export (json, csv, html, png)
        """
        if export_formats is None:
            export_formats = ['json', 'html']

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for fmt in export_formats:
            try:
                if fmt == 'json':
                    self._export_json(factors, evaluation, timestamp)
                elif fmt == 'csv':
                    self._export_csv(factors, timestamp)
                elif fmt == 'html':
                    self._export_html(factors, evaluation, timestamp)
                elif fmt == 'png':
                    self._export_charts(factors, timestamp)
                else:
                    logger.warning(f"Unknown export format: {fmt}")

            except Exception as e:
                logger.error(f"Failed to export in {fmt} format: {e}")

    def _export_json(self, factors: List[Dict[str, Any]], evaluation: Dict[str, Any], timestamp: str):
        """Export results as JSON"""

        results = {
            'factors': factors,
            'evaluation': evaluation,
            'export_time': datetime.now().isoformat(),
            'version': '1.0'
        }

        filepath = os.path.join(self.results_dir, f'chain_of_alpha_results_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Exported JSON results to {filepath}")

    def _export_csv(self, factors: List[Dict[str, Any]], timestamp: str):
        """Export factor summary as CSV"""

        # Extract key information for CSV
        csv_data = []
        for factor in factors:
            metrics = factor.get('backtest_results', {}).get('metrics', {})

            row = {
                'factor_id': factor['id'],
                'expression': factor['expression'],
                'explanation': factor.get('explanation', ''),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'annual_return': metrics.get('annual_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0)
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        filepath = os.path.join(self.results_dir, f'factor_summary_{timestamp}.csv')
        df.to_csv(filepath, index=False)

        logger.info(f"Exported CSV summary to {filepath}")

    def _export_html(self, factors: List[Dict[str, Any]], evaluation: Dict[str, Any], timestamp: str):
        """Export results as HTML report"""

        html_content = self._generate_html_report(factors, evaluation, timestamp)

        filepath = os.path.join(self.results_dir, f'chain_of_alpha_report_{timestamp}.html')
        with open(filepath, 'w') as f:
            f.write(html_content)

        logger.info(f"Exported HTML report to {filepath}")

    def _generate_html_report(self, factors: List[Dict[str, Any]], evaluation: Dict[str, Any], timestamp: str) -> str:
        """Generate HTML report content"""

        summary = evaluation.get('summary', {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chain-of-Alpha Results Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .factor {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .metrics {{ display: flex; flex-wrap: wrap; }}
        .metric {{ background-color: #fff; padding: 10px; margin: 5px; border-radius: 3px; min-width: 150px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Chain-of-Alpha Alpha Discovery Results</h1>
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Factors:</strong> {summary.get('total_factors', 0)}
            </div>
            <div class="metric">
                <strong>Average Sharpe:</strong> {summary.get('avg_sharpe_ratio', 0):.3f}
            </div>
            <div class="metric">
                <strong>Average Return:</strong> {summary.get('avg_annual_return', 0):.1%}
            </div>
            <div class="metric">
                <strong>Positive Sharpe:</strong> {summary.get('positive_sharpe_factors', 0)}
            </div>
        </div>
    </div>

    <h2>Top Performing Factors</h2>
    <table>
        <tr>
            <th>Factor ID</th>
            <th>Expression</th>
            <th>Sharpe Ratio</th>
            <th>Annual Return</th>
            <th>Max Drawdown</th>
            <th>Win Rate</th>
        </tr>
"""

        # Sort factors by Sharpe ratio
        sorted_factors = sorted(factors,
                              key=lambda x: x.get('backtest_results', {}).get('metrics', {}).get('sharpe_ratio', 0),
                              reverse=True)

        for factor in sorted_factors[:10]:  # Top 10
            metrics = factor.get('backtest_results', {}).get('metrics', {})
            html += f"""
        <tr>
            <td>{factor['id']}</td>
            <td><code>{factor['expression'][:50]}...</code></td>
            <td>{metrics.get('sharpe_ratio', 0):.3f}</td>
            <td>{metrics.get('annual_return', 0):.1%}</td>
            <td>{metrics.get('max_drawdown', 0):.1%}</td>
            <td>{metrics.get('win_rate', 0):.1%}</td>
        </tr>
"""

        html += """
    </table>

    <h2>Recommendations</h2>
    <ul>
"""

        recommendations = evaluation.get('recommendations', [])
        for rec in recommendations:
            html += f"        <li>{rec}</li>\n"

        html += """
    </ul>
</body>
</html>
"""

        return html

    def _export_charts(self, factors: List[Dict[str, Any]], timestamp: str):
        """Export performance charts"""

        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Sharpe ratio distribution
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Extract metrics
            sharpe_ratios = []
            returns = []
            drawdowns = []
            win_rates = []

            for factor in factors:
                metrics = factor.get('backtest_results', {}).get('metrics', {})
                if metrics:
                    sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                    returns.append(metrics.get('annual_return', 0))
                    drawdowns.append(metrics.get('max_drawdown', 0))
                    win_rates.append(metrics.get('win_rate', 0))

            if sharpe_ratios:
                # Sharpe ratio histogram
                axes[0, 0].hist(sharpe_ratios, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Sharpe Ratio Distribution')
                axes[0, 0].set_xlabel('Sharpe Ratio')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].axvline(np.mean(sharpe_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(sharpe_ratios):.3f}')
                axes[0, 0].legend()

                # Return vs Sharpe scatter
                axes[0, 1].scatter(returns, sharpe_ratios, alpha=0.6)
                axes[0, 1].set_title('Return vs Sharpe Ratio')
                axes[0, 1].set_xlabel('Annual Return')
                axes[0, 1].set_ylabel('Sharpe Ratio')

                # Drawdown distribution
                axes[1, 0].hist(drawdowns, bins=20, alpha=0.7, edgecolor='black', color='orange')
                axes[1, 0].set_title('Maximum Drawdown Distribution')
                axes[1, 0].set_xlabel('Max Drawdown')
                axes[1, 0].set_ylabel('Frequency')

                # Win rate distribution
                axes[1, 1].hist(win_rates, bins=20, alpha=0.7, edgecolor='black', color='green')
                axes[1, 1].set_title('Win Rate Distribution')
                axes[1, 1].set_xlabel('Win Rate')
                axes[1, 1].set_ylabel('Frequency')

                plt.tight_layout()

                # Save chart
                chart_path = os.path.join(self.results_dir, f'performance_charts_{timestamp}.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Exported performance charts to {chart_path}")

        except Exception as e:
            logger.error(f"Failed to export charts: {e}")

    def save_evaluation_report(self, evaluation: Dict[str, Any], filepath: str):
        """Save evaluation report to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(evaluation, f, indent=2, default=str)
            logger.info(f"Saved evaluation report to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")