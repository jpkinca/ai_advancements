#!/usr/bin/env python3
"""
Factor Correlation Analysis and Portfolio Construction
Analyzes correlations between AI-generated factors and constructs optimal portfolios
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FactorPortfolioConstructor:
    """Constructs optimal portfolios from AI-generated factors"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.factors_data = {}
        self.correlation_matrix = None
        self.selected_factors = []

    def load_latest_results(self) -> Dict[str, Any]:
        """Load the most recent factor results"""
        json_files = list(self.results_dir.glob("chain_of_alpha_results_*.json"))
        if not json_files:
            raise FileNotFoundError("No results files found")

        # Get latest file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, 'r') as f:
            results = json.load(f)

        print(f"Loaded results from: {latest_file}")
        return results

    def extract_factor_values(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Extract factor values from results and create correlation matrix"""
        factors = results.get('factors', [])

        # Extract factor values for each factor
        factor_series = {}
        all_dates = set()
        all_tickers = set()

        for factor in factors:
            factor_id = factor['id']
            factor_values_str = factor.get('values', '')

            if factor_values_str:
                # Parse the string representation of pandas Series with MultiIndex
                try:
                    # Split by lines and clean up
                    raw_lines = factor_values_str.split('\n')
                    # Keep original spacing to detect continuation lines, but also have a stripped version for checks
                    lines = [ln for ln in raw_lines if ln and ln.strip()]

                    # Find the data lines (skip header)
                    data_lines = []
                    for line in lines:
                        sline = line.strip()
                        # Skip the header line "ticker  Date"
                        if 'ticker' in sline and 'Date' in sline:
                            continue
                        # Skip pandas footer lines like "Length: 12072, dtype: float64"
                        if sline.startswith('Length:') or sline.startswith('dtype:'):
                            continue
                        # Skip empty lines
                        if not sline:
                            continue
                        data_lines.append(line)

                    # Parse each data line
                    factor_data = {}
                    current_ticker = None

                    for line in data_lines:
                        # Split on whitespace; continuation lines will have only 2 parts (date, value)
                        parts = line.split()
                        if len(parts) >= 3:
                            # Line contains ticker, date, value
                            ticker, date, value = parts[0], parts[1], parts[2]
                            current_ticker = ticker  # crucial to capture for following continuation lines
                            key = (ticker, date)
                            try:
                                factor_data[key] = float(value)
                                all_tickers.add(ticker)
                                all_dates.add(date)
                            except ValueError:
                                # Skip lines that don't parse to float
                                continue
                        elif len(parts) == 2 and current_ticker:
                            # Continuation line: date, value for the current ticker
                            date, value = parts[0], parts[1]
                            key = (current_ticker, date)
                            try:
                                factor_data[key] = float(value)
                                all_dates.add(date)
                            except ValueError:
                                continue

                    # Create MultiIndex Series
                    if factor_data:
                        # Convert dates to datetime for consistent indexing
                        tuples = []
                        values = []
                        for (t, d), v in factor_data.items():
                            try:
                                dt = pd.to_datetime(d)
                            except Exception:
                                dt = d
                            tuples.append((t, dt))
                            values.append(v)
                        index = pd.MultiIndex.from_tuples(tuples, names=['ticker', 'Date'])
                        series = pd.Series(values, index=index)
                        # Sort by both levels
                        series = series.sort_index()
                        factor_series[factor_id] = series

                except Exception as e:
                    print(f"Error parsing factor {factor_id}: {e}")
                    continue

        # Create DataFrame from factor series
        if factor_series:
            # Get all unique dates and tickers
            # Ensure datetime for dates
            try:
                all_dates = sorted(pd.to_datetime(list(all_dates)))
            except Exception:
                all_dates = sorted(list(all_dates))
            all_tickers = sorted(list(all_tickers))

            # Create MultiIndex for the DataFrame
            multiindex = pd.MultiIndex.from_product([all_tickers, all_dates], names=['ticker', 'Date'])

            # Create DataFrame
            df_factors = pd.DataFrame(index=multiindex)

            for factor_id, series in factor_series.items():
                df_factors[factor_id] = series

            # Fill NaN values with 0 (factors might not have values for all ticker-date combinations)
            df_factors = df_factors.fillna(0)

            # Basic sanity checks
            obs = len(df_factors)
            uniq_dates = df_factors.index.get_level_values('Date').nunique() if obs > 0 else 0
            uniq_tickers = df_factors.index.get_level_values('ticker').nunique() if obs > 0 else 0
            print(f"Extracted {len(df_factors.columns)} factors with {obs} observations "
                  f"({uniq_tickers} tickers x {uniq_dates} dates)")
            return df_factors

        return pd.DataFrame()

    def calculate_correlations(self, df_factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate pairwise correlations between factors"""
        if df_factors.empty:
            return pd.DataFrame()

        # Drop near-constant columns to avoid NaN correlations
        variances = df_factors.var(axis=0)
        valid_cols = variances[variances > 1e-12].index.tolist()
        if len(valid_cols) < df_factors.shape[1]:
            dropped = set(df_factors.columns) - set(valid_cols)
            if dropped:
                print(f"Warning: Dropping {len(dropped)} near-constant factor(s): {sorted(dropped)}")

        # Calculate correlation matrix on valid columns
        corr_matrix = df_factors[valid_cols].corr() if valid_cols else pd.DataFrame()

        # Display correlation matrix
        print("\n=== FACTOR CORRELATION MATRIX ===")
        print(corr_matrix.round(3))

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Factor Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'factor_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.correlation_matrix = corr_matrix
        return corr_matrix

    def select_uncorrelated_factors(self, corr_matrix: pd.DataFrame,
                                   max_correlation: float = 0.3,
                                   min_sharpe: float = 0.0) -> List[str]:
        """Select factors with low correlation and positive Sharpe ratios"""

        # Load performance data
        csv_files = list(self.results_dir.glob("factor_summary_*.csv"))
        if not csv_files:
            print("No summary files found, selecting based on correlation only")
            return self._select_by_correlation_only(corr_matrix, max_correlation)

        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)

        # Read performance data
        perf_df = pd.read_csv(latest_csv)
        # Ensure numeric types for metrics
        for col in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'win_rate', 'total_trades']:
            if col in perf_df.columns:
                perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')
        perf_df.set_index('factor_id', inplace=True)

        print("\n=== FACTOR PERFORMANCE SUMMARY ===")
        print(perf_df[['sharpe_ratio', 'annual_return', 'max_drawdown']].round(3))

        # Filter factors with positive Sharpe and low correlation
        selected = []
        remaining_factors = list(corr_matrix.columns)

        while remaining_factors:
            # Find best remaining factor by Sharpe ratio
            best_factor = None
            best_sharpe = -np.inf

            for factor in remaining_factors:
                if factor in perf_df.index:
                    sharpe = float(perf_df.loc[factor, 'sharpe_ratio'])
                    if sharpe > float(min_sharpe) and sharpe > float(best_sharpe):
                        best_sharpe = sharpe
                        best_factor = factor

            if best_factor is None:
                break

            # Check correlation with already selected factors
            is_uncorrelated = True
            for selected_factor in selected:
                corr = float(corr_matrix.loc[best_factor, selected_factor])
                if abs(corr) > float(max_correlation):
                    is_uncorrelated = False
                    break

            if is_uncorrelated:
                selected.append(best_factor)
                print(f"Selected {best_factor}: Sharpe={best_sharpe:.3f}")

            remaining_factors.remove(best_factor)

            # Limit to top 5 factors
            if len(selected) >= 5:
                break

        self.selected_factors = selected
        return selected

    def _select_by_correlation_only(self, corr_matrix: pd.DataFrame,
                                   max_correlation: float = 0.3) -> List[str]:
        """Fallback selection based on correlation only"""
        selected = []
        remaining_factors = list(corr_matrix.columns)

        while remaining_factors and len(selected) < 5:
            # Select first uncorrelated factor
            factor = remaining_factors[0]
            is_uncorrelated = True

            for selected_factor in selected:
                corr = float(corr_matrix.loc[factor, selected_factor])
                if abs(corr) > float(max_correlation):
                    is_uncorrelated = False
                    break

            if is_uncorrelated:
                selected.append(factor)

            remaining_factors.remove(factor)

        return selected

    def construct_risk_parity_portfolio(self, selected_factors: List[str],
                                       df_factors: pd.DataFrame) -> Dict[str, Any]:
        """Construct risk-parity weighted portfolio"""

        if not selected_factors:
            return {}

        # Extract selected factor returns
        portfolio_factors = df_factors[selected_factors].copy()

        # Calculate factor volatilities (rolling or expanding)
        factor_vols = portfolio_factors.rolling(window=20, min_periods=5).std()

        # Risk parity weights: inverse volatility
        weights = 1.0 / factor_vols

        # Normalize weights to sum to 1
        weights = weights.div(weights.sum(axis=1), axis=0)

        # Handle NaN weights (early periods)
        weights = weights.fillna(1.0 / len(selected_factors))

        # Calculate portfolio returns
        portfolio_returns = (portfolio_factors * weights).sum(axis=1)

        # Calculate portfolio metrics
        portfolio_stats = self._calculate_portfolio_stats(portfolio_returns)
        # Correct number of factors in stats
        portfolio_stats['n_factors'] = len(selected_factors)

        return {
            'weights': weights,
            'portfolio_returns': portfolio_returns,
            'selected_factors': selected_factors,
            'stats': portfolio_stats
        }

    def _calculate_portfolio_stats(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate portfolio performance statistics"""
        if returns.empty:
            return {}

        # Basic metrics
        # Ensure returns is numeric Series
        returns = pd.to_numeric(returns, errors='coerce').fillna(0.0)
        total_return = (1.0 + returns).prod() - 1.0
        annual_return = float((1.0 + total_return) ** (252.0 / max(1, len(returns))) - 1.0)
        volatility = float(returns.std() * np.sqrt(252.0))
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Max drawdown
    cumulative = (1.0 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_factors': len(returns)
        }

    def run_analysis(self):
        """Run complete factor analysis and portfolio construction"""
        print("🔍 Starting Factor Correlation Analysis & Portfolio Construction")
        print("=" * 70)

        # Load results
        results = self.load_latest_results()

        # Extract factor values
        df_factors = self.extract_factor_values(results)

        if df_factors.empty:
            print("❌ No factor data found")
            return

        # Calculate correlations
        corr_matrix = self.calculate_correlations(df_factors)

        # Select uncorrelated factors
        selected_factors = self.select_uncorrelated_factors(corr_matrix)

        if not selected_factors:
            print("❌ No suitable factors selected")
            return

        print(f"\n✅ Selected {len(selected_factors)} uncorrelated factors: {selected_factors}")

        # Construct portfolio
        portfolio = self.construct_risk_parity_portfolio(selected_factors, df_factors)

        if portfolio:
            print("\n🎯 PORTFOLIO CONSTRUCTION RESULTS")
            print("-" * 40)
            print(f"Selected Factors: {portfolio['selected_factors']}")
            stats = portfolio['stats']
            if stats:
                print(f"Total Return: {stats.get('total_return', 0):.1%}")
                print(f"Annual Return: {stats.get('annual_return', 0):.1%}")
                print(f"Annual Volatility: {stats.get('annual_volatility', 0):.1%}")
                print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {stats.get('max_drawdown', 0):.1%}")

            # Save portfolio results
            self._save_portfolio_results(portfolio)

        print("\n✅ Analysis Complete!")
        print(f"📊 Correlation heatmap saved to: {self.results_dir / 'factor_correlations.png'}")

    def _save_portfolio_results(self, portfolio: Dict[str, Any]):
        """Save portfolio construction results"""
        results_file = self.results_dir / 'portfolio_construction_results.json'

        # Convert to serializable format
        serializable_results = {
            'selected_factors': portfolio['selected_factors'],
            'stats': portfolio['stats'],
            'portfolio_returns': portfolio['portfolio_returns'].tolist() if 'portfolio_returns' in portfolio else [],
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"💾 Portfolio results saved to: {results_file}")


def main():
    """Main execution function"""
    analyzer = FactorPortfolioConstructor()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()