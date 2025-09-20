#!/usr/bin/env python3
"""
IBD 50 Database Manager

Python interface for managing IBD 50 stocks in the database.
Replaces the hardcoded list approach with database-driven stock universe management.

Features:
- Load IBD 50 stocks from database
- Update rankings and metadata
- Track historical changes
- Sector and industry analysis
- Integration with existing VLM training pipeline
"""

import os
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text

# Configure logging
logger = logging.getLogger(__name__)

class IBD50DatabaseManager:
    """
    Database manager for IBD 50 stock universe
    
    Handles all database operations for IBD 50 stocks including:
    - Loading current stock list
    - Updating rankings
    - Historical tracking
    - Sector analysis
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: PostgreSQL connection string. If None, uses environment variable.
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            logger.warning("No database URL provided. Using fallback hardcoded list.")
            self.use_database = False
        else:
            self.use_database = True
            self.engine = create_engine(self.database_url)
        
        # Fallback to hardcoded list if database unavailable
        self.fallback_stocks = [
            'IREN', 'CLS', 'ALAB', 'FUTU', 'PLTR', 'RKLB', 'RDDT', 'AMSC', 'HOOD', 'FIX',
            'AGX', 'RYTM', 'MIRM', 'OUST', 'GFI', 'WLDN', 'AFRM', 'BZ', 'ANET', 'WGS',
            'TFPM', 'APH', 'TARS', 'ATAT', 'LIF', 'AEM', 'RMBS', 'ANIP', 'GH', 'SOFI',
            'KGC', 'EME', 'AU', 'NVDA', 'TBBK', 'MEDP', 'DOCS', 'ONC', 'KNSA', 'STNE',
            'XPEV', 'CCJ', 'EGO', 'CVNA', 'BROS', 'TEM', 'BAP', 'WPM', 'IBKR', 'PODD'
        ]
    
    def test_connection(self) -> bool:
        """Test database connection"""
        if not self.use_database:
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def get_ibd50_stocks(self, as_dataframe: bool = False) -> List[str] | pd.DataFrame:
        """
        Get current IBD 50 stock list
        
        Args:
            as_dataframe: If True, returns DataFrame with full metadata
            
        Returns:
            List of stock symbols or DataFrame with metadata
        """
        if not self.use_database or not self.test_connection():
            logger.warning("Using fallback hardcoded IBD 50 list")
            if as_dataframe:
                return pd.DataFrame({'symbol': self.fallback_stocks})
            return self.fallback_stocks
        
        try:
            query = """
            SELECT 
                symbol,
                company_name,
                sector,
                industry,
                market_cap_category,
                exchange,
                rank_position,
                metadata,
                added_date
            FROM ai_trading.v_current_ibd50 
            ORDER BY rank_position
            """
            
            df = pd.read_sql_query(query, self.engine)
            
            if as_dataframe:
                return df
            else:
                return df['symbol'].tolist()
                
        except Exception as e:
            logger.error(f"Failed to load IBD 50 from database: {e}")
            logger.warning("Using fallback hardcoded list")
            if as_dataframe:
                return pd.DataFrame({'symbol': self.fallback_stocks})
            return self.fallback_stocks
    
    def get_ibd50_with_ratings(self) -> pd.DataFrame:
        """Get IBD 50 stocks with latest ratings"""
        if not self.use_database or not self.test_connection():
            logger.warning("Database unavailable. Returning basic stock list.")
            return pd.DataFrame({'symbol': self.fallback_stocks})
        
        try:
            query = """
            SELECT * FROM ai_trading.v_ibd50_with_ratings 
            ORDER BY rank_position
            """
            
            return pd.read_sql_query(query, self.engine)
            
        except Exception as e:
            logger.error(f"Failed to load IBD 50 ratings: {e}")
            return pd.DataFrame({'symbol': self.fallback_stocks})
    
    def get_stocks_by_sector(self, sector: str = None) -> pd.DataFrame:
        """
        Get IBD 50 stocks filtered by sector
        
        Args:
            sector: Sector to filter by. If None, returns all with sector grouping.
        """
        df = self.get_ibd50_stocks(as_dataframe=True)
        
        if sector:
            return df[df['sector'] == sector]
        else:
            # Return sector summary with simple column structure
            sector_summary = df.groupby('sector').agg({
                'symbol': 'count',
                'rank_position': 'min'
            }).round(2)
            
            # Flatten column names for easier access
            sector_summary.columns = ['stock_count', 'best_rank']
            sector_summary = sector_summary.reset_index()
            
            return sector_summary
    
    def get_top_stocks(self, limit: int = 10, metric: str = 'rank_position') -> pd.DataFrame:
        """
        Get top IBD 50 stocks by specified metric
        
        Args:
            limit: Number of stocks to return
            metric: Metric to sort by ('rank_position', 'composite_rating', etc.)
        """
        df = self.get_ibd50_with_ratings()
        
        if metric == 'rank_position':
            return df.head(limit)
        elif metric in df.columns:
            return df.nlargest(limit, metric)
        else:
            logger.warning(f"Metric '{metric}' not found. Using rank_position.")
            return df.head(limit)
    
    def update_stock_rankings(self, rankings: List[Dict[str, Any]], 
                            ranking_date: date = None) -> bool:
        """
        Update IBD 50 rankings for a specific date
        
        Args:
            rankings: List of dictionaries with ranking data
            ranking_date: Date for the rankings. If None, uses current date.
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_database or not self.test_connection():
            logger.error("Database not available for rankings update")
            return False
        
        if ranking_date is None:
            ranking_date = date.today()
        
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    # Delete existing rankings for this date
                    conn.execute(text("""
                        DELETE FROM ai_trading.ibd50_rankings 
                        WHERE ranking_date = :date
                    """), {"date": ranking_date})
                    
                    # Insert new rankings
                    for ranking in rankings:
                        conn.execute(text("""
                            INSERT INTO ai_trading.ibd50_rankings 
                            (symbol, rank_position, ranking_date, composite_rating, 
                             eps_rating, rs_rating, group_rs_rating, smr_rating, acc_dis_rating)
                            VALUES (:symbol, :rank_position, :ranking_date, :composite_rating,
                                   :eps_rating, :rs_rating, :group_rs_rating, :smr_rating, :acc_dis_rating)
                        """), {
                            "symbol": ranking['symbol'],
                            "rank_position": ranking['rank_position'],
                            "ranking_date": ranking_date,
                            "composite_rating": ranking.get('composite_rating'),
                            "eps_rating": ranking.get('eps_rating'),
                            "rs_rating": ranking.get('rs_rating'),
                            "group_rs_rating": ranking.get('group_rs_rating'),
                            "smr_rating": ranking.get('smr_rating'),
                            "acc_dis_rating": ranking.get('acc_dis_rating')
                        })
                    
                    logger.info(f"Updated {len(rankings)} IBD 50 rankings for {ranking_date}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to update rankings: {e}")
            return False
    
    def get_historical_rankings(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical rankings for a specific stock
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            DataFrame with historical rankings
        """
        if not self.use_database or not self.test_connection():
            logger.warning("Database not available for historical data")
            return pd.DataFrame()
        
        try:
            query = """
            SELECT * FROM ai_trading.ibd50_rankings 
            WHERE symbol = %s 
                AND ranking_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY ranking_date DESC
            """
            
            return pd.read_sql_query(query, self.engine, params=[symbol, days])
            
        except Exception as e:
            logger.error(f"Failed to get historical rankings: {e}")
            return pd.DataFrame()
    
    def analyze_sector_performance(self) -> pd.DataFrame:
        """Analyze IBD 50 performance by sector"""
        df = self.get_ibd50_with_ratings()
        
        if 'composite_rating' in df.columns:
            sector_analysis = df.groupby('sector').agg({
                'symbol': 'count',
                'composite_rating': ['mean', 'median', 'std'],
                'rank_position': ['min', 'max', 'mean']
            }).round(2)
            
            sector_analysis.columns = [
                'stock_count', 'avg_composite', 'median_composite', 'std_composite',
                'best_rank', 'worst_rank', 'avg_rank'
            ]
            
            return sector_analysis.sort_values('avg_composite', ascending=False)
        else:
            return df.groupby('sector').size().to_frame('stock_count')
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export current IBD 50 data as dictionary for compatibility"""
        df = self.get_ibd50_stocks(as_dataframe=True)
        
        return {
            'stocks': df['symbol'].tolist(),
            'metadata': df.to_dict('records'),
            'last_updated': datetime.now().isoformat(),
            'total_stocks': len(df),
            'source': 'database' if self.use_database else 'fallback'
        }
    
    def print_summary(self):
        """Print summary of current IBD 50 universe"""
        print("\n" + "="*60)
        print("IBD 50 STOCK UNIVERSE SUMMARY")
        print("="*60)
        
        df = self.get_ibd50_stocks(as_dataframe=True)
        
        print(f"Total Stocks: {len(df)}")
        print(f"Data Source: {'Database' if self.use_database else 'Fallback List'}")
        
        if 'sector' in df.columns:
            print(f"\nSector Breakdown:")
            sector_counts = df['sector'].value_counts()
            for sector, count in sector_counts.items():
                print(f"  {sector}: {count} stocks")
        
        print(f"\nTop 10 Stocks:")
        top_stocks = df.head(10)
        for idx, row in top_stocks.iterrows():
            if 'rank_position' in row:
                print(f"  {row['rank_position']:2d}. {row['symbol']:<6} {row.get('company_name', 'N/A')}")
            else:
                print(f"  {idx+1:2d}. {row['symbol']}")
        
        print("="*60)

# Integration function for backward compatibility
def get_ibd50_universe() -> List[str]:
    """
    Get IBD 50 stock universe (backward compatible function)
    
    Returns:
        List of stock symbols
    """
    manager = IBD50DatabaseManager()
    return manager.get_ibd50_stocks()

# Enhanced function for detailed data
def get_ibd50_universe_detailed() -> pd.DataFrame:
    """
    Get detailed IBD 50 stock universe data
    
    Returns:
        DataFrame with stock metadata
    """
    manager = IBD50DatabaseManager()
    return manager.get_ibd50_stocks(as_dataframe=True)

# Usage examples and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IBD 50 Database Manager")
    parser.add_argument('--action', choices=['summary', 'list', 'sectors', 'ratings'], 
                       default='summary', help='Action to perform')
    parser.add_argument('--sector', type=str, help='Filter by sector')
    parser.add_argument('--limit', type=int, default=10, help='Limit results')
    
    args = parser.parse_args()
    
    manager = IBD50DatabaseManager()
    
    if args.action == 'summary':
        manager.print_summary()
    elif args.action == 'list':
        stocks = manager.get_ibd50_stocks()
        print(f"IBD 50 Stocks ({len(stocks)}):")
        for i, symbol in enumerate(stocks, 1):
            print(f"{i:2d}. {symbol}")
    elif args.action == 'sectors':
        sector_data = manager.get_stocks_by_sector(args.sector)
        print(sector_data)
    elif args.action == 'ratings':
        ratings_data = manager.get_ibd50_with_ratings()
        print(ratings_data.head(args.limit))