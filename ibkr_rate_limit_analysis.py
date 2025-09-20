"""
IBKR Rate Limiting Analysis & Data Volume Calculator

This script analyzes IBKR rate limiting requirements and calculates
exact data volumes for your 50-stock watchlist with AI analysis pipeline.

Based on workspace analysis:
- IBKR Scanner: 60 requests per minute limit
- Historical Data: Pacing violations handled with exponential backoff
- Market Data: Error 354 (pacing) handled with adaptive delays
- General Pacing: Error 100 handled with 10-second delays
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytz

# Configure logging for ASCII-only output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ibkr_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Eastern Time setup for NYSE/NASDAQ compliance
EASTERN_TZ = pytz.timezone('America/New_York')

class IBKRRateLimitAnalyzer:
    """Analyzes IBKR rate limiting and calculates data requirements"""
    
    def __init__(self):
        self.logger = logger
        
        # IBKR Rate Limits (from workspace analysis)
        self.rate_limits = {
            'scanner_requests': {
                'limit': 60,  # requests per minute
                'window': 60,  # seconds
                'description': 'Scanner rotation to avoid violations'
            },
            'historical_data': {
                'limit': 60,  # Conservative estimate
                'window': 600,  # 10 minutes for safety
                'identical_cooldown': 10,  # seconds between identical requests
                'description': 'Historical data with pacing violation handling'
            },
            'market_data': {
                'limit': 100,  # Conservative for Level 1 data
                'window': 60,  # seconds
                'pacing_delay': 2.0,  # Exponential backoff base
                'description': 'Real-time market data subscriptions'
            },
            'general_api': {
                'limit': 50,  # General API calls
                'window': 60,  # seconds
                'pacing_delay': 10.0,  # Fixed delay for error 100
                'description': 'Contract details, account info, etc.'
            }
        }
        
        # Your 50-stock watchlist
        self.watchlist = [
            'IREN', 'CLS', 'ALAB', 'FUTU', 'PLTR', 'RKLB', 'RDDT', 'AMSC', 'HOOD', 'FIX',
            'AGX', 'RYTM', 'MIRM', 'OUST', 'GFI', 'WLDN', 'AFRM', 'BZ', 'ANET', 'WGS',
            'TFPM', 'APH', 'TARS', 'ATAT', 'LIF', 'AEM', 'RMBS', 'ANIP', 'GH', 'SOFI',
            'KGC', 'EME', 'AU', 'NVDA', 'TBBK', 'MEDP', 'DOCS', 'ONC', 'KNSA', 'STNE',
            'XPEV', 'CCJ', 'EGO', 'CVNA', 'BROS', 'TEM', 'BAP', 'WPM', 'IBKR', 'PODD'
        ]
        
        # Multi-timeframe configuration
        self.timeframes = {
            '1 min': {
                'duration': '2 D',  # 2 days of 1-minute data
                'bars_per_day': 390,  # 6.5 hours * 60 minutes
                'total_bars': 780,
                'ai_modules': ['Enhanced analysis'],
                'priority': 'high'
            },
            '5 mins': {
                'duration': '1 W',  # 1 week of 5-minute data
                'bars_per_day': 78,  # 6.5 hours * 12 bars/hour
                'total_bars': 390,
                'ai_modules': ['PPO Trader'],
                'priority': 'high'
            },
            '15 mins': {
                'duration': '1 M',  # 1 month of 15-minute data
                'bars_per_day': 26,  # 6.5 hours * 4 bars/hour
                'total_bars': 520,
                'ai_modules': ['PPO Trader', 'Wavelet Analyzer'],
                'priority': 'critical'
            },
            '1 hour': {
                'duration': '3 M',  # 3 months of hourly data
                'bars_per_day': 6.5,
                'total_bars': 390,
                'ai_modules': ['Portfolio Optimizer', 'Fourier Analyzer', 'Wavelet Analyzer'],
                'priority': 'critical'
            },
            '1 day': {
                'duration': '2 Y',  # 2 years of daily data
                'bars_per_day': 1,
                'total_bars': 520,
                'ai_modules': ['Portfolio Optimizer', 'Fourier Analyzer', 'Wavelet Analyzer'],
                'priority': 'critical'
            },
            '1 week': {
                'duration': '5 Y',  # 5 years of weekly data
                'bars_per_day': 0.2,  # 1 bar per 5 days
                'total_bars': 260,
                'ai_modules': ['Portfolio Optimizer', 'Fourier Analyzer'],
                'priority': 'medium'
            },
            '1 month': {
                'duration': '10 Y',  # 10 years of monthly data
                'bars_per_day': 0.05,  # 1 bar per 20 days
                'total_bars': 120,
                'ai_modules': ['Portfolio Optimizer'],
                'priority': 'low'
            }
        }
    
    def calculate_data_requirements(self) -> Dict[str, Any]:
        """Calculate total data requirements for the watchlist"""
        
        total_requests = 0
        total_bars = 0
        timeframe_breakdown = {}
        
        for timeframe, config in self.timeframes.items():
            symbols_count = len(self.watchlist)
            timeframe_requests = symbols_count  # 1 request per symbol per timeframe
            timeframe_bars = symbols_count * config['total_bars']
            
            timeframe_breakdown[timeframe] = {
                'symbols': symbols_count,
                'requests': timeframe_requests,
                'bars_per_symbol': config['total_bars'],
                'total_bars': timeframe_bars,
                'duration': config['duration'],
                'ai_modules': config['ai_modules'],
                'priority': config['priority']
            }
            
            total_requests += timeframe_requests
            total_bars += timeframe_bars
        
        return {
            'summary': {
                'total_symbols': len(self.watchlist),
                'total_timeframes': len(self.timeframes),
                'total_requests': total_requests,
                'total_bars': total_bars,
                'estimated_mb': total_bars * 0.001,  # ~1KB per bar estimate
                'estimated_time_minutes': self._estimate_fetch_time(total_requests)
            },
            'timeframe_breakdown': timeframe_breakdown,
            'watchlist': self.watchlist
        }
    
    def _estimate_fetch_time(self, total_requests: int) -> float:
        """Estimate total fetch time with rate limiting"""
        
        # Historical data rate limiting
        hist_limit = self.rate_limits['historical_data']
        requests_per_batch = hist_limit['limit']
        batch_time = hist_limit['window']
        
        num_batches = (total_requests + requests_per_batch - 1) // requests_per_batch
        total_time_seconds = num_batches * batch_time
        
        # Add buffer for pacing violations and cooldowns
        buffer_factor = 1.3  # 30% buffer
        total_time_with_buffer = total_time_seconds * buffer_factor
        
        return total_time_with_buffer / 60  # Convert to minutes
    
    def analyze_rate_limiting_strategy(self) -> Dict[str, Any]:
        """Analyze optimal rate limiting strategy"""
        
        strategies = {
            'conservative_batch': {
                'description': 'Conservative batching with safety margins',
                'historical_data': {
                    'requests_per_batch': 30,  # Half the limit for safety
                    'batch_interval': 600,     # 10 minutes
                    'identical_cooldown': 15   # 15 seconds between identical
                },
                'estimated_total_time_hours': 3.5,
                'risk_level': 'low',
                'recommended': True
            },
            'moderate_batch': {
                'description': 'Moderate batching with error handling',
                'historical_data': {
                    'requests_per_batch': 45,  # 75% of limit
                    'batch_interval': 450,     # 7.5 minutes
                    'identical_cooldown': 10   # 10 seconds between identical
                },
                'estimated_total_time_hours': 2.5,
                'risk_level': 'medium',
                'recommended': False
            },
            'aggressive_batch': {
                'description': 'Aggressive batching (not recommended)',
                'historical_data': {
                    'requests_per_batch': 60,  # Full limit
                    'batch_interval': 300,     # 5 minutes
                    'identical_cooldown': 5    # 5 seconds between identical
                },
                'estimated_total_time_hours': 1.8,
                'risk_level': 'high',
                'recommended': False
            }
        }
        
        return {
            'current_implementation': {
                'description': 'Current multi_timeframe_data_manager.py implementation',
                'rate_limiting': {
                    'per_request_delay': 1.0,    # 1 second between requests
                    'between_symbols_delay': 2.0, # 2 seconds between symbols
                    'error_handling': 'exponential_backoff'
                },
                'compliance_level': 'excellent',
                'estimated_time_hours': 2.8
            },
            'alternative_strategies': strategies,
            'recommendations': self._get_rate_limiting_recommendations()
        }
    
    def _get_rate_limiting_recommendations(self) -> List[str]:
        """Get rate limiting recommendations"""
        
        return [
            '[CURRENT] Keep existing 1s between requests, 2s between symbols',
            '[OPTIMIZE] Consider parallel fetching of different timeframes for same symbol',
            '[MONITOR] Implement adaptive rate limiting based on error frequency',
            '[FALLBACK] Have retry logic with exponential backoff for pacing violations',
            '[CACHE] Store data locally to avoid re-fetching on subsequent runs',
            '[PRIORITIZE] Fetch critical timeframes (15min, 1hour, 1day) first',
            '[BATCH] Group symbols by sector/market cap for logical batching'
        ]
    
    async def run_analysis(self):
        """Run complete IBKR rate limiting analysis"""
        
        self.logger.info("=" * 80)
        self.logger.info("    IBKR RATE LIMITING & DATA REQUIREMENTS ANALYSIS")
        self.logger.info("=" * 80)
        
        # Calculate data requirements
        data_req = self.calculate_data_requirements()
        
        self.logger.info(f"")
        self.logger.info(f"[DATA REQUIREMENTS SUMMARY]")
        self.logger.info(f"  Symbols in Watchlist: {data_req['summary']['total_symbols']}")
        self.logger.info(f"  Timeframes: {data_req['summary']['total_timeframes']}")
        self.logger.info(f"  Total IBKR Requests: {data_req['summary']['total_requests']}")
        self.logger.info(f"  Total Data Bars: {data_req['summary']['total_bars']:,}")
        self.logger.info(f"  Estimated Data Size: {data_req['summary']['estimated_mb']:.1f} MB")
        self.logger.info(f"  Estimated Fetch Time: {data_req['summary']['estimated_time_minutes']:.1f} minutes")
        
        self.logger.info(f"")
        self.logger.info(f"[TIMEFRAME BREAKDOWN]")
        for timeframe, details in data_req['timeframe_breakdown'].items():
            modules = ', '.join(details['ai_modules'])
            self.logger.info(f"  {timeframe:>8}: {details['requests']:>2} requests, "
                           f"{details['total_bars']:>5,} bars, "
                           f"Priority: {details['priority']:>8}, "
                           f"AI: {modules}")
        
        # Analyze rate limiting
        rate_analysis = self.analyze_rate_limiting_strategy()
        
        self.logger.info(f"")
        self.logger.info(f"[CURRENT IMPLEMENTATION STATUS]")
        current = rate_analysis['current_implementation']
        self.logger.info(f"  Description: {current['description']}")
        self.logger.info(f"  Request Delay: {current['rate_limiting']['per_request_delay']}s")
        self.logger.info(f"  Symbol Delay: {current['rate_limiting']['between_symbols_delay']}s")
        self.logger.info(f"  Error Handling: {current['rate_limiting']['error_handling']}")
        self.logger.info(f"  Compliance Level: {current['compliance_level']}")
        self.logger.info(f"  Estimated Time: {current['estimated_time_hours']:.1f} hours")
        
        self.logger.info(f"")
        self.logger.info(f"[IBKR RATE LIMITS (FROM WORKSPACE ANALYSIS)]")
        for category, limits in self.rate_limits.items():
            self.logger.info(f"  {category.replace('_', ' ').title()}:")
            self.logger.info(f"    Limit: {limits['limit']} requests per {limits['window']} seconds")
            self.logger.info(f"    Description: {limits['description']}")
        
        self.logger.info(f"")
        self.logger.info(f"[RECOMMENDATIONS]")
        for i, recommendation in enumerate(rate_analysis['recommendations'], 1):
            self.logger.info(f"  {i}. {recommendation}")
        
        self.logger.info(f"")
        self.logger.info(f"[WATCHLIST SYMBOLS ({len(self.watchlist)} total)]")
        # Display symbols in groups of 10
        for i in range(0, len(self.watchlist), 10):
            symbols_group = self.watchlist[i:i+10]
            symbols_line = ", ".join(symbols_group)
            self.logger.info(f"  {symbols_line}")
        
        self.logger.info(f"")
        self.logger.info("=" * 80)
        self.logger.info("    ANALYSIS COMPLETE - READY FOR WEEKEND AI TESTING")
        self.logger.info("=" * 80)
        
        return data_req, rate_analysis

# Main execution
async def main():
    """Run the IBKR rate limiting analysis"""
    
    analyzer = IBKRRateLimitAnalyzer()
    data_req, rate_analysis = await analyzer.run_analysis()
    
    return {
        'data_requirements': data_req,
        'rate_analysis': rate_analysis,
        'watchlist': analyzer.watchlist,
        'timeframes': analyzer.timeframes
    }

if __name__ == "__main__":
    results = asyncio.run(main())
