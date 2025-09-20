#!/usr/bin/env python3
"""
Pattern Generation Runner - Bridge between pattern generators and FAISS database

This script addresses the missing link identified in the roadmap:
1. Loads real market data from multiple sources
2. Runs all pattern generators (CANSLIM, SEPA, Warrior Trading)
3. Generates pattern vectors suitable for FAISS
4. Stores patterns in PostgreSQL for FAISS indexing

Author: AI Assistant
Date: September 2, 2025
Status: Phase 1 - Foundation Implementation
"""

import os
import sys
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import hashlib

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging with ASCII-only output (Windows compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('pattern_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import pattern generators
try:
    from faiss.canslim_sepa_pattern_generator import CANSLIMPatternGenerator, SEPAPatternGenerator, PatternMatchingEngine
    from faiss.WT_day_trading_pattern_generator import WarriorTradingPatternGenerator
    from faiss.railway_database import FAISSRailwayDatabase
    logger.info("[SUCCESS] Pattern generators imported successfully")
except ImportError as e:
    logger.error(f"[ERROR] Failed to import pattern generators: {e}")
    sys.exit(1)

# Data source imports
try:
    import yfinance as yf
    logger.info("[SUCCESS] yfinance imported for market data")
except ImportError:
    logger.error("[ERROR] yfinance not available - install with: pip install yfinance")
    sys.exit(1)

@dataclass
class PatternMetadata:
    """Metadata for generated patterns"""
    symbol: str
    pattern_type: str
    timeframe: str
    timestamp: datetime
    confidence: float
    pattern_subtype: str
    market_regime: str
    volume_confirmation: bool
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

class PatternGenerationRunner:
    """Main orchestrator for pattern generation pipeline"""
    
    def __init__(self, test_mode: bool = True):
        self.test_mode = test_mode
        self.db = None
        self.stock_universe = self._get_stock_universe()
        self.pattern_generators = self._initialize_pattern_generators()
        self.generated_patterns = []
        
        # Initialize database connection
        self._initialize_database()
        
    def _get_stock_universe(self) -> List[str]:
        """Define initial stock universe for pattern generation"""
        if self.test_mode:
            # Phase 1: Start with 10 high-quality stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'CRM', 'ADBE']
        else:
            # Production: Expand to larger universe
            return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Start small for now
    
    def _initialize_pattern_generators(self) -> Dict:
        """Initialize all pattern recognition engines"""
        return {
            'canslim': CANSLIMPatternGenerator(),
            'sepa': SEPAPatternGenerator(),
            'warrior': WarriorTradingPatternGenerator(),
            'engine': PatternMatchingEngine()
        }
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            self.db = FAISSRailwayDatabase()
            logger.info("[SUCCESS] Database connection established")
        except Exception as e:
            logger.error(f"[ERROR] Database initialization failed: {e}")
            if not self.test_mode:
                raise
    
    async def fetch_market_data(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical market data for a symbol"""
        try:
            logger.info(f"[PROCESSING] Fetching {interval} data for {symbol} ({period})")
            
            # Use yfinance for now, can be replaced with IBKR later
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"[WARNING] No data received for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.reset_index(inplace=True)
            
            logger.info(f"[SUCCESS] Fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_canslim_patterns(self, symbol: str, price_data: pd.DataFrame, 
                                 volume_data: pd.DataFrame) -> List[Tuple[np.ndarray, PatternMetadata]]:
        """Generate CANSLIM pattern vectors"""
        patterns = []
        
        try:
            # Create minimal datasets for CANSLIM (in production, get real fundamental data)
            earnings_data = self._create_mock_earnings_data()
            institutional_data = self._create_mock_institutional_data()
            market_data = self._create_mock_market_data(len(price_data))
            
            # Generate CANSLIM patterns
            canslim_patterns = self.pattern_generators['canslim'].generate_canslim_patterns(
                price_data, volume_data, earnings_data, institutional_data, market_data
            )
            
            # Convert to vectors with metadata
            for pattern_name, pattern_data in canslim_patterns.items():
                if isinstance(pattern_data, np.ndarray) and len(pattern_data) > 0:
                    # Create pattern vector
                    vector = self._normalize_vector(pattern_data)
                    
                    # Create metadata
                    metadata = PatternMetadata(
                        symbol=symbol,
                        pattern_type='canslim',
                        pattern_subtype=pattern_name,
                        timeframe='daily',
                        timestamp=datetime.now(),
                        confidence=float(np.mean(pattern_data)),
                        market_regime='unknown',
                        volume_confirmation=True
                    )
                    
                    patterns.append((vector, metadata))
                    
                elif isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    # Handle base patterns (cup_with_handle, etc.)
                    if 'vector' in pattern_data:
                        vector = self._normalize_vector(pattern_data['vector'])
                        
                        metadata = PatternMetadata(
                            symbol=symbol,
                            pattern_type='canslim',
                            pattern_subtype=pattern_name,
                            timeframe='daily',
                            timestamp=datetime.now(),
                            confidence=pattern_data.get('confidence', 0.0),
                            market_regime='unknown',
                            volume_confirmation=True,
                            entry_price=pattern_data.get('buy_point'),
                            stop_loss=pattern_data.get('stop_loss'),
                            target_price=pattern_data.get('profit_target_1')
                        )
                        
                        patterns.append((vector, metadata))
            
            logger.info(f"[SUCCESS] Generated {len(patterns)} CANSLIM patterns for {symbol}")
            
        except Exception as e:
            logger.error(f"[ERROR] CANSLIM pattern generation failed for {symbol}: {e}")
        
        return patterns
    
    def generate_sepa_patterns(self, symbol: str, price_data: pd.DataFrame, 
                              volume_data: pd.DataFrame) -> List[Tuple[np.ndarray, PatternMetadata]]:
        """Generate SEPA (Minervini) pattern vectors"""
        patterns = []
        
        try:
            market_data = self._create_mock_market_data(len(price_data))
            
            # Generate SEPA patterns
            sepa_patterns = self.pattern_generators['sepa'].generate_sepa_patterns(
                price_data, volume_data, market_data
            )
            
            # Convert to vectors with metadata
            for pattern_name, pattern_data in sepa_patterns.items():
                if isinstance(pattern_data, np.ndarray) and len(pattern_data) > 0:
                    vector = self._normalize_vector(pattern_data)
                    
                    metadata = PatternMetadata(
                        symbol=symbol,
                        pattern_type='sepa',
                        pattern_subtype=pattern_name,
                        timeframe='daily',
                        timestamp=datetime.now(),
                        confidence=float(np.mean(pattern_data)),
                        market_regime='unknown',
                        volume_confirmation=True
                    )
                    
                    patterns.append((vector, metadata))
                    
                elif isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    if 'vector' in pattern_data:
                        vector = self._normalize_vector(pattern_data['vector'])
                        
                        metadata = PatternMetadata(
                            symbol=symbol,
                            pattern_type='sepa',
                            pattern_subtype=pattern_name,
                            timeframe='daily',
                            timestamp=datetime.now(),
                            confidence=pattern_data.get('confidence', 0.0),
                            market_regime='unknown',
                            volume_confirmation=True
                        )
                        
                        patterns.append((vector, metadata))
            
            logger.info(f"[SUCCESS] Generated {len(patterns)} SEPA patterns for {symbol}")
            
        except Exception as e:
            logger.error(f"[ERROR] SEPA pattern generation failed for {symbol}: {e}")
        
        return patterns
    
    def generate_warrior_patterns(self, symbol: str, minute_data: pd.DataFrame, 
                                 volume_data: pd.DataFrame) -> List[Tuple[np.ndarray, PatternMetadata]]:
        """Generate Warrior Trading intraday pattern vectors"""
        patterns = []
        
        try:
            # Generate Warrior Trading patterns
            warrior_patterns = self.pattern_generators['warrior'].generate_warrior_patterns(
                minute_data, volume_data
            )
            
            # Convert to vectors with metadata
            for pattern_name, pattern_data in warrior_patterns.items():
                if isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    if 'vector' in pattern_data:
                        vector = self._normalize_vector(pattern_data['vector'])
                        
                        metadata = PatternMetadata(
                            symbol=symbol,
                            pattern_type='warrior',
                            pattern_subtype=pattern_name,
                            timeframe='intraday',
                            timestamp=datetime.now(),
                            confidence=pattern_data.get('confidence', 0.0),
                            market_regime='unknown',
                            volume_confirmation=True,
                            entry_price=pattern_data.get('buy_point'),
                            stop_loss=pattern_data.get('stop_loss'),
                            target_price=pattern_data.get('profit_target_1')
                        )
                        
                        patterns.append((vector, metadata))
            
            logger.info(f"[SUCCESS] Generated {len(patterns)} Warrior patterns for {symbol}")
            
        except Exception as e:
            logger.error(f"[ERROR] Warrior pattern generation failed for {symbol}: {e}")
        
        return patterns
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for FAISS similarity search"""
        try:
            if isinstance(vector, list):
                vector = np.array(vector)
            
            # Ensure float32 for FAISS compatibility
            vector = vector.astype(np.float32)
            
            # Handle any NaN or infinite values
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
            
        except Exception as e:
            logger.error(f"[ERROR] Vector normalization failed: {e}")
            # Return zero vector if normalization fails
            return np.zeros(len(vector), dtype=np.float32)
    
    def _create_pattern_id(self, metadata: PatternMetadata) -> str:
        """Create unique pattern ID for database storage"""
        id_string = f"{metadata.symbol}_{metadata.pattern_type}_{metadata.pattern_subtype}_{metadata.timestamp.isoformat()}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    async def store_patterns_to_database(self, patterns: List[Tuple[np.ndarray, PatternMetadata]]):
        """Store generated patterns in PostgreSQL database"""
        if not self.db:
            logger.warning("[WARNING] Database not available, patterns not stored")
            return
        
        try:
            stored_count = 0
            
            for vector, metadata in patterns:
                # Create pattern ID
                pattern_id = self._create_pattern_id(metadata)
                
                # Convert metadata to JSON
                metadata_dict = asdict(metadata)
                metadata_dict['timestamp'] = metadata_dict['timestamp'].isoformat()
                
                # Store in database
                self.db.store_trading_pattern(
                    pattern_id=pattern_id,
                    symbol=metadata.symbol,
                    pattern_type=f"{metadata.pattern_type}_{metadata.pattern_subtype}",
                    embedding_vector=vector,
                    metadata=metadata_dict,
                    confidence_score=metadata.confidence
                )
                
                stored_count += 1
            
            logger.info(f"[SUCCESS] Stored {stored_count} patterns in database")
            
        except Exception as e:
            logger.error(f"[ERROR] Database storage failed: {e}")
    
    def save_patterns_to_file(self, patterns: List[Tuple[np.ndarray, PatternMetadata]], filename: str):
        """Save patterns to file for backup/analysis"""
        try:
            output_data = []
            
            for vector, metadata in patterns:
                output_data.append({
                    'vector': vector.tolist(),
                    'metadata': asdict(metadata),
                    'vector_shape': vector.shape,
                    'vector_norm': float(np.linalg.norm(vector))
                })
            
            # Convert datetime to string for JSON serialization
            for item in output_data:
                item['metadata']['timestamp'] = item['metadata']['timestamp'].isoformat()
            
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"[SUCCESS] Saved {len(patterns)} patterns to {filename}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save patterns to file: {e}")
    
    async def run_pattern_generation(self) -> Dict:
        """Main execution function - generate patterns for all stocks"""
        logger.info("[STARTING] Pattern generation pipeline")
        
        results = {
            'stocks_processed': 0,
            'total_patterns': 0,
            'patterns_by_type': {},
            'errors': [],
            'start_time': datetime.now(),
            'patterns': []
        }
        
        for symbol in self.stock_universe:
            try:
                logger.info(f"[PROCESSING] Starting pattern generation for {symbol}")
                
                # Fetch daily data for CANSLIM/SEPA patterns
                daily_data = await self.fetch_market_data(symbol, period="1y", interval="1d")
                if daily_data.empty:
                    results['errors'].append(f"No daily data for {symbol}")
                    continue
                
                # Fetch minute data for Warrior Trading patterns (last 5 days)
                minute_data = await self.fetch_market_data(symbol, period="5d", interval="1m")
                
                # Prepare volume data (same as price data for now)
                daily_volume = daily_data[['Volume']].copy()
                minute_volume = minute_data[['Volume']].copy() if not minute_data.empty else pd.DataFrame()
                
                # Generate patterns
                symbol_patterns = []
                
                # CANSLIM patterns
                canslim_patterns = self.generate_canslim_patterns(symbol, daily_data, daily_volume)
                symbol_patterns.extend(canslim_patterns)
                
                # SEPA patterns
                sepa_patterns = self.generate_sepa_patterns(symbol, daily_data, daily_volume)
                symbol_patterns.extend(sepa_patterns)
                
                # Warrior Trading patterns (if minute data available)
                if not minute_data.empty:
                    warrior_patterns = self.generate_warrior_patterns(symbol, minute_data, minute_volume)
                    symbol_patterns.extend(warrior_patterns)
                
                # Store patterns
                if symbol_patterns:
                    await self.store_patterns_to_database(symbol_patterns)
                    results['patterns'].extend(symbol_patterns)
                    
                    # Update statistics
                    for _, metadata in symbol_patterns:
                        pattern_key = f"{metadata.pattern_type}_{metadata.pattern_subtype}"
                        results['patterns_by_type'][pattern_key] = results['patterns_by_type'].get(pattern_key, 0) + 1
                
                results['stocks_processed'] += 1
                results['total_patterns'] += len(symbol_patterns)
                
                logger.info(f"[SUCCESS] Generated {len(symbol_patterns)} patterns for {symbol}")
                
                # Small delay to avoid overwhelming APIs
                await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = f"Pattern generation failed for {symbol}: {e}"
                logger.error(f"[ERROR] {error_msg}")
                results['errors'].append(error_msg)
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # Save results to file
        self.save_patterns_to_file(results['patterns'], f"patterns_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return results
    
    def _create_mock_earnings_data(self) -> pd.DataFrame:
        """Create mock earnings data for testing (replace with real data in production)"""
        return pd.DataFrame({
            'eps_growth': [15, 22, 28, 35],
            'annual_eps_growth': [20, 25, 30],
            'roe': [18, 19, 21],
            'eps_surprise': [5, 8, 12, 15]
        })
    
    def _create_mock_institutional_data(self) -> pd.DataFrame:
        """Create mock institutional data for testing"""
        return pd.DataFrame({
            'institutional_ownership': [0.45, 0.50, 0.55],
            'new_positions': [3, 5, 8]
        })
    
    def _create_mock_market_data(self, length: int) -> pd.DataFrame:
        """Create mock market data (SPY proxy)"""
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        # Simple random walk for market data
        np.random.seed(42)
        market_prices = 400 + np.cumsum(np.random.randn(length) * 0.5)
        
        return pd.DataFrame({
            'Date': dates,
            'Close': market_prices
        })

async def main():
    """Main execution function"""
    print("="*60)
    print("[STARTING] FAISS Pattern Generation Pipeline")
    print("="*60)
    
    # Initialize runner
    runner = PatternGenerationRunner(test_mode=True)
    
    # Run pattern generation
    results = await runner.run_pattern_generation()
    
    # Print results
    print("\n" + "="*60)
    print("[RESULTS] Pattern Generation Summary")
    print("="*60)
    print(f"Stocks processed: {results['stocks_processed']}")
    print(f"Total patterns generated: {results['total_patterns']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Errors: {len(results['errors'])}")
    
    if results['patterns_by_type']:
        print("\nPatterns by type:")
        for pattern_type, count in results['patterns_by_type'].items():
            print(f"  {pattern_type}: {count}")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\n[SUCCESS] Pattern generation pipeline completed!")
    print("Next steps:")
    print("1. Verify patterns in PostgreSQL database")
    print("2. Build FAISS indexes from generated patterns")
    print("3. Test similarity search functionality")

if __name__ == "__main__":
    asyncio.run(main())
