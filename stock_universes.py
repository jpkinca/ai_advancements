"""
Stock Universe Configurations for Weekend AI Testing

Predefined stock lists optimized for different testing scenarios.
Customize these or add your own for weekend AI module testing.
"""

# Technology giants - High correlation, good for testing diversification
TECH_GIANTS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
    'NFLX', 'META', 'NVDA', 'ADBE', 'CRM'
]

# S&P 500 ETF sectors - Lower correlation, good for portfolio optimization
SP500_SECTORS = [
    'SPY',   # S&P 500
    'QQQ',   # NASDAQ 100
    'XLF',   # Financial
    'XLE',   # Energy
    'XLK',   # Technology
    'XLV',   # Healthcare
    'XLI',   # Industrial
    'XLP',   # Consumer Staples
    'XLY',   # Consumer Discretionary
    'XLU',   # Utilities
    'XLB',   # Materials
    'XLRE'   # Real Estate
]

# Diverse blue chips - Mixed sectors, good for comprehensive testing
DIVERSE_BLUE_CHIPS = [
    'AAPL', 'JPM', 'JNJ', 'XOM', 'PG',      # Mega caps
    'DIS', 'V', 'UNH', 'HD', 'WMT',         # Consumer/Services
    'PFE', 'BAC', 'KO', 'PEP', 'T',         # Traditional value
    'INTC', 'CSCO', 'CVX', 'MRK', 'ABT',    # Industrial/Tech
    'GE', 'F', 'C', 'ORCL', 'IBM'           # Cyclicals
]

# High volatility names - Good for testing pattern recognition
HIGH_VOLATILITY = [
    'TSLA', 'NVDA', 'AMD', 'NFLX', 'ZOOM',
    'PTON', 'ROKU', 'SQ', 'ARKK', 'MEME',
    'GME', 'AMC', 'BB', 'NOK', 'PLTR'
]

# Commodities and materials - Good for cycle analysis
COMMODITIES_MATERIALS = [
    'GLD', 'SLV', 'USO', 'UNG', 'DBA',      # Commodity ETFs
    'FCX', 'NEM', 'AA', 'X', 'CLF',         # Mining/Metals
    'XOM', 'CVX', 'COP', 'EOG', 'PSX'       # Energy
]

# Small test set - Quick testing (5-10 symbols)
SMALL_TEST = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'
]

# Medium test set - Balanced testing (15-20 symbols)
MEDIUM_TEST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'JPM', 'JNJ', 'V', 'PG', 'UNH',
    'SPY', 'QQQ', 'XLF', 'XLK', 'XLV'
]

# Large test set - Comprehensive testing (30+ symbols)
LARGE_TEST = DIVERSE_BLUE_CHIPS

# Your custom universe - Updated with your 50-stock production watchlist
YOUR_CUSTOM_UNIVERSE = [
    'IREN', 'CLS', 'ALAB', 'FUTU', 'PLTR', 'RKLB', 'RDDT', 'AMSC', 'HOOD', 'FIX',
    'AGX', 'RYTM', 'MIRM', 'OUST', 'GFI', 'WLDN', 'AFRM', 'BZ', 'ANET', 'WGS',
    'TFPM', 'APH', 'TARS', 'ATAT', 'LIF', 'AEM', 'RMBS', 'ANIP', 'GH', 'SOFI',
    'KGC', 'EME', 'AU', 'NVDA', 'TBBK', 'MEDP', 'DOCS', 'ONC', 'KNSA', 'STNE',
    'XPEV', 'CCJ', 'EGO', 'CVNA', 'BROS', 'TEM', 'BAP', 'WPM', 'IBKR', 'PODD'
]

# Production Watchlist Subsets
PRODUCTION_HIGH_PRIORITY = [
    'NVDA', 'PLTR', 'HOOD', 'RKLB', 'IREN', 'ANET', 'FUTU', 'RDDT', 'DOCS', 'SOFI', 
    'IBKR', 'STNE', 'TARS', 'AMSC', 'ALAB', 'MEDP', 'PODD', 'CCJ'
]

PRODUCTION_TECH_FOCUSED = [
    'NVDA', 'PLTR', 'ANET', 'FUTU', 'RDDT', 'DOCS', 'CLS', 'FIX', 'AGX', 'OUST', 
    'WLDN', 'BZ', 'WGS', 'APH', 'ATAT', 'GH', 'KNSA'
]

PRODUCTION_FINANCIAL = [
    'HOOD', 'SOFI', 'IBKR', 'STNE', 'AFRM', 'BAP', 'TBBK'
]

PRODUCTION_HEALTHCARE = [
    'ALAB', 'MEDP', 'PODD', 'ONC', 'ANIP', 'RMBS', 'RYTM', 'MIRM', 'LIF'
]

PRODUCTION_MINING = [
    'GFI', 'AEM', 'KGC', 'AU', 'WPM', 'EGO', 'CCJ'
]

PRODUCTION_ENERGY_AEROSPACE = [
    'IREN', 'AMSC', 'RKLB', 'TARS'
]

# All available universes
STOCK_UNIVERSES = {
    'production_watchlist': YOUR_CUSTOM_UNIVERSE,
    'production_high_priority': PRODUCTION_HIGH_PRIORITY,
    'production_tech': PRODUCTION_TECH_FOCUSED,
    'production_financial': PRODUCTION_FINANCIAL,
    'production_healthcare': PRODUCTION_HEALTHCARE,
    'production_mining': PRODUCTION_MINING,
    'production_energy_aerospace': PRODUCTION_ENERGY_AEROSPACE,
    'tech_giants': TECH_GIANTS,
    'sp500_sectors': SP500_SECTORS,
    'diverse_blue_chips': DIVERSE_BLUE_CHIPS,
    'high_volatility': HIGH_VOLATILITY,
    'commodities_materials': COMMODITIES_MATERIALS,
    'small_test': SMALL_TEST,
    'medium_test': MEDIUM_TEST,
    'large_test': LARGE_TEST,
    'custom': YOUR_CUSTOM_UNIVERSE
}

def get_universe(name: str):
    """Get a stock universe by name"""
    return STOCK_UNIVERSES.get(name, SMALL_TEST)

def list_universes():
    """List all available universes"""
    print("Available Stock Universes:")
    print("-" * 50)
    for name, symbols in STOCK_UNIVERSES.items():
        print(f"{name:20}: {len(symbols):2d} symbols - {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

def print_universe(name: str):
    """Print details of a specific universe"""
    if name in STOCK_UNIVERSES:
        symbols = STOCK_UNIVERSES[name]
        print(f"\n{name.upper()} Universe ({len(symbols)} symbols):")
        print("-" * 50)
        for i, symbol in enumerate(symbols, 1):
            print(f"{i:2d}. {symbol}")
    else:
        print(f"Universe '{name}' not found")

# Usage recommendations
USAGE_RECOMMENDATIONS = {
    'production_watchlist': "Full 50-stock watchlist - 2.8 hours runtime, 149MB data",
    'production_high_priority': "Critical & high priority stocks - 1.0 hour runtime, 53MB data",
    'production_tech': "Technology sector focus - 0.9 hours runtime, 50MB data",
    'production_financial': "Financial services focus - 0.4 hours runtime, 21MB data",
    'production_healthcare': "Healthcare & biotech focus - 0.5 hours runtime, 27MB data",
    'production_mining': "Mining & resources focus - 0.4 hours runtime, 21MB data",
    'production_energy_aerospace': "Energy & aerospace focus - 0.2 hours runtime, 12MB data",
    'small_test': "Quick testing - 5-10 minutes runtime",
    'medium_test': "Balanced testing - 10-20 minutes runtime", 
    'large_test': "Comprehensive testing - 20-40 minutes runtime",
    'tech_giants': "High correlation testing - Good for diversification algorithms",
    'sp500_sectors': "Low correlation testing - Good for portfolio optimization",
    'diverse_blue_chips': "Mixed testing - Good for comprehensive analysis",
    'high_volatility': "Pattern testing - Good for technical analysis",
    'commodities_materials': "Cycle testing - Good for frequency domain analysis"
}

def get_recommendation(name: str):
    """Get usage recommendation for a universe"""
    return USAGE_RECOMMENDATIONS.get(name, "General testing universe")

if __name__ == "__main__":
    list_universes()
    print("\nUsage Recommendations:")
    print("-" * 50)
    for name, rec in USAGE_RECOMMENDATIONS.items():
        print(f"{name:20}: {rec}")
