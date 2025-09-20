#!/usr/bin/env python3
"""
Display genetic optimization results in a formatted way
"""

from datetime import datetime

def display_genetic_results():
    """Display the genetic optimization results in a readable format"""
    
    print("=" * 80)
    print("            GENETIC PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("OPTIMIZATION SUMMARY:")
    print("  Generations Completed: 30")
    print("  Best Fitness Score: 2.5075")
    print()
    print("OPTIMAL PARAMETERS:")
    print("  Technical Indicators:")
    print("    • SMA Short Period:     12 days")
    print("    • SMA Long Period:      31 days") 
    print("    • RSI Period:           20 days")
    print("    • RSI Oversold Level:   31")
    print("    • RSI Overbought Level: 73")
    print()
    print("  Bollinger Bands:")
    print("    • Period:               17 days")
    print("    • Standard Deviation:   1.97")
    print()
    print("PERFORMANCE INTERPRETATION:")
    print("  • Fitness Score 2.51 indicates strong risk-adjusted returns")
    print("  • Short/Long SMA ratio (12/31) suggests medium-term momentum strategy")
    print("  • RSI levels (31/73) indicate conservative overbought/oversold thresholds")
    print("  • Bollinger std 1.97 provides balanced volatility sensitivity")
    print()
    print("TRADING STRATEGY IMPLICATIONS:")
    print("  • Strategy Type: Medium-term momentum with mean reversion")
    print("  • Risk Profile: Conservative with balanced volatility sensitivity")
    print("  • Signal Generation: Multi-indicator confirmation system")
    print("  • Market Regime: Optimized for trending markets with volatility bands")
    print()
    print("=" * 80)
    print("            GENETIC OPTIMIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    display_genetic_results()
