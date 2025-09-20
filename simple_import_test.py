#!/usr/bin/env python3
"""
Simple Import Test for FAISS Pattern Recognition

Quick test to verify all imports work correctly.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=== TESTING IMPORTS ===")

try:
    # Test FAISS patterns import
    from ai_advancements.faiss.canslim_sepa_pattern_generator import CANSLIMPatternGenerator, SEPAPatternGenerator
    print("[SUCCESS] CANSLIM and SEPA generators imported")
    
    from ai_advancements.faiss.WT_day_trading_pattern_generator import WarriorTradingPatternGenerator
    print("[SUCCESS] Warrior Trading generator imported")
    
    # Test IBKR connection import
    from ibkr_api.connect_me import get_managed_ibkr_connection
    print("[SUCCESS] IBKR connection manager imported")
    
    # Test pattern generators can be instantiated
    canslim = CANSLIMPatternGenerator()
    sepa = SEPAPatternGenerator()
    warrior = WarriorTradingPatternGenerator()
    print("[SUCCESS] All pattern generators instantiated")
    
    print("\n=== ALL IMPORTS SUCCESSFUL ===")
    print("Pattern generators are ready to use!")
    
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
