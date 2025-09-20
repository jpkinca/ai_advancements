#!/usr/bin/env python3
"""
Quick Setup Script for FAISS Pattern Recognition System

This script helps you get started with Phase 1 of the implementation roadmap.
It checks dependencies, verifies database connection, and runs a test pattern generation.

Usage:
    python quick_setup.py

Author: AI Assistant
Date: September 2, 2025
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major  0:
                logger.info(f"[SUCCESS] Generated {results['total_patterns']} test patterns")
                return True
            else:
                logger.warning("[WARNING] No patterns generated in test")
                return False
        
        # Run the test
        result = asyncio.run(test_runner())
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Test pattern generation failed: {e}")
        return False

def create_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'data_collection',
        'pattern_runners', 
        'faiss_engine',
        'tests',
        'logs',
        'output'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"[CREATED] Directory: {directory}")

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("FAISS PATTERN RECOGNITION SETUP COMPLETE")
    print("="*60)
    print("\nNext steps to implement the roadmap:")
    print("\n1. IMMEDIATE (Today):")
    print("   - Run: python pattern_generation_runner.py")
    print("   - Check generated patterns in PostgreSQL")
    print("   - Verify FAISS compatibility")
    
    print("\n2. THIS WEEK:")
    print("   - Create FAISS index manager")
    print("   - Build similarity search engine")
    print("   - Implement real-time pattern detection")
    
    print("\n3. WEEK 2:")
    print("   - Scale to 100+ stocks")
    print("   - Add performance monitoring")
    print("   - Integrate with trading pipeline")
    
    print("\n4. RESOURCES:")
    print("   - Roadmap: FAISS_IMPLEMENTATION_ROADMAP.md")
    print("   - Logs: pattern_generation.log")
    print("   - Database: Railway PostgreSQL")
    
    print("\n5. SUPPORT:")
    print("   - Check logs for any issues")
    print("   - Verify DATABASE_URL environment variable")
    print("   - Ensure market data access (IBKR/yfinance)")
    
    print("\n[READY] You're now ready to start Phase 1 implementation!")

def main():
    """Main setup function"""
    print("FAISS Pattern Recognition System - Quick Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create project structure
    create_project_structure()
    
    # Check database connection
    db_ok = check_database_connection()
    
    # Check pattern generators
    generators_ok = check_pattern_generators()
    
    # Run test if everything is OK
    if db_ok and generators_ok:
        test_ok = run_test_pattern_generation()
        if test_ok:
            print("\n[SUCCESS] All systems operational!")
        else:
            print("\n[WARNING] Setup complete but test failed - check logs")
    else:
        print("\n[WARNING] Setup issues detected - check error messages above")
    
    # Show next steps regardless
    display_next_steps()

if __name__ == "__main__":
    main()
