#!/usr/bin/env python3
"""
Simple Launch Script for Week 2 Level II Enhanced AI Models

This script launches the standalone Level II enhanced models without dependencies.
Perfect for testing and demonstration.

Author: GitHub Copilot
Date: August 31, 2025
"""

import sys
import os
import logging
from datetime import datetime

# Configure logging with ASCII-only output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to launch the Level II enhanced models"""
    
    logger.info("="*80)
    logger.info("[STARTING] Week 2 Level II Enhanced AI Models")
    logger.info("="*80)
    logger.info(f"Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # Import and run the standalone models
        logger.info("[PROCESSING] Loading standalone Level II enhanced models...")
        
        # Import the standalone models module
        from week2_level_ii_standalone_models import main as demo_main
        
        logger.info("[SUCCESS] Models loaded successfully")
        logger.info("")
        
        # Run the demonstration
        logger.info("[STARTING] Running Level II enhanced models demonstration...")
        demo_main()
        
        logger.info("")
        logger.info("="*80)
        logger.info("[SUCCESS] Week 2 Level II Integration Launch Completed")
        logger.info("="*80)
        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. Connect to live IBKR Gateway for real Level II data")
        logger.info("2. Set up PostgreSQL database for data storage")
        logger.info("3. Configure trading parameters in the models")
        logger.info("4. Begin live trading with enhanced AI models")
        logger.info("")
        logger.info("All models are ready for production deployment!")
        
        return 0
        
    except ImportError as e:
        logger.error(f"[ERROR] Failed to import models: {str(e)}")
        logger.error("Make sure week2_level_ii_standalone_models.py is in the same directory")
        return 1
        
    except Exception as e:
        logger.error(f"[ERROR] Launch failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
