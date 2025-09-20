#!/usr/bin/env python3
"""
Create IBD 50 Database Schema

This script creates the necessary database tables and views for IBD 50 stock management.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ibd50_schema():
    """Create IBD 50 database schema"""

    try:
        # Import database manager
        from modules.database.railway_db_manager import RailwayPostgreSQLManager

        # Initialize database connection
        db_manager = RailwayPostgreSQLManager()
        session = db_manager.get_session()

        # Read SQL schema file
        schema_file = Path(__file__).parent / "create_ibd50_schema.sql"
        with open(schema_file, 'r') as f:
            sql_content = f.read()

        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

        # Execute each statement
        for i, statement in enumerate(statements, 1):
            if statement:
                try:
                    logger.info(f"Executing statement {i}/{len(statements)}...")
                    session.execute(statement)
                    logger.info(f"Statement {i} executed successfully")
                except Exception as e:
                    logger.warning(f"Statement {i} failed (may already exist): {e}")

        # Commit changes
        session.commit()
        logger.info("IBD 50 schema creation completed successfully")

        # Test the view
        logger.info("Testing v_current_ibd50 view...")
        result = session.execute("SELECT COUNT(*) FROM ai_trading.v_current_ibd50")
        count = result.fetchone()[0]
        logger.info(f"v_current_ibd50 view accessible, {count} records")

        session.close()
        return True

    except ImportError:
        logger.warning("RailwayPostgreSQLManager not found, using direct connection")

        # Fallback to direct psycopg2 connection
        try:
            import psycopg2
            from psycopg2 import sql

            DATABASE_URL = "postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway"

            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            cursor = conn.cursor()

            # Read and execute SQL file
            schema_file = Path(__file__).parent / "create_ibd50_schema.sql"
            with open(schema_file, 'r') as f:
                sql_content = f.read()

            cursor.execute(sql_content)
            logger.info("IBD 50 schema created successfully via direct connection")

            # Test the view
            cursor.execute("SELECT COUNT(*) FROM ai_trading.v_current_ibd50")
            count = cursor.fetchone()[0]
            logger.info(f"v_current_ibd50 view accessible, {count} records")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to create schema via direct connection: {e}")
            return False

    except Exception as e:
        logger.error(f"Failed to create IBD 50 schema: {e}")
        return False

def populate_sample_data():
    """Populate sample IBD 50 data for testing"""

    try:
        from modules.database.railway_db_manager import RailwayPostgreSQLManager

        db_manager = RailwayPostgreSQLManager()
        session = db_manager.get_session()

        # Sample IBD 50 stocks with metadata
        sample_stocks = [
            ('IREN', 'Iris Energy Limited', 'Financial Services', 'Capital Markets', 'Small', 'NASDAQ'),
            ('CLS', 'Celestica Inc.', 'Technology', 'Electronic Components', 'Mid', 'NYSE'),
            ('ALAB', 'Astera Labs, Inc.', 'Technology', 'Semiconductors', 'Small', 'NASDAQ'),
            ('FUTU', 'Futu Holdings Limited', 'Financial Services', 'Capital Markets', 'Mid', 'NASDAQ'),
            ('PLTR', 'Palantir Technologies Inc.', 'Technology', 'Software-Infrastructure', 'Large', 'NYSE'),
            ('RKLB', 'Rocket Lab USA, Inc.', 'Industrials', 'Aerospace & Defense', 'Small', 'NASDAQ'),
            ('RDDT', 'Reddit, Inc.', 'Communication Services', 'Internet Content & Information', 'Mid', 'NYSE'),
            ('AMSC', 'American Superconductor Corporation', 'Industrials', 'Electrical Equipment', 'Small', 'NASDAQ'),
            ('HOOD', 'Robinhood Markets, Inc.', 'Financial Services', 'Capital Markets', 'Large', 'NASDAQ'),
            ('FIX', 'Comfort Systems USA, Inc.', 'Industrials', 'Construction & Engineering', 'Small', 'NYSE'),
            ('AGX', 'Argan, Inc.', 'Industrials', 'Engineering & Construction', 'Small', 'NYSE'),
            ('RYTM', 'Rhythm Pharmaceuticals, Inc.', 'Healthcare', 'Biotechnology', 'Small', 'NASDAQ'),
            ('MIRM', 'Mirum Pharmaceuticals, Inc.', 'Healthcare', 'Biotechnology', 'Small', 'NASDAQ'),
            ('OUST', 'Ouster, Inc.', 'Technology', 'Scientific & Technical Instruments', 'Small', 'NYSE'),
            ('GFI', 'Gold Fields Limited', 'Basic Materials', 'Gold', 'Large', 'NYSE'),
            ('WLDN', 'Willdan Group, Inc.', 'Industrials', 'Engineering & Construction', 'Small', 'NASDAQ'),
            ('AFRM', 'Affirm Holdings, Inc.', 'Financial Services', 'Consumer Finance', 'Mid', 'NASDAQ'),
            ('BZ', 'Kanzhun Limited', 'Communication Services', 'Internet Content & Information', 'Mid', 'NASDAQ'),
            ('ANET', 'Arista Networks, Inc.', 'Technology', 'Computer Hardware', 'Large', 'NYSE'),
            ('WGS', 'Wearable Health Solutions, Inc.', 'Healthcare', 'Medical Devices', 'Small', 'NASDAQ'),
            ('TFPM', 'Triple Flag Precious Metals Corp.', 'Basic Materials', 'Other Precious Metals & Mining', 'Small', 'NYSE'),
            ('APH', 'Amphenol Corporation', 'Technology', 'Electronic Components', 'Large', 'NYSE'),
            ('TARS', 'Tarsus Pharmaceuticals, Inc.', 'Healthcare', 'Drug Manufacturers-General', 'Small', 'NASDAQ'),
            ('ATAT', 'Atour Lifestyle Holdings Limited', 'Consumer Cyclical', 'Travel Services', 'Small', 'NASDAQ'),
            ('LIF', 'Life360, Inc.', 'Technology', 'Software-Application', 'Small', 'NASDAQ'),
            ('AEM', 'Agnico Eagle Mines Limited', 'Basic Materials', 'Gold', 'Large', 'NYSE'),
            ('RMBS', 'Rambus, Inc.', 'Technology', 'Semiconductors', 'Mid', 'NASDAQ'),
            ('ANIP', 'ANI Pharmaceuticals, Inc.', 'Healthcare', 'Drug Manufacturers-Specialty & Generic', 'Small', 'NASDAQ'),
            ('GH', 'Guardant Health, Inc.', 'Healthcare', 'Diagnostics & Research', 'Mid', 'NASDAQ'),
            ('SOFI', 'SoFi Technologies, Inc.', 'Financial Services', 'Credit Services', 'Large', 'NASDAQ'),
            ('KGC', 'Kinross Gold Corporation', 'Basic Materials', 'Gold', 'Large', 'NYSE'),
            ('EME', 'EMCOR Group, Inc.', 'Industrials', 'Engineering & Construction', 'Mid', 'NYSE'),
            ('AU', 'AngloGold Ashanti Limited', 'Basic Materials', 'Gold', 'Large', 'NYSE'),
            ('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 'Large', 'NASDAQ'),
            ('TBBK', 'The Bancorp, Inc.', 'Financial Services', 'Banks-Regional', 'Small', 'NASDAQ'),
            ('MEDP', 'Medpace Holdings, Inc.', 'Healthcare', 'Diagnostics & Research', 'Mid', 'NASDAQ'),
            ('DOCS', 'Doximity, Inc.', 'Healthcare', 'Health Information Services', 'Mid', 'NASDAQ'),
            ('ONC', 'Onconetix, Inc.', 'Healthcare', 'Biotechnology', 'Small', 'NASDAQ'),
            ('KNSA', 'Kiniksa Pharmaceuticals, Ltd.', 'Healthcare', 'Biotechnology', 'Small', 'NASDAQ'),
            ('STNE', 'StoneCo Ltd.', 'Technology', 'Software-Infrastructure', 'Mid', 'NASDAQ'),
            ('XPEV', 'XPeng Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 'Large', 'NYSE'),
            ('CCJ', 'Cameco Corporation', 'Energy', 'Uranium', 'Large', 'NYSE'),
            ('EGO', 'Eldorado Gold Corporation', 'Basic Materials', 'Gold', 'Mid', 'NYSE'),
            ('CVNA', 'Carvana Co.', 'Consumer Cyclical', 'Internet Retail', 'Large', 'NYSE'),
            ('BROS', 'Dutch Bros, Inc.', 'Consumer Cyclical', 'Restaurants', 'Small', 'NYSE'),
            ('TEM', 'Tempus AI, Inc.', 'Healthcare', 'Health Information Services', 'Small', 'NASDAQ'),
            ('BAP', 'Credicorp Ltd.', 'Financial Services', 'Banks-Regional', 'Large', 'NYSE'),
            ('WPM', 'Wheaton Precious Metals Corp.', 'Basic Materials', 'Other Precious Metals & Mining', 'Large', 'NYSE'),
            ('IBKR', 'Interactive Brokers Group, Inc.', 'Financial Services', 'Capital Markets', 'Large', 'NASDAQ'),
            ('PODD', 'Insulet Corporation', 'Healthcare', 'Medical Devices', 'Large', 'NASDAQ')
        ]

        # Insert stock metadata
        for symbol, company_name, sector, industry, market_cap, exchange in sample_stocks:
            session.execute("""
                INSERT INTO ai_trading.stock_metadata
                (symbol, company_name, sector, industry, market_cap_category, exchange)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap_category = EXCLUDED.market_cap_category,
                    exchange = EXCLUDED.exchange,
                    last_updated = CURRENT_TIMESTAMP
            """, (symbol, company_name, sector, industry, market_cap, exchange))

        # Insert sample rankings
        from datetime import date
        today = date.today()

        for i, (symbol, _, _, _, _, _) in enumerate(sample_stocks, 1):
            session.execute("""
                INSERT INTO ai_trading.ibd50_rankings
                (symbol, rank_position, ranking_date, composite_rating, eps_rating, rs_rating)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, ranking_date) DO UPDATE SET
                    rank_position = EXCLUDED.rank_position,
                    composite_rating = EXCLUDED.composite_rating,
                    eps_rating = EXCLUDED.eps_rating,
                    rs_rating = EXCLUDED.rs_rating,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, i, today, 95.0 - i * 0.5, 90.0 - i * 0.3, 85.0 - i * 0.4))

        session.commit()
        logger.info(f"Populated {len(sample_stocks)} IBD 50 stocks with sample data")

        session.close()
        return True

    except Exception as e:
        logger.error(f"Failed to populate sample data: {e}")
        return False

if __name__ == "__main__":
    print("Creating IBD 50 Database Schema...")
    print("=" * 50)

    if create_ibd50_schema():
        print("\nPopulating sample data...")
        populate_sample_data()

        print("\n" + "=" * 50)
        print("IBD 50 Database Setup Complete!")
        print("You can now run: python daily_launcher.py --health-check")
    else:
        print("Schema creation failed. Check logs for details.")
        sys.exit(1)