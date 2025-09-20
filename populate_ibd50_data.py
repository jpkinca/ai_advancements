#!/usr/bin/env python3
"""
Populate IBD 50 Sample Data

This script populates the IBD 50 database with sample stock data.
"""

import psycopg2
from datetime import date

def main():
    # Connect to database
    conn = psycopg2.connect('postgresql://postgres:TAqEkujnMknVURCcrYTIDOzQXbgBNtSX@turntable.proxy.rlwy.net:10410/railway')
    cursor = conn.cursor()

    # Sample IBD 50 stocks
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

    print(f"Populating {len(sample_stocks)} IBD 50 stocks...")

    # Insert stock metadata
    for symbol, company_name, sector, industry, market_cap, exchange in sample_stocks:
        cursor.execute('''
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
        ''', (symbol, company_name, sector, industry, market_cap, exchange))

    # Insert sample rankings
    today = date.today()
    for i, (symbol, _, _, _, _, _) in enumerate(sample_stocks, 1):
        cursor.execute('''
            INSERT INTO ai_trading.ibd50_rankings
            (symbol, rank_position, ranking_date, composite_rating, eps_rating, rs_rating)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, ranking_date) DO UPDATE SET
                rank_position = EXCLUDED.rank_position,
                composite_rating = EXCLUDED.composite_rating,
                eps_rating = EXCLUDED.eps_rating,
                rs_rating = EXCLUDED.rs_rating,
                updated_at = CURRENT_TIMESTAMP
        ''', (symbol, i, today, 95.0 - i * 0.5, 90.0 - i * 0.3, 85.0 - i * 0.4))

    conn.commit()
    print(f'Successfully populated {len(sample_stocks)} IBD 50 stocks')

    # Test the view
    cursor.execute('SELECT COUNT(*) FROM ai_trading.v_current_ibd50')
    result = cursor.fetchone()
    count = result[0] if result else 0
    print(f'v_current_ibd50 view now has {count} records')

    # Test sector breakdown
    cursor.execute('SELECT sector, COUNT(*) FROM ai_trading.v_current_ibd50 GROUP BY sector ORDER BY sector')
    sectors = cursor.fetchall()
    print("\nSector breakdown:")
    for sector, count in sectors:
        print(f"  {sector}: {count} stocks")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()