import asyncio
import asyncpg

async def check_data():
    conn = await asyncpg.connect('postgresql://postgres:vSqZyIKyhWUCLPXPclkHlFtXyLXCRmEF@crossover.proxy.rlwy.net:30738/railway')

    # Check what symbols exist
    symbols = await conn.fetch('SELECT DISTINCT symbol FROM ai_historical_market_data ORDER BY symbol')
    print('Symbols in ai_historical_market_data:')
    for row in symbols:
        print(f'  {row["symbol"]}')

    # Check data counts
    counts = await conn.fetch('SELECT symbol, timeframe, COUNT(*) as bars FROM ai_historical_market_data GROUP BY symbol, timeframe ORDER BY symbol, timeframe')
    print('\nData counts:')
    for row in counts:
        print(f'  {row["symbol"]} {row["timeframe"]}: {row["bars"]} bars')

    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_data())