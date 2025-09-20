-- IBD 50 Database Schema Setup
-- Creates tables and views for IBD 50 stock universe management

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS ai_trading;

-- Create IBD 50 rankings table
CREATE TABLE IF NOT EXISTS ai_trading.ibd50_rankings (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    rank_position INTEGER NOT NULL,
    ranking_date DATE NOT NULL,
    composite_rating DECIMAL(3,1),
    eps_rating DECIMAL(3,1),
    rs_rating DECIMAL(3,1),
    group_rs_rating DECIMAL(3,1),
    smr_rating DECIMAL(3,1),
    acc_dis_rating DECIMAL(3,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, ranking_date)
);

-- Create stock metadata table
CREATE TABLE IF NOT EXISTS ai_trading.stock_metadata (
    symbol VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap_category VARCHAR(50),
    exchange VARCHAR(20),
    metadata JSONB,
    added_date DATE DEFAULT CURRENT_DATE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create view for current IBD 50 stocks
CREATE OR REPLACE VIEW ai_trading.v_current_ibd50 AS
SELECT
    r.symbol,
    COALESCE(m.company_name, r.symbol) as company_name,
    COALESCE(m.sector, 'Unknown') as sector,
    COALESCE(m.industry, 'Unknown') as industry,
    COALESCE(m.market_cap_category, 'Unknown') as market_cap_category,
    COALESCE(m.exchange, 'NASDAQ') as exchange,
    r.rank_position,
    m.metadata,
    COALESCE(m.added_date, CURRENT_DATE) as added_date
FROM ai_trading.ibd50_rankings r
LEFT JOIN ai_trading.stock_metadata m ON r.symbol = m.symbol
WHERE r.ranking_date = (
    SELECT MAX(ranking_date)
    FROM ai_trading.ibd50_rankings
    WHERE ranking_date <= CURRENT_DATE
)
ORDER BY r.rank_position;

-- Create view for IBD 50 with ratings
CREATE OR REPLACE VIEW ai_trading.v_ibd50_with_ratings AS
SELECT
    r.*,
    m.company_name,
    m.sector,
    m.industry,
    m.market_cap_category,
    m.exchange,
    m.metadata
FROM ai_trading.ibd50_rankings r
LEFT JOIN ai_trading.stock_metadata m ON r.symbol = m.symbol
WHERE r.ranking_date = (
    SELECT MAX(ranking_date)
    FROM ai_trading.ibd50_rankings
    WHERE ranking_date <= CURRENT_DATE
)
ORDER BY r.rank_position;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ibd50_rankings_date ON ai_trading.ibd50_rankings(ranking_date);
CREATE INDEX IF NOT EXISTS idx_ibd50_rankings_symbol ON ai_trading.ibd50_rankings(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_metadata_sector ON ai_trading.stock_metadata(sector);
CREATE INDEX IF NOT EXISTS idx_stock_metadata_exchange ON ai_trading.stock_metadata(exchange);