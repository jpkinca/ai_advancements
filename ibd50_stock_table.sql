-- IBD 50 Stock Universe Table
-- PostgreSQL table for managing IBD 50 stock universe with metadata

-- Stock universes table for organizing different stock lists
CREATE TABLE IF NOT EXISTS ai_trading.stock_universes (
    universe_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    universe_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50), -- 'ibd50', 'production', 'test', 'sector_focus'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Individual stocks table with metadata
CREATE TABLE IF NOT EXISTS ai_trading.stocks (
    stock_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(200),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap_category VARCHAR(20), -- 'mega', 'large', 'mid', 'small', 'micro'
    exchange VARCHAR(10), -- 'NYSE', 'NASDAQ', 'AMEX'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Junction table for stocks in universes with rankings/metadata
CREATE TABLE IF NOT EXISTS ai_trading.universe_stocks (
    universe_stock_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    universe_id UUID NOT NULL REFERENCES ai_trading.stock_universes(universe_id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES ai_trading.stocks(stock_id) ON DELETE CASCADE,
    rank_position INTEGER, -- Position in the universe (1-50 for IBD 50)
    added_date TIMESTAMPTZ DEFAULT NOW(),
    removed_date TIMESTAMPTZ, -- NULL if still active
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB, -- Additional metadata like priority, weight, etc.
    
    UNIQUE(universe_id, stock_id, added_date)
);

-- Historical rankings table for tracking IBD 50 changes over time
CREATE TABLE IF NOT EXISTS ai_trading.ibd50_rankings (
    ranking_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    rank_position INTEGER NOT NULL CHECK (rank_position >= 1 AND rank_position <= 50),
    ranking_date DATE NOT NULL,
    composite_rating INTEGER CHECK (composite_rating >= 1 AND composite_rating <= 99),
    eps_rating INTEGER CHECK (eps_rating >= 1 AND eps_rating <= 99),
    rs_rating INTEGER CHECK (rs_rating >= 1 AND rs_rating <= 99),
    group_rs_rating CHARACTER(1) CHECK (group_rs_rating IN ('A', 'B', 'C', 'D', 'E')),
    smr_rating CHARACTER(1) CHECK (smr_rating IN ('A', 'B', 'C', 'D', 'E')),
    acc_dis_rating CHARACTER(1) CHECK (acc_dis_rating IN ('A', 'B', 'C', 'D', 'E')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(symbol, ranking_date),
    FOREIGN KEY (symbol) REFERENCES ai_trading.stocks(symbol)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_universe_stocks_universe_id ON ai_trading.universe_stocks(universe_id);
CREATE INDEX IF NOT EXISTS idx_universe_stocks_stock_id ON ai_trading.universe_stocks(stock_id);
CREATE INDEX IF NOT EXISTS idx_universe_stocks_active ON ai_trading.universe_stocks(is_active);
CREATE INDEX IF NOT EXISTS idx_ibd50_rankings_date ON ai_trading.ibd50_rankings(ranking_date);
CREATE INDEX IF NOT EXISTS idx_ibd50_rankings_symbol ON ai_trading.ibd50_rankings(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON ai_trading.stocks(symbol);

-- Insert the IBD 50 universe
INSERT INTO ai_trading.stock_universes (universe_name, description, category) 
VALUES (
    'IBD 50', 
    'Investor''s Business Daily 50 - Top growth stocks ranked by fundamental and technical criteria',
    'ibd50'
) ON CONFLICT (universe_name) DO NOTHING;

-- Insert current IBD 50 stocks with company information
INSERT INTO ai_trading.stocks (symbol, company_name, sector, industry, market_cap_category, exchange) VALUES
    ('IREN', 'Iris Energy Limited', 'Technology', 'Data Mining & Bitcoin', 'mid', 'NASDAQ'),
    ('CLS', 'Celestica Inc.', 'Technology', 'Electronic Manufacturing Services', 'mid', 'NYSE'),
    ('ALAB', 'Astera Labs Inc.', 'Technology', 'Semiconductors', 'mid', 'NASDAQ'),
    ('FUTU', 'Futu Holdings Limited', 'Financial Services', 'Financial Technology', 'mid', 'NASDAQ'),
    ('PLTR', 'Palantir Technologies Inc.', 'Technology', 'Software', 'large', 'NYSE'),
    ('RKLB', 'Rocket Lab USA Inc.', 'Industrials', 'Aerospace & Defense', 'mid', 'NASDAQ'),
    ('RDDT', 'Reddit Inc.', 'Communication Services', 'Social Media', 'mid', 'NYSE'),
    ('AMSC', 'American Superconductor Corporation', 'Technology', 'Electrical Equipment', 'small', 'NASDAQ'),
    ('HOOD', 'Robinhood Markets Inc.', 'Financial Services', 'Financial Technology', 'mid', 'NASDAQ'),
    ('FIX', 'Comfort Systems USA Inc.', 'Industrials', 'Engineering & Construction', 'mid', 'NYSE'),
    
    ('AGX', 'Argan Inc.', 'Industrials', 'Engineering & Construction', 'small', 'NYSE'),
    ('RYTM', 'Rhythm Pharmaceuticals Inc.', 'Healthcare', 'Biotechnology', 'small', 'NASDAQ'),
    ('MIRM', 'Mirum Pharmaceuticals Inc.', 'Healthcare', 'Biotechnology', 'small', 'NASDAQ'),
    ('OUST', 'Ouster Inc.', 'Technology', 'Electronic Components', 'small', 'NYSE'),
    ('GFI', 'Gold Fields Limited', 'Basic Materials', 'Gold Mining', 'mid', 'NYSE'),
    ('WLDN', 'Willdan Group Inc.', 'Industrials', 'Engineering Services', 'small', 'NASDAQ'),
    ('AFRM', 'Affirm Holdings Inc.', 'Financial Services', 'Financial Technology', 'mid', 'NASDAQ'),
    ('BZ', 'KANZHUN Limited', 'Technology', 'Internet Services', 'mid', 'NASDAQ'),
    ('ANET', 'Arista Networks Inc.', 'Technology', 'Computer Networking', 'large', 'NYSE'),
    ('WGS', 'GeneDx Holdings Corp.', 'Healthcare', 'Diagnostics & Research', 'small', 'NASDAQ'),
    
    ('TFPM', 'Triple Flag Precious Metals Corp.', 'Basic Materials', 'Precious Metals', 'mid', 'NYSE'),
    ('APH', 'Amphenol Corporation', 'Technology', 'Electronic Components', 'large', 'NYSE'),
    ('TARS', 'Tarsus Pharmaceuticals Inc.', 'Healthcare', 'Biotechnology', 'small', 'NASDAQ'),
    ('ATAT', 'Atour Lifestyle Holdings Limited', 'Consumer Cyclical', 'Hotels & Lodging', 'small', 'NASDAQ'),
    ('LIF', 'Life360 Inc.', 'Technology', 'Software', 'small', 'NASDAQ'),
    ('AEM', 'Agnico Eagle Mines Limited', 'Basic Materials', 'Gold Mining', 'large', 'NYSE'),
    ('RMBS', 'Rambus Inc.', 'Technology', 'Semiconductors', 'mid', 'NASDAQ'),
    ('ANIP', 'ANI Pharmaceuticals Inc.', 'Healthcare', 'Drug Manufacturers', 'small', 'NASDAQ'),
    ('GH', 'Guardant Health Inc.', 'Healthcare', 'Diagnostics & Research', 'mid', 'NASDAQ'),
    ('SOFI', 'SoFi Technologies Inc.', 'Financial Services', 'Financial Technology', 'mid', 'NASDAQ'),
    
    ('KGC', 'Kinross Gold Corporation', 'Basic Materials', 'Gold Mining', 'mid', 'NYSE'),
    ('EME', 'EMCOR Group Inc.', 'Industrials', 'Engineering & Construction', 'mid', 'NYSE'),
    ('AU', 'AngloGold Ashanti Limited', 'Basic Materials', 'Gold Mining', 'mid', 'NYSE'),
    ('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 'mega', 'NASDAQ'),
    ('TBBK', 'The Bancorp Inc.', 'Financial Services', 'Regional Banks', 'small', 'NASDAQ'),
    ('MEDP', 'Medpace Holdings Inc.', 'Healthcare', 'Medical Devices', 'mid', 'NASDAQ'),
    ('DOCS', 'Doximity Inc.', 'Healthcare', 'Health Information Services', 'small', 'NYSE'),
    ('ONC', 'Oncocyte Corporation', 'Healthcare', 'Diagnostics & Research', 'micro', 'NASDAQ'),
    ('KNSA', 'Kiniksa Pharmaceuticals Ltd.', 'Healthcare', 'Biotechnology', 'small', 'NASDAQ'),
    ('STNE', 'StoneCo Ltd.', 'Financial Services', 'Financial Technology', 'mid', 'NASDAQ'),
    
    ('XPEV', 'XPeng Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 'mid', 'NYSE'),
    ('CCJ', 'Cameco Corporation', 'Energy', 'Uranium', 'mid', 'NYSE'),
    ('EGO', 'Eldorado Gold Corporation', 'Basic Materials', 'Gold Mining', 'mid', 'NYSE'),
    ('CVNA', 'Carvana Co.', 'Consumer Cyclical', 'Auto Dealerships', 'mid', 'NYSE'),
    ('BROS', 'Dutch Bros Inc.', 'Consumer Cyclical', 'Restaurants', 'mid', 'NYSE'),
    ('TEM', 'Tempus AI Inc.', 'Healthcare', 'Health Information Services', 'mid', 'NASDAQ'),
    ('BAP', 'Credicorp Ltd.', 'Financial Services', 'Regional Banks', 'mid', 'NYSE'),
    ('WPM', 'Wheaton Precious Metals Corp.', 'Basic Materials', 'Precious Metals', 'mid', 'NYSE'),
    ('IBKR', 'Interactive Brokers Group Inc.', 'Financial Services', 'Capital Markets', 'large', 'NASDAQ'),
    ('PODD', 'Insulet Corporation', 'Healthcare', 'Medical Devices', 'mid', 'NASDAQ')
ON CONFLICT (symbol) DO UPDATE SET
    company_name = EXCLUDED.company_name,
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    market_cap_category = EXCLUDED.market_cap_category,
    exchange = EXCLUDED.exchange,
    updated_at = NOW();

-- Insert current IBD 50 stocks into the universe (with ranking positions)
INSERT INTO ai_trading.universe_stocks (universe_id, stock_id, rank_position, metadata)
SELECT 
    u.universe_id,
    s.stock_id,
    ROW_NUMBER() OVER (ORDER BY 
        CASE s.symbol 
            WHEN 'IREN' THEN 1 WHEN 'CLS' THEN 2 WHEN 'ALAB' THEN 3 WHEN 'FUTU' THEN 4 WHEN 'PLTR' THEN 5
            WHEN 'RKLB' THEN 6 WHEN 'RDDT' THEN 7 WHEN 'AMSC' THEN 8 WHEN 'HOOD' THEN 9 WHEN 'FIX' THEN 10
            WHEN 'AGX' THEN 11 WHEN 'RYTM' THEN 12 WHEN 'MIRM' THEN 13 WHEN 'OUST' THEN 14 WHEN 'GFI' THEN 15
            WHEN 'WLDN' THEN 16 WHEN 'AFRM' THEN 17 WHEN 'BZ' THEN 18 WHEN 'ANET' THEN 19 WHEN 'WGS' THEN 20
            WHEN 'TFPM' THEN 21 WHEN 'APH' THEN 22 WHEN 'TARS' THEN 23 WHEN 'ATAT' THEN 24 WHEN 'LIF' THEN 25
            WHEN 'AEM' THEN 26 WHEN 'RMBS' THEN 27 WHEN 'ANIP' THEN 28 WHEN 'GH' THEN 29 WHEN 'SOFI' THEN 30
            WHEN 'KGC' THEN 31 WHEN 'EME' THEN 32 WHEN 'AU' THEN 33 WHEN 'NVDA' THEN 34 WHEN 'TBBK' THEN 35
            WHEN 'MEDP' THEN 36 WHEN 'DOCS' THEN 37 WHEN 'ONC' THEN 38 WHEN 'KNSA' THEN 39 WHEN 'STNE' THEN 40
            WHEN 'XPEV' THEN 41 WHEN 'CCJ' THEN 42 WHEN 'EGO' THEN 43 WHEN 'CVNA' THEN 44 WHEN 'BROS' THEN 45
            WHEN 'TEM' THEN 46 WHEN 'BAP' THEN 47 WHEN 'WPM' THEN 48 WHEN 'IBKR' THEN 49 WHEN 'PODD' THEN 50
        END
    ) as rank_position,
    jsonb_build_object(
        'priority', 'high',
        'category', 'ibd50_current',
        'date_added', NOW()::date
    )
FROM ai_trading.stock_universes u
CROSS JOIN ai_trading.stocks s
WHERE u.universe_name = 'IBD 50'
    AND s.symbol IN (
        'IREN', 'CLS', 'ALAB', 'FUTU', 'PLTR', 'RKLB', 'RDDT', 'AMSC', 'HOOD', 'FIX',
        'AGX', 'RYTM', 'MIRM', 'OUST', 'GFI', 'WLDN', 'AFRM', 'BZ', 'ANET', 'WGS',
        'TFPM', 'APH', 'TARS', 'ATAT', 'LIF', 'AEM', 'RMBS', 'ANIP', 'GH', 'SOFI',
        'KGC', 'EME', 'AU', 'NVDA', 'TBBK', 'MEDP', 'DOCS', 'ONC', 'KNSA', 'STNE',
        'XPEV', 'CCJ', 'EGO', 'CVNA', 'BROS', 'TEM', 'BAP', 'WPM', 'IBKR', 'PODD'
    )
ON CONFLICT (universe_id, stock_id, added_date) DO NOTHING;

-- Insert sample IBD 50 ranking data (current snapshot)
INSERT INTO ai_trading.ibd50_rankings (symbol, rank_position, ranking_date, composite_rating, eps_rating, rs_rating, group_rs_rating, smr_rating, acc_dis_rating)
SELECT 
    symbol,
    ROW_NUMBER() OVER (ORDER BY 
        CASE symbol 
            WHEN 'IREN' THEN 1 WHEN 'CLS' THEN 2 WHEN 'ALAB' THEN 3 WHEN 'FUTU' THEN 4 WHEN 'PLTR' THEN 5
            WHEN 'RKLB' THEN 6 WHEN 'RDDT' THEN 7 WHEN 'AMSC' THEN 8 WHEN 'HOOD' THEN 9 WHEN 'FIX' THEN 10
            WHEN 'AGX' THEN 11 WHEN 'RYTM' THEN 12 WHEN 'MIRM' THEN 13 WHEN 'OUST' THEN 14 WHEN 'GFI' THEN 15
            WHEN 'WLDN' THEN 16 WHEN 'AFRM' THEN 17 WHEN 'BZ' THEN 18 WHEN 'ANET' THEN 19 WHEN 'WGS' THEN 20
            WHEN 'TFPM' THEN 21 WHEN 'APH' THEN 22 WHEN 'TARS' THEN 23 WHEN 'ATAT' THEN 24 WHEN 'LIF' THEN 25
            WHEN 'AEM' THEN 26 WHEN 'RMBS' THEN 27 WHEN 'ANIP' THEN 28 WHEN 'GH' THEN 29 WHEN 'SOFI' THEN 30
            WHEN 'KGC' THEN 31 WHEN 'EME' THEN 32 WHEN 'AU' THEN 33 WHEN 'NVDA' THEN 34 WHEN 'TBBK' THEN 35
            WHEN 'MEDP' THEN 36 WHEN 'DOCS' THEN 37 WHEN 'ONC' THEN 38 WHEN 'KNSA' THEN 39 WHEN 'STNE' THEN 40
            WHEN 'XPEV' THEN 41 WHEN 'CCJ' THEN 42 WHEN 'EGO' THEN 43 WHEN 'CVNA' THEN 44 WHEN 'BROS' THEN 45
            WHEN 'TEM' THEN 46 WHEN 'BAP' THEN 47 WHEN 'WPM' THEN 48 WHEN 'IBKR' THEN 49 WHEN 'PODD' THEN 50
        END
    ) as rank_position,
    CURRENT_DATE as ranking_date,
    FLOOR(75 + RANDOM() * 24)::INTEGER as composite_rating, -- Random ratings 75-99
    FLOOR(70 + RANDOM() * 29)::INTEGER as eps_rating,      -- Random ratings 70-99
    FLOOR(80 + RANDOM() * 19)::INTEGER as rs_rating,       -- Random ratings 80-99
    (ARRAY['A', 'A', 'A', 'B', 'B'])[FLOOR(1 + RANDOM() * 5)] as group_rs_rating, -- Mostly A/B
    (ARRAY['A', 'A', 'B', 'B', 'C'])[FLOOR(1 + RANDOM() * 5)] as smr_rating,      -- Mostly A/B/C
    (ARRAY['A', 'A', 'A', 'B', 'C'])[FLOOR(1 + RANDOM() * 5)] as acc_dis_rating   -- Mostly A/B
FROM (
    SELECT unnest(ARRAY[
        'IREN', 'CLS', 'ALAB', 'FUTU', 'PLTR', 'RKLB', 'RDDT', 'AMSC', 'HOOD', 'FIX',
        'AGX', 'RYTM', 'MIRM', 'OUST', 'GFI', 'WLDN', 'AFRM', 'BZ', 'ANET', 'WGS',
        'TFPM', 'APH', 'TARS', 'ATAT', 'LIF', 'AEM', 'RMBS', 'ANIP', 'GH', 'SOFI',
        'KGC', 'EME', 'AU', 'NVDA', 'TBBK', 'MEDP', 'DOCS', 'ONC', 'KNSA', 'STNE',
        'XPEV', 'CCJ', 'EGO', 'CVNA', 'BROS', 'TEM', 'BAP', 'WPM', 'IBKR', 'PODD'
    ]) as symbol
) symbols
ON CONFLICT (symbol, ranking_date) DO NOTHING;

-- Create views for easy access
CREATE OR REPLACE VIEW ai_trading.v_current_ibd50 AS
SELECT 
    s.symbol,
    s.company_name,
    s.sector,
    s.industry,
    s.market_cap_category,
    s.exchange,
    us.rank_position,
    us.metadata,
    us.added_date
FROM ai_trading.stocks s
JOIN ai_trading.universe_stocks us ON s.stock_id = us.stock_id
JOIN ai_trading.stock_universes u ON us.universe_id = u.universe_id
WHERE u.universe_name = 'IBD 50' 
    AND us.is_active = TRUE
ORDER BY us.rank_position;

CREATE OR REPLACE VIEW ai_trading.v_ibd50_with_ratings AS
SELECT 
    v.symbol,
    v.company_name,
    v.sector,
    v.industry,
    v.rank_position,
    r.composite_rating,
    r.eps_rating,
    r.rs_rating,
    r.group_rs_rating,
    r.smr_rating,
    r.acc_dis_rating,
    r.ranking_date
FROM ai_trading.v_current_ibd50 v
LEFT JOIN ai_trading.ibd50_rankings r ON v.symbol = r.symbol
    AND r.ranking_date = (
        SELECT MAX(ranking_date) 
        FROM ai_trading.ibd50_rankings r2 
        WHERE r2.symbol = v.symbol
    )
ORDER BY v.rank_position;

-- Sample queries
/*
-- Get current IBD 50 list
SELECT * FROM ai_trading.v_current_ibd50;

-- Get IBD 50 with latest ratings
SELECT * FROM ai_trading.v_ibd50_with_ratings;

-- Get top 10 by composite rating
SELECT symbol, company_name, composite_rating 
FROM ai_trading.v_ibd50_with_ratings 
WHERE composite_rating IS NOT NULL
ORDER BY composite_rating DESC 
LIMIT 10;

-- Get stocks by sector
SELECT sector, COUNT(*) as stock_count, array_agg(symbol ORDER BY rank_position) as symbols
FROM ai_trading.v_current_ibd50 
GROUP BY sector 
ORDER BY stock_count DESC;

-- Get technology stocks
SELECT * FROM ai_trading.v_current_ibd50 WHERE sector = 'Technology' ORDER BY rank_position;
*/