-- ============================================
-- FinMind - Database Schema
-- TimescaleDB + PostgreSQL
-- ============================================

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 文本搜索
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- GIN索引

-- ============================================
-- 市场数据表 (时序数据)
-- ============================================

-- 股票价格数据
CREATE TABLE IF NOT EXISTS stock_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DECIMAL(18, 4),
    high DECIMAL(18, 4),
    low DECIMAL(18, 4),
    close DECIMAL(18, 4),
    volume BIGINT,
    adj_close DECIMAL(18, 4),
    source VARCHAR(50) DEFAULT 'yfinance',
    CONSTRAINT stock_prices_pkey PRIMARY KEY (time, symbol)
);

-- 转换为TimescaleDB超表
SELECT create_hypertable('stock_prices', 'time', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices (symbol, time DESC);

-- 设置数据保留策略 (保留5年)
SELECT add_retention_policy('stock_prices', INTERVAL '5 years', if_not_exists => TRUE);

-- 压缩策略 (7天后压缩)
ALTER TABLE stock_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('stock_prices', INTERVAL '7 days', if_not_exists => TRUE);


-- ============================================
-- 财务数据表
-- ============================================

-- 公司基本信息
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    country VARCHAR(50),
    exchange VARCHAR(50),
    currency VARCHAR(10),
    market_cap BIGINT,
    employees INT,
    website VARCHAR(255),
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies (sector);
CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies (industry);

-- 财务报表数据
CREATE TABLE IF NOT EXISTS financial_statements (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    period_type VARCHAR(20) NOT NULL,  -- 'annual', 'quarterly'
    period_end DATE NOT NULL,
    report_type VARCHAR(20) NOT NULL,  -- 'income', 'balance', 'cashflow'
    data JSONB NOT NULL,
    source VARCHAR(50),
    filed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT financial_statements_unique UNIQUE (symbol, period_type, period_end, report_type)
);

CREATE INDEX IF NOT EXISTS idx_financial_statements_symbol ON financial_statements (symbol, period_end DESC);
CREATE INDEX IF NOT EXISTS idx_financial_statements_data ON financial_statements USING GIN (data);


-- ============================================
-- 分析结果表
-- ============================================

-- 分析任务
CREATE TABLE IF NOT EXISTS analysis_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    chain_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed
    parameters JSONB,
    result JSONB,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_tasks_symbol ON analysis_tasks (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_status ON analysis_tasks (status);

-- Agent输出
CREATE TABLE IF NOT EXISTS agent_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES analysis_tasks(id) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    output_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    confidence DECIMAL(5, 4),
    reasoning_chain JSONB,
    data_sources JSONB,
    warnings TEXT[],
    execution_time_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_outputs_task ON agent_outputs (task_id);
CREATE INDEX IF NOT EXISTS idx_agent_outputs_agent ON agent_outputs (agent_name, created_at DESC);


-- ============================================
-- 新闻和情绪数据
-- ============================================

-- 新闻文章
CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    url TEXT UNIQUE,
    source VARCHAR(100),
    published_at TIMESTAMPTZ,
    content TEXT,
    summary TEXT,
    symbols TEXT[],  -- 相关股票
    sentiment_score DECIMAL(5, 4),
    topics TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_articles_published ON news_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_symbols ON news_articles USING GIN (symbols);
CREATE INDEX IF NOT EXISTS idx_news_articles_title ON news_articles USING GIN (title gin_trgm_ops);


-- ============================================
-- 用户和配置
-- ============================================

-- 用户
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    password_hash VARCHAR(255),
    api_key VARCHAR(64) UNIQUE,
    preferences JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 观察列表
CREATE TABLE IF NOT EXISTS watchlists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    symbols TEXT[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_watchlists_user ON watchlists (user_id);

-- 分析模板
CREATE TABLE IF NOT EXISTS analysis_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    chain_config JSONB NOT NULL,
    is_public BOOLEAN DEFAULT FALSE,
    usage_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================
-- 投资组合历史数据 (时序数据)
-- ============================================

-- 投资组合快照（用于计算最大回撤等指标）
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time TIMESTAMPTZ NOT NULL,
    account_id VARCHAR(100) NOT NULL DEFAULT 'default',
    total_assets DECIMAL(18, 4) NOT NULL,
    total_cash DECIMAL(18, 4) NOT NULL,
    total_market_value DECIMAL(18, 4) NOT NULL,
    total_unrealized_pnl DECIMAL(18, 4) DEFAULT 0,
    total_realized_pnl DECIMAL(18, 4) DEFAULT 0,
    position_count INT DEFAULT 0,
    broker_count INT DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT portfolio_snapshots_pkey PRIMARY KEY (time, account_id)
);

-- 转换为TimescaleDB超表
SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_account
    ON portfolio_snapshots (account_id, time DESC);

-- 设置数据保留策略 (保留3年)
SELECT add_retention_policy('portfolio_snapshots', INTERVAL '3 years', if_not_exists => TRUE);

-- 压缩策略 (30天后压缩)
ALTER TABLE portfolio_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'account_id'
);
SELECT add_compression_policy('portfolio_snapshots', INTERVAL '30 days', if_not_exists => TRUE);


-- 持仓历史（每日快照）
CREATE TABLE IF NOT EXISTS position_history (
    time TIMESTAMPTZ NOT NULL,
    account_id VARCHAR(100) NOT NULL DEFAULT 'default',
    symbol VARCHAR(20) NOT NULL,
    market VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 6) NOT NULL,
    avg_cost DECIMAL(18, 4) NOT NULL,
    current_price DECIMAL(18, 4) NOT NULL,
    market_value DECIMAL(18, 4) NOT NULL,
    unrealized_pnl DECIMAL(18, 4) DEFAULT 0,
    unrealized_pnl_percent DECIMAL(10, 4) DEFAULT 0,
    weight DECIMAL(10, 6) DEFAULT 0,  -- 持仓权重
    broker VARCHAR(50),
    CONSTRAINT position_history_pkey PRIMARY KEY (time, account_id, symbol, market)
);

-- 转换为TimescaleDB超表
SELECT create_hypertable('position_history', 'time', if_not_exists => TRUE);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_position_history_account_symbol
    ON position_history (account_id, symbol, time DESC);

-- 数据保留和压缩
SELECT add_retention_policy('position_history', INTERVAL '3 years', if_not_exists => TRUE);
ALTER TABLE position_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'account_id,symbol'
);
SELECT add_compression_policy('position_history', INTERVAL '30 days', if_not_exists => TRUE);


-- 交易记录
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL,
    account_id VARCHAR(100) NOT NULL DEFAULT 'default',
    symbol VARCHAR(20) NOT NULL,
    market VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- 'buy' or 'sell'
    quantity DECIMAL(18, 6) NOT NULL,
    price DECIMAL(18, 4) NOT NULL,
    total_value DECIMAL(18, 4) NOT NULL,
    commission DECIMAL(18, 4) DEFAULT 0,
    currency VARCHAR(10) DEFAULT 'USD',
    broker VARCHAR(50),
    order_id VARCHAR(100),
    execution_id VARCHAR(100),
    realized_pnl DECIMAL(18, 4),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_account_time
    ON trades (account_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol
    ON trades (symbol, time DESC);


-- ============================================
-- 投资组合分析视图
-- ============================================

-- 最新投资组合快照
CREATE OR REPLACE VIEW latest_portfolio_snapshot AS
SELECT DISTINCT ON (account_id)
    time,
    account_id,
    total_assets,
    total_cash,
    total_market_value,
    total_unrealized_pnl,
    total_realized_pnl,
    position_count,
    broker_count
FROM portfolio_snapshots
ORDER BY account_id, time DESC;

-- 投资组合每日收益率
CREATE OR REPLACE VIEW portfolio_daily_returns AS
SELECT
    time::DATE as date,
    account_id,
    total_assets,
    LAG(total_assets) OVER (PARTITION BY account_id ORDER BY time) as prev_assets,
    CASE
        WHEN LAG(total_assets) OVER (PARTITION BY account_id ORDER BY time) > 0
        THEN (total_assets - LAG(total_assets) OVER (PARTITION BY account_id ORDER BY time))
             / LAG(total_assets) OVER (PARTITION BY account_id ORDER BY time)
        ELSE 0
    END as daily_return
FROM (
    SELECT DISTINCT ON (time::DATE, account_id)
        time, account_id, total_assets
    FROM portfolio_snapshots
    ORDER BY time::DATE, account_id, time DESC
) daily_snapshots;


-- ============================================
-- 缓存表
-- ============================================

CREATE TABLE IF NOT EXISTS cache_entries (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at);


-- ============================================
-- 审计日志
-- ============================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs (action, timestamp DESC);


-- ============================================
-- 函数和触发器
-- ============================================

-- 更新 updated_at 时间戳
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 应用触发器
CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watchlists_updated_at
    BEFORE UPDATE ON watchlists
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================
-- 初始数据
-- ============================================

-- 插入示例用户
INSERT INTO users (email, name, api_key, preferences)
VALUES (
    'admin@financeai.local',
    'Admin',
    'fai_demo_key_12345678901234567890123456789012',
    '{"theme": "dark", "language": "zh-CN"}'
)
ON CONFLICT (email) DO NOTHING;

-- 插入一些常用股票
INSERT INTO companies (symbol, name, sector, industry, exchange)
VALUES
    ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 'NASDAQ'),
    ('MSFT', 'Microsoft Corporation', 'Technology', 'Software - Infrastructure', 'NASDAQ'),
    ('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Content & Information', 'NASDAQ'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical', 'Internet Retail', 'NASDAQ'),
    ('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 'NASDAQ'),
    ('TSLA', 'Tesla Inc.', 'Consumer Cyclical', 'Auto Manufacturers', 'NASDAQ'),
    ('META', 'Meta Platforms Inc.', 'Technology', 'Internet Content & Information', 'NASDAQ'),
    ('BRK.B', 'Berkshire Hathaway Inc.', 'Financial Services', 'Insurance - Diversified', 'NYSE'),
    ('JPM', 'JPMorgan Chase & Co.', 'Financial Services', 'Banks - Diversified', 'NYSE'),
    ('V', 'Visa Inc.', 'Financial Services', 'Credit Services', 'NYSE')
ON CONFLICT (symbol) DO NOTHING;


-- ============================================
-- 常用视图
-- ============================================

-- 最新价格视图
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (symbol)
    symbol,
    time,
    open,
    high,
    low,
    close,
    volume,
    adj_close
FROM stock_prices
ORDER BY symbol, time DESC;

-- 分析任务摘要视图
CREATE OR REPLACE VIEW analysis_summary AS
SELECT
    at.id,
    at.symbol,
    at.chain_name,
    at.status,
    at.created_at,
    at.completed_at,
    COUNT(ao.id) as agent_count,
    AVG(ao.confidence) as avg_confidence,
    SUM(ao.execution_time_ms) as total_execution_time_ms
FROM analysis_tasks at
LEFT JOIN agent_outputs ao ON ao.task_id = at.id
GROUP BY at.id;


-- ============================================
-- 授权
-- ============================================

-- 如果需要创建只读用户
-- CREATE USER financeai_readonly WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE financeai TO financeai_readonly;
-- GRANT USAGE ON SCHEMA public TO financeai_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO financeai_readonly;


COMMENT ON TABLE stock_prices IS '股票价格时序数据，使用TimescaleDB自动分区';
COMMENT ON TABLE companies IS '公司基本信息';
COMMENT ON TABLE financial_statements IS '财务报表数据，存储为JSONB格式';
COMMENT ON TABLE analysis_tasks IS 'AI分析任务记录';
COMMENT ON TABLE agent_outputs IS '各Agent的分析输出';
