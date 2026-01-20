"""
FinanceAI Pro - pytest配置和共享fixtures
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Async支持
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# 基础Fixtures
# =============================================================================

@pytest.fixture
def sample_context():
    """示例分析上下文"""
    from src.core.base import AnalysisContext
    return AnalysisContext(
        target="AAPL",
        analysis_date=datetime(2024, 1, 15),
        parameters={"scenarios": ["bull", "base", "bear"]}
    )


@pytest.fixture
def sample_financial_data() -> Dict[str, Any]:
    """示例财务数据"""
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "revenue": 385e9,
        "revenue_growth": 0.03,
        "gross_profit": 169e9,
        "operating_income": 115e9,
        "net_income": 97e9,
        "ebitda": 125e9,
        "free_cash_flow": 110e9,
        "total_assets": 352e9,
        "total_liabilities": 290e9,
        "total_equity": 62e9,
        "total_debt": 120e9,
        "cash": 65e9,
        "shares_outstanding": 15.5e9,
        "current_price": 175.0,
        "market_cap": 2.75e12,
        "pe_ratio": 28.5,
        "pb_ratio": 44.3,
        "ps_ratio": 7.1,
        "ev_ebitda": 21.0,
        "ev_sales": 7.2,
        "dividend_yield": 0.005,
        "gross_margin": 0.44,
        "operating_margin": 0.30,
        "net_margin": 0.25,
        "roe": 1.56,
        "roa": 0.28,
        "debt_equity": 1.93,
        "current_ratio": 0.99,
        "revenue_history": [
            {"year": 2019, "value": 260e9},
            {"year": 2020, "value": 275e9},
            {"year": 2021, "value": 365e9},
            {"year": 2022, "value": 394e9},
            {"year": 2023, "value": 385e9}
        ],
        "fcf_history": [
            {"year": 2019, "value": 58e9},
            {"year": 2020, "value": 73e9},
            {"year": 2021, "value": 93e9},
            {"year": 2022, "value": 111e9},
            {"year": 2023, "value": 110e9}
        ]
    }


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """示例市场数据"""
    return {
        "ticker": "AAPL",
        "price": 175.0,
        "open": 173.5,
        "high": 176.2,
        "low": 172.8,
        "close": 175.0,
        "volume": 45_000_000,
        "avg_volume_20d": 52_000_000,
        "market_cap": 2.75e12,
        "beta": 1.25,
        "52w_high": 198.0,
        "52w_low": 142.0,
        "sma_20": 174.0,
        "sma_50": 172.0,
        "sma_200": 165.0,
        "ema_12": 174.5,
        "ema_26": 173.0,
        "rsi_14": 55.0,
        "macd": 1.5,
        "macd_signal": 1.2,
        "macd_histogram": 0.3,
        "bb_upper": 180.0,
        "bb_middle": 174.0,
        "bb_lower": 168.0,
        "atr_14": 3.5
    }


@pytest.fixture
def sample_peers() -> list:
    """示例可比公司数据"""
    return [
        {
            "ticker": "MSFT",
            "name": "Microsoft Corp.",
            "market_cap": 2.9e12,
            "pe": 34,
            "ev_ebitda": 24,
            "ev_sales": 12,
            "revenue_growth": 0.12,
            "gross_margin": 0.69,
            "operating_margin": 0.42
        },
        {
            "ticker": "GOOGL",
            "name": "Alphabet Inc.",
            "market_cap": 1.7e12,
            "pe": 24,
            "ev_ebitda": 14,
            "ev_sales": 5.5,
            "revenue_growth": 0.08,
            "gross_margin": 0.56,
            "operating_margin": 0.27
        },
        {
            "ticker": "META",
            "name": "Meta Platforms Inc.",
            "market_cap": 1.2e12,
            "pe": 26,
            "ev_ebitda": 15,
            "ev_sales": 8.5,
            "revenue_growth": 0.15,
            "gross_margin": 0.81,
            "operating_margin": 0.35
        }
    ]


@pytest.fixture
def sample_macro_data() -> Dict[str, float]:
    """示例宏观经济数据"""
    return {
        "gdp_growth": 2.1,
        "unemployment_rate": 3.9,
        "cpi_yoy": 3.2,
        "core_cpi_yoy": 3.5,
        "pce_yoy": 2.8,
        "fed_funds_rate": 5.25,
        "treasury_10y": 4.35,
        "treasury_2y": 4.85,
        "yield_curve_spread": -0.50,
        "ism_manufacturing": 48.5,
        "ism_services": 52.3,
        "consumer_confidence": 102.5
    }


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def valuation_agent():
    """估值Agent"""
    from src.agents.valuation_agent import ValuationAgent
    return ValuationAgent()


@pytest.fixture
def technical_agent():
    """技术分析Agent"""
    from src.agents.technical_agent import TechnicalAgent
    return TechnicalAgent()


@pytest.fixture
def macro_agent():
    """宏观Agent"""
    from src.agents.macro_agent import MacroAgent
    return MacroAgent()


@pytest.fixture
def sector_agent():
    """行业Agent"""
    from src.agents.sector_agent import SectorAgent
    return SectorAgent()


# =============================================================================
# 模拟数据Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """模拟LLM响应"""
    return {
        "content": "这是一个测试响应",
        "model": "claude-sonnet",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }


@pytest.fixture
def mock_price_history():
    """模拟价格历史"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(42)
    
    prices = 150 + np.cumsum(np.random.randn(252) * 2)
    volumes = np.random.randint(30_000_000, 60_000_000, 252)
    
    return pd.DataFrame({
        'date': dates,
        'open': prices - np.random.rand(252),
        'high': prices + np.random.rand(252) * 2,
        'low': prices - np.random.rand(252) * 2,
        'close': prices,
        'volume': volumes
    })


# =============================================================================
# 配置Fixtures
# =============================================================================

@pytest.fixture
def config_path(tmp_path):
    """临时配置目录"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # 创建子目录
    (config_dir / "agents").mkdir()
    (config_dir / "chains").mkdir()
    (config_dir / "methodologies").mkdir()
    (config_dir / "prompts").mkdir()
    
    return config_dir


@pytest.fixture
def sample_agent_config(config_path):
    """示例Agent配置"""
    config_content = """
version: "1.0.0"
agent_name: "test_agent"
capabilities:
  analysis: true
llm_config:
  preferred_model: "claude-sonnet"
"""
    config_file = config_path / "agents" / "test_agent.yaml"
    config_file.write_text(config_content)
    return config_file


# =============================================================================
# 清理Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup():
    """每个测试后的清理"""
    yield
    # 清理代码（如果需要）


# =============================================================================
# 标记
# =============================================================================

def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "api: API测试")
