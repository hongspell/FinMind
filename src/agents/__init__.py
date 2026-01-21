"""
FinMind - Agents模块

包含所有专业分析Agent的实现：
- ValuationAgent: 估值分析
- TechnicalAgent: 技术分析
- EarningsAgent: 财报分析
- SentimentAgent: 情绪分析
- RiskAgent: 风险分析
- StrategyAgent: 策略综合
- MacroAgent: 宏观经济分析
- SectorAgent: 行业分析
"""

from .valuation_agent import ValuationAgent
from .technical_agent import (
    TechnicalAgent,
    Timeframe,
    TimeframeAnalysis,
    TrendDirection,
    SignalStrength
)
from .earnings_agent import EarningsAgent
from .sentiment_risk_agent import SentimentAgent, RiskAgent
from .strategy_agent import StrategyAgent
from .macro_agent import MacroAgent
from .sector_agent import SectorAgent

__all__ = [
    "ValuationAgent",
    "TechnicalAgent",
    "Timeframe",
    "TimeframeAnalysis",
    "TrendDirection",
    "SignalStrength",
    "EarningsAgent",
    "SentimentAgent",
    "RiskAgent",
    "StrategyAgent",
    "MacroAgent",
    "SectorAgent"
]

# Agent注册表
AGENT_REGISTRY = {
    "valuation": ValuationAgent,
    "technical": TechnicalAgent,
    "earnings": EarningsAgent,
    "sentiment": SentimentAgent,
    "risk": RiskAgent,
    "strategy": StrategyAgent,
    "macro": MacroAgent,
    "sector": SectorAgent
}


def get_agent(agent_name: str, config: dict = None):
    """
    根据名称获取Agent实例
    
    Args:
        agent_name: Agent名称
        config: 可选配置
        
    Returns:
        Agent实例
        
    Raises:
        ValueError: 未知的Agent名称
    """
    agent_name = agent_name.lower()
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Available agents: {list(AGENT_REGISTRY.keys())}"
        )
    
    agent_class = AGENT_REGISTRY[agent_name]
    return agent_class(config or {})


def list_agents() -> list:
    """列出所有可用的Agent"""
    return list(AGENT_REGISTRY.keys())
