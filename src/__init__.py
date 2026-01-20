"""
FinanceAI Pro - 模块化金融AI分析平台

这是一个生产级的金融分析AI平台，采用配置驱动的设计理念，
支持可插拔的数据源、可组合的分析Agent和完整的推理链追溯。
"""

__version__ = "0.1.0"
__author__ = "FinanceAI Team"

from src.core.base import (
    AgentOutput,
    AnalysisContext,
    ConfidenceScorer,
    ConfidenceLevel,
    BaseAgent,
)
from src.core.config_loader import ConfigLoader
from src.core.data_and_chain import (
    FinanceAI,
    DataProvider,
    DataProviderRegistry,
    ChainExecutor,
)
from src.llm.gateway import LLMGateway

__all__ = [
    "FinanceAI",
    "AgentOutput",
    "AnalysisContext",
    "ConfidenceScorer",
    "ConfidenceLevel",
    "BaseAgent",
    "ConfigLoader",
    "DataProvider",
    "DataProviderRegistry",
    "ChainExecutor",
    "LLMGateway",
]