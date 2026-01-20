"""
FinanceAI Pro - Core模块

核心框架组件：
- base: 基础类定义（AgentOutput, AnalysisContext, ConfidenceScore等）
- config_loader: 配置加载和验证
- data_and_chain: 数据提供者和分析链执行器
"""

from .base import (
    AgentOutput,
    AnalysisContext,
    ConfidenceScorer,
    ConfidenceLevel,
    ReasoningStep,
    DataSource,
    Uncertainty,
    BaseAgent
)

from .config_loader import (
    ConfigLoader,
    ConfigValidator,
    ValidationError
)

from .data_and_chain import (
    DataProvider,
    DataProviderRegistry,
    YFinanceProvider,
    SECProvider,
    NewsProvider,
    ChainExecutor,
    FinanceAI,
    AnalysisResult
)

__all__ = [
    # Base classes
    "AgentOutput",
    "AnalysisContext",
    "ConfidenceScorer",
    "ConfidenceLevel",
    "ReasoningStep",
    "DataSource",
    "Uncertainty",
    "BaseAgent",
    
    # Config
    "ConfigLoader",
    "ConfigValidator",
    "ValidationError",
    
    # Data & Chain
    "DataProvider",
    "DataProviderRegistry",
    "YFinanceProvider",
    "SECProvider",
    "NewsProvider",
    "ChainExecutor",
    "FinanceAI",
    "AnalysisResult"
]
