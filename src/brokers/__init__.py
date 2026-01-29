"""
FinMind - 券商集成模块

支持的券商:
- IBKR (盈透证券) - TWS API
- Futu (富途证券) - OpenD Gateway
- Tiger (老虎证券) - Tiger Open API

所有券商适配器实现统一的 BrokerAdapter 接口，
提供投资组合、持仓、账户余额等只读功能。
"""

from .base import (
    BrokerAdapter,
    BrokerConfig,
    Position,
    AccountBalance,
    PortfolioSummary,
    BrokerError,
    AuthenticationError,
    ConnectionError,
)
from .portfolio import UnifiedPortfolio

__all__ = [
    # Base classes
    "BrokerAdapter",
    "BrokerConfig",
    "Position",
    "AccountBalance",
    "PortfolioSummary",
    # Errors
    "BrokerError",
    "AuthenticationError",
    "ConnectionError",
    # Unified interface
    "UnifiedPortfolio",
]
