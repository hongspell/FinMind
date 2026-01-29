"""
FinMind - 统一投资组合接口

聚合多个券商的持仓和账户信息，提供统一的投资组合视图。
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Type
import logging

from .base import (
    BrokerAdapter,
    BrokerConfig,
    Position,
    AccountBalance,
    PortfolioSummary,
    BrokerError,
    Currency,
    Market,
)
from .ibkr import IBKRAdapter, IBKRMockAdapter
from .futu import FutuAdapter, FutuMockAdapter
from .tiger import TigerAdapter, TigerMockAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class AggregatedPosition:
    """聚合持仓（合并多个券商同一股票的持仓）"""
    symbol: str
    market: Market
    total_quantity: float
    avg_cost: float
    current_price: float
    total_market_value: float
    total_unrealized_pnl: float
    unrealized_pnl_percent: float
    currency: Currency
    # 来源明细
    positions_by_broker: Dict[str, Position] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market.value,
            "total_quantity": self.total_quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "total_market_value": self.total_market_value,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "currency": self.currency.value,
            "brokers": list(self.positions_by_broker.keys()),
        }


@dataclass
class UnifiedPortfolioSummary:
    """统一投资组合摘要"""
    total_assets: float
    total_cash: float
    total_market_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    # 分券商数据
    broker_summaries: List[PortfolioSummary]
    # 聚合持仓
    aggregated_positions: List[AggregatedPosition]
    # 分配分析
    broker_allocation: Dict[str, float]  # 券商资产分布
    market_allocation: Dict[str, float]  # 市场分布
    currency_exposure: Dict[str, float]  # 货币敞口
    top_holdings: List[Dict]             # 前N大持仓
    # 元数据
    broker_count: int
    position_count: int
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_assets": self.total_assets,
            "total_cash": self.total_cash,
            "total_market_value": self.total_market_value,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "broker_summaries": [s.to_dict() for s in self.broker_summaries],
            "aggregated_positions": [p.to_dict() for p in self.aggregated_positions],
            "broker_allocation": self.broker_allocation,
            "market_allocation": self.market_allocation,
            "currency_exposure": self.currency_exposure,
            "top_holdings": self.top_holdings,
            "broker_count": self.broker_count,
            "position_count": self.position_count,
            "last_updated": self.last_updated.isoformat(),
        }


# =============================================================================
# 券商注册表
# =============================================================================

BROKER_ADAPTERS: Dict[str, Type[BrokerAdapter]] = {
    "ibkr": IBKRAdapter,
    "futu": FutuAdapter,
    "tiger": TigerAdapter,
}

MOCK_ADAPTERS: Dict[str, Type[BrokerAdapter]] = {
    "ibkr": IBKRMockAdapter,
    "futu": FutuMockAdapter,
    "tiger": TigerMockAdapter,
}


def create_broker_adapter(config: BrokerConfig, mock: bool = False) -> BrokerAdapter:
    """
    创建券商适配器

    Args:
        config: 券商配置
        mock: 是否使用模拟适配器

    Returns:
        BrokerAdapter: 券商适配器实例
    """
    broker_type = config.broker_type.lower()
    adapters = MOCK_ADAPTERS if mock else BROKER_ADAPTERS

    if broker_type not in adapters:
        raise ValueError(f"Unknown broker type: {broker_type}")

    return adapters[broker_type](config)


# =============================================================================
# 统一投资组合
# =============================================================================

class UnifiedPortfolio:
    """
    统一投资组合管理器

    聚合多个券商账户，提供统一的投资组合视图和分析。

    Example:
        ```python
        portfolio = UnifiedPortfolio()

        # 添加券商
        portfolio.add_broker(BrokerConfig(broker_type="ibkr", ...))
        portfolio.add_broker(BrokerConfig(broker_type="futu", ...))

        # 连接所有券商
        await portfolio.connect_all()

        # 获取统一视图
        summary = await portfolio.get_unified_summary()
        print(f"Total assets: ${summary.total_assets:,.2f}")

        # 获取特定股票在所有账户的持仓
        positions = await portfolio.get_position_across_brokers("AAPL")

        # 断开所有连接
        await portfolio.disconnect_all()
        ```
    """

    def __init__(self, use_mock: bool = False):
        """
        初始化

        Args:
            use_mock: 是否使用模拟适配器
        """
        self.use_mock = use_mock
        self._adapters: Dict[str, BrokerAdapter] = {}
        self._configs: Dict[str, BrokerConfig] = {}

    @property
    def broker_names(self) -> List[str]:
        """已配置的券商列表"""
        return list(self._adapters.keys())

    @property
    def connected_brokers(self) -> List[str]:
        """已连接的券商列表"""
        return [name for name, adapter in self._adapters.items() if adapter.is_connected]

    def add_broker(self, config: BrokerConfig) -> None:
        """
        添加券商配置

        Args:
            config: 券商配置
        """
        broker_type = config.broker_type.lower()

        # 如果已存在且已连接，不要覆盖
        existing = self._adapters.get(broker_type)
        if existing and existing.is_connected:
            logger.info(f"Broker {broker_type} already connected, skipping add")
            return

        adapter = create_broker_adapter(config, self.use_mock)
        self._adapters[broker_type] = adapter
        self._configs[broker_type] = config
        logger.info(f"Added broker: {broker_type}")

    def remove_broker(self, broker_type: str) -> None:
        """
        移除券商

        Args:
            broker_type: 券商类型
        """
        broker_type = broker_type.lower()
        if broker_type in self._adapters:
            del self._adapters[broker_type]
            del self._configs[broker_type]
            logger.info(f"Removed broker: {broker_type}")

    async def connect_all(self) -> Dict[str, bool]:
        """
        连接所有券商

        Returns:
            Dict[str, bool]: 各券商连接状态
        """
        results = {}
        tasks = []

        for name, adapter in self._adapters.items():
            tasks.append(self._connect_broker(name, adapter))

        for result in await asyncio.gather(*tasks, return_exceptions=True):
            name, success = result if isinstance(result, tuple) else (None, False)
            if name:
                results[name] = success

        return results

    async def _connect_broker(self, name: str, adapter: BrokerAdapter) -> tuple:
        """连接单个券商"""
        try:
            success = await adapter.connect()
            return (name, success)
        except Exception as e:
            logger.error(f"Failed to connect to {name}: {e}")
            return (name, False)

    async def disconnect_all(self) -> None:
        """断开所有连接"""
        tasks = [adapter.disconnect() for adapter in self._adapters.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def health_check_all(self) -> Dict[str, bool]:
        """
        检查所有券商连接状态

        Returns:
            Dict[str, bool]: 各券商健康状态
        """
        results = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = await adapter.health_check()
            except Exception:
                results[name] = False
        return results

    async def get_broker_summary(self, broker_type: str) -> Optional[PortfolioSummary]:
        """
        获取单个券商的投资组合摘要

        Args:
            broker_type: 券商类型

        Returns:
            PortfolioSummary | None: 投资组合摘要
        """
        adapter = self._adapters.get(broker_type.lower())
        if not adapter or not adapter.is_connected:
            return None

        try:
            return await adapter.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Failed to get summary from {broker_type}: {e}")
            return None

    async def get_all_summaries(self) -> List[PortfolioSummary]:
        """
        获取所有券商的投资组合摘要

        Returns:
            List[PortfolioSummary]: 各券商摘要列表
        """
        tasks = []
        broker_names = []

        for name, adapter in self._adapters.items():
            if adapter.is_connected:
                tasks.append(adapter.get_portfolio_summary())
                broker_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        summaries = []

        for name, result in zip(broker_names, results):
            if isinstance(result, PortfolioSummary):
                summaries.append(result)
            else:
                logger.error(f"Failed to get summary from {name}: {result}")

        return summaries

    async def get_unified_summary(self) -> UnifiedPortfolioSummary:
        """
        获取统一投资组合摘要

        聚合所有券商数据，计算总资产、总持仓等。

        Returns:
            UnifiedPortfolioSummary: 统一摘要
        """
        summaries = await self.get_all_summaries()

        # 计算总资产
        total_assets = sum(s.balance.total_assets for s in summaries)
        total_cash = sum(s.balance.cash for s in summaries)
        total_market_value = sum(s.balance.market_value for s in summaries)
        total_unrealized_pnl = sum(s.balance.total_pnl for s in summaries)
        total_realized_pnl = sum(s.balance.day_pnl for s in summaries)

        # 聚合持仓
        aggregated = self._aggregate_positions(summaries)

        # 计算分配
        broker_allocation = {}
        market_allocation: Dict[str, float] = {}
        currency_exposure: Dict[str, float] = {}

        for s in summaries:
            # 券商分配
            if total_assets > 0:
                broker_allocation[s.broker] = s.balance.total_assets / total_assets * 100

            # 市场分配
            for pos in s.positions:
                market = pos.market.value
                market_allocation[market] = market_allocation.get(market, 0) + pos.market_value

            # 货币敞口
            for pos in s.positions:
                curr = pos.currency.value
                currency_exposure[curr] = currency_exposure.get(curr, 0) + pos.market_value

        # 转换为百分比
        if total_market_value > 0:
            market_allocation = {k: v / total_market_value * 100 for k, v in market_allocation.items()}
            currency_exposure = {k: v / total_market_value * 100 for k, v in currency_exposure.items()}

        # 前10大持仓
        top_holdings = sorted(
            [p.to_dict() for p in aggregated],
            key=lambda x: x['total_market_value'],
            reverse=True
        )[:10]

        return UnifiedPortfolioSummary(
            total_assets=total_assets,
            total_cash=total_cash,
            total_market_value=total_market_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            broker_summaries=summaries,
            aggregated_positions=aggregated,
            broker_allocation=broker_allocation,
            market_allocation=market_allocation,
            currency_exposure=currency_exposure,
            top_holdings=top_holdings,
            broker_count=len(summaries),
            position_count=len(aggregated),
            last_updated=datetime.now(),
        )

    def _aggregate_positions(self, summaries: List[PortfolioSummary]) -> List[AggregatedPosition]:
        """聚合各券商的持仓"""
        positions_map: Dict[str, AggregatedPosition] = {}

        for summary in summaries:
            for pos in summary.positions:
                key = f"{pos.symbol}_{pos.market.value}"

                if key not in positions_map:
                    positions_map[key] = AggregatedPosition(
                        symbol=pos.symbol,
                        market=pos.market,
                        total_quantity=0,
                        avg_cost=0,
                        current_price=pos.current_price,
                        total_market_value=0,
                        total_unrealized_pnl=0,
                        unrealized_pnl_percent=0,
                        currency=pos.currency,
                        positions_by_broker={},
                    )

                agg = positions_map[key]
                agg.positions_by_broker[summary.broker] = pos
                agg.total_quantity += pos.quantity
                agg.total_market_value += pos.market_value
                agg.total_unrealized_pnl += pos.unrealized_pnl
                agg.current_price = pos.current_price  # 使用最新价格

        # 计算加权平均成本和盈亏百分比
        for agg in positions_map.values():
            total_cost = sum(
                p.quantity * p.avg_cost for p in agg.positions_by_broker.values()
            )
            if agg.total_quantity > 0:
                agg.avg_cost = total_cost / agg.total_quantity
            if total_cost > 0:
                agg.unrealized_pnl_percent = agg.total_unrealized_pnl / total_cost * 100

        return list(positions_map.values())

    async def get_position_across_brokers(self, symbol: str) -> Dict[str, Optional[Position]]:
        """
        获取某只股票在所有券商的持仓

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Position | None]: 各券商的持仓
        """
        results = {}

        for name, adapter in self._adapters.items():
            if adapter.is_connected:
                try:
                    pos = await adapter.get_position(symbol)
                    results[name] = pos
                except Exception as e:
                    logger.error(f"Failed to get position from {name}: {e}")
                    results[name] = None

        return results

    async def __aenter__(self):
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect_all()
