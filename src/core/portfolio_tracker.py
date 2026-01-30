"""
FinMind - 投资组合追踪器

定期记录投资组合快照，用于计算最大回撤等历史指标。
"""

import asyncio
from datetime import datetime
from typing import Optional
import logging

from .database import (
    DatabaseService,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    get_database,
)

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    投资组合追踪器

    定期从券商获取投资组合数据并保存到数据库。

    Usage:
        ```python
        from src.brokers.portfolio import UnifiedPortfolio

        portfolio = UnifiedPortfolio()
        tracker = PortfolioTracker(portfolio)

        # 手动记录快照
        await tracker.record_snapshot()

        # 启动自动追踪（每小时记录一次）
        await tracker.start(interval_minutes=60)
        ```
    """

    def __init__(
        self,
        portfolio,  # UnifiedPortfolio
        account_id: str = 'default',
        db_service: DatabaseService = None,
    ):
        """
        初始化追踪器

        Args:
            portfolio: UnifiedPortfolio 实例
            account_id: 账户 ID，用于区分不同用户的数据
            db_service: 数据库服务，如果不提供则使用全局实例
        """
        self._portfolio = portfolio
        self._account_id = account_id
        self._db = db_service
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _get_db(self) -> DatabaseService:
        """获取数据库服务"""
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def record_snapshot(self) -> bool:
        """
        记录当前投资组合快照

        Returns:
            bool: 是否成功记录
        """
        try:
            # 检查是否有连接的券商
            if not self._portfolio.connected_brokers:
                logger.debug("No connected brokers, skipping snapshot")
                return False

            # 获取统一投资组合数据
            summary = await self._portfolio.get_unified_summary()

            # 创建投资组合快照
            snapshot = PortfolioSnapshot(
                time=datetime.now(),
                account_id=self._account_id,
                total_assets=summary.total_assets,
                total_cash=summary.total_cash,
                total_market_value=summary.total_market_value,
                total_unrealized_pnl=summary.total_unrealized_pnl,
                total_realized_pnl=summary.total_realized_pnl,
                position_count=summary.position_count,
                broker_count=summary.broker_count,
                metadata={
                    'brokers': list(summary.broker_allocation.keys()),
                    'markets': list(summary.market_allocation.keys()),
                },
            )

            # 保存到数据库
            db = await self._get_db()
            if not db.is_available:
                logger.warning("Database not available, snapshot not saved")
                return False

            success = await db.save_portfolio_snapshot(snapshot)
            if success:
                logger.info(f"Portfolio snapshot recorded: ${summary.total_assets:,.2f}")

            # 同时保存持仓快照
            if summary.aggregated_positions:
                position_snapshots = []
                for pos in summary.aggregated_positions:
                    weight = pos.total_market_value / summary.total_assets if summary.total_assets > 0 else 0
                    position_snapshots.append(PositionSnapshot(
                        time=snapshot.time,
                        account_id=self._account_id,
                        symbol=pos.symbol,
                        market=pos.market.value,
                        quantity=pos.total_quantity,
                        avg_cost=pos.avg_cost,
                        current_price=pos.current_price,
                        market_value=pos.total_market_value,
                        unrealized_pnl=pos.total_unrealized_pnl,
                        unrealized_pnl_percent=pos.unrealized_pnl_percent,
                        weight=weight,
                        broker=','.join(pos.positions_by_broker.keys()),
                    ))
                await db.save_position_snapshots(position_snapshots)

            return success

        except Exception as e:
            logger.error(f"Failed to record portfolio snapshot: {e}")
            return False

    async def record_trades(self) -> int:
        """
        记录新的交易

        从券商获取最近的交易记录并保存到数据库。

        Returns:
            int: 保存的交易数量
        """
        try:
            db = await self._get_db()
            if not db.is_available:
                return 0

            total_saved = 0
            for broker_name in self._portfolio.connected_brokers:
                adapter = self._portfolio.get_adapter(broker_name)
                if not adapter:
                    continue

                try:
                    trades = await adapter.get_trades(days=7)
                    if not trades:
                        continue

                    trade_records = [
                        TradeRecord(
                            time=t.trade_time or datetime.now(),
                            account_id=self._account_id,
                            symbol=t.symbol,
                            market=t.market.value,
                            action=t.action.value,
                            quantity=t.quantity,
                            price=t.price,
                            total_value=t.quantity * t.price,
                            commission=t.commission,
                            currency=t.currency.value,
                            broker=broker_name,
                            order_id=t.order_id,
                            execution_id=t.execution_id,
                            realized_pnl=t.realized_pnl,
                        )
                        for t in trades
                    ]

                    if await db.save_trades(trade_records):
                        total_saved += len(trade_records)

                except Exception as e:
                    logger.error(f"Failed to get trades from {broker_name}: {e}")
                    continue

            if total_saved > 0:
                logger.info(f"Recorded {total_saved} trades")
            return total_saved

        except Exception as e:
            logger.error(f"Failed to record trades: {e}")
            return 0

    async def start(self, interval_minutes: int = 60):
        """
        启动自动追踪

        Args:
            interval_minutes: 记录间隔（分钟）
        """
        if self._running:
            logger.warning("Tracker already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._track_loop(interval_minutes))
        logger.info(f"Portfolio tracker started (interval: {interval_minutes} minutes)")

    async def stop(self):
        """停止自动追踪"""
        self._running = False
        if self._task:
            if not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    logger.debug("Portfolio tracker task cancelled")
                except Exception as e:
                    logger.error(f"Error stopping tracker task: {e}")
            self._task = None
        logger.info("Portfolio tracker stopped")

    async def _track_loop(self, interval_minutes: int):
        """追踪循环"""
        while self._running:
            try:
                await self.record_snapshot()
                await self.record_trades()
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")

            # 等待下一次记录
            await asyncio.sleep(interval_minutes * 60)

    async def get_max_drawdown(self, days: int = 365) -> float:
        """
        获取最大回撤

        Args:
            days: 计算期间（天数）

        Returns:
            float: 最大回撤（0-1 之间的小数）
        """
        db = await self._get_db()
        if not db.is_available:
            return 0.0

        snapshots = await db.get_portfolio_history(
            account_id=self._account_id,
            days=days,
        )
        return db.calculate_max_drawdown(snapshots)

    async def get_volatility(self, days: int = 252) -> float:
        """
        获取波动率

        Args:
            days: 计算期间（天数）

        Returns:
            float: 年化波动率
        """
        db = await self._get_db()
        if not db.is_available:
            return 0.0

        snapshots = await db.get_portfolio_history(
            account_id=self._account_id,
            days=days,
        )
        return db.calculate_volatility(snapshots)

    async def get_sharpe_ratio(self, days: int = 252, risk_free_rate: float = 0.05) -> float:
        """
        获取夏普比率

        Args:
            days: 计算期间（天数）
            risk_free_rate: 无风险利率

        Returns:
            float: 夏普比率
        """
        db = await self._get_db()
        if not db.is_available:
            return 0.0

        snapshots = await db.get_portfolio_history(
            account_id=self._account_id,
            days=days,
        )
        return db.calculate_sharpe_ratio(snapshots, risk_free_rate)
