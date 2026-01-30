"""
FinMind - 数据库服务

提供 PostgreSQL/TimescaleDB 数据库操作接口。
用于存储和查询投资组合历史数据、交易记录等。
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 尝试导入 asyncpg，如果不可用则使用占位符
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not installed. Database features will be disabled.")


@dataclass
class PortfolioSnapshot:
    """投资组合快照"""
    time: datetime
    account_id: str
    total_assets: float
    total_cash: float
    total_market_value: float
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    position_count: int = 0
    broker_count: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class PositionSnapshot:
    """持仓快照"""
    time: datetime
    account_id: str
    symbol: str
    market: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    weight: float = 0.0
    broker: str = None


@dataclass
class TradeRecord:
    """交易记录"""
    time: datetime
    account_id: str
    symbol: str
    market: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    total_value: float
    commission: float = 0.0
    currency: str = 'USD'
    broker: str = None
    order_id: str = None
    execution_id: str = None
    realized_pnl: float = None


class DatabaseService:
    """
    数据库服务

    管理与 PostgreSQL/TimescaleDB 的连接和数据操作。

    Usage:
        ```python
        db = DatabaseService()
        await db.initialize()

        # 保存投资组合快照
        await db.save_portfolio_snapshot(snapshot)

        # 获取历史数据用于计算最大回撤
        history = await db.get_portfolio_history(days=365)
        max_drawdown = db.calculate_max_drawdown(history)

        await db.close()
        ```
    """

    def __init__(self, database_url: str = None):
        """
        初始化数据库服务

        Args:
            database_url: PostgreSQL 连接 URL，格式:
                postgresql://user:password@host:port/database
        """
        self._database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://financeai:financeai@localhost:5432/financeai'
        )
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """数据库是否可用"""
        return ASYNCPG_AVAILABLE and self._initialized and self._pool is not None

    async def initialize(self) -> bool:
        """
        初始化数据库连接池

        Returns:
            bool: 是否成功初始化
        """
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, database disabled")
            return False

        try:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            self._initialized = True
            logger.info("Database connection pool initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # 清理可能部分创建的连接池
            if self._pool:
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
            self._initialized = False
            return False

    async def close(self):
        """关闭数据库连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Database connection pool closed")

    # =========================================================================
    # 投资组合快照操作
    # =========================================================================

    async def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> bool:
        """
        保存投资组合快照

        Args:
            snapshot: 投资组合快照数据

        Returns:
            bool: 是否保存成功
        """
        if not self.is_available:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO portfolio_snapshots
                    (time, account_id, total_assets, total_cash, total_market_value,
                     total_unrealized_pnl, total_realized_pnl, position_count, broker_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (time, account_id)
                    DO UPDATE SET
                        total_assets = EXCLUDED.total_assets,
                        total_cash = EXCLUDED.total_cash,
                        total_market_value = EXCLUDED.total_market_value,
                        total_unrealized_pnl = EXCLUDED.total_unrealized_pnl,
                        total_realized_pnl = EXCLUDED.total_realized_pnl,
                        position_count = EXCLUDED.position_count,
                        broker_count = EXCLUDED.broker_count,
                        metadata = EXCLUDED.metadata
                ''',
                    snapshot.time,
                    snapshot.account_id,
                    snapshot.total_assets,
                    snapshot.total_cash,
                    snapshot.total_market_value,
                    snapshot.total_unrealized_pnl,
                    snapshot.total_realized_pnl,
                    snapshot.position_count,
                    snapshot.broker_count,
                    snapshot.metadata or {},
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")
            return False

    async def get_portfolio_history(
        self,
        account_id: str = 'default',
        days: int = 365,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[PortfolioSnapshot]:
        """
        获取投资组合历史数据

        Args:
            account_id: 账户 ID
            days: 获取最近多少天的数据（如果未指定 start_date/end_date）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[PortfolioSnapshot]: 投资组合快照列表
        """
        if not self.is_available:
            return []

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT time, account_id, total_assets, total_cash, total_market_value,
                           total_unrealized_pnl, total_realized_pnl, position_count, broker_count
                    FROM portfolio_snapshots
                    WHERE account_id = $1 AND time >= $2 AND time <= $3
                    ORDER BY time ASC
                ''', account_id, start_date, end_date)

                return [
                    PortfolioSnapshot(
                        time=row['time'],
                        account_id=row['account_id'],
                        total_assets=float(row['total_assets']),
                        total_cash=float(row['total_cash']),
                        total_market_value=float(row['total_market_value']),
                        total_unrealized_pnl=float(row['total_unrealized_pnl'] or 0),
                        total_realized_pnl=float(row['total_realized_pnl'] or 0),
                        position_count=row['position_count'],
                        broker_count=row['broker_count'],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return []

    async def get_latest_snapshot(self, account_id: str = 'default') -> Optional[PortfolioSnapshot]:
        """获取最新的投资组合快照"""
        if not self.is_available:
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow('''
                    SELECT time, account_id, total_assets, total_cash, total_market_value,
                           total_unrealized_pnl, total_realized_pnl, position_count, broker_count
                    FROM portfolio_snapshots
                    WHERE account_id = $1
                    ORDER BY time DESC
                    LIMIT 1
                ''', account_id)

                if row:
                    return PortfolioSnapshot(
                        time=row['time'],
                        account_id=row['account_id'],
                        total_assets=float(row['total_assets']),
                        total_cash=float(row['total_cash']),
                        total_market_value=float(row['total_market_value']),
                        total_unrealized_pnl=float(row['total_unrealized_pnl'] or 0),
                        total_realized_pnl=float(row['total_realized_pnl'] or 0),
                        position_count=row['position_count'],
                        broker_count=row['broker_count'],
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get latest snapshot: {e}")
            return None

    # =========================================================================
    # 持仓历史操作
    # =========================================================================

    async def save_position_snapshots(self, snapshots: List[PositionSnapshot]) -> bool:
        """批量保存持仓快照"""
        if not self.is_available or not snapshots:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany('''
                    INSERT INTO position_history
                    (time, account_id, symbol, market, quantity, avg_cost, current_price,
                     market_value, unrealized_pnl, unrealized_pnl_percent, weight, broker)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (time, account_id, symbol, market) DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        avg_cost = EXCLUDED.avg_cost,
                        current_price = EXCLUDED.current_price,
                        market_value = EXCLUDED.market_value,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        unrealized_pnl_percent = EXCLUDED.unrealized_pnl_percent,
                        weight = EXCLUDED.weight,
                        broker = EXCLUDED.broker
                ''', [
                    (s.time, s.account_id, s.symbol, s.market, s.quantity, s.avg_cost,
                     s.current_price, s.market_value, s.unrealized_pnl,
                     s.unrealized_pnl_percent, s.weight, s.broker)
                    for s in snapshots
                ])
            return True
        except Exception as e:
            logger.error(f"Failed to save position snapshots: {e}")
            return False

    # =========================================================================
    # 交易记录操作
    # =========================================================================

    async def save_trades(self, trades: List[TradeRecord]) -> bool:
        """批量保存交易记录"""
        if not self.is_available or not trades:
            return False

        try:
            async with self._pool.acquire() as conn:
                # 使用 execution_id 去重
                for trade in trades:
                    await conn.execute('''
                        INSERT INTO trades
                        (time, account_id, symbol, market, action, quantity, price,
                         total_value, commission, currency, broker, order_id, execution_id, realized_pnl)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        ON CONFLICT DO NOTHING
                    ''',
                        trade.time, trade.account_id, trade.symbol, trade.market,
                        trade.action, trade.quantity, trade.price, trade.total_value,
                        trade.commission, trade.currency, trade.broker,
                        trade.order_id, trade.execution_id, trade.realized_pnl,
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")
            return False

    async def get_trades(
        self,
        account_id: str = 'default',
        days: int = 30,
        symbol: str = None,
    ) -> List[TradeRecord]:
        """获取交易记录"""
        if not self.is_available:
            return []

        try:
            async with self._pool.acquire() as conn:
                query = '''
                    SELECT time, account_id, symbol, market, action, quantity, price,
                           total_value, commission, currency, broker, order_id, execution_id, realized_pnl
                    FROM trades
                    WHERE account_id = $1 AND time >= $2
                '''
                params = [account_id, datetime.now() - timedelta(days=days)]

                if symbol:
                    query += ' AND symbol = $3'
                    params.append(symbol)

                query += ' ORDER BY time DESC'

                rows = await conn.fetch(query, *params)
                return [
                    TradeRecord(
                        time=row['time'],
                        account_id=row['account_id'],
                        symbol=row['symbol'],
                        market=row['market'],
                        action=row['action'],
                        quantity=float(row['quantity']),
                        price=float(row['price']),
                        total_value=float(row['total_value']),
                        commission=float(row['commission'] or 0),
                        currency=row['currency'],
                        broker=row['broker'],
                        order_id=row['order_id'],
                        execution_id=row['execution_id'],
                        realized_pnl=float(row['realized_pnl']) if row['realized_pnl'] else None,
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    # =========================================================================
    # 分析计算方法
    # =========================================================================

    @staticmethod
    def calculate_max_drawdown(snapshots: List[PortfolioSnapshot]) -> float:
        """
        计算最大回撤

        Args:
            snapshots: 投资组合快照列表（按时间升序排列）

        Returns:
            float: 最大回撤（0-1 之间的小数）
        """
        if not snapshots or len(snapshots) < 2:
            return 0.0

        values = [s.total_assets for s in snapshots]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    @staticmethod
    def calculate_volatility(snapshots: List[PortfolioSnapshot], annualize: bool = True) -> float:
        """
        计算波动率

        Args:
            snapshots: 投资组合快照列表
            annualize: 是否年化

        Returns:
            float: 波动率
        """
        if not snapshots or len(snapshots) < 2:
            return 0.0

        import numpy as np

        values = [s.total_assets for s in snapshots]
        returns = np.diff(values) / values[:-1]

        volatility = np.std(returns)
        if annualize:
            volatility *= np.sqrt(252)  # 假设每天一个快照

        return float(volatility)

    @staticmethod
    def calculate_sharpe_ratio(
        snapshots: List[PortfolioSnapshot],
        risk_free_rate: float = 0.05,
    ) -> float:
        """
        计算夏普比率

        Args:
            snapshots: 投资组合快照列表
            risk_free_rate: 无风险利率（年化）

        Returns:
            float: 夏普比率
        """
        if not snapshots or len(snapshots) < 2:
            return 0.0

        import numpy as np

        values = [s.total_assets for s in snapshots]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns) * 252  # 年化
        std_return = np.std(returns) * np.sqrt(252)  # 年化

        if std_return == 0:
            return 0.0

        return float((mean_return - risk_free_rate) / std_return)


# 全局数据库服务实例（线程安全）
_db_service: Optional[DatabaseService] = None
_db_lock = asyncio.Lock()


async def get_database() -> DatabaseService:
    """获取数据库服务实例（并发安全）"""
    global _db_service
    if _db_service is not None:
        return _db_service
    async with _db_lock:
        # Double-check locking：锁内再次检查
        if _db_service is None:
            service = DatabaseService()
            await service.initialize()
            _db_service = service
    return _db_service


async def close_database():
    """关闭数据库连接"""
    global _db_service
    async with _db_lock:
        if _db_service:
            await _db_service.close()
            _db_service = None
