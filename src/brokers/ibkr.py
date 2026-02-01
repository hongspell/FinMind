"""
FinMind - IBKR (盈透证券) 适配器

使用 TWS API 连接到 IB Gateway 或 TWS。
这是 IBKR 的长期稳定 API，推荐用于生产环境。

需要安装: pip install ib_insync

使用前需要：
1. 安装并运行 IB Gateway（推荐）或 TWS
2. 在 API 设置中启用 Socket 连接
3. 配置允许的 IP 地址

端口说明：
- TWS Live: 7496
- TWS Paper: 7497
- IB Gateway Live: 4001
- IB Gateway Paper: 4002
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List
import logging
from concurrent.futures import ThreadPoolExecutor

from .base import (
    BrokerAdapter,
    BrokerConfig,
    Position,
    AccountBalance,
    PortfolioSummary,
    Trade,
    TradeAction,
    BrokerError,
    AuthenticationError,
    ConnectionError,
    Market,
    Currency,
    PositionSide,
)
from .trade_store import TradeStore

logger = logging.getLogger(__name__)

# 线程池用于运行 ib_insync 操作
# 必须使用 max_workers=1 确保所有 IB 操作在同一线程中运行
# 因为 ib_insync 的事件循环绑定到创建它的线程
_executor = ThreadPoolExecutor(max_workers=1)



class IBKRAdapter(BrokerAdapter):
    """
    IBKR TWS API 适配器

    使用 ib_insync 库连接到 IB Gateway/TWS。

    Example:
        ```python
        config = BrokerConfig(
            broker_type="ibkr",
            ibkr_host="127.0.0.1",
            ibkr_port=4001,  # IB Gateway paper trading
            ibkr_client_id=1,
        )

        async with IBKRAdapter(config) as broker:
            balance = await broker.get_account_balance()
            positions = await broker.get_positions()
            print(f"Total assets: ${balance.total_assets:,.2f}")
        ```
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._ib = None
        self._account_values = {}
        self._loop = None
        self._trade_store = TradeStore("ibkr_trades", dedup_keys="exec_id")

    @property
    def broker_name(self) -> str:
        return "IBKR"

    def _ensure_event_loop(self):
        """确保当前线程有事件循环（Python 3.14 兼容）

        Python 3.14 移除了 asyncio.get_event_loop() 在非主线程中自动创建
        事件循环的行为。ib_insync 内部依赖此行为，所以必须在每次调用
        ib_insync 方法前确保事件循环已设置。
        """
        if self._loop and not self._loop.is_closed():
            asyncio.set_event_loop(self._loop)
        else:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    # -------------------------------------------------------------------------
    # 交易记录本地持久化
    #
    # IB Gateway 的 reqExecutions / fills() 只返回当前 Gateway 会话的数据，
    # Gateway 重启后执行缓存就清空了。为了保留历史交易记录，我们在每次
    # 读取 fills 时将其持久化到本地 JSON 文件。
    # -------------------------------------------------------------------------

    def _persist_fills(self, fills) -> int:
        """将 ib_insync Fill 对象转为字典，委托 TradeStore 持久化"""
        trade_dicts = []
        for fill in fills:
            exec_id = fill.execution.execId
            if not exec_id:
                continue

            trade_time = None
            if fill.execution.time:
                trade_time = str(fill.execution.time)

            trade_dicts.append({
                'exec_id': exec_id,
                'symbol': fill.contract.symbol,
                'sec_type': fill.contract.secType,
                'exchange': fill.contract.exchange,
                'currency': fill.contract.currency,
                'side': fill.execution.side,
                'shares': fill.execution.shares,
                'price': fill.execution.price,
                'order_id': fill.execution.orderId,
                'time': trade_time,
                'commission': (
                    fill.commissionReport.commission
                    if fill.commissionReport else 0.0
                ),
                'realized_pnl': (
                    fill.commissionReport.realizedPNL
                    if fill.commissionReport else None
                ),
            })

        return self._trade_store.persist(trade_dicts)

    def _connect_sync(self) -> bool:
        """同步连接方法（在单独的事件循环中运行）"""
        # 必须先创建事件循环，再导入和实例化 IB
        self._ensure_event_loop()

        try:
            from ib_insync import IB
        except ImportError:
            raise BrokerError(
                "ib_insync library not installed. "
                "Install with: pip install ib_insync"
            )

        # 尝试连接，如果 client ID 冲突则尝试其他 ID
        client_id = self.config.ibkr_client_id
        max_attempts = 5  # 尝试 client ID 1-5
        last_error = None

        for attempt in range(max_attempts):
            try:
                self._ib = IB()

                # 直接使用同步连接
                self._ib.connect(
                    host=self.config.ibkr_host,
                    port=self.config.ibkr_port,
                    clientId=client_id,
                    timeout=min(self.config.timeout, 10),  # 最多等待 10 秒
                )

                # 获取账户列表
                accounts = self._ib.managedAccounts()
                if accounts:
                    self._account_id = accounts[0]
                    self._trade_store.set_account(self._account_id)
                    logger.info(f"Connected to IBKR, account: {self._account_id}, clientId: {client_id}")
                    self._connected = True
                    return True
                else:
                    raise AuthenticationError("No managed accounts found")

            except (TimeoutError, asyncio.TimeoutError) as e:
                # TimeoutError 通常表示 client ID 冲突
                last_error = e
                if attempt < max_attempts - 1:
                    logger.warning(f"Connection timeout with clientId {client_id}, trying {client_id + 1}")
                    client_id += 1
                    continue
                else:
                    raise ConnectionError(
                        f"Connection timeout. All client IDs ({self.config.ibkr_client_id}-{client_id}) may be in use. "
                        "Please close other TWS/Gateway connections or wait a moment."
                    )

            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                # 检查是否是 client ID 冲突
                if "client id" in error_msg and "already in use" in error_msg:
                    if attempt < max_attempts - 1:
                        logger.warning(f"Client ID {client_id} in use, trying {client_id + 1}")
                        client_id += 1
                        continue
                    else:
                        raise ConnectionError(
                            f"All client IDs ({self.config.ibkr_client_id}-{client_id}) are in use. "
                            "Please close other connections or use a different client ID."
                        )

                if "connection refused" in error_msg:
                    raise ConnectionError(
                        f"Cannot connect to IB Gateway/TWS at "
                        f"{self.config.ibkr_host}:{self.config.ibkr_port}. "
                        "Please ensure IB Gateway or TWS is running and API is enabled."
                    )
                raise ConnectionError(f"IBKR connection failed: {str(e)}")

        raise ConnectionError(f"IBKR connection failed after {max_attempts} attempts: {last_error}")

    async def connect(self) -> bool:
        """连接到 IB Gateway/TWS"""
        # 如果已经连接，直接返回
        if self._connected and self._ib and self._ib.isConnected():
            logger.info("Already connected to IBKR")
            return True

        # 如果有旧连接但已断开，清理它
        if self._ib:
            await self.disconnect()

        try:
            # 在线程池中运行同步连接
            # 所有 ib_insync 操作必须在同一个线程和事件循环中
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(_executor, self._connect_sync)
            return result

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"IBKR connection error: {error_msg}")
            raise ConnectionError(f"IBKR connection failed: {error_msg}")

    async def disconnect(self) -> None:
        """断开连接"""
        try:
            if self._ib:
                try:
                    if self._ib.isConnected():
                        self._ib.disconnect()
                except Exception as e:
                    logger.warning(f"Error during IBKR disconnect: {e}")
        finally:
            self._connected = False
            self._ib = None
            if self._loop:
                try:
                    if not self._loop.is_closed():
                        self._loop.close()
                except Exception:
                    pass
                self._loop = None
            logger.info("Disconnected from IBKR")

    async def health_check(self) -> bool:
        """健康检查"""
        if not self._ib:
            return False
        return self._ib.isConnected()

    def _get_account_balance_sync(self) -> AccountBalance:
        """同步获取账户余额"""
        if not self._connected or not self._ib:
            raise BrokerError("Not connected to IBKR")

        self._ensure_event_loop()

        # 使用 accountValues() 获取缓存的账户数据（不会创建新订阅）
        # 注意：不要使用 accountSummary()，它在 Python 3.14 中会因
        # 事件循环问题导致订阅泄漏（"Maximum number of account summary requests exceeded"）
        account_values = self._ib.accountValues(self._account_id)

        # 解析账户值
        values = {}
        for av in account_values:
            if av.value:
                try:
                    values[av.tag] = float(av.value)
                except (ValueError, TypeError):
                    values[av.tag] = av.value  # 保留字符串值

        # 获取货币
        base_currency = values.get("BaseCurrency", "USD")
        currency = self._parse_currency(base_currency)

        return AccountBalance(
            total_assets=values.get("NetLiquidation", 0.0),
            cash=values.get("TotalCashValue", 0.0),
            market_value=values.get("GrossPositionValue", 0.0),
            buying_power=values.get("BuyingPower", 0.0),
            currency=currency,
            day_pnl=values.get("DailyPnL", 0.0),
            total_pnl=values.get("RealizedPnL", 0.0) + values.get("UnrealizedPnL", 0.0),
            margin_used=values.get("MaintMarginReq", 0.0),
            margin_available=values.get("AvailableFunds", 0.0),
            maintenance_margin=values.get("MaintMarginReq", 0.0),
            last_updated=datetime.now(),
        )

    async def get_account_balance(self) -> AccountBalance:
        """获取账户余额"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._get_account_balance_sync)

    def _get_positions_sync(self) -> List[Position]:
        """同步获取持仓列表"""
        if not self._connected or not self._ib:
            raise BrokerError("Not connected to IBKR")

        self._ensure_event_loop()

        positions = []

        # 使用 portfolio() 获取持仓 - 它包含实时市值和盈亏
        # portfolio() 返回 PortfolioItem 列表，包含 marketValue, unrealizedPNL 等
        try:
            portfolio_items = self._ib.portfolio(self._account_id)
            logger.info(f"Found {len(portfolio_items)} portfolio items")

            for item in portfolio_items:
                contract = item.contract
                symbol = contract.symbol

                # 跳过非股票/ETF类型（如期权、期货等）
                if contract.secType not in ('STK', 'ETF'):
                    continue

                # 确定市场
                market = self._resolve_market(contract.exchange, contract.currency)

                # 确定货币
                currency = self._parse_currency(contract.currency)

                # 从 portfolio item 获取数据（IBKR 直接提供）
                quantity = item.position
                avg_cost = item.averageCost
                market_value = item.marketValue
                unrealized_pnl = item.unrealizedPNL

                # 计算当前价格
                if quantity != 0:
                    current_price = market_value / quantity
                else:
                    current_price = avg_cost

                # 计算盈亏百分比
                cost_basis = abs(quantity * avg_cost)
                unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    market=market,
                    quantity=abs(quantity),
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=abs(market_value),
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                    currency=currency,
                    company_name=contract.localSymbol or symbol,
                    last_updated=datetime.now(),
                ))

            logger.info(f"Processed {len(positions)} stock/ETF positions")

        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            # 回退到 positions() 方法
            ib_positions = self._ib.positions()  # 不传 account 获取所有持仓
            logger.info(f"Fallback: Found {len(ib_positions)} positions")

            for pos in ib_positions:
                contract = pos.contract
                if contract.secType not in ('STK', 'ETF'):
                    continue

                symbol = contract.symbol
                market = self._resolve_market(contract.exchange, contract.currency)
                currency = self._parse_currency(contract.currency)

                quantity = pos.position
                avg_cost = pos.avgCost

                positions.append(Position(
                    symbol=symbol,
                    market=market,
                    quantity=abs(quantity),
                    avg_cost=avg_cost,
                    current_price=avg_cost,  # 无法获取实时价格
                    market_value=abs(quantity * avg_cost),
                    unrealized_pnl=0,
                    unrealized_pnl_percent=0,
                    side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                    currency=currency,
                    company_name=contract.localSymbol or symbol,
                    last_updated=datetime.now(),
                ))

        return positions

    async def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._get_positions_sync)

    def _get_trades_sync(self, days: int = 7) -> List[Trade]:
        """获取交易历史（当前会话 fills + 本地持久化记录）

        IB Gateway 的 fills/reqExecutions 只保留当前会话的数据，
        Gateway 重启后就丢失了。所以我们：
        1. 获取当前会话的 fills 并持久化到本地
        2. 从本地文件读取所有历史交易记录
        3. 合并后返回
        """
        if not self._connected or not self._ib:
            raise BrokerError("Not connected to IBKR")

        self._ensure_event_loop()

        # Step 1: 捕获当前会话的 fills 并持久化
        try:
            self._ib.sleep(0)  # 处理待处理消息
            session_fills = self._ib.fills()
            if session_fills:
                new_count = self._persist_fills(session_fills)
                logger.info(
                    f"Session has {len(session_fills)} fills, "
                    f"{new_count} newly persisted"
                )
        except Exception as e:
            logger.warning(f"Error capturing session fills: {e}")

        # Step 2: 从本地持久化存储读取所有交易记录
        cutoff_time = datetime.now() - timedelta(days=days)
        persisted = self._trade_store.load()

        trades = []
        for t in persisted:
            if t.get('sec_type') not in ('STK', 'ETF'):
                continue

            trade_time = None
            if t.get('time'):
                try:
                    trade_time = datetime.fromisoformat(str(t['time']))
                except Exception:
                    pass

            if trade_time and trade_time < cutoff_time:
                continue

            market = self._resolve_market(
                t.get('exchange', ''), t.get('currency', '')
            )
            currency = self._parse_currency(t.get('currency', ''))
            action = (
                TradeAction.BUY if t.get('side') == 'BOT'
                else TradeAction.SELL
            )

            trades.append(Trade(
                symbol=t['symbol'],
                market=market,
                action=action,
                quantity=t.get('shares', 0),
                price=t.get('price', 0),
                commission=t.get('commission', 0),
                currency=currency,
                trade_time=trade_time,
                order_id=(
                    str(t['order_id']) if t.get('order_id') else None
                ),
                execution_id=t.get('exec_id'),
                realized_pnl=t.get('realized_pnl'),
            ))

        trades.sort(key=lambda t: t.trade_time or datetime.min, reverse=True)
        logger.info(
            f"Returning {len(trades)} trades "
            f"(from {len(persisted)} persisted records)"
        )
        return trades

    async def get_trades(self, days: int = 7) -> List[Trade]:
        """获取交易历史"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, lambda: self._get_trades_sync(days))


# =============================================================================
# 模拟适配器（用于测试）
# =============================================================================

class IBKRMockAdapter(BrokerAdapter):
    """
    IBKR 模拟适配器

    用于测试和开发，不需要实际连接到 IB Gateway。
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._mock_positions = []
        self._mock_balance = None
        self._mock_trades = []

    @property
    def broker_name(self) -> str:
        return "IBKR (Mock)"

    async def connect(self) -> bool:
        """模拟连接"""
        self._connected = True
        self._account_id = "DU1234567"
        self._setup_mock_data()
        logger.info("Connected to IBKR (Mock)")
        return True

    async def disconnect(self) -> None:
        """模拟断开"""
        self._connected = False
        logger.info("Disconnected from IBKR (Mock)")

    async def health_check(self) -> bool:
        """健康检查"""
        return self._connected

    def _setup_mock_data(self):
        """设置模拟数据"""
        self._mock_balance = AccountBalance(
            total_assets=150000.0,
            cash=25000.0,
            market_value=125000.0,
            buying_power=50000.0,
            currency=Currency.USD,
            day_pnl=1250.0,
            total_pnl=15000.0,
            margin_used=0.0,
            margin_available=50000.0,
            last_updated=datetime.now(),
        )

        self._mock_positions = [
            Position(
                symbol="AAPL",
                market=Market.US,
                quantity=100,
                avg_cost=165.0,
                current_price=178.50,
                market_value=17850.0,
                unrealized_pnl=1350.0,
                unrealized_pnl_percent=8.18,
                currency=Currency.USD,
                company_name="Apple Inc.",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="MSFT",
                market=Market.US,
                quantity=50,
                avg_cost=380.0,
                current_price=415.25,
                market_value=20762.50,
                unrealized_pnl=1762.50,
                unrealized_pnl_percent=9.28,
                currency=Currency.USD,
                company_name="Microsoft Corporation",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="GOOGL",
                market=Market.US,
                quantity=30,
                avg_cost=140.0,
                current_price=175.80,
                market_value=5274.0,
                unrealized_pnl=1074.0,
                unrealized_pnl_percent=25.57,
                currency=Currency.USD,
                company_name="Alphabet Inc.",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="TSLA",
                market=Market.US,
                quantity=25,
                avg_cost=200.0,
                current_price=248.50,
                market_value=6212.50,
                unrealized_pnl=1212.50,
                unrealized_pnl_percent=24.25,
                currency=Currency.USD,
                company_name="Tesla, Inc.",
                sector="Consumer Cyclical",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="NVDA",
                market=Market.US,
                quantity=40,
                avg_cost=450.0,
                current_price=875.50,
                market_value=35020.0,
                unrealized_pnl=17020.0,
                unrealized_pnl_percent=94.56,
                currency=Currency.USD,
                company_name="NVIDIA Corporation",
                sector="Technology",
                last_updated=datetime.now(),
            ),
        ]

        self._mock_trades = [
            Trade(
                symbol="AAPL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=50,
                price=162.30,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=120),
                order_id="MOCK-1001",
                execution_id="EXEC-1001",
            ),
            Trade(
                symbol="AAPL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=50,
                price=168.50,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=60),
                order_id="MOCK-1002",
                execution_id="EXEC-1002",
            ),
            Trade(
                symbol="MSFT",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=50,
                price=380.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=90),
                order_id="MOCK-1003",
                execution_id="EXEC-1003",
            ),
            Trade(
                symbol="GOOGL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=30,
                price=140.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=180),
                order_id="MOCK-1004",
                execution_id="EXEC-1004",
            ),
            Trade(
                symbol="TSLA",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=25,
                price=200.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=150),
                order_id="MOCK-1005",
                execution_id="EXEC-1005",
            ),
            Trade(
                symbol="NVDA",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=40,
                price=450.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=200),
                order_id="MOCK-1006",
                execution_id="EXEC-1006",
            ),
            Trade(
                symbol="AMZN",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=20,
                price=178.50,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=45),
                order_id="MOCK-1007",
                execution_id="EXEC-1007",
            ),
            Trade(
                symbol="AMZN",
                market=Market.US,
                action=TradeAction.SELL,
                quantity=20,
                price=195.20,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=10),
                order_id="MOCK-1008",
                execution_id="EXEC-1008",
                realized_pnl=334.0,
            ),
        ]

    async def get_account_balance(self) -> AccountBalance:
        """获取模拟账户余额"""
        if not self._connected:
            raise BrokerError("Not connected")
        return self._mock_balance

    async def get_positions(self) -> List[Position]:
        """获取模拟持仓"""
        if not self._connected:
            raise BrokerError("Not connected")
        return self._mock_positions

    async def get_trades(self, days: int = 365) -> List[Trade]:
        """获取模拟交易历史"""
        if not self._connected:
            raise BrokerError("Not connected")
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in self._mock_trades if t.trade_time and t.trade_time >= cutoff]
