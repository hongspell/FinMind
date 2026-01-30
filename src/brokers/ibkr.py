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
from datetime import datetime
from typing import Optional, List
import logging
import threading
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

logger = logging.getLogger(__name__)

# 线程池用于运行 ib_insync 操作
_executor = ThreadPoolExecutor(max_workers=2)


def _run_in_new_loop(coro):
    """在新的事件循环中运行协程（用于解决 Python 3.14 兼容性问题）"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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

    @property
    def broker_name(self) -> str:
        return "IBKR"

    def _connect_sync(self) -> bool:
        """同步连接方法（在单独的事件循环中运行）"""
        # 必须先创建事件循环，再导入和实例化 IB
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

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

        # 请求账户摘要
        account_values = self._ib.accountSummary(self._account_id)

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
                market = self._get_market(contract.exchange, contract.currency)

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
                market = self._get_market(contract.exchange, contract.currency)
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

    def _get_market(self, exchange: str, currency: str) -> Market:
        """根据交易所和货币确定市场"""
        exchange_upper = exchange.upper() if exchange else ""

        # 美股交易所
        us_exchanges = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "IEX", "SMART"}
        if exchange_upper in us_exchanges or currency == "USD":
            return Market.US

        # 港股交易所
        hk_exchanges = {"SEHK", "HKEX"}
        if exchange_upper in hk_exchanges or currency == "HKD":
            return Market.HK

        # A股交易所
        cn_exchanges = {"SSE", "SZSE", "SHSE"}
        if exchange_upper in cn_exchanges or currency == "CNY":
            return Market.CN

        # 默认美股
        return Market.US

    def _get_trades_sync(self, days: int = 7) -> List[Trade]:
        """同步获取交易历史"""
        if not self._connected or not self._ib:
            raise BrokerError("Not connected to IBKR")

        trades = []

        try:
            # 使用 fills() 获取成交记录 - 返回 Fill 对象列表
            fills = self._ib.fills()
            logger.info(f"Found {len(fills)} fills")

            for fill in fills:
                contract = fill.contract
                execution = fill.execution

                # 只处理股票和ETF
                if contract.secType not in ('STK', 'ETF'):
                    continue

                # 确定市场和货币
                market = self._get_market(contract.exchange, contract.currency)
                currency = self._parse_currency(contract.currency)

                # 确定交易方向
                action = TradeAction.BUY if execution.side == 'BOT' else TradeAction.SELL

                # 获取佣金和实现盈亏
                commission = 0.0
                realized_pnl = None
                if fill.commissionReport:
                    commission = fill.commissionReport.commission or 0.0
                    realized_pnl = fill.commissionReport.realizedPNL

                # 解析交易时间
                trade_time = None
                if execution.time:
                    try:
                        if isinstance(execution.time, datetime):
                            trade_time = execution.time
                        else:
                            trade_time = datetime.fromisoformat(str(execution.time))
                    except:
                        pass

                trades.append(Trade(
                    symbol=contract.symbol,
                    market=market,
                    action=action,
                    quantity=execution.shares,
                    price=execution.price,
                    commission=commission,
                    currency=currency,
                    trade_time=trade_time,
                    order_id=str(execution.orderId) if execution.orderId else None,
                    execution_id=execution.execId,
                    realized_pnl=realized_pnl,
                ))

            # 按时间降序排列
            trades.sort(key=lambda t: t.trade_time or datetime.min, reverse=True)
            logger.info(f"Processed {len(trades)} stock/ETF trades")

        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
