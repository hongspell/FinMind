"""
FinMind - IBKR Client Portal API 适配器

通过 Client Portal Gateway (localhost:5000) REST API 获取数据。
相比 TWS API (ib_insync)，Client Portal API 可获取历史交易记录。

使用前需要：
1. 下载并运行 Client Portal Gateway
2. 在浏览器打开 https://localhost:5000 完成登录和 2FA
3. Gateway 必须在本地运行

速率限制：10 req/s 全局，部分端点 1 req/5s
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List

from .base import (
    BrokerAdapter,
    BrokerConfig,
    Position,
    AccountBalance,
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


class IBKRClientPortalAdapter(BrokerAdapter):
    """
    IBKR Client Portal API 适配器

    使用 httpx.AsyncClient 发起 REST 请求连接到 Client Portal Gateway。

    Example:
        ```python
        config = BrokerConfig(
            broker_type="ibkr_cp",
            ibkr_cp_base_url="https://localhost:5000/v1/api",
        )

        async with IBKRClientPortalAdapter(config) as broker:
            balance = await broker.get_account_balance()
            positions = await broker.get_positions()
        ```
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._client = None
        self._tickle_task: Optional[asyncio.Task] = None

    @property
    def broker_name(self) -> str:
        return "IBKR-CP"

    async def _ensure_client(self):
        """确保 httpx client 已创建"""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise BrokerError(
                    "httpx library not installed. "
                    "Install with: pip install httpx"
                )
            self._client = httpx.AsyncClient(
                base_url=self.config.ibkr_cp_base_url,
                verify=False,  # 跳过自签名证书验证
                timeout=self.config.timeout,
            )

    async def _tickle_loop(self):
        """后台 tickle 保活任务，每 55 秒调用一次 /tickle"""
        while True:
            try:
                await asyncio.sleep(55)
                if self._client:
                    await self._client.post("/tickle")
                    logger.debug("IBKR CP tickle sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"IBKR CP tickle error (ignored): {e}")

    async def connect(self) -> bool:
        """连接到 Client Portal Gateway"""
        await self._ensure_client()

        try:
            # 初始化 brokerage session
            resp = await self._client.post("/iserver/auth/ssodh/init")
            if resp.status_code not in (200, 201):
                logger.warning(f"SSO init returned {resp.status_code}, attempting to continue")

            # 获取账户列表
            resp = await self._client.get("/portfolio/accounts")
            if resp.status_code != 200:
                raise ConnectionError(
                    f"Failed to get accounts: HTTP {resp.status_code}. "
                    "Please ensure you are logged in at https://localhost:5000"
                )

            accounts = resp.json()
            if not accounts:
                raise AuthenticationError("No accounts found")

            # 取第一个账户
            account = accounts[0]
            self._account_id = account.get("accountId") or account.get("id")
            if not self._account_id:
                raise AuthenticationError("Account ID not found in response")

            self._connected = True

            # 启动 tickle 保活任务
            self._tickle_task = asyncio.create_task(self._tickle_loop())

            logger.info(f"Connected to IBKR Client Portal, account: {self._account_id}")
            return True

        except (AuthenticationError, ConnectionError):
            raise
        except Exception as e:
            error_msg = str(e)
            if "connect" in error_msg.lower():
                raise ConnectionError(
                    f"Cannot connect to Client Portal Gateway at "
                    f"{self.config.ibkr_cp_base_url}. "
                    "Please ensure the Gateway is running and you are logged in."
                )
            raise ConnectionError(f"IBKR Client Portal connection failed: {error_msg}")

    async def disconnect(self) -> None:
        """断开连接"""
        # 取消 tickle 任务
        if self._tickle_task:
            self._tickle_task.cancel()
            try:
                await self._tickle_task
            except asyncio.CancelledError:
                pass
            self._tickle_task = None

        # 关闭 httpx client
        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False
        logger.info("Disconnected from IBKR Client Portal")

    async def health_check(self) -> bool:
        """健康检查 - 检查认证状态"""
        if not self._client or not self._connected:
            return False
        try:
            resp = await self._client.post("/iserver/auth/status")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("authenticated", False)
            return False
        except Exception:
            return False

    async def get_account_balance(self) -> AccountBalance:
        """获取账户余额"""
        if not self._connected or not self._client:
            raise BrokerError("Not connected to IBKR Client Portal")

        try:
            resp = await self._client.get(f"/portfolio/{self._account_id}/summary")
            if resp.status_code != 200:
                raise BrokerError(f"Failed to get account summary: HTTP {resp.status_code}")

            data = resp.json()

            def _val(key: str, default=0.0) -> float:
                """从 summary 中提取数值"""
                item = data.get(key, {})
                if isinstance(item, dict):
                    return float(item.get("amount", default))
                return float(item) if item else default

            currency = self._parse_currency(
                data.get("baseCurrency", {}).get("value", "USD")
                if isinstance(data.get("baseCurrency"), dict)
                else str(data.get("baseCurrency", "USD"))
            )

            return AccountBalance(
                total_assets=_val("netliquidation"),
                cash=_val("totalcashvalue"),
                market_value=_val("grosspositionvalue"),
                buying_power=_val("buyingpower"),
                currency=currency,
                day_pnl=_val("dailypnl"),
                total_pnl=_val("realizedpnl") + _val("unrealizedpnl"),
                margin_used=_val("maintmarginreq"),
                margin_available=_val("availablefunds"),
                maintenance_margin=_val("maintmarginreq"),
                last_updated=datetime.now(),
            )

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get account balance: {e}")

    async def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        if not self._connected or not self._client:
            raise BrokerError("Not connected to IBKR Client Portal")

        try:
            resp = await self._client.get(f"/portfolio/{self._account_id}/positions/0")
            if resp.status_code != 200:
                raise BrokerError(f"Failed to get positions: HTTP {resp.status_code}")

            items = resp.json()
            if not isinstance(items, list):
                return []

            positions = []
            for item in items:
                # 只处理股票/ETF
                asset_class = item.get("assetClass", "")
                if asset_class not in ("STK", "ETF"):
                    continue

                symbol = item.get("ticker") or item.get("contractDesc", "")
                currency_str = item.get("currency", "USD")
                currency = self._parse_currency(currency_str)
                market = self._resolve_market(item.get("listingExchange", ""), currency_str)

                quantity = float(item.get("position", 0))
                avg_cost = float(item.get("avgCost", 0))
                mkt_value = float(item.get("mktValue", 0))
                unrealized_pnl = float(item.get("unrealizedPnl", 0))

                # 当前价格
                if quantity != 0:
                    current_price = mkt_value / quantity
                else:
                    current_price = float(item.get("mktPrice", avg_cost))

                # 盈亏百分比
                cost_basis = abs(quantity * avg_cost)
                pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    market=market,
                    quantity=abs(quantity),
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=abs(mkt_value),
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=pnl_pct,
                    side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                    currency=currency,
                    company_name=item.get("name"),
                    last_updated=datetime.now(),
                ))

            return positions

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get positions: {e}")

    async def get_trades(self, days: int = 7) -> List[Trade]:
        """获取当日交易记录"""
        if not self._connected or not self._client:
            raise BrokerError("Not connected to IBKR Client Portal")

        try:
            resp = await self._client.get("/iserver/account/trades")
            if resp.status_code != 200:
                raise BrokerError(f"Failed to get trades: HTTP {resp.status_code}")

            items = resp.json()
            if not isinstance(items, list):
                return []

            cutoff_time = datetime.now() - timedelta(days=days)
            trades = []

            for item in items:
                # 解析交易时间
                trade_time = None
                time_str = item.get("trade_time_r") or item.get("trade_time")
                if time_str:
                    try:
                        if isinstance(time_str, (int, float)):
                            trade_time = datetime.fromtimestamp(time_str / 1000)
                        else:
                            trade_time = datetime.fromisoformat(str(time_str))
                    except Exception:
                        pass

                if trade_time and trade_time < cutoff_time:
                    continue

                symbol = item.get("symbol", "")
                currency_str = item.get("currency", "USD")
                currency = self._parse_currency(currency_str)
                market = self._resolve_market(item.get("exchange", ""), currency_str)

                side = str(item.get("side", "")).upper()
                action = TradeAction.BUY if side in ("B", "BOT", "BUY") else TradeAction.SELL

                trades.append(Trade(
                    symbol=symbol,
                    market=market,
                    action=action,
                    quantity=abs(float(item.get("size", 0))),
                    price=float(item.get("price", 0)),
                    commission=float(item.get("commission", 0)),
                    currency=currency,
                    trade_time=trade_time,
                    order_id=str(item.get("order_ref", "")),
                    execution_id=item.get("execution_id"),
                    realized_pnl=item.get("realized_pnl"),
                ))

            trades.sort(key=lambda t: t.trade_time or datetime.min, reverse=True)
            return trades

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get trades: {e}")

# =============================================================================
# 模拟适配器（用于测试）
# =============================================================================

class IBKRClientPortalMockAdapter(BrokerAdapter):
    """
    IBKR Client Portal 模拟适配器

    用于测试和开发，不需要实际运行 Client Portal Gateway。
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._mock_positions = []
        self._mock_balance = None
        self._mock_trades = []

    @property
    def broker_name(self) -> str:
        return "IBKR-CP (Mock)"

    async def connect(self) -> bool:
        """模拟连接"""
        self._connected = True
        self._account_id = "DU9876543"
        self._setup_mock_data()
        logger.info("Connected to IBKR Client Portal (Mock)")
        return True

    async def disconnect(self) -> None:
        """模拟断开"""
        self._connected = False
        logger.info("Disconnected from IBKR Client Portal (Mock)")

    async def health_check(self) -> bool:
        """健康检查"""
        return self._connected

    def _setup_mock_data(self):
        """设置模拟数据"""
        self._mock_balance = AccountBalance(
            total_assets=180000.0,
            cash=30000.0,
            market_value=150000.0,
            buying_power=60000.0,
            currency=Currency.USD,
            day_pnl=1800.0,
            total_pnl=18000.0,
            margin_used=0.0,
            margin_available=60000.0,
            last_updated=datetime.now(),
        )

        self._mock_positions = [
            Position(
                symbol="AAPL",
                market=Market.US,
                quantity=120,
                avg_cost=170.0,
                current_price=185.50,
                market_value=22260.0,
                unrealized_pnl=1860.0,
                unrealized_pnl_percent=9.12,
                currency=Currency.USD,
                company_name="Apple Inc.",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="MSFT",
                market=Market.US,
                quantity=60,
                avg_cost=390.0,
                current_price=420.00,
                market_value=25200.0,
                unrealized_pnl=1800.0,
                unrealized_pnl_percent=7.69,
                currency=Currency.USD,
                company_name="Microsoft Corporation",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="NVDA",
                market=Market.US,
                quantity=35,
                avg_cost=500.0,
                current_price=880.00,
                market_value=30800.0,
                unrealized_pnl=13300.0,
                unrealized_pnl_percent=76.0,
                currency=Currency.USD,
                company_name="NVIDIA Corporation",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="AMZN",
                market=Market.US,
                quantity=45,
                avg_cost=170.0,
                current_price=192.00,
                market_value=8640.0,
                unrealized_pnl=990.0,
                unrealized_pnl_percent=12.94,
                currency=Currency.USD,
                company_name="Amazon.com, Inc.",
                sector="Consumer Cyclical",
                last_updated=datetime.now(),
            ),
        ]

        self._mock_trades = [
            Trade(
                symbol="AAPL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=60,
                price=168.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=100),
                order_id="CP-MOCK-001",
                execution_id="CP-EXEC-001",
            ),
            Trade(
                symbol="AAPL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=60,
                price=172.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=50),
                order_id="CP-MOCK-002",
                execution_id="CP-EXEC-002",
            ),
            Trade(
                symbol="MSFT",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=60,
                price=390.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=80),
                order_id="CP-MOCK-003",
                execution_id="CP-EXEC-003",
            ),
            Trade(
                symbol="NVDA",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=35,
                price=500.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=180),
                order_id="CP-MOCK-004",
                execution_id="CP-EXEC-004",
            ),
            Trade(
                symbol="AMZN",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=45,
                price=170.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=30),
                order_id="CP-MOCK-005",
                execution_id="CP-EXEC-005",
            ),
            Trade(
                symbol="GOOGL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=25,
                price=145.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=40),
                order_id="CP-MOCK-006",
                execution_id="CP-EXEC-006",
            ),
            Trade(
                symbol="GOOGL",
                market=Market.US,
                action=TradeAction.SELL,
                quantity=25,
                price=178.00,
                commission=1.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=5),
                order_id="CP-MOCK-007",
                execution_id="CP-EXEC-007",
                realized_pnl=825.0,
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
