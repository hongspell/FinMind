"""
FinMind - 富途证券适配器

使用 Futu OpenAPI 连接到 OpenD Gateway。

需要安装: pip install futu-api

使用前需要：
1. 下载并安装 FutuOpenD: https://www.futunn.com/download/openAPI
2. 启动 OpenD 并登录
3. 如需交易功能，需要配置 RSA 密钥和交易密码

端口说明：
- OpenD 默认端口: 11111
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
import logging

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
from .trade_store import TradeStore

logger = logging.getLogger(__name__)


class FutuAdapter(BrokerAdapter):
    """
    富途 OpenAPI 适配器

    使用 futu-api 库连接到 OpenD Gateway。

    Example:
        ```python
        config = BrokerConfig(
            broker_type="futu",
            futu_host="127.0.0.1",
            futu_port=11111,
        )

        async with FutuAdapter(config) as broker:
            balance = await broker.get_account_balance()
            positions = await broker.get_positions()
        ```
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._quote_ctx = None
        self._trade_ctx = None
        self._trd_env = None
        self._trade_store = TradeStore("futu_trades", dedup_keys="deal_id")

    @property
    def broker_name(self) -> str:
        return "Futu"

    async def connect(self) -> bool:
        """连接到 OpenD"""
        try:
            # 延迟导入
            from futu import (
                OpenQuoteContext,
                OpenSecTradeContext,
                TrdEnv,
                TrdMarket,
                RET_OK,
            )

            # 连接行情上下文
            self._quote_ctx = OpenQuoteContext(
                host=self.config.futu_host,
                port=self.config.futu_port,
            )

            # 确定交易环境（模拟或真实）
            self._trd_env = TrdEnv.SIMULATE  # 默认使用模拟环境

            # 连接交易上下文
            self._trade_ctx = OpenSecTradeContext(
                host=self.config.futu_host,
                port=self.config.futu_port,
            )

            # 如果提供了交易密码，解锁交易
            if self.config.futu_trade_password:
                ret, data = self._trade_ctx.unlock_trade(
                    password=self.config.futu_trade_password
                )
                if ret != RET_OK:
                    logger.warning(f"Failed to unlock trade: {data}")

            # 获取账户列表
            ret, data = self._trade_ctx.get_acc_list()
            if ret == RET_OK and len(data) > 0:
                self._account_id = str(data.iloc[0]['acc_id'])
                self._trade_store.set_account(self._account_id)
                logger.info(f"Connected to Futu, account: {self._account_id}")
            else:
                raise AuthenticationError(f"Failed to get account list: {data}")

            self._connected = True
            return True

        except ImportError:
            raise BrokerError(
                "futu-api library not installed. "
                "Install with: pip install futu-api"
            )
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower():
                raise ConnectionError(
                    f"Cannot connect to OpenD at "
                    f"{self.config.futu_host}:{self.config.futu_port}. "
                    "Please ensure OpenD is running."
                )
            raise ConnectionError(f"Futu connection failed: {error_msg}")

    async def disconnect(self) -> None:
        """断开连接"""
        if self._quote_ctx:
            self._quote_ctx.close()
        if self._trade_ctx:
            self._trade_ctx.close()
        self._connected = False
        logger.info("Disconnected from Futu")

    async def health_check(self) -> bool:
        """健康检查"""
        if not self._trade_ctx:
            return False
        try:
            from futu import RET_OK
            ret, _ = self._trade_ctx.get_acc_list()
            return ret == RET_OK
        except Exception:
            return False

    async def get_account_balance(self) -> AccountBalance:
        """获取账户余额"""
        if not self._connected or not self._trade_ctx:
            raise BrokerError("Not connected to Futu")

        try:
            from futu import RET_OK, TrdMarket

            # 获取账户资金
            ret, data = self._trade_ctx.accinfo_query(
                trd_env=self._trd_env,
                acc_id=int(self._account_id) if self._account_id else 0,
            )

            if ret != RET_OK:
                raise BrokerError(f"Failed to get account info: {data}")

            if len(data) == 0:
                raise BrokerError("No account data returned")

            row = data.iloc[0]

            # 确定货币
            currency = self._parse_currency(
                str(row['currency']) if 'currency' in row else '',
                default=Currency.HKD,
            )

            return AccountBalance(
                total_assets=float(row.get('total_assets', 0)),
                cash=float(row.get('cash', 0)),
                market_value=float(row.get('market_val', 0)),
                buying_power=float(row.get('power', 0)),
                currency=currency,
                day_pnl=float(row.get('realized_pl', 0)),
                total_pnl=float(row.get('unrealized_pl', 0)),
                margin_used=float(row.get('frozen_cash', 0)),
                margin_available=float(row.get('avl_withdrawal_cash', 0)),
                last_updated=datetime.now(),
            )

        except Exception as e:
            if "RET_OK" not in str(e):
                raise BrokerError(f"Failed to get account balance: {e}")
            raise

    async def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        if not self._connected or not self._trade_ctx:
            raise BrokerError("Not connected to Futu")

        try:
            from futu import RET_OK

            # 获取持仓
            ret, data = self._trade_ctx.position_list_query(
                trd_env=self._trd_env,
                acc_id=int(self._account_id) if self._account_id else 0,
            )

            if ret != RET_OK:
                raise BrokerError(f"Failed to get positions: {data}")

            positions = []
            for _, row in data.iterrows():
                code = str(row.get('code', ''))
                # 解析股票代码和市场
                symbol, market = self._parse_futu_code(code)

                # 确定货币
                market_currency_map = {Market.US: Currency.USD, Market.CN: Currency.CNY}
                currency = market_currency_map.get(market, Currency.HKD)

                quantity = float(row.get('qty', 0))
                avg_cost = float(row.get('cost_price', 0))
                current_price = float(row.get('nominal_price', 0))
                market_value = float(row.get('market_val', 0))
                unrealized_pnl = float(row.get('pl_val', 0))
                unrealized_pnl_percent = float(row.get('pl_ratio', 0)) * 100

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
                    company_name=str(row.get('stock_name', '')),
                    last_updated=datetime.now(),
                ))

            return positions

        except Exception as e:
            if "RET_OK" not in str(e):
                raise BrokerError(f"Failed to get positions: {e}")
            raise

    async def get_trades(self, days: int = 365) -> List[Trade]:
        """获取交易历史（当前会话 + 本地持久化记录）"""
        if not self._connected or not self._trade_ctx:
            raise BrokerError("Not connected to Futu")

        try:
            from futu import RET_OK

            cutoff_time = datetime.now() - timedelta(days=days)

            # Step 1: 从 API 获取当前交易并持久化
            ret, data = self._trade_ctx.deal_list_query(
                trd_env=self._trd_env,
                acc_id=int(self._account_id) if self._account_id else 0,
            )

            if ret == RET_OK and len(data) > 0:
                trade_dicts = []
                for _, row in data.iterrows():
                    trade_dicts.append({
                        "deal_id": str(row.get("deal_id", "")),
                        "code": str(row.get("code", "")),
                        "trd_side": str(row.get("trd_side", "")),
                        "qty": float(row.get("qty", 0)),
                        "price": float(row.get("price", 0)),
                        "order_id": str(row.get("order_id", "")),
                        "create_time": str(row.get("create_time", "")),
                    })
                self._trade_store.persist(trade_dicts)

            # Step 2: 从本地持久化存储读取所有交易记录
            persisted = self._trade_store.load()

            trades = []
            for t in persisted:
                code = t.get("code", "")
                symbol, market = self._parse_futu_code(code)

                market_currency_map = {Market.US: Currency.USD, Market.CN: Currency.CNY}
                currency = market_currency_map.get(market, Currency.HKD)

                trd_side = str(t.get("trd_side", "")).upper()
                action = TradeAction.BUY if "BUY" in trd_side else TradeAction.SELL

                trade_time = None
                create_time = t.get("create_time")
                if create_time:
                    try:
                        trade_time = datetime.strptime(str(create_time), "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass

                # 日期过滤
                if trade_time and trade_time < cutoff_time:
                    continue

                trades.append(Trade(
                    symbol=symbol,
                    market=market,
                    action=action,
                    quantity=float(t.get("qty", 0)),
                    price=float(t.get("price", 0)),
                    commission=0.0,
                    currency=currency,
                    trade_time=trade_time,
                    order_id=str(t.get("order_id", "")),
                    execution_id=str(t.get("deal_id", "")),
                ))

            trades.sort(key=lambda t: t.trade_time or datetime.min, reverse=True)
            return trades

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to get trades: {e}")

    def _parse_futu_code(self, code: str) -> tuple:
        """
        解析富途股票代码

        富途代码格式: 市场.代码
        - US.AAPL -> AAPL, Market.US
        - HK.00700 -> 00700, Market.HK
        - SH.600519 -> 600519, Market.CN
        """
        if '.' in code:
            market_str, symbol = code.split('.', 1)
            market_str = market_str.upper()

            if market_str == 'US':
                return symbol, Market.US
            elif market_str == 'HK':
                return symbol, Market.HK
            elif market_str in ('SH', 'SZ'):
                return symbol, Market.CN
            elif market_str == 'SG':
                return symbol, Market.SG

        return code, Market.US  # 默认美股


# =============================================================================
# 模拟适配器（用于测试）
# =============================================================================

class FutuMockAdapter(BrokerAdapter):
    """
    富途模拟适配器

    用于测试和开发。
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._mock_positions = []
        self._mock_balance = None
        self._mock_trades: List[Trade] = []

    @property
    def broker_name(self) -> str:
        return "Futu (Mock)"

    async def connect(self) -> bool:
        """模拟连接"""
        self._connected = True
        self._account_id = "88888888"
        self._setup_mock_data()
        logger.info("Connected to Futu (Mock)")
        return True

    async def disconnect(self) -> None:
        """模拟断开"""
        self._connected = False
        logger.info("Disconnected from Futu (Mock)")

    async def health_check(self) -> bool:
        """健康检查"""
        return self._connected

    def _setup_mock_data(self):
        """设置模拟数据"""
        self._mock_balance = AccountBalance(
            total_assets=500000.0,
            cash=100000.0,
            market_value=400000.0,
            buying_power=150000.0,
            currency=Currency.HKD,
            day_pnl=5000.0,
            total_pnl=50000.0,
            last_updated=datetime.now(),
        )

        self._mock_positions = [
            Position(
                symbol="00700",
                market=Market.HK,
                quantity=500,
                avg_cost=320.0,
                current_price=368.50,
                market_value=184250.0,
                unrealized_pnl=24250.0,
                unrealized_pnl_percent=15.16,
                currency=Currency.HKD,
                company_name="腾讯控股",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="09988",
                market=Market.HK,
                quantity=300,
                avg_cost=75.0,
                current_price=88.50,
                market_value=26550.0,
                unrealized_pnl=4050.0,
                unrealized_pnl_percent=18.0,
                currency=Currency.HKD,
                company_name="阿里巴巴-SW",
                sector="Consumer Cyclical",
                last_updated=datetime.now(),
            ),
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
        ]

        self._mock_trades = [
            Trade(
                symbol="00700",
                market=Market.HK,
                action=TradeAction.BUY,
                quantity=500,
                price=320.00,
                commission=50.0,
                currency=Currency.HKD,
                trade_time=datetime.now() - timedelta(days=200),
                order_id="FT-MOCK-001",
                execution_id="FT-EXEC-001",
            ),
            Trade(
                symbol="09988",
                market=Market.HK,
                action=TradeAction.BUY,
                quantity=300,
                price=75.00,
                commission=30.0,
                currency=Currency.HKD,
                trade_time=datetime.now() - timedelta(days=130),
                order_id="FT-MOCK-002",
                execution_id="FT-EXEC-002",
            ),
            Trade(
                symbol="AAPL",
                market=Market.US,
                action=TradeAction.BUY,
                quantity=100,
                price=165.00,
                commission=2.0,
                currency=Currency.USD,
                trade_time=datetime.now() - timedelta(days=75),
                order_id="FT-MOCK-003",
                execution_id="FT-EXEC-003",
            ),
            Trade(
                symbol="00700",
                market=Market.HK,
                action=TradeAction.SELL,
                quantity=100,
                price=370.00,
                commission=50.0,
                currency=Currency.HKD,
                trade_time=datetime.now() - timedelta(days=15),
                order_id="FT-MOCK-004",
                execution_id="FT-EXEC-004",
                realized_pnl=5000.0,
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
