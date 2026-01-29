"""
FinMind - 老虎证券适配器

使用 Tiger Open API 连接到老虎证券。

需要安装: pip install tigeropen

使用前需要：
1. 在老虎证券开发者平台注册应用: https://quant.itigerup.com
2. 获取 Tiger ID 和私钥
3. 配置账户信息

文档: https://quant.itiger.com/openapi/py-docs/zh-cn/
"""

import asyncio
from datetime import datetime
from typing import Optional, List
import logging

from .base import (
    BrokerAdapter,
    BrokerConfig,
    Position,
    AccountBalance,
    BrokerError,
    AuthenticationError,
    ConnectionError,
    Market,
    Currency,
    PositionSide,
)

logger = logging.getLogger(__name__)


class TigerAdapter(BrokerAdapter):
    """
    老虎证券 Open API 适配器

    使用 tigeropen 库连接到老虎证券。

    Example:
        ```python
        config = BrokerConfig(
            broker_type="tiger",
            tiger_id="your_tiger_id",
            tiger_account="your_account",
            tiger_private_key_path="/path/to/private_key.pem",
        )

        async with TigerAdapter(config) as broker:
            balance = await broker.get_account_balance()
            positions = await broker.get_positions()
        ```
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._client = None
        self._trade_client = None

    @property
    def broker_name(self) -> str:
        return "Tiger"

    async def connect(self) -> bool:
        """连接到老虎证券"""
        try:
            # 延迟导入
            from tigeropen.common.consts import Language
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.trade.trade_client import TradeClient
            from tigeropen.quote.quote_client import QuoteClient

            # 验证配置
            if not self.config.tiger_id:
                raise AuthenticationError("Tiger ID is required")
            if not self.config.tiger_private_key_path:
                raise AuthenticationError("Tiger private key path is required")

            # 创建配置
            client_config = TigerOpenClientConfig()
            client_config.tiger_id = self.config.tiger_id
            client_config.account = self.config.tiger_account
            client_config.private_key = self._read_private_key(
                self.config.tiger_private_key_path
            )
            client_config.language = Language.zh_CN
            client_config.timeout = self.config.timeout

            # 创建交易客户端
            self._trade_client = TradeClient(client_config)

            # 获取账户信息验证连接
            accounts = self._trade_client.get_managed_accounts()
            if accounts:
                self._account_id = accounts[0]
                logger.info(f"Connected to Tiger, account: {self._account_id}")
            else:
                raise AuthenticationError("No managed accounts found")

            self._connected = True
            return True

        except ImportError:
            raise BrokerError(
                "tigeropen library not installed. "
                "Install with: pip install tigeropen"
            )
        except FileNotFoundError:
            raise AuthenticationError(
                f"Private key file not found: {self.config.tiger_private_key_path}"
            )
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "signature" in error_msg.lower():
                raise AuthenticationError(f"Tiger authentication failed: {error_msg}")
            raise ConnectionError(f"Tiger connection failed: {error_msg}")

    def _read_private_key(self, path: str) -> str:
        """读取私钥文件"""
        with open(path, 'r') as f:
            return f.read()

    async def disconnect(self) -> None:
        """断开连接"""
        self._trade_client = None
        self._connected = False
        logger.info("Disconnected from Tiger")

    async def health_check(self) -> bool:
        """健康检查"""
        if not self._trade_client:
            return False
        try:
            accounts = self._trade_client.get_managed_accounts()
            return len(accounts) > 0
        except Exception:
            return False

    async def get_account_balance(self) -> AccountBalance:
        """获取账户余额"""
        if not self._connected or not self._trade_client:
            raise BrokerError("Not connected to Tiger")

        try:
            # 获取账户资产
            assets = self._trade_client.get_assets()

            if not assets:
                raise BrokerError("No asset data returned")

            # 获取美元账户（主要账户）
            asset = None
            for a in assets:
                if a.currency == 'USD':
                    asset = a
                    break
            if not asset:
                asset = assets[0]

            # 确定货币
            currency = Currency.USD
            if asset.currency == 'HKD':
                currency = Currency.HKD
            elif asset.currency == 'CNH':
                currency = Currency.CNY

            return AccountBalance(
                total_assets=float(asset.summary.net_liquidation or 0),
                cash=float(asset.summary.cash or 0),
                market_value=float(asset.summary.gross_position_value or 0),
                buying_power=float(asset.summary.buying_power or 0),
                currency=currency,
                day_pnl=float(asset.summary.realized_pnl or 0),
                total_pnl=float(asset.summary.unrealized_pnl or 0),
                margin_used=float(asset.summary.maintain_margin_req or 0),
                margin_available=float(asset.summary.excess_liquidity or 0),
                maintenance_margin=float(asset.summary.maintain_margin_req or 0),
                last_updated=datetime.now(),
            )

        except Exception as e:
            raise BrokerError(f"Failed to get account balance: {e}")

    async def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        if not self._connected or not self._trade_client:
            raise BrokerError("Not connected to Tiger")

        try:
            # 获取持仓
            tiger_positions = self._trade_client.get_positions()

            positions = []
            for pos in tiger_positions:
                symbol = pos.contract.symbol
                market = self._get_market(pos.contract.market)

                # 确定货币
                currency = Currency.USD
                if pos.contract.currency == 'HKD':
                    currency = Currency.HKD
                elif pos.contract.currency == 'CNH':
                    currency = Currency.CNY

                quantity = float(pos.quantity or 0)
                avg_cost = float(pos.average_cost or 0)
                current_price = float(pos.market_price or avg_cost)
                market_value = float(pos.market_value or 0)
                unrealized_pnl = float(pos.unrealized_pnl or 0)

                # 计算盈亏百分比
                cost_basis = quantity * avg_cost
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
                    realized_pnl=float(pos.realized_pnl or 0),
                    side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                    currency=currency,
                    last_updated=datetime.now(),
                ))

            return positions

        except Exception as e:
            raise BrokerError(f"Failed to get positions: {e}")

    def _get_market(self, market_str: str) -> Market:
        """转换老虎市场标识"""
        market_upper = market_str.upper() if market_str else ""

        if market_upper in ('US', 'NYSE', 'NASDAQ', 'AMEX'):
            return Market.US
        elif market_upper in ('HK', 'HKEX'):
            return Market.HK
        elif market_upper in ('CN', 'SH', 'SZ'):
            return Market.CN
        elif market_upper == 'SG':
            return Market.SG

        return Market.US  # 默认美股


# =============================================================================
# 模拟适配器（用于测试）
# =============================================================================

class TigerMockAdapter(BrokerAdapter):
    """
    老虎证券模拟适配器

    用于测试和开发。
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        self._mock_positions = []
        self._mock_balance = None

    @property
    def broker_name(self) -> str:
        return "Tiger (Mock)"

    async def connect(self) -> bool:
        """模拟连接"""
        self._connected = True
        self._account_id = "20000000"
        self._setup_mock_data()
        logger.info("Connected to Tiger (Mock)")
        return True

    async def disconnect(self) -> None:
        """模拟断开"""
        self._connected = False
        logger.info("Disconnected from Tiger (Mock)")

    async def health_check(self) -> bool:
        """健康检查"""
        return self._connected

    def _setup_mock_data(self):
        """设置模拟数据"""
        self._mock_balance = AccountBalance(
            total_assets=200000.0,
            cash=50000.0,
            market_value=150000.0,
            buying_power=80000.0,
            currency=Currency.USD,
            day_pnl=2500.0,
            total_pnl=25000.0,
            last_updated=datetime.now(),
        )

        self._mock_positions = [
            Position(
                symbol="AAPL",
                market=Market.US,
                quantity=150,
                avg_cost=160.0,
                current_price=178.50,
                market_value=26775.0,
                unrealized_pnl=2775.0,
                unrealized_pnl_percent=11.56,
                currency=Currency.USD,
                company_name="Apple Inc.",
                sector="Technology",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="AMZN",
                market=Market.US,
                quantity=30,
                avg_cost=165.0,
                current_price=185.50,
                market_value=5565.0,
                unrealized_pnl=615.0,
                unrealized_pnl_percent=12.42,
                currency=Currency.USD,
                company_name="Amazon.com, Inc.",
                sector="Consumer Cyclical",
                last_updated=datetime.now(),
            ),
            Position(
                symbol="META",
                market=Market.US,
                quantity=50,
                avg_cost=380.0,
                current_price=505.25,
                market_value=25262.50,
                unrealized_pnl=6262.50,
                unrealized_pnl_percent=32.96,
                currency=Currency.USD,
                company_name="Meta Platforms, Inc.",
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
