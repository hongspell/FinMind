"""
FinMind - 券商适配器基类

定义所有券商适配器的统一接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


# =============================================================================
# 异常类
# =============================================================================

class BrokerError(Exception):
    """券商操作基础异常"""
    pass


class AuthenticationError(BrokerError):
    """认证失败"""
    pass


class ConnectionError(BrokerError):
    """连接失败"""
    pass


class RateLimitError(BrokerError):
    """请求频率限制"""
    pass


# =============================================================================
# 数据类
# =============================================================================

class Currency(str, Enum):
    """货币类型"""
    USD = "USD"
    HKD = "HKD"
    CNY = "CNY"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


class Market(str, Enum):
    """市场类型"""
    US = "US"           # 美股
    HK = "HK"           # 港股
    CN = "CN"           # A股
    SG = "SG"           # 新加坡
    JP = "JP"           # 日本


class PositionSide(str, Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class BrokerConfig:
    """券商配置"""
    broker_type: str                    # ibkr, futu, tiger
    # IBKR 配置
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497               # TWS: 7497, IB Gateway: 4001
    ibkr_client_id: int = 1
    # Futu 配置
    futu_host: str = "127.0.0.1"
    futu_port: int = 11111
    futu_rsa_path: Optional[str] = None
    futu_trade_password: Optional[str] = None
    # Tiger 配置
    tiger_id: Optional[str] = None
    tiger_account: Optional[str] = None
    tiger_private_key_path: Optional[str] = None
    # 通用配置
    timeout: int = 30
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（隐藏敏感信息）"""
        return {
            "broker_type": self.broker_type,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        }


@dataclass
class Position:
    """持仓信息"""
    symbol: str                         # 股票代码
    market: Market                      # 市场
    quantity: float                     # 持仓数量
    avg_cost: float                     # 平均成本
    current_price: float                # 当前价格
    market_value: float                 # 市值
    unrealized_pnl: float               # 未实现盈亏
    unrealized_pnl_percent: float       # 未实现盈亏百分比
    realized_pnl: float = 0.0           # 已实现盈亏
    side: PositionSide = PositionSide.LONG
    currency: Currency = Currency.USD
    # 可选字段
    company_name: Optional[str] = None
    sector: Optional[str] = None
    last_updated: Optional[datetime] = None

    @property
    def cost_basis(self) -> float:
        """总成本"""
        return self.quantity * self.avg_cost

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "market": self.market.value,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "realized_pnl": self.realized_pnl,
            "side": self.side.value,
            "currency": self.currency.value,
            "company_name": self.company_name,
            "sector": self.sector,
            "cost_basis": self.cost_basis,
        }


@dataclass
class AccountBalance:
    """账户余额"""
    total_assets: float                 # 总资产
    cash: float                         # 现金
    market_value: float                 # 持仓市值
    buying_power: float                 # 购买力
    currency: Currency = Currency.USD
    # 盈亏
    day_pnl: float = 0.0                # 当日盈亏
    total_pnl: float = 0.0              # 总盈亏
    # 保证金（如适用）
    margin_used: float = 0.0
    margin_available: float = 0.0
    maintenance_margin: float = 0.0
    # 时间戳
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_assets": self.total_assets,
            "cash": self.cash,
            "market_value": self.market_value,
            "buying_power": self.buying_power,
            "currency": self.currency.value,
            "day_pnl": self.day_pnl,
            "total_pnl": self.total_pnl,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
        }


@dataclass
class PortfolioSummary:
    """投资组合摘要"""
    broker: str                         # 券商名称
    account_id: str                     # 账户ID
    balance: AccountBalance             # 账户余额
    positions: List[Position]           # 持仓列表
    position_count: int = 0             # 持仓数量
    # 分析数据
    top_holdings: List[Dict] = field(default_factory=list)  # 前N大持仓
    sector_allocation: Dict[str, float] = field(default_factory=dict)  # 行业分布
    market_allocation: Dict[str, float] = field(default_factory=dict)  # 市场分布
    currency_exposure: Dict[str, float] = field(default_factory=dict)  # 货币敞口
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        self.position_count = len(self.positions)
        self._calculate_allocations()

    def _calculate_allocations(self):
        """计算各种分配比例"""
        if not self.positions or self.balance.total_assets == 0:
            return

        total = self.balance.total_assets

        # 前N大持仓
        sorted_positions = sorted(
            self.positions,
            key=lambda p: p.market_value,
            reverse=True
        )
        self.top_holdings = [
            {
                "symbol": p.symbol,
                "market_value": p.market_value,
                "weight": p.market_value / total * 100,
                "pnl_percent": p.unrealized_pnl_percent,
            }
            for p in sorted_positions[:10]
        ]

        # 市场分布
        market_values: Dict[str, float] = {}
        for p in self.positions:
            market = p.market.value
            market_values[market] = market_values.get(market, 0) + p.market_value
        self.market_allocation = {
            k: v / total * 100 for k, v in market_values.items()
        }

        # 货币敞口
        currency_values: Dict[str, float] = {}
        for p in self.positions:
            curr = p.currency.value
            currency_values[curr] = currency_values.get(curr, 0) + p.market_value
        self.currency_exposure = {
            k: v / total * 100 for k, v in currency_values.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "broker": self.broker,
            "account_id": self.account_id,
            "balance": self.balance.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "position_count": self.position_count,
            "top_holdings": self.top_holdings,
            "market_allocation": self.market_allocation,
            "currency_exposure": self.currency_exposure,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class TradeAction(str, Enum):
    """交易动作"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """交易记录"""
    symbol: str                              # 股票代码
    market: Market                           # 市场
    action: TradeAction                      # 买入/卖出
    quantity: float                          # 数量
    price: float                             # 成交价格
    commission: float = 0.0                  # 佣金
    currency: Currency = Currency.USD        # 货币
    trade_time: Optional[datetime] = None    # 成交时间
    order_id: Optional[str] = None           # 订单ID
    execution_id: Optional[str] = None       # 成交ID
    realized_pnl: Optional[float] = None     # 实现盈亏（卖出时）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "market": self.market.value,
            "action": self.action.value,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "currency": self.currency.value,
            "trade_time": self.trade_time.isoformat() if self.trade_time else None,
            "order_id": self.order_id,
            "execution_id": self.execution_id,
            "realized_pnl": self.realized_pnl,
            "total_value": self.quantity * self.price,
        }


# =============================================================================
# 抽象基类
# =============================================================================

class BrokerAdapter(ABC):
    """
    券商适配器抽象基类

    所有券商适配器必须实现这个接口。
    仅提供只读功能，不支持交易。
    """

    _CURRENCY_MAP: Dict[str, 'Currency'] = {
        "USD": Currency.USD,
        "HKD": Currency.HKD,
        "CNY": Currency.CNY,
        "CNH": Currency.CNY,
        "EUR": Currency.EUR,
        "GBP": Currency.GBP,
        "JPY": Currency.JPY,
    }

    def __init__(self, config: BrokerConfig):
        self.config = config
        self._connected = False
        self._account_id: Optional[str] = None

    @staticmethod
    def _parse_currency(currency_str: str, default: 'Currency' = Currency.USD) -> 'Currency':
        """解析货币字符串为 Currency 枚举"""
        return BrokerAdapter._CURRENCY_MAP.get(currency_str.upper(), default) if currency_str else default

    @property
    def broker_name(self) -> str:
        """券商名称"""
        return self.config.broker_type.upper()

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    @property
    def account_id(self) -> Optional[str]:
        """账户ID"""
        return self._account_id

    # -------------------------------------------------------------------------
    # 连接管理
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到券商API

        Returns:
            bool: 连接是否成功

        Raises:
            AuthenticationError: 认证失败
            ConnectionError: 连接失败
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 连接是否健康
        """
        pass

    # -------------------------------------------------------------------------
    # 账户信息（只读）
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_account_balance(self) -> AccountBalance:
        """
        获取账户余额

        Returns:
            AccountBalance: 账户余额信息

        Raises:
            BrokerError: 获取失败
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        获取持仓列表

        Returns:
            List[Position]: 持仓列表

        Raises:
            BrokerError: 获取失败
        """
        pass

    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取单个持仓

        Args:
            symbol: 股票代码

        Returns:
            Position | None: 持仓信息，不存在返回None
        """
        positions = await self.get_positions()
        for p in positions:
            if p.symbol.upper() == symbol.upper():
                return p
        return None

    async def get_portfolio_summary(self) -> PortfolioSummary:
        """
        获取投资组合摘要

        Returns:
            PortfolioSummary: 投资组合摘要
        """
        balance = await self.get_account_balance()
        positions = await self.get_positions()

        return PortfolioSummary(
            broker=self.broker_name,
            account_id=self._account_id or "unknown",
            balance=balance,
            positions=positions,
            last_updated=datetime.now(),
        )

    async def get_trades(self, days: int = 7) -> List['Trade']:
        """
        获取交易历史

        Args:
            days: 获取最近多少天的交易记录，默认7天

        Returns:
            List[Trade]: 交易记录列表
        """
        # 默认返回空列表，子类可以覆盖实现
        return []

    # -------------------------------------------------------------------------
    # 上下文管理器
    # -------------------------------------------------------------------------

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
